
#pip install pdfplumber sentence_transformers jieba rank_bm25 openpyxl
import pandas as pd
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import jieba
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

#path
DESKTOP_PATH = os.path.join(os.path.expanduser('~'), '/openbayes/home')
EXCEL_PATH = os.path.join(DESKTOP_PATH, 'Goal.xlsx')
PDF_DIR = os.path.join(DESKTOP_PATH, 'ESG_Reports')
df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
keywords = df.columns[1:].tolist()
print(f"Keywords: {keywords}")


#Embeding model
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
#Rerank model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#encoder
def encode_texts(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

keyword_embeddings = encode_texts(keywords)
keyword_embeddings.shape


#Process PDF
def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        chunks = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                chunks.extend(paragraphs)
    return chunks


#Search Top k contents
def semantic_search(query_emb, doc_embeddings, top_k=5):

    query_emb = query_emb / np.linalg.norm(query_emb)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    similarities = np.dot(doc_embeddings, query_emb)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]


#BM25
def bm25_search(query, chunks, top_k=10):
    tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in top_indices]


#Rerank
def rerank_results(query, candidates):
    pairs = [(query, cand) for cand, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = [(candidates[i][0], scores[i]) for i in np.argsort(scores)[::-1]]
    return reranked[:5]


#Main model
for index, row in df.iterrows():
    company = row['Company']
    if pd.isna(company):
        continue

    print(f"\nCompany name: {company}")
    pdf_path = os.path.join(PDF_DIR, f"{company}_ESG.pdf")

    if not os.path.exists(pdf_path):
        print(f"No found: {pdf_path}")
        continue

    chunks = process_pdf(pdf_path)
    print(f"Chunk number: {len(chunks)}")

    chunk_embeddings = encode_texts(chunks)

    for keyword in keywords:
        #BM_25
        bm25_results = bm25_search(keyword, chunks, top_k=20)

        #embeddings
        keyword_emb = keyword_embeddings[keywords.index(keyword)]
        semantic_results = semantic_search(keyword_emb, chunk_embeddings, top_k=20)

        combined_results = bm25_results + semantic_results
        unique_results = {chunk: score for chunk, score in combined_results}
        candidates = list(unique_results.items())

        # Rerank
        if candidates:
            reranked_results = rerank_results(keyword, candidates)

            best_chunk, best_score = reranked_results[0]
            print(f"keyword: {keyword} | Confidence level: {best_score:.4f}")

            df.at[index, keyword] = best_chunk[:500]
        else:
            print(f"No found: {keyword}")
            df.at[index, keyword] = "No found"


print("\nFinished table:")
print(df)


