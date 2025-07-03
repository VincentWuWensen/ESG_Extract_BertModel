#pip install pdfplumber sentence_transformers jieba rank_bm25 openpyxl
import os
import pandas as pd
import pdfplumber
import torch
import jieba
import openpyxl
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rank_bm25 import BM25Okapi

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

#path
DESKTOP_PATH = os.path.join(os.path.expanduser('~'), '/openbayes/home')
EXCEL_PATH = os.path.join(DESKTOP_PATH, 'Goal.xlsx')
PDF_DIR = os.path.join(DESKTOP_PATH, 'ESG_Reports')
df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
keywords = df.columns[1:].tolist()
print(f"Keywords: {keywords}")

# Qwen model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    return_full_text=False
)


#Process PDF
def process_pdf(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:
        chunks = []
        current_chunk = ""

        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            paragraphs = re.split(r'\n{2,}', text)

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                if len(paragraph) > 1000:
                    sentences = re.split(r'(?<=[。！？；])', paragraph)
                    current_sentences = []
                    current_length = 0

                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue

                        sent_length = len(sent)
                        if current_length + sent_length <= 1000:
                            current_sentences.append(sent)
                            current_length += sent_length
                        else:
                            if current_sentences:
                                chunks.append("".join(current_sentences))
                            current_sentences = [sent]
                            current_length = sent_length

                    if current_sentences:
                        chunks.append("".join(current_sentences))
                else:
                    chunks.append(paragraph)

    return chunks

#BM25
def bm25_search(query, chunks, top_k=10):
    tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

#extract_with_qwen by learning samples
def extract_with_qwen(keyword, context_chunks, example_content=None):

    prompt = """
<task>
  作为专业ESG分析师，请从公司ESG报告中提取与特定关键词直接相关的内容和数据，
  请返回JSON格式结果，包含至少以下三个字段，但可以拓展为与对应示例中的类似结构的回答（例如是或者否）：
1. "关键词" - 原始查询关键词
2. "关键句子" - 原文中最相关的1-3个关键句子原句
3. "关键数值" - 关键指标数值（若有）
</task>

<rules>
  1. 仅返回原文中最相关的1-3个关键句子原句
  2. 如果示范案例为int数值结果，返回1-3个最相关的关键数值
  3. 禁止添加任何解释、总结或额外文字，也不需要提供进一步的解释分析与提醒
  4. 若无相关信息请返回固定短语：'未找到相关信息'
  5. 选择包含完整信息的连续句子
</rules>
"""
    if example_content:
        prompt += f"""
<example>
  关键词：{keyword}
  文本内容：{example_content}
  有效输出："对应示例中的实际引用句子"  # 实际使用时应替换为真实示例输出
</example>
"""

    prompt += f"""
<context>
{" ".join(context_chunks[:5])}
</context>

<keyword>{keyword}</keyword>

<output>
  请基于上述要求和上下文返回提取结果：
</output>
"""

    try:
        response = generator(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )[0]['generated_text']

        output_start = response.find("<output>") + len("<output>")
        output_end = response.find("</output>", output_start)

        if output_end == -1:
            return "未找到相关信息" if not response.strip() else response

        output = response[output_start:output_end].strip()

        if not output or "未找到相关信息" in output or len(output) < 15:
            return "未找到相关信息"

        # 清理并截取关键句子
        sentences = re.split(r'[。！？]', output)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return '。'.join(valid_sentences[:3]) + '。' if valid_sentences else output

    except Exception as e:
        print(f"生成模型出错: {str(e)}")
        return "模型处理失败"

#Sample
example_row = df.iloc[0]
company = example_row['Company']
print(f"\nSample: {company}")

example_mapping = {}
for keyword in keywords:
    example_content = example_row[keyword]
    example_mapping[keyword] = example_content

    content_str = str(example_content) if not pd.isna(example_content) else ""
    preview = content_str[:100] + '...' if len(content_str) > 100 else content_str
    print(f"关键词: {keyword} | 示范内容: {preview}")

print("\n示范加载完成")

#Process
if len(df) > 1:
    target_index = 1
    target_row = df.iloc[target_index]
    company = target_row['Company']

    if pd.isna(company):
        print("名称为空，跳过处理")
    else:
        print(f"\n处理目标公司: {company}")
        pdf_path = os.path.join(PDF_DIR, f"{company}_ESG.pdf")

        if not os.path.exists(pdf_path):
            print(f"报告不存在: {pdf_path}")
            for keyword in keywords:
                df.at[target_index, keyword] = "报告缺失"
        else:
            try:
                chunks = process_pdf(pdf_path)
                print(f"提取文本块数: {len(chunks)}")

                for keyword in keywords:
                    print(f"处理关键词: {keyword}")

                    example_content = example_mapping.get(keyword, "")

                    candidate_chunks = bm25_search(keyword, chunks, top_k=10)

                    if not candidate_chunks:
                        print(f"未找到相关文本")
                        df.at[target_index, keyword] = "未找到相关信息"
                        continue

                    result = extract_with_qwen(keyword, candidate_chunks, example_content)

                    preview = result[:100] + '...' if len(result) > 100 else result
                    print(f"结果: {preview}")

                    df.at[target_index, keyword] = result

                print("目标公司处理完成")
            except Exception as e:
                print(f"处理出错: {str(e)}")
                for keyword in keywords:
                    df.at[target_index, keyword] = "处理错误"
else:
    print("Excel中没有第二行需要处理")

print("\n表格更新:")
print(df)

output_path = os.path.join(DESKTOP_PATH, "ESG_Results.xlsx")
df.to_excel(output_path, index=False)
print(f"保存成功: {output_path}")

