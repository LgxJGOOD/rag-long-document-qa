"""
RAG 系统 - 运行脚本
从 hw2.ipynb 提取，直接运行生成结果文件
"""

import json
import torch
from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ========== 配置 ==========
STUDENT_ID = "202320108026"
CHUNK_SIZE = 512
STRIDE = 256
TOP_K = 15


# ========== 加载查询 ==========
def load_queries(json_path: str) -> List[str]:
    queries = []
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        for item in data:
            queries.append(item['query'])
    return queries


# ========== 加载知识库 ==========
def load_chunks(txt_path: str, chunk_size: int = CHUNK_SIZE, stride: int = STRIDE) -> List[str]:
    chunks = []
    with open(txt_path, "r", encoding="utf-8") as fp:
        passage = fp.read()

    length = len(passage)
    for i in range(0, length, stride):
        chunk = passage[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


# ========== 检索模型 ==========
def init_bm25(chunks: List[str]) -> BM25Okapi:
    tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def init_embedding_model(model_name: str = "intfloat/multilingual-e5-large") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_chunk_embeddings(chunks: List[str], embedding_model: SentenceTransformer) -> List:
    return embedding_model.encode(chunks, show_progress_bar=True)


# ========== 检索函数 ==========
def bm25_retrieve(query: str, chunks: List[str], bm25: BM25Okapi) -> List[str]:
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    rank = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in rank]


def embedding_retrieve(query: str, chunks: List[str], chunks_embeddings: List, embedding_model: SentenceTransformer) -> List[str]:
    query_embeddings = embedding_model.encode(query, prompt_name="query")
    scores = list(embedding_model.similarity(query_embeddings, chunks_embeddings)[0])
    rank = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in rank]


def reciprocal_rank_fusion(*ranked_lists, k=60) -> List[str]:
    scores = {}
    for rl in ranked_lists:
        for rank, doc_id in enumerate(rl, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in fused]


# ========== 加载 LLM ==========
def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen2.5-Coder-3B-Instruct",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
    )

    llm_pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return llm_pipe


# ========== 提示词 ==========
SYSTEM_PROMPT = """
请根据**问题**，从用户提供的**《回声》小说段落**找出与问题相关的内容，直接提供精简并准确的解答，避免与问题无关的内容。
""".strip()

USER_PROMPT = """
### 问题
{query}
### 《回声》小说段落
{relevant_chunks}
""".strip()


# ========== 主流程 ==========
def main():
    print("加载查询...")
    queries = load_queries("rag_questions.json")

    print("加载知识库...")
    chunks = load_chunks("novel_echo.txt")

    print("初始化 BM25 模型...")
    bm25 = init_bm25(chunks)

    print("加载向量模型...")
    embedding_model = init_embedding_model()
    print("计算 chunk 向量...")
    chunks_embeddings = get_chunk_embeddings(chunks, embedding_model)

    print("加载 LLM...")
    llm_pipe = load_llm()

    print("开始生成回答...")
    responses = []
    for i, query in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] 处理中...")

        relevant_chunks_bm25 = bm25_retrieve(query=query, chunks=chunks, bm25=bm25)
        relevant_chunks_embedding = embedding_retrieve(
            query=query,
            chunks=chunks,
            chunks_embeddings=chunks_embeddings,
            embedding_model=embedding_model
        )
        relevant_chunks = "\n\n".join(reciprocal_rank_fusion(relevant_chunks_bm25, relevant_chunks_embedding)[:TOP_K])

        chats = [
            {'role': "system", 'content': SYSTEM_PROMPT},
            {'role': "user", 'content': USER_PROMPT.format(query=query, relevant_chunks=relevant_chunks)},
        ]
        response = llm_pipe(chats)[0]['generated_text'][-1]['content'].strip()
        responses.append(response[:512])
        print(f"  回答: {response[:100]}...")

    # 保存结果
    output_file = f"hw2-result-{STUDENT_ID}.json"
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(responses, fp, ensure_ascii=False, indent=2)

    print(f"\n完成！结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
