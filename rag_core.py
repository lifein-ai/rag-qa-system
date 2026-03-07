import os
import faiss
from sentence_transformers import SentenceTransformer

from storage import (
    calculate_file_hash,
    save_faiss_index,
    load_faiss_index,
    save_paragraphs,
    load_paragraphs,
)
from llm_client import generate_answer_openai

# 使用本地模型路径加载模型
local_model_path = "D:/sentencetransformers"  # 这里填写你的本地模型路径
embed_model = SentenceTransformer(local_model_path)

def split_document(text: str):
    paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if p.strip()]

def embed_paragraphs(paragraphs):
    return embed_model.encode(paragraphs)

def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def process_document(document_path, data_dir="data"):
    # 定义存储 FAISS 索引和段落的文件夹路径
    faiss_index_dir = os.path.join(data_dir, "faiss_indexes")
    paragraphs_dir = os.path.join(data_dir, "paragraphs")

    # 如果目录不存在，则创建
    os.makedirs(faiss_index_dir, exist_ok=True)
    os.makedirs(paragraphs_dir, exist_ok=True)

    # 计算文档的哈希值，用来作为文件名的一部分，保证文件唯一
    doc_hash = calculate_file_hash(document_path)
    faiss_index_file = os.path.join(faiss_index_dir, f"{doc_hash}_faiss.index")  # FAISS 索引文件
    paragraphs_file = os.path.join(paragraphs_dir, f"{doc_hash}_paragraphs.pkl")  # 段落文件

    # 如果索引和段落文件存在，直接加载
    if os.path.exists(faiss_index_file) and os.path.exists(paragraphs_file):
        print("Using existing FAISS index and paragraphs...")
        return load_faiss_index(faiss_index_file), load_paragraphs(paragraphs_file)

    # 如果文件不存在，则进行新文档处理
    print("Creating new FAISS index and paragraphs...")

    # 读取文档内容
    with open(document_path, "r", encoding="utf-8") as f:
        document = f.read()

    # 文档切分成段落
    paragraphs = split_document(document)
    # 向量化段落
    vectors = embed_paragraphs(paragraphs)
    # 创建 FAISS 索引
    faiss_index = create_faiss_index(vectors)

    # 将段落转换为带有元数据的格式（包括文档来源和段落的索引）
    paragraphs_with_metadata = [
        {"text": p, "source": document_path, "chunk_id": i}  # 每个段落都包含文档路径和该段落的索引
        for i, p in enumerate(paragraphs)
    ]

    # 保存 FAISS 索引和段落数据
    save_faiss_index(faiss_index, faiss_index_file)
    save_paragraphs(paragraphs_with_metadata, paragraphs_file)

    return faiss_index, paragraphs_with_metadata

def search_query(query: str, index, top_k=5):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]


def rag_answer(query, index, paragraphs_with_metadata, top_k=3, dist_threshold=1.2):
    # 进行检索，返回最相似的 top_k 个段落
    distances, idxs = search_query(query, index, top_k=top_k)

    # 获取最小距离值，如果大于设定的阈值，则返回“不能回答”
    best = min(distances)
    if best > dist_threshold:
        return {
            "answer": "我在当前资料中没有找到足够相关的内容，建议换一种更具体的问法。",
            "hits": [],
            "empty_retrieval": True  # 如果没有检索到相关内容，标记为空检索
        }

    hits = []  # 用于存储检索到的段落
    for d, i in zip(distances, idxs):
        if i == -1:  # 如果索引无效，跳过
            continue
        hit_data = paragraphs_with_metadata[int(i)]  # 获取段落的详细信息
        hits.append({
            "chunk_id": hit_data["chunk_id"],  # 段落的索引
            "source": hit_data["source"],  # 段落所在的文档路径
            "distance": float(d),  # 距离，表示与查询的相似度
            "text": hit_data["text"]  # 段落内容
        })

    # 生成答案时使用检索到的相关上下文
    context = "\n".join([hit["text"] for hit in hits])
    answer = generate_answer_openai(query, context)

    return {
        "answer": answer,  # 返回生成的答案
        "hits": hits,  # 返回命中的段落信息
        "empty_retrieval": False  # 表示有有效的检索结果
    }

def ask(query, index, paragraphs_with_metadata, top_k=3, dist_threshold=1.2):
    return rag_answer(
        query=query,
        index=index,
        paragraphs_with_metadata=paragraphs_with_metadata,
        top_k=top_k,
        dist_threshold=dist_threshold
    )