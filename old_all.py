import hashlib
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# ===== 1) Embedding + FAISS =====
# 加载预训练模型
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 计算文档的哈希值
def calculate_file_hash(file_path):
    """计算文档的哈希值"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()  # 返回文件的哈希值（32个字符的十六进制字符串）

# 保存 FAISS 索引
def save_faiss_index(index, filename):
    faiss.write_index(index, filename)

# 加载 FAISS 索引
def load_faiss_index(filename):
    return faiss.read_index(filename)

# 保存段落
def save_paragraphs(paragraphs, filename):
    with open(filename, "wb") as f:
        pickle.dump(paragraphs, f)

# 加载段落
def load_paragraphs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# 处理文档，生成索引并保存
def process_document(document_path, data_dir="data"):
    # 在 data 目录下定义两个子文件夹
    faiss_index_dir = os.path.join(data_dir, "faiss_indexes")
    paragraphs_dir = os.path.join(data_dir, "paragraphs")

    # 确保目录存在，如果不存在则创建
    os.makedirs(faiss_index_dir, exist_ok=True)
    os.makedirs(paragraphs_dir, exist_ok=True)

    # 计算文档的哈希值
    doc_hash = calculate_file_hash(document_path)
    faiss_index_file = os.path.join(faiss_index_dir, f"{doc_hash}_faiss.index")  # 索引文件命名为哈希值
    paragraphs_file = os.path.join(paragraphs_dir, f"{doc_hash}_paragraphs.pkl")  # 段落文件命名为哈希值

    # 检查文档的索引是否已经存在，若存在且哈希值一致，跳过处理
    if os.path.exists(faiss_index_file) and os.path.exists(paragraphs_file):
        print("Using existing FAISS index and paragraphs...")
        return load_faiss_index(faiss_index_file), load_paragraphs(paragraphs_file)

    print("Creating new FAISS index and paragraphs...")
    # 新文档处理
    with open(document_path, "r", encoding="utf-8") as f:
        document = f.read()  # 读取文档内容
    
    paragraphs = split_document(document)  # 切分文档
    vectors = embed_paragraphs(paragraphs)  # 向量化段落
    faiss_index = create_faiss_index(vectors)  # 创建 FAISS 索引

    # 保存新的索引和段落
    save_faiss_index(faiss_index, faiss_index_file)
    save_paragraphs(paragraphs, paragraphs_file)

    return faiss_index, paragraphs

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

def search_query(query: str, index, top_k=5):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]   # 返回一维数组更好用

def rag_answer(query, index, paragraphs, top_k=3, dist_threshold=1.2):
    distances, idxs = search_query(query, index, top_k=top_k)

    # 取最小距离（最相似）
    best = min(distances)
    if best > dist_threshold:
        return "不知道（检索到的内容不足以回答该问题）"

    hits = []
    for d, i in zip(distances, idxs):
        if i == -1:
            continue
        hits.append(paragraphs[i])

    context = "\n".join(hits)
    return generate_answer_openai(query, context)

# ===== 2) LLM (DeepSeek via OpenAI SDK) =====
load_dotenv(dotenv_path=r"D:\RAG\.env")  # 你之前已经验证过这样可用

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def generate_answer_openai(query: str, context: str) -> str:
    system_prompt = (
        "你是一个严谨的中文问答助手。只能依据给定【上下文】回答；"
        "如果信息，不要编造，答案尽量不超过200字。"
    )

    user_prompt = f"""【上下文】
{context}

【问题】
{query}
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ===== 3) 可运行入口 =====
def main():
    document_path = input("请输入文档路径：").strip().strip('"').strip("'")
    index, paragraphs = process_document(document_path)  # 只做一次（会自动缓存）

    print("\n文档已加载完成，可以开始提问。输入 exit 退出。\n")

    while True:
        query = input("请输入你的问题：").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        answer = rag_answer(query, index, paragraphs, top_k=3, dist_threshold=1.2)
        print("\n=== ANSWER ===")
        print(answer)
        print()

if __name__ == "__main__":
    main()