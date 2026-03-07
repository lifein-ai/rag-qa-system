import hashlib
import pickle
import faiss

def calculate_file_hash(file_path):
    """计算文档哈希值"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def save_faiss_index(index, filename):
    faiss.write_index(index, filename)

def load_faiss_index(filename):
    return faiss.read_index(filename)

def save_paragraphs(paragraphs_with_metadata, filename):
    with open(filename, "wb") as f:
        pickle.dump(paragraphs_with_metadata, f)

def load_paragraphs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)