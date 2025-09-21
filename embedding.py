import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
import chromadb


CHROMA_DB_PATH = "/home/sa/bar-exam-housing/processed_data/passage/chroma_db_large2/"
COLLECTION_NAME = "legal_docs_large_collection2"
MODEL_NAME = "/home/sa/bar-exam-housing/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




class ChromaVectorSearchEngine:

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    def search(self, query_str: str, top_k: int = 1) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            query_str, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]  # 直接包含文档内容
        )

        if not results['ids'][0]:
            return []

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]
            })

        return formatted_results




class SystemInitializer:
    def __init__(self):
        print("=" * 20 + " 检索系统初始化开始 " + "=" * 20)
        self.collection = self._connect_to_chromadb()
        self.engine = self._initialize_engine()
        print("=" * 20 + " ✅ 检索系统初始化完成 " + "=" * 20 + "\n")

    def _connect_to_chromadb(self):
        print("--- 正在连接到 ChromaDB... ---")
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(f"错误: 数据库路径 '{CHROMA_DB_PATH}' 不存在。请先运行新的建库脚本。")

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ 成功连接到集合'{COLLECTION_NAME}'，其中包含 {collection.count()} 个文本块。\n")
            return collection
        except Exception as e:
            raise ValueError(f"错误: 无法获取集合 '{COLLECTION_NAME}'。错误信息: {e}")

    def _initialize_engine(self) -> ChromaVectorSearchEngine:
        print("--- 正在加载查询模型... ---")
        query_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        engine = ChromaVectorSearchEngine(
            self.collection,
            query_model
        )
        print("✅ 查询模型加载完成。\n")
        return engine



try:
    print("正在初始化检索系统模块...")
    _system_instance = SystemInitializer()
except Exception as e:
    print(f"致命错误：检索系统初始化失败！错误信息: {e}")
    _system_instance = None


def query(query_text: str, top_k: int = 1, **kwargs) -> List[Dict[str, Any]]:
    if _system_instance is None:
        print("错误：检索系统未被成功初始化，无法执行查询。")
        return []

    if not query_text or not isinstance(query_text, str):
        print("错误：查询文本无效。")
        return []

    return _system_instance.engine.search(query_str=query_text, top_k=top_k)
