from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from .text_utils import chunk_text

class VectorIndex:
    def __init__(self, store: FAISS):
        self.store = store

    @classmethod
    def from_texts(cls, texts: List[str], chunk_size: int = 800, chunk_overlap: int = 100):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        store = FAISS.from_texts(chunks, embedding)
        return cls(store)

    def retrieve(self, query: str, k: int = 4):
        retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(8, k*2)}
        )
        return retriever.get_relevant_documents(query)
