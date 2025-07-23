from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from process_data import load_and_chunk_texts
from config import EMBEDDING_MODEL, FAISS_INDEX_DIR
import os

def vectorize_and_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents = load_and_chunk_texts()
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    return db

if __name__ == "__main__":
    db = vectorize_and_store()
    print(f"FAISS index created in {FAISS_INDEX_DIR}")