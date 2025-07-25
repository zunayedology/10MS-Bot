from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from process_data import load_and_chunk_texts
from config import EMBEDDING_MODEL, FAISS_INDEX_DIR
import os

def vectorize_and_store():
    try:
        documents = load_and_chunk_texts()
        if not documents:
            raise ValueError("No documents to vectorize.")
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"}
        )
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(FAISS_INDEX_DIR)
        print(f"FAISS index saved to {FAISS_INDEX_DIR}")
        return db
    except Exception as e:
        print(f"Vectorization error: {str(e)}")

if __name__ == "__main__":
    vectorize_and_store()