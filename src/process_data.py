from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from config import TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text):
    return text.strip().replace("\n", " ").replace("  ", " ")

def load_and_chunk_texts():
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for text_file in os.listdir(TEXT_DIR):
        if text_file.endswith(".txt"):
            with open(os.path.join(TEXT_DIR, text_file), "r", encoding="utf-8") as f:
                book_text = f.read()
            cleaned_text = clean_text(book_text)
            docs = text_splitter.create_documents([cleaned_text], metadatas=[{"source": text_file}])
            documents.extend(docs)
    return documents

if __name__ == "__main__":
    docs = load_and_chunk_texts()
    print(f"Created {len(docs)} chunks")