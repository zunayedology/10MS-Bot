from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os
from config import TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text):
    return text.strip().replace("\n", " ").replace("  ", " ")

def load_and_chunk_texts():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["।", "!", "?", "\n", " "],
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )

    documents = []
    for file in os.listdir(TEXT_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_DIR, file), "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                chunks = splitter.create_documents([text], metadatas=[{"source": file}])
                documents.extend(chunks)
    return documents

if __name__ == "__main__":
    docs = load_and_chunk_texts()
    print(f"Created {len(docs)} chunks.")
