import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")

# Local LLM name via Ollama
MODEL_NAME = "BanglaGPT"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
