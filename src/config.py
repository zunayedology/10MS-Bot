import os
from pathlib import Path

# Set project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")
MODEL_NAME = "bigscience/bloom-560m"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 2