import os

# RAG Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Search Settings
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.5

# Generation Settings
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000
TEMPERATURE = 0.3

# Paths
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdf")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_store")
