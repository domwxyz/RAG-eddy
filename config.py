import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
ARCHIVE_DIR = BASE_DIR / "archive"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"

# Create directories
ARCHIVE_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
DEFAULT_LLM_URL = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# LLM settings
NUM_THREADS = int(os.getenv("NUM_THREADS", 4))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))

# Vector store settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents. 
Be concise and accurate in your responses. If the answer isn't in the documents, say so."""