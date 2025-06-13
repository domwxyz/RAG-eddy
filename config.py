import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VERSION = "1.0.0"

# Base paths
BASE_DIR = Path(__file__).parent
ARCHIVE_DIR = BASE_DIR / "archive"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"

# Create directories
ARCHIVE_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.parent.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
# Default to Qwen 2.5 3B - a good balance of performance and size

DEFAULT_LLM_URL = "https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf"

# Alternative models (uncomment to use):
# DEFAULT_LLM_URL = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# DEFAULT_LLM_URL = "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"  # Smaller, faster
# DEFAULT_LLM_URL = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # Larger, better

DEFAULT_EMBEDDING_URL = "BAAI/bge-m3"

LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_URL)

# LLM settings
NUM_THREADS = int(os.getenv("NUM_THREADS", os.cpu_count() or 4))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

# Vector store settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# System prompt
SYSTEM_PROMPT = """You are RAG-eddy, a helpful assistant that answers questions based on the provided documents. 
Your responses should be:
- Accurate and based on the document content
- Clear and easy to understand
- Concise but complete
- Friendly and conversational

If you cannot find the answer in the provided context, be honest about it and suggest what information might be helpful."""

if __name__ == "__main__":
    print(f"RAG-eddy v{VERSION} Configuration")
