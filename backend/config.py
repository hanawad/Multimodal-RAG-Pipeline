import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = str(BASE_DIR / "data" / "uploads")
FAISS_INDEX_PATH = str(BASE_DIR / "storage" / "faiss_index")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-20b:free"
TEMPERATURE = 0