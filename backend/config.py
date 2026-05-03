import os
from dotenv import load_dotenv

load_dotenv()

# --- paths ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
UPLOAD_DIR  = os.getenv("UPLOAD_DIR", "data/uploaded_docs")

# sentence-transformers model; no API key required
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- chunking ---
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# --- llm ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- api ---
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
]

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set. Check your .env file.")
