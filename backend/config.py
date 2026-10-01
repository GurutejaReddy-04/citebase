"""Application configuration loaded from environment variables (.env)."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- paths ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
UPLOAD_DIR  = os.getenv("UPLOAD_DIR", "data/uploaded_docs")

# sentence-transformers model; no API key required
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- chunking ---
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", 60))
MIN_CHUNK_SIZE      = int(os.getenv("MIN_CHUNK_SIZE", 20))
INCLUDE_BREADCRUMBS = os.getenv("INCLUDE_BREADCRUMBS", "true").lower() in ("true", "1", "yes")

# --- retrieval & reranking ---
TOP_K_RESULTS       = int(os.getenv("TOP_K_RESULTS", 4))
INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", 10))
ENABLE_RERANKER     = os.getenv("ENABLE_RERANKER", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL      = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K      = int(os.getenv("RERANKER_TOP_K", 4))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.20"))

# --- web search fallback ---
ENABLE_WEB_SEARCH             = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("true", "1", "yes")
WEB_SEARCH_PROVIDER           = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").lower()  # "duckduckgo" or "tavily"
TAVILY_API_KEY                = os.getenv("TAVILY_API_KEY")
# Provisional fallback threshold (sigmoid 0..1); to be re-tuned empirically once Phase 7 eval framework produces benchmark data
WEB_SEARCH_FALLBACK_THRESHOLD = float(os.getenv("WEB_SEARCH_FALLBACK_THRESHOLD", "0.35"))
WEB_SEARCH_MAX_RESULTS        = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 4))

# --- llm ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- environment & auth ---
ENV                    = os.getenv("ENV", "development").lower()
AUTH_ENABLED           = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
DATABASE_URL           = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
REDIS_URL              = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_RATE_LIMIT_RPM = int(os.getenv("DEFAULT_RATE_LIMIT_RPM", 60))
BOOTSTRAP_API_KEY = os.getenv("BOOTSTRAP_API_KEY")  # Must be set in .env

# --- caching ---
CACHE_ENABLED          = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
CACHE_TTL_SECONDS      = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour default TTL

# --- api ---
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
]

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set. Check your .env file.")

DISALLOWED_PRODUCTION_BOOTSTRAP_KEYS = {
    "your_bootstrap_api_key_here",
    "your_api_key_here",
    "your_bootstrap_master_api_key_here",
}

def is_insecure_bootstrap_key(key: str | None) -> bool:
    if not key or not key.strip():
        return True
    k = key.strip().lower()
    if k in DISALLOWED_PRODUCTION_BOOTSTRAP_KEYS:
        return True
    # Disallow common development placeholders and insecure patterns
    placeholder_tokens = ("replace", "placeholder", "master_12345", "test_key", "change_me")
    if any(token in k for token in placeholder_tokens):
        return True
    if len(k) < 24:
        return True
    return False

if ENV == "production":
    if is_insecure_bootstrap_key(BOOTSTRAP_API_KEY):
        raise EnvironmentError(
            "FATAL: When ENV=production, BOOTSTRAP_API_KEY must be explicitly configured with a secure, "
            "high-entropy key in your environment. Insecure default keys and development placeholders are disallowed."
        )
