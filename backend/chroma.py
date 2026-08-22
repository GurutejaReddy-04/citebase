"""
One PersistentClient for the whole process - ChromaDB manages the SQLite
pool internally, so spinning up a new client per request wastes resources
and risks hitting open-file limits.
"""

import threading
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from config import CHROMA_PATH

_client: Optional[ClientAPI] = None
_lock = threading.Lock()


def get_chroma_client() -> ClientAPI:
    """Return the process-wide PersistentClient, creating it on first call."""
    global _client
    if _client is None:
        with _lock:
            # Double-checked locking so two threads can't both pass the None check.
            if _client is None:
                _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client
