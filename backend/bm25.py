"""
BM25Okapi sparse keyword retrieval.

Indexes are rebuilt on ingestion to eliminate incremental index drift, keeping
keyword scoring deterministic without SQLite FTS overhead.
Serialized safely via JSON with atomic file replacement to eliminate pickle vulnerabilities.
"""

import json
import logging
import os
import re
import tempfile
import threading
from typing import Any, Optional

import rank_bm25

from chroma import get_chroma_client
from config import CHROMA_PATH

logger = logging.getLogger(__name__)

# In-memory cache of loaded BM25 indexes: collection_name -> {"bm25": BM25Okapi, "chunks": [...], "metadatas": [...]}
_INDEX_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()

# Tokenizer pattern for alphanumeric tokens, acronyms, and codes
_TOKEN_PATTERN = re.compile(r'\b\w+\b')


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 matching."""
    return _TOKEN_PATTERN.findall(text.lower())


def _get_index_file_path(collection_name: str) -> str:
    """Get the absolute path for a collection's BM25 JSON index file."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return os.path.join(CHROMA_PATH, f"{collection_name}_bm25.json")


def build_and_save_bm25_index(
    collection_name: str,
    chunks: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """
    Build a BM25Okapi index from all chunks in a collection and persist to disk via atomic JSON.
    """
    if not chunks:
        logger.warning("No chunks provided to build BM25 index for collection '%s'", collection_name)
        return

    tokenized_corpus = [tokenize(chunk) for chunk in chunks]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

    corpus_chunk_ids = [
        m.get("chunk_id", f"{m.get('source', 'chunk')}_idx{i}")
        for i, m in enumerate(metadatas)
    ]
    doc_freqs = [dict(df) for df in bm25.doc_freqs]

    index_json_data = {
        "doc_freqs": doc_freqs,
        "corpus_chunk_ids": corpus_chunk_ids,
        "avgdl": float(bm25.avgdl),
        "corpus_size": int(bm25.corpus_size),
        "chunks": chunks,
        "metadatas": metadatas,
    }

    file_path = _get_index_file_path(collection_name)
    dir_name = os.path.dirname(file_path) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dir_name, delete=False, suffix=".tmp") as tmp:
            json.dump(index_json_data, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, file_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise

    index_runtime_data = {
        "bm25": bm25,
        "chunks": chunks,
        "metadatas": metadatas,
    }

    with _CACHE_LOCK:
        _INDEX_CACHE[collection_name] = index_runtime_data

    logger.info(
        "Built and atomically persisted JSON BM25 index for collection '%s' (%d chunks) -> '%s'",
        collection_name,
        len(chunks),
        file_path,
    )


def rebuild_bm25_from_chroma(collection_name: str) -> bool:
    """
    Pull all documents and metadata from ChromaDB for collection_name and rebuild BM25.
    """
    client = get_chroma_client()
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        logger.warning("Collection '%s' does not exist in ChromaDB; cannot rebuild BM25", collection_name)
        return False

    records = coll.get(include=["documents", "metadatas"])
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    if not documents:
        logger.warning("No documents found in collection '%s' to build BM25", collection_name)
        return False

    build_and_save_bm25_index(collection_name, documents, metadatas)
    return True


def load_bm25_index(collection_name: str) -> Optional[dict[str, Any]]:
    """
    Retrieve BM25 index for collection_name from cache, JSON file, or rebuild from Chroma.
    Migrates any legacy pickle files to JSON safely and removes them.
    """
    with _CACHE_LOCK:
        if collection_name in _INDEX_CACHE:
            return _INDEX_CACHE[collection_name]

    file_path = _get_index_file_path(collection_name)
    legacy_pkl_path = os.path.join(CHROMA_PATH, f"{collection_name}_bm25.pkl")

    # Migration: if legacy .pkl exists and no .json exists, convert and remove .pkl
    if not os.path.exists(file_path) and os.path.exists(legacy_pkl_path):
        try:
            import pickle
            with open(legacy_pkl_path, "rb") as f:
                old_data = pickle.load(f)
            chunks = old_data.get("chunks") or []
            metadatas = old_data.get("metadatas") or []
            if chunks:
                build_and_save_bm25_index(collection_name, chunks, metadatas)
        except Exception:
            logger.warning("Could not migrate legacy pickle file '%s'", legacy_pkl_path, exc_info=True)
        finally:
            try:
                if os.path.exists(legacy_pkl_path):
                    os.remove(legacy_pkl_path)
                    logger.info("Migrated and removed legacy pickle file '%s'", legacy_pkl_path)
            except Exception:
                pass

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if (
                not isinstance(raw_data, dict)
                or "chunks" not in raw_data
                or "metadatas" not in raw_data
                or "avgdl" not in raw_data
            ):
                raise ValueError("BM25 index JSON file schema invalid")

            chunks = raw_data["chunks"]
            metadatas = raw_data["metadatas"]
            tokenized_corpus = [tokenize(chunk) for chunk in chunks]
            bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

            index_data = {
                "bm25": bm25,
                "chunks": chunks,
                "metadatas": metadatas,
            }
            with _CACHE_LOCK:
                _INDEX_CACHE[collection_name] = index_data
            return index_data
        except Exception:
            logger.exception("Failed to load BM25 index from '%s', attempting rebuild from ChromaDB", file_path)

    # If file not present or failed to load, rebuild from Chroma
    if rebuild_bm25_from_chroma(collection_name):
        with _CACHE_LOCK:
            return _INDEX_CACHE.get(collection_name)

    return None


# Backward-compatible alias
get_bm25_index = load_bm25_index


def search_bm25(
    query: str,
    collection_name: str,
    top_k: int = 20,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Perform BM25 sparse keyword search across the collection.

    Returns a list of matching candidates sorted by BM25 score.
    """
    index_data = get_bm25_index(collection_name)
    if not index_data:
        logger.warning("No BM25 index available for collection '%s'", collection_name)
        return []

    bm25 = index_data["bm25"]
    chunks = index_data["chunks"]
    metadatas = index_data["metadatas"]

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    scored_items = []
    for idx, score in enumerate(scores):
        if score <= 0.0:
            continue

        meta = metadatas[idx] if idx < len(metadatas) else {}

        # Apply metadata filters if specified
        if filters:
            match = True
            for k, v in filters.items():
                meta_val = meta.get(k)
                if isinstance(v, str) and isinstance(meta_val, str):
                    if meta_val.strip().lower() != v.strip().lower():
                        match = False
                        break
                elif meta_val != v:
                    match = False
                    break
            if not match:
                continue

        scored_items.append((score, idx))

    # Sort descending by score
    scored_items.sort(key=lambda x: x[0], reverse=True)
    top_items = scored_items[:top_k]

    results = []
    for rank, (score, idx) in enumerate(top_items, start=1):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        results.append({
            "content":     chunks[idx],
            "page":        meta.get("page"),
            "source":      meta.get("source"),
            "doc_title":   meta.get("doc_title", meta.get("source")),
            "category":    meta.get("category", "uncategorized"),
            "upload_date": meta.get("upload_date", ""),
            "section":     meta.get("section", "General"),
            "breadcrumb":  meta.get("breadcrumb", "General"),
            "chunk_id":    meta.get("chunk_id", f"{meta.get('source', '')}_idx{idx}"),
            "bm25_score":  round(float(score), 4),
            "bm25_rank":   rank,
        })

    logger.info("BM25 retrieved %d candidates for query '%s' in collection '%s'", len(results), query, collection_name)
    return results


def delete_bm25_index(collection_name: str) -> None:
    """Delete the BM25 index file and clear from cache."""
    with _CACHE_LOCK:
        _INDEX_CACHE.pop(collection_name, None)

    for ext in (".json", ".pkl"):
        file_path = os.path.join(CHROMA_PATH, f"{collection_name}_bm25{ext}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Deleted BM25 index file '%s'", file_path)
            except Exception:
                logger.warning("Failed to delete BM25 file '%s'", file_path, exc_info=True)
