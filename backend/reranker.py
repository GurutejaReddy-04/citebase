"""
Cross-Encoder reranking for candidate passages.

Evaluates query-document pairs with full bidirectional cross-attention, capturing nuanced
semantic relevance that bi-encoder cosine similarity misses.
"""

import math
import time
import logging
import threading
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANKER_TOP_K

logger = logging.getLogger(__name__)

_RERANKER_INSTANCE: Optional[CrossEncoder] = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> CrossEncoder:
    """Singleton getter for the CrossEncoder model with thread-safe lazy loading."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        with _RERANKER_LOCK:
            if _RERANKER_INSTANCE is None:
                logger.info("Loading CrossEncoder model: %s", RERANKER_MODEL)
                _RERANKER_INSTANCE = CrossEncoder(RERANKER_MODEL)
                logger.info("CrossEncoder model loaded successfully.")
    return _RERANKER_INSTANCE


def sigmoid(x: float) -> float:
    """Compute sigmoid score for logit normalization (0.0 to 1.0)."""
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = RERANKER_TOP_K,
    min_score: float = 0.0,
) -> tuple[list[dict[str, Any]], float]:
    """
    Rerank a list of candidate documents using the CrossEncoder model.

    Optionally drops any candidates whose sigmoid normalized score is below min_score.

    Returns:
        tuple of (reranked_documents[:top_k], latency_ms)
    """
    if not documents or not query.strip():
        return [], 0.0

    model = get_reranker()

    pairs: Any = [(str(query), str(doc.get("content", ""))) for doc in documents]

    start_time = time.perf_counter()
    raw_scores = model.predict(pairs)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    scored_docs = []
    for idx, (doc, raw_score) in enumerate(zip(documents, raw_scores)):
        logit = float(raw_score)
        norm_score = round(sigmoid(logit), 4)

        scored_docs.append({
            **doc,
            "rerank_logit": round(logit, 4),
            "rerank_score": norm_score,
            "pre_rerank_rank": idx + 1,
            "rerank_latency_ms": latency_ms,
        })

    scored_docs.sort(key=lambda x: x["rerank_logit"], reverse=True)

    for rank, doc in enumerate(scored_docs, start=1):
        doc["rerank_rank"] = rank
        doc["score"] = round(2.0 * (1.0 - doc["rerank_score"]), 4)

    # Filter out low-confidence distractors below min_score threshold
    if min_score > 0.0:
        qualified_docs = [doc for doc in scored_docs if doc["rerank_score"] >= min_score]
        dropped_count = len(scored_docs) - len(qualified_docs)
        if dropped_count > 0:
            logger.info("Dropped %d low-confidence candidates below threshold %.4f", dropped_count, min_score)
        scored_docs = qualified_docs

    top_results = scored_docs[:top_k]
    logger.info(
        "Reranked %d candidates in %.2fms -> top %d qualified returned (best logit: %.4f, score: %.4f)",
        len(documents),
        latency_ms,
        len(top_results),
        top_results[0]["rerank_logit"] if top_results else 0.0,
        top_results[0]["rerank_score"] if top_results else 0.0,
    )

    return top_results, latency_ms
