"""
Hybrid dense and sparse retrieval.

Fuses ChromaDB vector embeddings with BM25Okapi keyword matching via Reciprocal Rank Fusion,
ensuring technical acronyms aren't lost to semantic generalities.
"""

import logging
import threading
from typing import Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import EMBEDDING_MODEL, TOP_K_RESULTS, INITIAL_RETRIEVAL_K, ENABLE_RERANKER, MIN_RELEVANCE_SCORE
from chroma import get_chroma_client
from bm25 import search_bm25
from reranker import rerank_documents

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant (TREC benchmark standard)
RRF_K = 60

_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None
_EMBEDDINGS_LOCK = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton getter for HuggingFaceEmbeddings model to avoid per-query model reloading."""
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        with _EMBEDDINGS_LOCK:
            if _EMBEDDINGS_INSTANCE is None:
                _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS_INSTANCE


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    top_k: int = TOP_K_RESULTS,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse dense and sparse result lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (rrf_k + rank)) across available retrieval channels.
    """
    fused_candidates: dict[str, dict[str, Any]] = {}

    # Helper to derive a unique candidate key
    def get_candidate_key(item: dict[str, Any]) -> str:
        if item.get("chunk_id"):
            return item["chunk_id"]
        source = item.get("source", "")
        page = item.get("page", 0)
        content_preview = item.get("content", "")[:60]
        return f"{source}_p{page}_{hash(content_preview)}"

    # 1. Process Dense Results
    for rank, item in enumerate(dense_results, start=1):
        key = get_candidate_key(item)
        rrf_delta = 1.0 / (rrf_k + rank)
        if key not in fused_candidates:
            fused_candidates[key] = {
                **item,
                "rrf_score": rrf_delta,
                "dense_rank": rank,
                "dense_score": item.get("score"),  # cosine distance
                "bm25_rank": None,
                "bm25_score": None,
                "retrieval_channels": ["dense"],
            }
        else:
            cand = fused_candidates[key]
            cand["rrf_score"] += rrf_delta
            cand["dense_rank"] = rank
            cand["dense_score"] = item.get("score")
            if "dense" not in cand["retrieval_channels"]:
                cand["retrieval_channels"].append("dense")

    # 2. Process Sparse (BM25) Results
    for rank, item in enumerate(sparse_results, start=1):
        key = get_candidate_key(item)
        rrf_delta = 1.0 / (rrf_k + rank)
        if key not in fused_candidates:
            fused_candidates[key] = {
                **item,
                "rrf_score": rrf_delta,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank,
                "bm25_score": item.get("bm25_score"),
                "retrieval_channels": ["bm25"],
            }
        else:
            cand = fused_candidates[key]
            cand["rrf_score"] += rrf_delta
            cand["bm25_rank"] = rank
            cand["bm25_score"] = item.get("bm25_score")
            if "bm25" not in cand["retrieval_channels"]:
                cand["retrieval_channels"].append("bm25")

    # Sort fused candidates descending by RRF score
    sorted_candidates = sorted(
        fused_candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    max_possible_rrf = 2.0 / (rrf_k + 1)  # Rank 1 in both dense and sparse

    results = []
    for cand in sorted_candidates[:top_k]:
        # Convert RRF score to normalized 0-2 distance scale for frontend compatibility
        normalized_sim = min(1.0, cand["rrf_score"] / max_possible_rrf)
        synthetic_distance = round(2.0 * (1.0 - normalized_sim), 4)

        results.append({
            "content":            cand["content"],
            "page":               cand.get("page"),
            "source":             cand.get("source"),
            "doc_title":          cand.get("doc_title", cand.get("source")),
            "category":           cand.get("category", "uncategorized"),
            "upload_date":        cand.get("upload_date", ""),
            "section":            cand.get("section", "General"),
            "breadcrumb":         cand.get("breadcrumb", "General"),
            "chunk_id":           cand.get("chunk_id", ""),
            "score":              synthetic_distance,
            "rrf_score":          round(cand["rrf_score"], 6),
            "dense_rank":         cand.get("dense_rank"),
            "dense_score":        cand.get("dense_score"),
            "bm25_rank":          cand.get("bm25_rank"),
            "bm25_score":         cand.get("bm25_score"),
            "retrieval_channels": cand.get("retrieval_channels", []),
        })

    return results


def retrieve_context(
    query: str,
    collection_name: str | list[str],
    top_k: int = TOP_K_RESULTS,
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = ENABLE_RERANKER,
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict[str, Any]]:
    """
    Perform hybrid retrieval (ChromaDB dense search + BM25 sparse search fused via RRF)
    across one or more collections, followed by optional Cross-Encoder reranking
    and minimum confidence threshold filtering.
    """
    collections = [collection_name] if isinstance(collection_name, str) else list(collection_name)
    if not collections:
        return []

    candidate_pool_size = max(INITIAL_RETRIEVAL_K, top_k * 4)
    all_fused_candidates: list[dict[str, Any]] = []

    for coll in collections:
        if not coll or not coll.strip():
            continue
        coll = coll.strip()

        # 1. Dense Vector Search via ChromaDB
        dense_results = []
        try:
            embeddings = get_embeddings()
            client = get_chroma_client()

            vectorstore = Chroma(
                client=client,
                collection_name=coll,
                embedding_function=embeddings,
            )

            search_kwargs = {"k": candidate_pool_size}
            if filters and isinstance(filters, dict) and len(filters) > 0:
                search_kwargs["filter"] = filters

            chroma_res = vectorstore.similarity_search_with_score(query, **search_kwargs)
            for doc, dist in chroma_res:
                dense_results.append({
                    "content":         doc.page_content,
                    "page":            doc.metadata.get("page"),
                    "source":          doc.metadata.get("source"),
                    "doc_title":       doc.metadata.get("doc_title", doc.metadata.get("source")),
                    "category":        doc.metadata.get("category", "uncategorized"),
                    "upload_date":     doc.metadata.get("upload_date", ""),
                    "section":         doc.metadata.get("section", "General"),
                    "breadcrumb":      doc.metadata.get("breadcrumb", "General"),
                    "chunk_id":        doc.metadata.get("chunk_id", ""),
                    "collection_name": coll,
                    "score":           round(dist, 4),  # Cosine distance
                })
            logger.info("Dense search returned %d candidates from '%s'", len(dense_results), coll)
        except Exception:
            logger.warning("Dense search failed for collection '%s'", coll, exc_info=True)

        # 2. Sparse Search via BM25
        sparse_results = []
        try:
            sparse_results = search_bm25(
                query=query,
                collection_name=coll,
                top_k=candidate_pool_size,
                filters=filters,
            )
            for item in sparse_results:
                item["collection_name"] = coll
            logger.info("BM25 search returned %d candidates from '%s'", len(sparse_results), coll)
        except Exception:
            logger.warning("BM25 search failed for collection '%s'", coll, exc_info=True)

        # 3. Fuse via Reciprocal Rank Fusion (RRF) for this collection
        coll_fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=candidate_pool_size if (enable_rerank or len(collections) > 1) else top_k,
        )
        for item in coll_fused:
            item["collection_name"] = coll

        all_fused_candidates.extend(coll_fused)

    if not all_fused_candidates:
        logger.warning("No hybrid results found across collections %s (filters: %s)", collections, filters)
        return []

    # 4. Cross-Encoder Reranking with Minimum Relevance Filtering
    if enable_rerank:
        reranked_results, latency_ms = rerank_documents(
            query=query,
            documents=all_fused_candidates,
            top_k=top_k,
            min_score=min_relevance_score,
        )
        logger.info(
            "Retrieved & reranked %d chunks across %d collection(s) in %.2fms (top rerank score: %.4f, channels: %s)",
            len(reranked_results),
            len(collections),
            latency_ms,
            reranked_results[0].get("rerank_score", 0.0) if reranked_results else 0.0,
            reranked_results[0].get("retrieval_channels", []) if reranked_results else [],
        )
        return reranked_results

    # If reranking is disabled, sort aggregated multi-collection candidates by RRF score
    all_fused_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return all_fused_candidates[:top_k]
