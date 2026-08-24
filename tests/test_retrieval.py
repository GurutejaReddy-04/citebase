"""
Unit tests for hybrid retrieval and Reciprocal Rank Fusion (RRF).
Tests fusion logic, deduplication, score normalization, and channel attribution in isolation.
"""

import pytest
from retriever import reciprocal_rank_fusion, RRF_K


def test_rrf_with_dense_and_sparse():
    """Verify RRF fuses distinct dense and sparse results correctly."""
    dense_results = [
        {"chunk_id": "chunk_1", "content": "Dense chunk one", "source": "doc1.pdf", "page": 1, "score": 0.12},
        {"chunk_id": "chunk_2", "content": "Dense chunk two", "source": "doc1.pdf", "page": 2, "score": 0.25},
    ]
    sparse_results = [
        {"chunk_id": "chunk_3", "content": "Sparse chunk three", "source": "doc2.pdf", "page": 1, "bm25_score": 5.4},
        {"chunk_id": "chunk_4", "content": "Sparse chunk four", "source": "doc2.pdf", "page": 2, "bm25_score": 3.2},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=4)

    assert len(fused) == 4
    chunk_ids = [c["chunk_id"] for c in fused]
    assert "chunk_1" in chunk_ids
    assert "chunk_3" in chunk_ids

    # RRF scores must be positive and sorted descending
    scores = [c["rrf_score"] for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplication_and_channel_attribution():
    """Verify duplicate chunks across dense and sparse channels are merged with combined RRF scores."""
    shared_chunk_id = "shared_chunk_alpha"
    dense_results = [
        {"chunk_id": shared_chunk_id, "content": "Shared content", "source": "guide.pdf", "page": 1, "score": 0.05},
        {"chunk_id": "dense_only", "content": "Dense only content", "source": "guide.pdf", "page": 2, "score": 0.15},
    ]
    sparse_results = [
        {"chunk_id": shared_chunk_id, "content": "Shared content", "source": "guide.pdf", "page": 1, "bm25_score": 8.0},
        {"chunk_id": "sparse_only", "content": "Sparse only content", "source": "guide.pdf", "page": 3, "bm25_score": 4.0},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=5)

    # 3 unique candidates total
    assert len(fused) == 3

    top_result = fused[0]
    assert top_result["chunk_id"] == shared_chunk_id
    assert "dense" in top_result["retrieval_channels"]
    assert "bm25" in top_result["retrieval_channels"]

    # Expected RRF score = 1/(60+1) + 1/(60+1) = 2/61
    expected_score = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 1))
    assert pytest.approx(top_result["rrf_score"], abs=1e-5) == expected_score


def test_rrf_with_empty_inputs():
    """Verify RRF handles empty result sets gracefully without errors."""
    # Both empty
    assert reciprocal_rank_fusion([], [], top_k=5) == []

    # Dense only
    dense_only = [{"chunk_id": "d1", "content": "Dense content", "source": "doc.pdf", "page": 1, "score": 0.1}]
    fused_dense = reciprocal_rank_fusion(dense_only, [], top_k=5)
    assert len(fused_dense) == 1
    assert fused_dense[0]["chunk_id"] == "d1"
    assert fused_dense[0]["retrieval_channels"] == ["dense"]

    # Sparse only
    sparse_only = [{"chunk_id": "s1", "content": "Sparse content", "source": "doc.pdf", "page": 1, "bm25_score": 2.5}]
    fused_sparse = reciprocal_rank_fusion([], sparse_only, top_k=5)
    assert len(fused_sparse) == 1
    assert fused_sparse[0]["chunk_id"] == "s1"
    assert fused_sparse[0]["retrieval_channels"] == ["bm25"]


def test_rrf_score_normalization_and_top_k():
    """Verify output score normalization scale (0-2) and strict top_k truncation."""
    dense_results = [
        {"chunk_id": f"chunk_{i}", "content": f"Content {i}", "source": "doc.pdf", "page": i, "score": 0.1 * i}
        for i in range(1, 10)
    ]
    sparse_results = [
        {"chunk_id": f"chunk_{i}", "content": f"Content {i}", "source": "doc.pdf", "page": i, "bm25_score": 10.0 - i}
        for i in range(1, 10)
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=3)

    # Truncation check
    assert len(fused) == 3

    # Normalized synthetic distance score: 0 <= score <= 2.0
    for item in fused:
        assert 0.0 <= item["score"] <= 2.0
        assert item["rrf_score"] > 0.0
