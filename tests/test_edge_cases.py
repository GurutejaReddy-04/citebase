"""
Unit tests for Edge Cases and Exception Handling (EDGE-001 through EDGE-014).
"""

import json
import os
import sys
import pytest
from fastapi import HTTPException

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import generator
import main
from auth import TenantContext
from bm25 import (
    build_and_save_bm25_index,
    delete_bm25_index,
    load_bm25_index,
    _get_index_file_path,
)
from cache import get_redis_client, invalidate_tenant_cache
from rate_limiter import check_rate_limit, _memory_counters, _memory_lock


def test_question_length_limit(client):
    """EDGE-005: Verify questions exceeding 2000 characters return HTTP 422."""
    long_question = "A" * 2001
    response = client.post(
        "/query",
        json={"question": long_question, "collection_name": "test_hybrid_coll"},
    )
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("question" in str(err) for err in errors)


def test_upload_missing_filename(client):
    """EDGE-007: Verify file upload with missing/non-pdf filename returns HTTP 400 without AttributeError."""
    # 1. Via HTTP client with non-pdf filename
    response = client.post(
        "/upload",
        files={"file": ("unnamed_file", b"%PDF-1.4 sample content", "application/pdf")},
        data={"collection_name": "test_edge_coll"},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

    # 2. Directly invoke handler with file.filename = None to ensure no AttributeError crash
    from unittest.mock import MagicMock
    from fastapi import BackgroundTasks, Response
    mock_file = MagicMock()
    mock_file.filename = None
    mock_tenant = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        main.upload_document(
            response=Response(),
            background_tasks=BackgroundTasks(),
            file=mock_file,
            collection_name="test_edge_coll",
            tenant=mock_tenant,
            db=MagicMock(),
        )
    assert exc_info.value.status_code == 400
    assert "Only PDF files are supported" in exc_info.value.detail


def test_health_chromadb_field(client):
    """EDGE-011, EDGE-012: Verify /health includes chromadb field and correct boolean checks."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "chromadb" in data
    assert "database" in data
    assert "cache" in data
    assert data["chromadb"] is True
    assert data["database"] is True
    assert data["status"] in ("ok", "degraded")


def test_rate_limiter_fallback(monkeypatch):
    """EDGE-014: Verify rate limiter falls back to in-memory dict when Redis fails."""
    # Create isolated tenant with tight limit of 2 requests per minute
    tenant = TenantContext(
        tenant_id="fallback_tenant_id",
        tenant_name="Fallback Tenant",
        api_key_id="fallback_key_id",
        key_prefix="sk_live_fb99",
        rate_limit_rpm=2,
    )

    # Clear any previous memory counts for this prefix
    with _memory_lock:
        _memory_counters.pop(tenant.key_prefix, None)

    # Mock get_redis_client to return a client whose pipeline raises an exception
    class BrokenRedis:
        def pipeline(self):
            raise ConnectionError("Simulated Redis outage")

    monkeypatch.setattr("rate_limiter.get_redis_client", lambda: BrokenRedis())

    # First request: should succeed via in-memory fallback
    res1 = check_rate_limit(tenant)
    assert res1.tenant_id == tenant.tenant_id

    # Second request: should succeed
    res2 = check_rate_limit(tenant)
    assert res2.tenant_id == tenant.tenant_id

    # Third request: should raise HTTP 429 via in-memory fallback
    with pytest.raises(HTTPException) as exc_info:
        check_rate_limit(tenant)
    assert exc_info.value.status_code == 429
    assert "Rate limit of 2 requests per minute exceeded" in exc_info.value.detail


def test_cache_invalidation_scan():
    """EDGE-006: Verify invalidate_tenant_cache uses SCAN and purges matching tenant keys."""
    r = get_redis_client()
    target_tenant = "tenant_scan_test"
    other_tenant = "tenant_scan_other"

    key1 = f"rag_cache:{target_tenant}:abc123"
    key2 = f"rag_cache:{target_tenant}:def456"
    other_key = f"rag_cache:{other_tenant}:xyz789"

    r.set(key1, json.dumps({"answer": "1"}))
    r.set(key2, json.dumps({"answer": "2"}))
    r.set(other_key, json.dumps({"answer": "keep"}))

    deleted = invalidate_tenant_cache(target_tenant)
    assert deleted == 2

    assert r.get(key1) is None
    assert r.get(key2) is None
    assert r.get(other_key) is not None

    # Cleanup
    r.delete(other_key)


def test_bm25_atomic_write_and_schema():
    """EDGE-001, EDGE-009: Verify BM25 index is serialized via JSON with atomic writes and schema."""
    coll_name = "test_atomic_json_coll"
    chunks = ["Python asynchronous concurrency", "FastAPI web framework with Pydantic"]
    metadatas = [
        {"source": "test.pdf", "page": 1, "chunk_id": "c1"},
        {"source": "test.pdf", "page": 2, "chunk_id": "c2"},
    ]

    # 1. Build and save
    build_and_save_bm25_index(coll_name, chunks, metadatas)

    file_path = _get_index_file_path(coll_name)
    assert file_path.endswith(".json")
    assert os.path.exists(file_path)

    # 2. Verify JSON structure
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "doc_freqs" in data
    assert "corpus_chunk_ids" in data
    assert "avgdl" in data
    assert "chunks" in data
    assert "metadatas" in data
    assert data["corpus_size"] == 2

    # 3. Load from JSON
    loaded = load_bm25_index(coll_name)
    assert loaded is not None
    assert "bm25" in loaded
    assert len(loaded["chunks"]) == 2

    # Cleanup
    delete_bm25_index(coll_name)
    assert not os.path.exists(file_path)


def test_gemini_lazy_client(monkeypatch):
    """EDGE-010: Verify Gemini client is lazily loaded via _get_client()."""
    # Reset singleton
    generator._client = None

    assert generator._client is None

    # Calling _get_client initializes it
    client = generator._get_client()
    assert client is not None
    assert generator._client is not None


def test_upload_file_size_limit(client, monkeypatch):
    """EDGE-004: Verify uploads exceeding MAX_UPLOAD_BYTES return HTTP 413."""
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100")

    # 500 bytes payload exceeds 100 bytes limit
    oversized_pdf = b"%PDF-1.4 " + b"X" * 500
    response = client.post(
        "/upload",
        files={"file": ("large_doc.pdf", oversized_pdf, "application/pdf")},
        data={"collection_name": "test_edge_coll"},
    )
    assert response.status_code == 413
    assert "File exceeds maximum size" in response.json()["detail"]


def test_upload_path_traversal_sanitization(client):
    """EDGE-002: Verify filename traversal sequences are sanitized."""
    pdf_bytes = b"%PDF-1.4 normal content"
    response = client.post(
        "/upload",
        files={"file": ("../../../../etc/malicious.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": "test_edge_coll"},
    )
    assert response.status_code in (200, 202)
    filename_in_resp = response.json().get("filename", "")
    assert ".." not in filename_in_resp
    assert "/" not in filename_in_resp
    assert "\\" not in filename_in_resp
