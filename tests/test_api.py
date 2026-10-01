"""
Comprehensive PyTest Test Suite for FastAPI RAG Backend Endpoints.

Tests:
- Collection discovery & health
- Upload file validation
- Query validation (empty questions, missing collections)
- Single collection querying
- Multi-document cross-collection querying
- Category & metadata filtering
- Reranker toggle (ON vs. OFF)
- Web search fallback triggering & blending
- Collection deletion
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import main
from main import app
from config import BOOTSTRAP_API_KEY

TEST_API_KEY = BOOTSTRAP_API_KEY or "sk_test_fixture_master_key_internal_only"

# Mock generator so tests run independently of Gemini API quota
main.generate_answer = lambda q, ctx: f"Mock Grounded Answer with {len(ctx)} sources for: {q}"

client = TestClient(app)
client.headers.update({"X-API-Key": TEST_API_KEY})


def test_collections_endpoint():
    """Verify GET /collections returns a valid JSON list of collections."""
    response = client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert isinstance(data["collections"], list)


def test_query_validation_empty_question():
    """Verify POST /query rejects empty or whitespace-only questions."""
    response = client.post("/query", json={"question": "   ", "collection_name": "test_hybrid_coll"})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


def test_query_validation_missing_collection():
    """Verify POST /query rejects requests with no collection specified."""
    response = client.post("/query", json={"question": "What is EE101?"})
    assert response.status_code == 400
    assert "At least one collection_name or collection_names must be provided" in response.json()["detail"]


def test_upload_invalid_filetype():
    """Verify POST /upload rejects non-PDF files."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"Hello world", "text/plain")},
        data={"collection_name": "test_txt_coll"},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_single_document_query():
    """Verify POST /query returns cited sources for an in-domain single-collection query."""
    response = client.post(
        "/query",
        json={
            "question": "What are the prerequisites for Elements of Electrical Engineering EE101?",
            "collection_name": "test_hybrid_coll",
            "enable_rerank": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["retrieval_mode"] == "local_document"
    assert not data["fallback_triggered"]
    assert len(data["sources"]) > 0
    assert data["sources"][0]["page"] == 135


def test_multi_collection_query():
    """Verify POST /query supports collection_names array across multiple collections."""
    response = client.post(
        "/query",
        json={
            "question": "What is the frequency reuse bandwidth equation in cellular networks, and what are the Ku-band frequencies in satellite communication?",
            "collection_names": ["telecom_architecture_coll", "ece_curriculum_coll"],
            "enable_rerank": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["collection_name"] in ["telecom_architecture_coll", "ece_curriculum_coll"]


def test_metadata_filtering():
    """Verify POST /query respects metadata filters."""
    response = client.post(
        "/query",
        json={
            "question": "What are the course outcomes?",
            "collection_name": "telecom_architecture_coll",
            "filters": {"category": "telecom"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    for s in data["sources"]:
        if s.get("source_type") == "document" and s.get("category"):
            assert s["category"] == "telecom"


def test_reranker_runtime_toggle():
    """Verify enable_rerank=False returns un-reranked RRF scores without error."""
    response = client.post(
        "/query",
        json={
            "question": "What are the course outcomes and topics in digital modulation?",
            "collection_name": "test_hybrid_coll",
            "enable_rerank": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) > 0


def test_web_search_fallback(monkeypatch):
    """Verify out-of-domain query triggers web fallback with live URLs."""
    mock_web_results = [
        {
            "content": "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+.",
            "title": "FastAPI Documentation",
            "url": "https://fastapi.tiangolo.com/",
            "provider": "duckduckgo",
            "source_type": "web",
            "score": 1.0,
        },
        {
            "content": "FastAPI framework release history and changelog details.",
            "title": "FastAPI - PyPI",
            "url": "https://pypi.org/project/fastapi/",
            "provider": "duckduckgo",
            "source_type": "web",
            "score": 1.0,
        },
    ]
    monkeypatch.setattr(main, "search_web", lambda **kwargs: mock_web_results)

    response = client.post(
        "/query",
        json={
            "question": "What are the latest features and release history of Python FastAPI framework?",
            "collection_name": "telecom_architecture_coll",
            "enable_web_fallback": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_triggered"] is True
    assert data["retrieval_mode"] == "web_fallback"
    assert len(data["sources"]) > 0
    assert any(s["source_type"] == "web" for s in data["sources"])
    assert any(s.get("url") for s in data["sources"])


def test_delete_nonexistent_collection_error():
    """Verify DELETE /collections/{name} errors appropriately when collection is not found."""
    response = client.delete("/collections/nonexistent_dummy_collection_xyz")
    assert response.status_code == 404


def test_delete_collection_success():
    """Verify DELETE /collections/{name} successfully deletes an existing collection."""
    from ingest import embed_and_store
    from database import SessionLocal
    from models import Tenant

    db = SessionLocal()
    tenant = db.query(Tenant).filter(Tenant.name == "default_org").first()
    tenant_prefix = tenant.id[:12] if tenant else "default_tena"
    db.close()

    coll_name = "test_deletion_target"
    scoped_name = f"t_{tenant_prefix}_{coll_name}"
    embed_and_store(
        ["Temporary chunk for deletion testing"],
        [{"source": "temp.pdf", "page": 1, "chunk_id": "temp_c1"}],
        scoped_name,
        "temp.pdf",
        force=True,
    )

    # Verify collection appears in listing
    list_res = client.get("/collections")
    assert list_res.status_code == 200
    assert coll_name in list_res.json()["collections"]

    # Delete the collection
    del_res = client.delete(f"/collections/{coll_name}")
    assert del_res.status_code == 200
    assert "deleted" in del_res.json()["message"]

    # Verify collection no longer exists
    list_after = client.get("/collections")
    assert coll_name not in list_after.json()["collections"]


def test_health_endpoint():
    """Verify GET /health returns service status and dependency checks."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "database" in data
    assert "cache" in data

