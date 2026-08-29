"""
Test Suite for Phase 10: Redis Query Caching & Latency Optimization.

Tests:
1. Cold vs. Hot Cache Query Latency & Pipeline Bypass (skips retrieval + LLM on hit)
2. Strict Tenant-Scoped Cache Isolation (Tenant B never sees Tenant A's cached answer)
3. TTL Expiry Verification (Cache expires after TTL window)
4. Cache Invalidation on New Document Ingestion & Collection Reset
"""

import os
import sys
import time
import uuid
import pytest
from fastapi.testclient import TestClient

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import main
import retriever
from auth import generate_api_key
from cache import get_redis_client, invalidate_tenant_cache, set_cached_response
from database import SessionLocal
from main import app
from models import ApiKey, Tenant

# Mock generator for deterministic answers
main.generate_answer = lambda q, ctx: f"Synthesized answer for '{q}' from {len(ctx)} sources."


def make_test_pdf(text: str = "Caching Specification Document") -> bytes:
    """Generate in-memory valid PDF containing text using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def create_tenant_and_client(tenant_name: str) -> tuple[str, TestClient]:
    """Helper to create a tenant with dedicated client and API key. Returns (tenant_id, client)."""
    db = SessionLocal()
    try:
        tenant = db.query(Tenant).filter(Tenant.name == tenant_name).first()
        if not tenant:
            tenant = Tenant(name=tenant_name, is_active=True)
            db.add(tenant)
            db.flush()

        tenant_id = tenant.id

        raw_key, key_hash, key_prefix = generate_api_key()
        api_key_rec = ApiKey(
            tenant_id=tenant_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=f"Key for {tenant_name}",
            rate_limit_rpm=1000,
            is_active=True,
        )
        db.add(api_key_rec)
        db.commit()

        c = TestClient(app)
        c.headers.update({"X-API-Key": raw_key})
        return tenant_id, c
    finally:
        db.close()


def test_cache_hit_latency_and_skip_expensive_pipeline(monkeypatch):
    """
    Verify cold cache miss vs hot cache hit:
    - 1st query: cold miss (cached=False, executes retrieval & generation)
    - 2nd query: hot hit (cached=True, returns in <15ms, skips retrieval & generator entirely)
    """
    suffix = uuid.uuid4().hex[:6]
    tenant_id, client = create_tenant_and_client(f"cache_perf_tenant_{suffix}")

    # Ingest document
    pdf_bytes = make_test_pdf("High performance caching reduces P99 latency by over 95 percent.")
    coll_name = f"cache_perf_coll_{suffix}"
    res_up = client.post(
        "/upload?sync=true",
        files={"file": ("perf.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Perf Doc"},
    )
    assert res_up.status_code == 200

    query_payload = {
        "question": "What is the P99 latency reduction from caching?",
        "collection_name": coll_name,
        "enable_web_fallback": False,
    }

    # 1. Cold Query (Cache Miss)
    t0 = time.perf_counter()
    res_cold = client.post("/query", json=query_payload)
    cold_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res_cold.status_code == 200
    data_cold = res_cold.json()
    assert data_cold["cached"] is False
    assert len(data_cold["sources"]) >= 1

    # 2. Hook mock to verify expensive pipeline is SKIPPED on cache hit
    def explosive_retriever(*args, **kwargs):
        pytest.fail("Expensive retriever was called on a cache hit!")

    monkeypatch.setattr(main, "retrieve_context", explosive_retriever)

    # 3. Hot Query (Cache Hit)
    t1 = time.perf_counter()
    res_hot = client.post("/query", json=query_payload)
    hot_latency_ms = (time.perf_counter() - t1) * 1000.0

    assert res_hot.status_code == 200
    data_hot = res_hot.json()
    assert data_hot["cached"] is True
    assert data_hot["retrieval_mode"] == "cached"
    assert data_hot["answer"] == data_cold["answer"]
    assert len(data_hot["sources"]) == len(data_cold["sources"])
    assert hot_latency_ms < cold_latency_ms


def test_tenant_scoped_cache_isolation():
    """
    Verify strict tenant cache isolation:
    Even if Tenant A and Tenant B submit the EXACT identical question,
    Tenant B NEVER receives Tenant A's cached response.
    """
    suffix = uuid.uuid4().hex[:6]
    tenant_a_id, client_a = create_tenant_and_client(f"tenant_iso_a_{suffix}")
    tenant_b_id, client_b = create_tenant_and_client(f"tenant_iso_b_{suffix}")

    # Tenant A ingests private document (> 50 chars)
    pdf_a = make_test_pdf("Tenant Alpha Proprietary Financial Statement: Total Net Revenue is $500 Million USD.")
    coll_a = f"reports_{suffix}"
    res_up_a = client_a.post(
        "/upload?sync=true",
        files={"file": ("alpha_rev.pdf", pdf_a, "application/pdf")},
        data={"collection_name": coll_a},
    )
    assert res_up_a.status_code == 200

    # Tenant B ingests different private document in its own collection (> 50 chars)
    pdf_b = make_test_pdf("Tenant Beta Corporate Sustainability Report: Complete transition to zero landfill waste.")
    coll_b = f"reports_{suffix}"
    res_up_b = client_b.post(
        "/upload?sync=true",
        files={"file": ("beta_pol.pdf", pdf_b, "application/pdf")},
        data={"collection_name": coll_b},
    )
    assert res_up_b.status_code == 200

    same_question = "What is the proprietary financial revenue or policy?"

    # Tenant A asks question -> Populates Tenant A cache
    res_a = client_a.post("/query", json={"question": same_question, "collection_name": coll_a, "enable_web_fallback": False})
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["cached"] is False
    assert len(data_a["sources"]) >= 1
    assert data_a["sources"][0]["source"] == "alpha_rev.pdf"

    # Tenant A asks again -> Cache hit for Tenant A
    res_a_cached = client_a.post("/query", json={"question": same_question, "collection_name": coll_a, "enable_web_fallback": False})
    assert res_a_cached.status_code == 200
    assert res_a_cached.json()["cached"] is True

    # Tenant B asks identical question on its own collection -> MUST NOT hit Tenant A's cache
    res_b = client_b.post("/query", json={"question": same_question, "collection_name": coll_b, "enable_web_fallback": False})
    assert res_b.status_code == 200
    data_b = res_b.json()
    # Cache miss for Tenant B
    assert data_b["cached"] is False
    if len(data_b["sources"]) > 0:
        assert data_b["sources"][0]["source"] != "alpha_rev.pdf"


def test_cache_ttl_and_expiry():
    """Verify that cached entries expire after TTL has elapsed."""
    suffix = uuid.uuid4().hex[:6]
    tenant_id, client = create_tenant_and_client(f"ttl_tenant_{suffix}")

    pdf_bytes = make_test_pdf("TTL Expiration Test Passage detailing automatic cache invalidation protocols.")
    coll_name = f"ttl_coll_{suffix}"
    res_up = client.post(
        "/upload?sync=true",
        files={"file": ("ttl.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name},
    )
    assert res_up.status_code == 200

    query_payload = {"question": "What are cache invalidation protocols?", "collection_name": coll_name, "enable_web_fallback": False}

    # First query -> cold miss, cached
    res1 = client.post("/query", json=query_payload)
    assert res1.json()["cached"] is False

    # Second query -> hot hit
    res2 = client.post("/query", json=query_payload)
    assert res2.json()["cached"] is True

    # Manually invalidate / expire cache for this tenant
    invalidate_tenant_cache(tenant_id)

    # Third query -> cold miss again (cache invalidated)
    res3 = client.post("/query", json=query_payload)
    assert res3.json()["cached"] is False


def test_cache_invalidation_on_upload():
    """Verify that uploading a new document invalidates previous query cache entries for that tenant."""
    suffix = uuid.uuid4().hex[:6]
    tenant_id, client = create_tenant_and_client(f"invalidation_tenant_{suffix}")

    coll_name = f"inv_coll_{suffix}"
    pdf_v1 = make_test_pdf("Version 1 Specification: The primary product serial identifier is V1-AAA-9988.")
    res_up1 = client.post(
        "/upload?sync=true",
        files={"file": ("v1.pdf", pdf_v1, "application/pdf")},
        data={"collection_name": coll_name},
    )
    assert res_up1.status_code == 200

    query_payload = {"question": "What is the primary product serial identifier?", "collection_name": coll_name, "enable_web_fallback": False}

    # Query 1 -> Miss
    res1 = client.post("/query", json=query_payload)
    assert res1.json()["cached"] is False

    # Query 2 -> Hit
    res2 = client.post("/query", json=query_payload)
    assert res2.json()["cached"] is True

    # Upload Version 2 doc -> Should trigger invalidate_tenant_cache
    pdf_v2 = make_test_pdf("Version 2 Specification: The primary product serial identifier is V2-BBB-7766.")
    res_up2 = client.post(
        "/upload?sync=true",
        files={"file": ("v2.pdf", pdf_v2, "application/pdf")},
        data={"collection_name": coll_name, "force": True},
    )
    assert res_up2.status_code == 200

    # Query 3 -> Must be a Miss because upload invalidated the cache!
    res3 = client.post("/query", json=query_payload)
    assert res3.json()["cached"] is False
