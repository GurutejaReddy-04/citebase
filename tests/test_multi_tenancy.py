"""
Comprehensive Multi-Tenancy, Authentication, and Rate Limiting Test Suite.

Tests:
1. High-entropy cryptographic API key generation & hashing
2. Missing & invalid API key rejection (401 Unauthorized)
3. Cross-tenant collection isolation (Tenant A data invisible to Tenant B)
4. Prefix-injection attack resistance (Tenant B querying Tenant A's internal raw name)
5. Per-key rate limiting (429 Too Many Requests with Retry-After header)
6. Relational query audit logging in Postgres/SQLite
"""

import hashlib
import io
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import main
from auth import generate_api_key, hash_api_key
from database import SessionLocal
from main import app
from models import ApiKey, DocumentRecord, QueryLog, Tenant

# Mock generator so tests run independently of Gemini API quota
main.generate_answer = lambda q, ctx: f"Mock Grounded Answer with {len(ctx)} sources for: {q}"


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_test_tenant_and_key(db_session, tenant_name: str, rpm: int = 60) -> tuple[Tenant, str]:
    """Helper to create a tenant with a dedicated high-entropy API key."""
    tenant = db_session.query(Tenant).filter(Tenant.name == tenant_name).first()
    if not tenant:
        tenant = Tenant(name=tenant_name, is_active=True)
        db_session.add(tenant)
        db_session.flush()

    raw_key, key_hash, key_prefix = generate_api_key()
    api_key_rec = ApiKey(
        tenant_id=tenant.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=f"Key for {tenant_name}",
        rate_limit_rpm=rpm,
        is_active=True,
    )
    db_session.add(api_key_rec)
    db_session.commit()
    return tenant, raw_key


def test_entropy_key_generation():
    """Verify API keys have 256-bit cryptographic entropy and correct SHA-256 hash."""
    raw_key, key_hash, key_prefix = generate_api_key()
    assert raw_key.startswith("sk_live_")
    assert len(raw_key) >= 48  # Sufficient entropy length
    assert key_prefix == raw_key[:12]
    assert hash_api_key(raw_key) == key_hash


def test_missing_and_invalid_api_key_401():
    """Verify unauthenticated or invalid API key requests receive HTTP 401."""
    unauth_client = TestClient(app)

    # 1. Missing key
    res1 = unauth_client.get("/collections")
    assert res1.status_code == 401
    assert "Missing API Key" in res1.json()["detail"]

    # 2. Invalid/fabricated key
    res2 = unauth_client.get("/collections", headers={"X-API-Key": "sk_live_fake_invalid_key_999"})
    assert res2.status_code == 401
    assert "Invalid or revoked API Key" in res2.json()["detail"]


def make_test_pdf(text: str = "Alpha Corp Confidential Q3 Financial Earnings Report") -> bytes:
    """Generate in-memory valid PDF containing text using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


import uuid


def test_tenant_collection_isolation(db_session):
    """
    Verify complete isolation between Tenant A and Tenant B:
    - Tenant A creates collection and uploads doc.
    - Tenant A sees collection in /collections.
    - Tenant B calls /collections and does not see Tenant A's collection.
    - Tenant B calls /query on collection and receives 0 results.
    """
    unique_suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"t_alpha_{unique_suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"t_beta_{unique_suffix}")

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    pdf_bytes = make_test_pdf("Alpha Corp Q3 Financial Performance: Total Net Revenue is $42 Million USD.")
    coll_name = f"alpha_reports_{unique_suffix}"

    # Tenant A uploads document to collection (sync=true in dev)
    res_upload = client_a.post(
        "/upload?sync=true",
        files={"file": ("alpha_q3.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Alpha Q3 Financials"},
    )
    assert res_upload.status_code == 200

    # Tenant A lists collections -> sees its collection
    res_a_colls = client_a.get("/collections")
    assert res_a_colls.status_code == 200
    assert coll_name in res_a_colls.json()["collections"]

    # Tenant B lists collections -> DOES NOT see Tenant A's collection
    res_b_colls = client_b.get("/collections")
    assert res_b_colls.status_code == 200
    assert coll_name not in res_b_colls.json()["collections"]

    # Tenant B queries collection -> receives 0 local results (Tenant B has no such collection)
    res_b_query = client_b.post(
        "/query",
        json={
            "question": "What is the total net revenue in Q3?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_b_query.status_code == 200
    data_b = res_b_query.json()
    assert len(data_b["sources"]) == 0  # Tenant B has 0 access to Tenant A's chunks


def test_tenant_scoping_cannot_be_bypassed_via_crafted_names(db_session):
    """
    Verify tenant scoping is strictly enforced by construction:
    The tenant ID is derived entirely from the authenticated API key, never from user-supplied input.
    Even if Tenant B crafts a collection name matching Tenant A's internal naming scheme ('t_{tenant_a_id}_secret_vault'),
    the backend namespaces it strictly under Tenant B's identity ('t_{tenant_b_id}_...'), preventing cross-tenant leakage.
    """
    unique_suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"t_victim_{unique_suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"t_attacker_{unique_suffix}")

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    # Tenant A uploads a secret document (sync=true in dev)
    pdf_bytes = make_test_pdf("Victim Corp Proprietary Secret Formula: Compound X-999.")
    coll_name = f"vault_{unique_suffix}"
    res_up = client_a.post(
        "/upload?sync=true",
        files={"file": ("victim_secret.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Victim Secrets"},
    )
    assert res_up.status_code == 200

    # Tenant B passes a crafted collection name mimicking Tenant A's internal scheme
    crafted_coll_name = f"t_{tenant_a.id[:12]}_{coll_name}"

    res = client_b.post(
        "/query",
        json={
            "question": "What is Compound X-999?",
            "collection_name": crafted_coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["sources"]) == 0  # Resolves under Tenant B's namespace, completely isolated


def test_rate_limiting_429(db_session):
    """
    Verify per-key RPM rate limiting:
    Configure a key with limit=3 RPM. Send 4 requests in rapid succession;
    the 4th request must return HTTP 429 Too Many Requests with 'Retry-After' header.
    """
    tenant, rate_limited_key = create_test_tenant_and_key(
        db_session,
        "tenant_rate_limited_corp",
        rpm=3,
    )

    client = TestClient(app)
    client.headers.update({"X-API-Key": rate_limited_key})

    # Requests 1, 2, 3 should succeed
    for i in range(3):
        res = client.get("/collections")
        assert res.status_code == 200, f"Request {i+1} failed unexpectedly"

    # Request 4 should be blocked with 429
    res_blocked = client.get("/collections")
    assert res_blocked.status_code == 429
    assert "Rate limit of 3 requests per minute exceeded" in res_blocked.json()["detail"]
    assert "Retry-After" in res_blocked.headers
    assert int(res_blocked.headers["Retry-After"]) > 0


def test_query_audit_logging(db_session):
    """Verify successful queries are logged to the relational audit log table."""
    tenant, key = create_test_tenant_and_key(db_session, "tenant_audit_corp")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    q_text = "Audit logging verification query test"
    res = client.post(
        "/query",
        json={
            "question": q_text,
            "collection_name": "test_hybrid_coll",
            "enable_web_fallback": False,
        },
    )
    assert res.status_code == 200

    # Query DB audit table
    log_entry = (
        db_session.query(QueryLog)
        .filter(QueryLog.tenant_id == tenant.id, QueryLog.question == q_text)
        .first()
    )
    assert log_entry is not None
    assert log_entry.retrieval_mode == "local_document"
    assert log_entry.latency_ms >= 0.0
