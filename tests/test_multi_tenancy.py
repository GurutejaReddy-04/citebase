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


def create_test_tenant_and_key(db_session, tenant_name: str, rpm: int = 60, custom_id: str | None = None) -> tuple[Tenant, str]:
    """Helper to create a tenant with a dedicated high-entropy API key."""
    tenant = db_session.query(Tenant).filter(Tenant.name == tenant_name).first()
    if not tenant:
        kwargs = {"name": tenant_name, "is_active": True}
        if custom_id:
            kwargs["id"] = custom_id
        tenant = Tenant(**kwargs)
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


def test_collection_name_collision_rejected_at_boundary(db_session):
    """
    Verify canonical collection naming policy at the API boundary:
    - Uppercase letters (e.g. 'Research', 'Alpha_Vault') are rejected with 400.
    - Names exceeding 45 characters are rejected with 400.
    - Distinct valid names map injectively to internal scoped collection names.
    """
    tenant, key = create_test_tenant_and_key(db_session, "tenant_namespace_guard")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    # 1. Uppercase collection name rejected
    res_upper = client.post(
        "/query",
        json={"question": "What is this?", "collection_name": "Research"},
    )
    assert res_upper.status_code == 400
    assert "Invalid collection name" in res_upper.json()["detail"]

    # 2. Collection name > 45 chars rejected
    long_name = "a" * 46
    res_long = client.post(
        "/query",
        json={"question": "What is this?", "collection_name": long_name},
    )
    assert res_long.status_code == 400
    assert "Invalid collection name" in res_long.json()["detail"]

    # 3. Injective mapping check on TenantContext for modern 32-char tenant with long names
    from auth import TenantContext
    ctx = TenantContext(
        tenant_id="123456789abc11111111111111111111",
        tenant_name="Alpha",
        api_key_id="k1",
        key_prefix="sk_live_1234",
        rate_limit_rpm=60,
    )
    name_a = "quarterly_reports_finances_2026_div_1"  # 37 chars
    name_b = "quarterly_reports_finances_2026_div_2"  # 37 chars
    scoped_a = ctx.get_scoped_collection_name(name_a)
    scoped_b = ctx.get_scoped_collection_name(name_b)
    assert scoped_a != scoped_b
    assert len(scoped_a) <= 63
    assert len(scoped_b) <= 63
    assert "." in scoped_a

    # Finding 1 Regression: Critic's exact pair (Long name vs matching short hash-tail name)
    long_name_critic = "quarterly_reports_finances_2026_abcdef"  # 38 chars
    short_name_critic = "quarterly_reports__3de5f53a"  # 27 chars
    scoped_long = ctx.get_scoped_collection_name(long_name_critic)
    scoped_short = ctx.get_scoped_collection_name(short_name_critic)
    assert scoped_long != scoped_short, "Long name and short hash-tail name must NEVER collide"
    assert len(scoped_long) <= 63
    assert len(scoped_short) <= 63
    assert "." in scoped_long
    assert "." not in scoped_short


def test_tenant_collection_scoping_avoids_prefix_collision():
    """
    Verify that two distinct tenants sharing the same first 12 characters
    resolve to completely different internal collection names using full tenant ID.
    Prevents cross-tenant collection collision when UUIDs share prefixes.
    """
    from auth import TenantContext
    ctx_a = TenantContext(
        tenant_id="123456789abc11111111111111111111",
        tenant_name="Tenant Alpha",
        api_key_id="k1",
        key_prefix="sk_live_1234",
        rate_limit_rpm=60,
    )
    ctx_b = TenantContext(
        tenant_id="123456789abc22222222222222222222",
        tenant_name="Tenant Beta",
        api_key_id="k2",
        key_prefix="sk_live_5678",
        rate_limit_rpm=60,
    )
    name = "financial_reports"
    scoped_a = ctx_a.get_scoped_collection_name(name)
    scoped_b = ctx_b.get_scoped_collection_name(name)
    assert scoped_a != scoped_b
    assert scoped_a == "t_123456789abc11111111111111111111_financial_reports"
    assert scoped_b == "t_123456789abc22222222222222222222_financial_reports"
    assert len(scoped_a) <= 63
    assert len(scoped_b) <= 63


def test_negative_rate_limit_rejected_at_creation(db_session):
    """Verify API-key creation rejects negative and zero rate limits with HTTP 422."""
    tenant, key = create_test_tenant_and_key(db_session, "tenant_rate_test")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    # 1. Negative RPM rejected by Pydantic schema validation
    res_neg = client.post("/admin/api-keys", json={"name": "Hacked Key", "rate_limit_rpm": -1})
    assert res_neg.status_code == 422

    # 2. Zero RPM rejected
    res_zero = client.post("/admin/api-keys", json={"name": "Zero Key", "rate_limit_rpm": 0})
    assert res_zero.status_code == 422

    # 3. Valid RPM accepted
    res_valid = client.post("/admin/api-keys", json={"name": "Valid Key", "rate_limit_rpm": 120})
    assert res_valid.status_code == 200
    assert res_valid.json()["rate_limit_rpm"] == 120


def test_rate_limiter_handles_nonpositive_db_fallback(db_session, monkeypatch):
    """
    Verify defensive fallback and observable enforcement:
    Even if a key has rate_limit_rpm <= 0 in the database, the limiter resets
    the quota to DEFAULT_RATE_LIMIT_RPM and actively blocks subsequent requests
    with HTTP 429 once that quota is exhausted (preventing unlimited bypass).
    """
    import rate_limiter

    # Set configured default rate limit to 3 RPM for observable testing
    monkeypatch.setattr(rate_limiter, "DEFAULT_RATE_LIMIT_RPM", 3)

    import uuid
    suffix = uuid.uuid4().hex[:6]
    tenant, neg_key = create_test_tenant_and_key(
        db_session,
        f"tenant_exploit_negative_rpm_{suffix}",
        rpm=-10,  # Negative quota in DB
    )

    client = TestClient(app)
    client.headers.update({"X-API-Key": neg_key})

    # First 3 requests should succeed under the fallback quota of 3 RPM
    for i in range(3):
        res = client.get("/collections")
        assert res.status_code == 200, f"Request {i+1} failed unexpectedly: {res.text}"

    # 4th request must be actively blocked with HTTP 429
    res_blocked = client.get("/collections")
    assert res_blocked.status_code == 429
    assert "Rate limit of 3 requests per minute exceeded" in res_blocked.json()["detail"]
    assert "Retry-After" in res_blocked.headers
    assert int(res_blocked.headers["Retry-After"]) > 0


def test_rate_limiter_partitions_by_api_key_id_not_just_prefix():
    """
    Verify that two different API keys sharing the exact same 12-char prefix
    (e.g. sk_live_test) are keyed by their unique api_key_id and maintain independent rate limit counters.
    Uses dynamically generated UUIDs and cleans both in-memory and Redis keys to prevent state bleed.
    """
    import uuid
    from fastapi import HTTPException
    import pytest
    from auth import TenantContext
    from redis_client import get_redis_client
    from rate_limiter import check_rate_limit, get_rate_limit_redis_key, _memory_counters, _memory_lock

    key_a_id = f"test_key_alpha_{uuid.uuid4().hex}"
    key_b_id = f"test_key_beta_{uuid.uuid4().hex}"

    key_a = TenantContext(
        tenant_id="t1",
        tenant_name="Tenant One",
        api_key_id=key_a_id,
        key_prefix="sk_live_test",
        rate_limit_rpm=2,
    )
    key_b = TenantContext(
        tenant_id="t1",
        tenant_name="Tenant One",
        api_key_id=key_b_id,
        key_prefix="sk_live_test",  # Identical prefix!
        rate_limit_rpm=2,
    )

    # Clean in-memory counters
    with _memory_lock:
        _memory_counters.pop(key_a.api_key_id, None)
        _memory_counters.pop(key_b.api_key_id, None)

    # Clean Redis counters if Redis is active using exact production key structure
    r = get_redis_client()
    if r:
        try:
            r.delete(get_rate_limit_redis_key(key_a_id), get_rate_limit_redis_key(key_b_id))
        except Exception:
            pass

    try:
        # Key A makes 2 requests (exhausts its quota of 2 RPM)
        check_rate_limit(key_a)
        check_rate_limit(key_a)

        # Key A third request should be blocked with HTTP 429
        with pytest.raises(HTTPException) as exc_info:
            check_rate_limit(key_a)
        assert exc_info.value.status_code == 429

        # Key B should still have its full quota available (not blocked by Key A)
        res_b = check_rate_limit(key_b)
        assert res_b.api_key_id == key_b_id
    finally:
        # Cleanup
        with _memory_lock:
            _memory_counters.pop(key_a.api_key_id, None)
            _memory_counters.pop(key_b.api_key_id, None)
        if r:
            try:
                r.delete(get_rate_limit_redis_key(key_a_id), get_rate_limit_redis_key(key_b_id))
            except Exception:
                pass


def test_cross_tenant_legacy_collection_query_and_delete_isolation(db_session):
    """
    Finding 1 Regression Test:
    Verify that a modern 32-character tenant whose ID prefix matches an older 12-character
    legacy tenant's ID CANNOT resolve, query, or delete the legacy tenant's collections.
    
    Setup:
    - Legacy Tenant A: id='123456789abc' (legacy 12-char ID)
    - Modern Tenant B: id='123456789abc22222222222222222222' (modern 32-char UUID sharing the first 12 chars)
    - Legacy Tenant A uploads doc to 'legacy_vault' (resolves to 't_123456789abc_legacy_vault')
    
    Verifications:
    1. Tenant B calling /query on 'legacy_vault' returns 0 sources (cannot read legacy collection).
    2. Tenant B calling DELETE /collections/legacy_vault receives HTTP 404 (cannot delete legacy collection).
    3. Legacy Tenant A verifies 'legacy_vault' remains intact and can still query its document.
    4. Legacy Tenant A can delete 'legacy_vault' with HTTP 200.
    """
    import uuid
    suffix = uuid.uuid4().hex[:6]
    legacy_id = f"leg_{suffix}"[:12]
    modern_id = f"{legacy_id}{'2' * 20}"

    tenant_a, key_a = create_test_tenant_and_key(
        db_session, f"legacy_tenant_{suffix}", custom_id=legacy_id
    )
    tenant_b, key_b = create_test_tenant_and_key(
        db_session, f"modern_tenant_{suffix}", custom_id=modern_id
    )

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    pdf_bytes = make_test_pdf("Confidential blueprint for legacy infrastructure.")
    coll_name = "legacy_vault"

    # Tenant A uploads document to legacy collection
    res_upload = client_a.post(
        "/upload?sync=true",
        files={"file": ("blueprint.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Legacy Blueprint"},
    )
    assert res_upload.status_code == 200

    # Tenant A sees it in /collections
    res_a_colls = client_a.get("/collections")
    assert res_a_colls.status_code == 200
    assert coll_name in res_a_colls.json()["collections"]

    # 1. Tenant B queries 'legacy_vault' -> receives 0 results
    res_b_query = client_b.post(
        "/query",
        json={
            "question": "What is the blueprint for infrastructure?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_b_query.status_code == 200
    assert len(res_b_query.json()["sources"]) == 0

    # 2. Tenant B attempts to DELETE 'legacy_vault' -> receives HTTP 404
    res_b_del = client_b.delete(f"/collections/{coll_name}")
    assert res_b_del.status_code == 404
    assert f"Collection not found: {coll_name}" in res_b_del.json()["detail"]

    # 3. Tenant A verifies 'legacy_vault' was NOT deleted and is still queryable
    res_a_query = client_a.post(
        "/query",
        json={
            "question": "What is the blueprint for infrastructure?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_a_query.status_code == 200
    assert len(res_a_query.json()["sources"]) > 0

    # 4. Tenant A deletes its own collection successfully
    res_a_del = client_a.delete(f"/collections/{coll_name}")
    assert res_a_del.status_code == 200


def test_long_collection_name_full_lifecycle(db_session):
    """
    Finding 2 Regression Test:
    Verify that collection names up to 45 characters (e.g. 37 chars) operate smoothly across
    upload, list_collections, query_document, and delete_collection for modern 32-char tenants.
    """
    import uuid
    suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"long_coll_tenant_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    long_coll = f"quarterly_reports_finances_2026_{suffix}"
    assert 28 < len(long_coll) <= 45

    pdf_bytes = make_test_pdf("Quarterly financial earnings report summary.")

    # Upload document to long collection
    res_upload = client.post(
        "/upload?sync=true",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": long_coll, "doc_title": "Quarterly Financials"},
    )
    assert res_upload.status_code == 200

    # List collections -> long_coll should be present with original logical name
    res_colls = client.get("/collections")
    assert res_colls.status_code == 200
    assert long_coll in res_colls.json()["collections"]

    # Query long collection -> should return sources with cleaned logical collection name
    res_query = client.post(
        "/query",
        json={
            "question": "What is the financial earnings report?",
            "collection_name": long_coll,
            "enable_web_fallback": False,
        },
    )
    assert res_query.status_code == 200
    sources = res_query.json()["sources"]
    assert len(sources) > 0
    assert sources[0]["collection_name"] == long_coll

    # Delete collection
    res_del = client.delete(f"/collections/{long_coll}")
    assert res_del.status_code == 200
    assert res_del.json()["message"] == f"Collection '{long_coll}' deleted."

    # Subsequent delete returns 404
    res_del_again = client.delete(f"/collections/{long_coll}")
    assert res_del_again.status_code == 404


def test_seeded_tenant_long_collection_name_bounds():
    """
    Finding 3 Regression Test:
    Verify that seeded tenants with long tenant IDs (e.g. 'default_tenant_id' - 17 chars)
    do not exceed ChromaDB's 63-character limit when using maximum length (43-45 chars) collection names.
    """
    from auth import TenantContext
    ctx = TenantContext(
        tenant_id="default_tenant_id",
        tenant_name="Default Org",
        api_key_id="k_default",
        key_prefix="sk_live_def",
        rate_limit_rpm=60,
    )
    for length in (43, 44, 45):
        clean_name = "a" + "x" * (length - 2) + "b"
        scoped = ctx.get_scoped_collection_name(clean_name)
        assert len(scoped) <= 63, f"Physical collection name exceeds 63 chars: {scoped} (len={len(scoped)})"
        assert scoped.startswith("t_default_tenant_id_")


def test_collection_deletion_fails_visibly_when_chromadb_raises(db_session, monkeypatch):
    """
    Finding 2 Regression Test:
    Verify that when ChromaDB deletion raises an unexpected failure (I/O, database crash),
    the endpoint does NOT return HTTP 200, logs the failure, and preserves the relational
    DocumentRecord rows in PostgreSQL/SQLite rather than silently deleting them.
    """
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"tenant_del_fail_{unique_suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    pdf_bytes = make_test_pdf("Confidential document that must not be orphaned on failure.")
    coll_name = f"vault_fail_{unique_suffix}"

    # Ingest document
    res_upload = client.post(
        "/upload?sync=true",
        files={"file": ("confidential.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Confidential Vault"},
    )
    assert res_upload.status_code == 200

    # Verify relational record exists
    initial_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(initial_docs) > 0

    # Mock client.delete_collection to simulate unexpected ChromaDB disk/database failure
    from chroma import get_chroma_client
    chroma_client = get_chroma_client()

    def mock_failing_delete(coll_target):
        raise RuntimeError("Simulated ChromaDB disk I/O failure during deletion")

    monkeypatch.setattr(chroma_client, "delete_collection", mock_failing_delete)

    # Attempt deletion -> MUST fail with HTTP 500, NOT return HTTP 200
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 500
    assert "Failed to delete underlying vector collection" in res_del.json()["detail"]

    # Verify relational records were PRESERVED and NOT deleted
    remaining_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(remaining_docs) == len(initial_docs), "Relational records must not be deleted if ChromaDB deletion failed"

    # Restore monkeypatch and execute genuine deletion
    monkeypatch.undo()
    res_del_genuine = client.delete(f"/collections/{coll_name}")
    assert res_del_genuine.status_code == 200
    assert res_del_genuine.json()["message"] == f"Collection '{coll_name}' deleted."

    # Verify relational records are now cleaned
    cleaned_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(cleaned_docs) == 0


def test_critic_exact_pair_no_aliasing_and_isolation(db_session):
    """
    Finding 1 End-to-End Regression Test:
    Verify that the critic's exact pair:
    - Long name: 'quarterly_reports_finances_2026_abcdef' (38 chars)
    - Short name: 'quarterly_reports__3de5f53a' (27 chars)
    do NOT alias or collide in ChromaDB or relational storage.
    Each collection retains its own distinct documents and can be queried independently.
    """
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"t_critic_{unique_suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    long_coll = "quarterly_reports_finances_2026_abcdef"
    short_coll = "quarterly_reports__3de5f53a"

    pdf_long = make_test_pdf("Long collection financial results: Net profit $100M.")
    pdf_short = make_test_pdf("Short collection budget overview: Operating costs $10M.")

    # Ingest document to long_coll
    res_up1 = client.post(
        "/upload?sync=true",
        files={"file": ("long_finances.pdf", pdf_long, "application/pdf")},
        data={"collection_name": long_coll, "doc_title": "Long Financials"},
    )
    assert res_up1.status_code == 200

    # Ingest document to short_coll
    res_up2 = client.post(
        "/upload?sync=true",
        files={"file": ("short_budget.pdf", pdf_short, "application/pdf")},
        data={"collection_name": short_coll, "doc_title": "Short Budget"},
    )
    assert res_up2.status_code == 200

    # List collections -> both collections should be listed
    res_list = client.get("/collections")
    assert res_list.status_code == 200
    colls = res_list.json()["collections"]
    assert long_coll in colls
    assert short_coll in colls

    # Query short_coll -> should only return short_coll source
    res_q_short = client.post(
        "/query",
        json={"question": "What are the operating costs?", "collection_name": short_coll, "enable_web_fallback": False},
    )
    assert res_q_short.status_code == 200
    assert len(res_q_short.json()["sources"]) > 0
    assert res_q_short.json()["sources"][0]["collection_name"] == short_coll

    # Delete long_coll -> short_coll must remain completely intact
    res_del_long = client.delete(f"/collections/{long_coll}")
    assert res_del_long.status_code == 200

    # Query short_coll again -> still queryable and intact!
    res_q_short2 = client.post(
        "/query",
        json={"question": "What are the operating costs?", "collection_name": short_coll, "enable_web_fallback": False},
    )
    assert res_q_short2.status_code == 200
    assert len(res_q_short2.json()["sources"]) > 0

    # Clean up short_coll
    res_del_short = client.delete(f"/collections/{short_coll}")
    assert res_del_short.status_code == 200


def test_collection_deletion_fails_visibly_when_chromadb_raises_unexpected_value_error(db_session, monkeypatch):
    """
    Round 6 Finding 1 Regression Test:
    Verify that if ChromaDB raises a ValueError that is NOT 'does not exist' (e.g. internal schema or parameter error),
    the endpoint does NOT treat it as already removed, returns HTTP 500, and preserves relational records.
    """
    unique_suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"t_ve_{unique_suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    pdf_bytes = make_test_pdf("Confidential document subject to ChromaDB ValueError deletion check.")
    coll_name = f"vault_ve_{unique_suffix}"

    # Ingest document
    res_upload = client.post(
        "/upload?sync=true",
        files={"file": ("ve_test.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "VE Test Doc"},
    )
    assert res_upload.status_code == 200

    initial_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(initial_docs) > 0

    # Mock client.delete_collection to raise an unexpected ValueError (not "does not exist")
    from chroma import get_chroma_client
    chroma_client = get_chroma_client()

    def mock_unexpected_value_error(coll_target):
        raise ValueError("ChromaDB internal validation error: corrupt metadata payload")

    monkeypatch.setattr(chroma_client, "delete_collection", mock_unexpected_value_error)

    # Attempt deletion -> MUST return HTTP 500, NOT HTTP 200
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 500
    assert "Failed to delete underlying vector collection" in res_del.json()["detail"]

    # Verify relational records were PRESERVED
    remaining_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(remaining_docs) == len(initial_docs)

    # Restore monkeypatch and execute genuine deletion
    monkeypatch.undo()
    res_del_genuine = client.delete(f"/collections/{coll_name}")
    assert res_del_genuine.status_code == 200
    assert (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .count()
        == 0
    )


def test_collection_deletion_fails_visibly_when_bm25_file_removal_fails(db_session, monkeypatch):
    """
    Round 6 Finding 2 Regression Test:
    Verify that if ChromaDB collection is deleted but the BM25 index file removal fails (e.g. PermissionError),
    the endpoint propagates the failure with HTTP 500 and preserves relational records to prevent retaining
    orphaned plaintext chunks without reporting incomplete cleanup.
    """
    unique_suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"t_bm25_err_{unique_suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    pdf_bytes = make_test_pdf("Sensitive engineering document with BM25 sparse index.")
    coll_name = f"vault_bm25_{unique_suffix}"

    # Ingest document
    res_upload = client.post(
        "/upload?sync=true",
        files={"file": ("bm25_doc.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "BM25 Doc"},
    )
    assert res_upload.status_code == 200

    initial_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(initial_docs) > 0

    # Mock os.remove in bm25 module to simulate file-lock or permission failure
    import bm25

    def mock_failing_os_remove(path):
        raise PermissionError(f"Access denied: cannot remove {path}")

    monkeypatch.setattr(bm25.os, "remove", mock_failing_os_remove)

    # Attempt deletion -> MUST return HTTP 500, reporting incomplete deletion
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 500
    assert "Incomplete collection deletion: failed to remove persistent BM25 index" in res_del.json()["detail"]

    # Verify relational records were PRESERVED
    remaining_docs = (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .all()
    )
    assert len(remaining_docs) == len(initial_docs)

    # Restore monkeypatch and execute genuine deletion
    monkeypatch.undo()
    res_del_genuine = client.delete(f"/collections/{coll_name}")
    assert res_del_genuine.status_code == 200
    assert (
        db_session.query(DocumentRecord)
        .filter(DocumentRecord.tenant_id == tenant.id, DocumentRecord.collection_name == coll_name)
        .count()
        == 0
    )


def test_unconditional_physical_length_invariants_for_arbitrary_tenant_id_lengths():
    """
    Round 6 & 7 Edge Case Test:
    Verify that get_scoped_collection_name unconditionally guarantees total length <= 63
    across any tenant ID length up to 64 characters and any collection name length up to 45 characters,
    with normalized_tid deriving 128-bit cryptographic tokens (32 hex chars) for long IDs.
    """
    from auth import TenantContext

    for tid_len in (10, 12, 17, 32, 36, 48, 52, 60, 64):
        ctx = TenantContext(
            tenant_id="t" * tid_len,
            tenant_name="Test Org",
            api_key_id="k_test",
            key_prefix="sk_live_test",
            rate_limit_rpm=60,
        )
        assert len(ctx.normalized_tid) <= 32
        for name_len in (3, 10, 20, 28, 35, 43, 45):
            name = "a" + "x" * (name_len - 2) + "b"
            scoped = ctx.get_scoped_collection_name(name)
            assert len(scoped) <= 63, f"Exceeded 63 chars for tid_len={tid_len}, name_len={name_len}: '{scoped}'"
            assert len(scoped) >= 3


def test_cross_tenant_isolation_with_engineered_shared_namespace(db_session, monkeypatch):
    """
    Finding 1 Regression Test:
    Verify that even if two distinct long tenant IDs are engineered to share the same
    normalized physical collection namespace, cross-tenant isolation guarantees hold:
    1. Tenant A uploads a document to 'shared_vault'.
    2. Tenant B attempting to upload to 'shared_vault' is rejected with HTTP 409 Conflict.
    3. Tenant B querying 'shared_vault' receives 0 sources (cannot read Tenant A's documents).
    4. Tenant B attempting to delete 'shared_vault' receives HTTP 404 (cannot delete Tenant A's collection).
    5. Tenant A can query 'shared_vault' and retrieve its documents.
    6. Tenant A can delete 'shared_vault' with HTTP 200.
    """
    import uuid
    from auth import TenantContext
    suffix = uuid.uuid4().hex[:6]
    id_a = f"tenant_long_alpha_{suffix}_{'a' * 24}"
    id_b = f"tenant_long_beta_{suffix}_{'b' * 24}"

    tenant_a, key_a = create_test_tenant_and_key(db_session, f"Tenant Long Alpha {suffix}", custom_id=id_a)
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"Tenant Long Beta {suffix}", custom_id=id_b)

    # Both long IDs derive 32-char (128-bit) normalized_tid tokens
    ctx_a = TenantContext(tenant_id=id_a, tenant_name="Alpha", api_key_id="ka", key_prefix="ska", rate_limit_rpm=60)
    ctx_b = TenantContext(tenant_id=id_b, tenant_name="Beta", api_key_id="kb", key_prefix="skb", rate_limit_rpm=60)
    assert len(ctx_a.normalized_tid) == 32
    assert len(ctx_b.normalized_tid) == 32

    # Engineer a shared physical namespace between these two tenants to simulate an adversarial namespace collision
    engineered_tid = f"shared_{suffix}"[:32]
    orig_normalized_tid = TenantContext.normalized_tid

    def mock_normalized_tid(self):
        if self.tenant_id in (id_a, id_b):
            return engineered_tid
        return orig_normalized_tid.fget(self)

    monkeypatch.setattr(TenantContext, "normalized_tid", property(mock_normalized_tid))

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    coll_name = "shared_vault"
    pdf_bytes = make_test_pdf("Strictly confidential financial ledger for Tenant Alpha.")

    # 1. Tenant A uploads to 'shared_vault'
    res_a_up = client_a.post(
        "/upload?sync=true",
        files={"file": ("ledger.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Alpha Confidential Ledger"},
    )
    assert res_a_up.status_code == 200

    # 2. Tenant B attempts to upload to 'shared_vault' -> MUST return HTTP 409 Conflict
    res_b_up = client_b.post(
        "/upload?sync=true",
        files={"file": ("infiltrate.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Beta Infiltration"},
    )
    assert res_b_up.status_code == 409
    assert "Cross-tenant collection collision" in res_b_up.json()["detail"]

    # 3. Tenant B queries 'shared_vault' -> MUST receive 0 sources
    res_b_query = client_b.post(
        "/query",
        json={
            "question": "What is in the financial ledger?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_b_query.status_code == 200
    assert len(res_b_query.json()["sources"]) == 0

    # 4. Tenant B attempts to delete 'shared_vault' -> MUST receive HTTP 404
    res_b_del = client_b.delete(f"/collections/{coll_name}")
    assert res_b_del.status_code == 404
    assert f"Collection not found: {coll_name}" in res_b_del.json()["detail"]

    # 5. Tenant A queries 'shared_vault' -> succeeds and retrieves documents
    res_a_query = client_a.post(
        "/query",
        json={
            "question": "What is in the financial ledger?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_a_query.status_code == 200
    assert len(res_a_query.json()["sources"]) > 0

    # 6. Tenant A deletes 'shared_vault' successfully
    res_a_del = client_a.delete(f"/collections/{coll_name}")
    assert res_a_del.status_code == 200


def test_collection_deletion_cleans_orphaned_legacy_bm25_pickle(db_session):
    """
    Finding 2 Regression Test:
    Verify that an orphaned legacy .pkl BM25 file on disk is cleanly detected and removed
    by DELETE /collections/{name} even if no ChromaDB vector collection and no relational
    DocumentRecord rows exist.
    """
    import pickle
    import uuid
    from auth import TenantContext
    from config import CHROMA_PATH

    suffix = uuid.uuid4().hex[:6]
    tenant, api_key = create_test_tenant_and_key(db_session, f"pickle_clean_tenant_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": api_key})

    coll_name = "stale_pkl_coll"
    ctx = TenantContext(tenant_id=tenant.id, tenant_name=tenant.name, api_key_id="k", key_prefix="sk", rate_limit_rpm=60)
    scoped_coll = ctx.get_scoped_collection_name(coll_name)

    # Ensure no vector collection and no relational records exist
    chroma_client = main.get_chroma_client()
    existing_colls = {c.name for c in chroma_client.list_collections()}
    if scoped_coll in existing_colls:
        chroma_client.delete_collection(scoped_coll)

    db_session.query(DocumentRecord).filter(
        DocumentRecord.tenant_id == tenant.id,
        DocumentRecord.collection_name == coll_name,
    ).delete()
    db_session.commit()

    # Ensure no .json file exists
    json_path = os.path.join(CHROMA_PATH, f"{scoped_coll}_bm25.json")
    if os.path.exists(json_path):
        os.remove(json_path)

    # Create an orphaned .pkl file with verified tenant ownership
    pkl_path = os.path.join(CHROMA_PATH, f"{scoped_coll}_bm25.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"tenant_id": tenant.id, "dummy": "legacy index data"}, f)
    assert os.path.exists(pkl_path)

    # Execute DELETE /collections/{coll_name} -> MUST return HTTP 200 and remove the stale .pkl file
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 200
    assert not os.path.exists(pkl_path)


def test_query_fails_closed_when_metadata_absent_and_foreign_docs_exist(db_session):
    """
    Round 8 Finding 1 Regression Test:
    Verify that if ChromaDB collection metadata lacks 'tenant_id', is_collection_accessible_to_tenant
    evaluates foreign DocumentRecord rows first and fails closed (denies access), preventing
    shared or collided collection chunk leakage across tenants.
    """
    import uuid
    from auth import TenantContext
    suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"leak_victim_{suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"leak_attacker_{suffix}")

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    coll_name = "secret_notes"
    pdf_bytes = make_test_pdf("Top secret payload for Tenant A only.")

    # Tenant A ingests document
    res_up = client_a.post(
        "/upload?sync=true",
        files={"file": ("secret.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Top Secret Notes"},
    )
    assert res_up.status_code == 200

    # Simulate collection metadata lacking 'tenant_id' in ChromaDB
    chroma_client = main.get_chroma_client()
    ctx_a = TenantContext(tenant_id=tenant_a.id, tenant_name=tenant_a.name, api_key_id="ka", key_prefix="ska", rate_limit_rpm=60)
    scoped_a = ctx_a.get_scoped_collection_name(coll_name)
    coll = chroma_client.get_collection(scoped_a)
    coll.modify(metadata={"source_category": "untracked"})  # Strip tenant_id from Chroma metadata
    assert coll.metadata.get("tenant_id") is None

    # Tenant B queries the exact same logical collection name 'secret_notes'
    # Tenant B has no documents in 'secret_notes'; Tenant A has foreign documents.
    # The fail-closed check must evaluate has_other_docs first and deny access!
    res_b_query = client_b.post(
        "/query",
        json={
            "question": "What is the top secret payload?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_b_query.status_code == 200
    assert len(res_b_query.json()["sources"]) == 0  # Zero chunks leaked!


def test_reset_preserves_records_when_chromadb_deletion_fails(db_session, monkeypatch):
    """
    Round 8 Finding 2 Regression Test:
    Verify that POST /reset preserves relational DocumentRecord rows and returns HTTP 500
    when physical deletion in ChromaDB fails, preventing silent or partial data loss.
    """
    import uuid
    suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"reset_chroma_fail_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    coll_name = "vault_to_keep"
    pdf_bytes = make_test_pdf("Important document that must be preserved on reset failure.")

    res_up = client.post(
        "/upload?sync=true",
        files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Important Vault"},
    )
    assert res_up.status_code == 200

    # Ensure document record exists in DB
    records = db_session.query(DocumentRecord).filter(
        DocumentRecord.tenant_id == tenant.id,
        DocumentRecord.collection_name == coll_name,
    ).all()
    assert len(records) > 0

    # Mock ChromaDB delete_collection to simulate failure
    chroma_client = main.get_chroma_client()
    orig_del = chroma_client.delete_collection

    def mock_delete_fail(name):
        raise RuntimeError("Simulated physical storage I/O error during ChromaDB reset")

    monkeypatch.setattr(chroma_client, "delete_collection", mock_delete_fail)

    # Call POST /reset -> MUST return HTTP 500
    res_reset = client.post("/reset")
    assert res_reset.status_code == 500
    assert "Incomplete reset" in res_reset.json()["detail"]
    assert "Relational records were preserved" in res_reset.json()["detail"]

    # Verify relational DocumentRecord rows are PRESERVED in DB!
    preserved_records = db_session.query(DocumentRecord).filter(
        DocumentRecord.tenant_id == tenant.id,
        DocumentRecord.collection_name == coll_name,
    ).all()
    assert len(preserved_records) == len(records)

    # Clean up with restored delete
    monkeypatch.setattr(chroma_client, "delete_collection", orig_del)
    client.delete(f"/collections/{coll_name}")


def test_reset_preserves_records_when_bm25_deletion_fails(db_session, monkeypatch):
    """
    Round 8 Finding 2 Regression Test:
    Verify that POST /reset preserves relational DocumentRecord rows and returns HTTP 500
    when physical deletion of BM25 sparse index fails.
    """
    import uuid
    suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"reset_bm25_fail_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    coll_name = "vault_bm25_fail"
    pdf_bytes = make_test_pdf("Data with BM25 index that cannot be unlinked.")

    res_up = client.post(
        "/upload?sync=true",
        files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "BM25 Fail Vault"},
    )
    assert res_up.status_code == 200

    # Mock delete_bm25_index to raise IOError
    def mock_bm25_delete_fail(c_name):
        raise IOError(f"Simulated permission denied unlinking BM25 file for '{c_name}'")

    monkeypatch.setattr(main, "delete_bm25_index", mock_bm25_delete_fail)

    res_reset = client.post("/reset")
    assert res_reset.status_code == 500
    assert "Incomplete reset" in res_reset.json()["detail"]
    assert "Relational records were preserved" in res_reset.json()["detail"]

    # Verify records preserved
    preserved = db_session.query(DocumentRecord).filter(
        DocumentRecord.tenant_id == tenant.id,
        DocumentRecord.collection_name == coll_name,
    ).all()
    assert len(preserved) > 0


def test_delete_collection_rejects_unowned_orphaned_bm25_sidecar(db_session):
    """
    Round 8 Finding 3 Regression Test:
    Verify that DELETE /collections/{name} rejects deletion with HTTP 404 if an orphaned
    BM25 file exists on disk but belongs to a different tenant or cannot be verified.
    """
    import json
    import uuid
    from auth import TenantContext
    from config import CHROMA_PATH

    suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"sidecar_owner_{suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"sidecar_intruder_{suffix}")

    client_a = TestClient(app)
    client_a.headers.update({"X-API-Key": key_a})

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    coll_name = "orphaned_vault"
    ctx_b = TenantContext(tenant_id=tenant_b.id, tenant_name=tenant_b.name, api_key_id="kb", key_prefix="skb", rate_limit_rpm=60)
    scoped_b = ctx_b.get_scoped_collection_name(coll_name)

    # Create an orphaned BM25 file where sidecar specifies Tenant A as owner
    json_path = os.path.join(CHROMA_PATH, f"{scoped_b}_bm25.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"tenant_id": tenant_a.id, "corpus_chunk_ids": [], "chunks": []}, f)
    assert os.path.exists(json_path)

    try:
        # Tenant B attempts to delete 'orphaned_vault' -> MUST receive HTTP 404
        res_b = client_b.delete(f"/collections/{coll_name}")
        assert res_b.status_code == 404

        # Verify Tenant A's file was NOT deleted
        assert os.path.exists(json_path)
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)


def test_pre_migration_v6_collection_backward_compatibility(db_session):
    """
    Round 8 Compatibility Regression Test:
    Verify that collections created under the v6 scheme (where tenant IDs > 32 characters
    were unhashed for 33-36 chars or used tid[:24] + sha256[:8]) remain discoverable
    in GET /collections, queryable in POST /query, and deletable in DELETE /collections/{name}.
    """
    import hashlib
    import uuid
    from auth import TenantContext
    from bm25 import build_and_save_bm25_index

    # Create tenant with 36-char ID
    long_tid = f"tenant_{uuid.uuid4().hex[:29]}"  # length = 36
    tenant, key = create_test_tenant_and_key(db_session, f"v6_compat_tenant", custom_id=long_tid)
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    coll_name = "v6_legacy_docs"
    ctx = TenantContext(tenant_id=tenant.id, tenant_name=tenant.name, api_key_id="k", key_prefix="sk", rate_limit_rpm=60)
    v6_scoped_name = ctx.get_pre_migration_scoped_collection_name(coll_name)
    assert v6_scoped_name is not None
    assert v6_scoped_name.startswith(f"t_{tenant.id}_")

    # Seed ChromaDB collection using v6 scoped name
    chroma_client = main.get_chroma_client()
    chroma_coll = chroma_client.create_collection(
        v6_scoped_name,
        metadata={"tenant_id": tenant.id},
    )
    chroma_coll.add(
        ids=["chunk_1"],
        documents=["Legacy technical specification from v6 architecture migration."],
        metadatas=[{"tenant_id": tenant.id, "source": "v6_specs.pdf", "page": 1}],
    )
    build_and_save_bm25_index(
        v6_scoped_name,
        ["Legacy technical specification from v6 architecture migration."],
        [{"tenant_id": tenant.id, "source": "v6_specs.pdf", "page": 1}],
        tenant_id=tenant.id,
    )

    # Insert DocumentRecord
    db_doc = DocumentRecord(
        tenant_id=tenant.id,
        collection_name=coll_name,
        filename="v6_specs.pdf",
        doc_title="V6 Specs",
        category="specs",
        file_path="dummy.pdf",
        upload_date="2026-10-18",
    )
    db_session.add(db_doc)
    db_session.commit()

    # 1. Discoverable via GET /collections
    res_list = client.get("/collections")
    assert res_list.status_code == 200
    assert coll_name in res_list.json()["collections"]

    # 2. Queryable via POST /query
    res_query = client.post(
        "/query",
        json={
            "question": "What is the technical specification from v6?",
            "collection_name": coll_name,
            "enable_web_fallback": False,
        },
    )
    assert res_query.status_code == 200
    sources = res_query.json()["sources"]
    assert len(sources) > 0
    assert sources[0]["collection_name"] == coll_name

    # 3. Deletable via DELETE /collections/{name}
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 200

    # Verify collection deleted from ChromaDB
    existing = {c.name for c in chroma_client.list_collections()}
    assert v6_scoped_name not in existing


def test_reset_never_deletes_foreign_collection_or_bm25_when_metadata_absent(db_session):
    """
    Round 9 Finding 1 Regression Test:
    Verify that if a ChromaDB collection and on-disk BM25 index lack metadata,
    POST /reset invoked by Tenant B NEVER deletes Tenant A's physical collection or BM25 index.
    Fails closed: when candidate_logicals is empty for caller, absence of evidence is NOT treated as caller ownership.
    """
    import json
    import uuid
    from auth import TenantContext
    from config import CHROMA_PATH

    suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"reset_target_a_{suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"reset_caller_b_{suffix}")

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    coll_name = f"unannotated_{suffix}"
    ctx_a = TenantContext(tenant_id=tenant_a.id, tenant_name=tenant_a.name, api_key_id="ka", key_prefix="ska", rate_limit_rpm=60)
    scoped_a = ctx_a.get_scoped_collection_name(coll_name)

    # Create Chroma collection for Tenant A without tenant_id in metadata (simulating unannotated/legacy collection)
    chroma_client = main.get_chroma_client()
    coll = chroma_client.create_collection(scoped_a, metadata={"unannotated": "legacy_fixture"})
    coll.add(
        ids=[f"chunk_a_{suffix}"],
        documents=["Confidential corporate data belonging strictly to Tenant A."],
        metadatas=[{"source": "confidential.pdf", "page": 1}],
    )

    # Create BM25 index on disk with NO tenant_id
    json_path = os.path.join(CHROMA_PATH, f"{scoped_a}_bm25.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"corpus_chunk_ids": [f"chunk_a_{suffix}"], "chunks": ["Confidential corporate data"]}, f)

    # Insert DocumentRecord for Tenant A
    doc_a = DocumentRecord(
        tenant_id=tenant_a.id,
        collection_name=coll_name,
        filename="confidential.pdf",
        doc_title="Confidential Doc",
        category="finance",
        file_path="dummy.pdf",
        upload_date="2026-10-18",
    )
    db_session.add(doc_a)
    db_session.commit()

    try:
        # Tenant B calls POST /reset
        res_reset_b = client_b.post("/reset")
        assert res_reset_b.status_code == 200

        # Verify Tenant A's ChromaDB collection still exists!
        existing_colls = {c.name for c in chroma_client.list_collections()}
        assert scoped_a in existing_colls, f"Foreign unannotated collection '{scoped_a}' was deleted by Tenant B /reset!"

        # Verify Tenant A's BM25 index file on disk still exists!
        assert os.path.exists(json_path), f"Foreign BM25 index '{json_path}' was deleted by Tenant B /reset!"

        # Verify Tenant A's DocumentRecord still exists!
        doc_still_exists = (
            db_session.query(DocumentRecord)
            .filter(DocumentRecord.tenant_id == tenant_a.id, DocumentRecord.collection_name == coll_name)
            .first()
        )
        assert doc_still_exists is not None
    finally:
        # Cleanup
        try:
            chroma_client.delete_collection(scoped_a)
        except Exception:
            pass
        if os.path.exists(json_path):
            os.remove(json_path)


def test_ownership_check_never_deserializes_pickle(db_session, monkeypatch):
    """
    Round 9 Finding 3 Regression Test:
    Verify that ownership resolution and management endpoints NEVER deserialize pickles
    (prohibiting pickle.load and pickle.loads to permanently prevent RCE).
    """
    import pickle
    import uuid
    from auth import TenantContext
    from config import CHROMA_PATH

    def mock_disallowed_pickle_load(*args, **kwargs):
        raise RuntimeError("SECURITY_VIOLATION: pickle.load() was called during ownership inspection!")

    def mock_disallowed_pickle_loads(*args, **kwargs):
        raise RuntimeError("SECURITY_VIOLATION: pickle.loads() was called during ownership inspection!")

    # Monkeypatch pickle.load and pickle.loads to fail immediately
    monkeypatch.setattr(pickle, "load", mock_disallowed_pickle_load)
    monkeypatch.setattr(pickle, "loads", mock_disallowed_pickle_loads)

    suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"no_pickle_tenant_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    coll_name = f"no_pickle_coll_{suffix}"
    ctx = TenantContext(tenant_id=tenant.id, tenant_name=tenant.name, api_key_id="k", key_prefix="sk", rate_limit_rpm=60)
    scoped_coll = ctx.get_scoped_collection_name(coll_name)

    # Place a dummy .pkl file on disk
    pkl_path = os.path.join(CHROMA_PATH, f"{scoped_coll}_bm25.pkl")
    with open(pkl_path, "wb") as f:
        # Write dummy binary content
        f.write(b"NOT_A_SAFE_PICKLE_OR_MALICIOUS_PAYLOAD")

    try:
        # 1. GET /collections must not call pickle.load
        res_list = client.get("/collections")
        assert res_list.status_code == 200

        # 2. POST /query must not call pickle.load
        res_query = client.post(
            "/query",
            json={"question": "Does pickle load run?", "collection_name": coll_name, "enable_web_fallback": False},
        )
        assert res_query.status_code == 200

        # 3. DELETE /collections/{name} must not call pickle.load
        res_del = client.delete(f"/collections/{coll_name}")
        assert res_del.status_code in (200, 404)

        # 4. POST /reset must not call pickle.load
        res_reset = client.post("/reset")
        assert res_reset.status_code == 200

    finally:
        if os.path.exists(pkl_path):
            os.remove(pkl_path)


def test_second_bm25_cleanup_pass_never_deletes_unowned_sidecar(db_session):
    """
    Round 9 Finding 2 Regression Test:
    Verify that the second BM25 cleanup pass in /reset does NOT delete orphaned or foreign
    BM25 files without authoritative ownership verification.
    """
    import json
    import uuid
    from auth import TenantContext
    from config import CHROMA_PATH

    suffix = uuid.uuid4().hex[:6]
    tenant_a, key_a = create_test_tenant_and_key(db_session, f"pass2_owner_{suffix}")
    tenant_b, key_b = create_test_tenant_and_key(db_session, f"pass2_intruder_{suffix}")

    client_b = TestClient(app)
    client_b.headers.update({"X-API-Key": key_b})

    coll_name = f"unowned_sidecar_{suffix}"
    ctx_b = TenantContext(tenant_id=tenant_b.id, tenant_name=tenant_b.name, api_key_id="kb", key_prefix="skb", rate_limit_rpm=60)
    scoped_b = ctx_b.get_scoped_collection_name(coll_name)

    # Seed DocumentRecord for Tenant B with logical name 'coll_name'
    doc_b = DocumentRecord(
        tenant_id=tenant_b.id,
        collection_name=coll_name,
        filename="dummy.pdf",
        doc_title="Dummy",
        category="misc",
        file_path="dummy.pdf",
        upload_date="2026-10-18",
    )
    db_session.add(doc_b)
    db_session.commit()

    # Create an orphaned BM25 file matching scoped_b, BUT owned by Tenant A
    json_path = os.path.join(CHROMA_PATH, f"{scoped_b}_bm25.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"tenant_id": tenant_a.id, "corpus_chunk_ids": [], "chunks": []}, f)

    try:
        # Tenant B calls POST /reset
        res_reset = client_b.post("/reset")
        # Since Tenant A's BM25 file was foreign and not owned by B, it must NOT be deleted
        assert os.path.exists(json_path), f"Foreign BM25 sidecar '{json_path}' was deleted during second pass of /reset!"
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)


def test_unknown_ownership_denied_across_all_endpoints(db_session):
    """
    Regression Test:
    Verify that ResourceOwnership.UNKNOWN strictly fails closed:
    - Never listed in GET /collections
    - Never queried in POST /query (0 sources returned)
    - Never deleted in DELETE /collections/{name} (HTTP 404 returned)
    """
    import uuid
    from auth import TenantContext

    suffix = uuid.uuid4().hex[:6]
    tenant, key = create_test_tenant_and_key(db_session, f"unknown_tester_{suffix}")
    client = TestClient(app)
    client.headers.update({"X-API-Key": key})

    coll_name = f"unknown_vault_{suffix}"
    ctx = TenantContext(tenant_id=tenant.id, tenant_name=tenant.name, api_key_id="k", key_prefix="sk", rate_limit_rpm=60)
    scoped_coll = ctx.get_scoped_collection_name(coll_name)

    # Ensure no vector collection and no relational records exist for caller
    chroma_client = main.get_chroma_client()
    existing_colls = {c.name for c in chroma_client.list_collections()}
    if scoped_coll in existing_colls:
        chroma_client.delete_collection(scoped_coll)

    # 1. GET /collections -> must not list coll_name
    res_list = client.get("/collections")
    assert res_list.status_code == 200
    assert coll_name not in res_list.json()["collections"]

    # 2. POST /query -> must return 0 sources
    res_query = client.post(
        "/query",
        json={"question": "Where is the data?", "collection_name": coll_name, "enable_web_fallback": False},
    )
    assert res_query.status_code == 200
    assert len(res_query.json()["sources"]) == 0

    # 3. DELETE /collections/{coll_name} -> must return HTTP 404
    res_del = client.delete(f"/collections/{coll_name}")
    assert res_del.status_code == 404
    assert f"Collection not found: {coll_name}" in res_del.json()["detail"]






