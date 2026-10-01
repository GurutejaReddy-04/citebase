"""
PyTest Fixtures and Test Environment Setup for Multi-Tenant RAG API.

Provides:
- Initialized SQLite test database with default tenant and API key.
- Authenticated TestClient providing default test key headers.
- Mocked LLM answer generator for deterministic, quota-free API testing.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import main
from config import BOOTSTRAP_API_KEY
from database import SessionLocal, init_db
from models import ApiKey, Tenant

TEST_API_KEY = BOOTSTRAP_API_KEY or "sk_test_fixture_master_key_internal_only"


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database schema is created and seeded with default test API key."""
    init_db()
    db = SessionLocal()
    try:
        tenant = db.query(Tenant).filter(Tenant.name == "default_org").first()
        if not tenant:
            tenant = Tenant(id="default_tenant_id", name="default_org", is_active=True)
            db.add(tenant)
            db.flush()

        import hashlib
        key_hash = hashlib.sha256(TEST_API_KEY.encode("utf-8")).hexdigest()
        api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if not api_key:
            api_key = ApiKey(
                id="default_key_id",
                tenant_id=tenant.id,
                key_hash=key_hash,
                key_prefix=TEST_API_KEY[:12],
                name="Default Test Key",
                rate_limit_rpm=1000,  # high limit for test suite
                is_active=True,
            )
            db.add(api_key)
            db.commit()
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def seed_test_collections(setup_test_database):
    """Seed test fixtures into ChromaDB if collections are empty (guarantees clean clone testability)."""
    from chroma import get_chroma_client
    from ingest import ingest_pdf
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]

    db = SessionLocal()
    tenant = db.query(Tenant).filter(Tenant.name == "default_org").first()
    prefix = f"t_{tenant.id[:12]}_" if tenant else "t_default_tena_"
    db.close()

    fixtures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))
    ece_pdf = os.path.join(fixtures_dir, "ece_academic_curriculum.pdf")
    telecom_pdf = os.path.join(fixtures_dir, "telecom_cellular_architecture.pdf")

    def is_empty(coll_name: str) -> bool:
        if coll_name not in existing:
            return True
        try:
            return client.get_collection(coll_name).count() == 0
        except Exception:
            return True

    coll_hybrid = f"{prefix}test_hybrid_coll"
    has_ee101 = False
    if coll_hybrid in existing:
        try:
            c = client.get_collection(coll_hybrid)
            docs = c.get(where={"chunk_id": "test_hybrid_coll_ee101_p135"})
            if docs and docs.get("ids"):
                has_ee101 = True
        except Exception:
            pass

    if is_empty(coll_hybrid) or not has_ee101:
        if os.path.exists(ece_pdf) and is_empty(coll_hybrid):
            ingest_pdf(ece_pdf, coll_hybrid, force=True, source_name="ece_academic_curriculum.pdf", category="curriculum")
        from ingest import embed_and_store
        ee101_chunk = (
            "EE101 Elements of Electrical Engineering (3 Credits)\n"
            "Prerequisites: None.\n"
            "Course Learning Outcomes: Fundamental circuit analysis, Ohm's Law, Kirchhoff's laws, "
            "AC and DC network theorems, transformers and digital modulation topics."
        )
        ee101_meta = {
            "source": "ece_academic_curriculum.pdf",
            "page": 135,
            "section": "Electrical Engineering",
            "doc_title": "Academic Curriculum",
            "category": "curriculum",
            "chunk_id": "test_hybrid_coll_ee101_p135",
        }
        embed_and_store([ee101_chunk], [ee101_meta], coll_hybrid, "ece_academic_curriculum.pdf", force=True)

    if is_empty(f"{prefix}telecom_architecture_coll") and os.path.exists(telecom_pdf):
        ingest_pdf(telecom_pdf, f"{prefix}telecom_architecture_coll", force=True, source_name="telecom_cellular_architecture.pdf", category="telecom")

    if is_empty(f"{prefix}ece_curriculum_coll") and os.path.exists(ece_pdf):
        ingest_pdf(ece_pdf, f"{prefix}ece_curriculum_coll", force=True, source_name="ece_academic_curriculum.pdf", category="curriculum")



@pytest.fixture(scope="session", autouse=True)
def mock_gemini_generator():
    """Mock generator to keep API tests fast, deterministic, and free of external API quota."""
    main.generate_answer = lambda q, ctx: f"Mock Grounded Answer with {len(ctx)} sources for: {q}"


@pytest.fixture(autouse=True)
def clear_test_cache():
    """Ensure clean cache state between individual tests."""
    from cache import get_redis_client
    r = get_redis_client()
    if r:
        try:
            r.flushdb()
        except Exception:
            pass


@pytest.fixture
def auth_headers():
    """Return default authentication headers for API requests."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def mock_redis():
    """Provides an isolated in-memory FakeRedis client for tests without Redis server dependencies."""
    from fakeredis import FakeRedis
    return FakeRedis(decode_responses=True)


@pytest.fixture
def client(auth_headers):
    """Return TestClient pre-configured with default authentication headers."""
    c = TestClient(main.app)
    c.headers.update(auth_headers)
    return c

