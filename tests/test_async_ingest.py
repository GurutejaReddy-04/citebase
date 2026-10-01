"""
Comprehensive Test Suite for Phase 9: Asynchronous Ingestion & Task Management.

Tests:
1. Async PDF upload returning HTTP 202 Accepted with task_id
2. Polling GET /tasks/{task_id} until completion
3. Querying asynchronously ingested document collection
4. Cross-tenant task isolation (Tenant B cannot poll Tenant A's tasks)
5. Staleness detection for interrupted/timed-out in-process tasks (> 5 minutes)
6. Gated dev-only sync=true execution
7. Error handling for failed background ingestion tasks
"""

import datetime
from datetime import timezone, timedelta
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

# Put backend on sys.path
sys.path.insert(0, os.path.abspath("backend"))

import main
from auth import generate_api_key
from database import SessionLocal
from main import app
from models import ApiKey, IngestionTask, Tenant, generate_uuid, get_utc_now

# Mock generator so tests run independently of Gemini API quota
main.generate_answer = lambda q, ctx: f"Mock Grounded Answer with {len(ctx)} sources for: {q}"


def make_test_pdf(text: str = "Async Ingestion Specification Document") -> bytes:
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


def test_async_upload_returns_202_and_polls_completed():
    """Verify POST /upload responds immediately with 202 Accepted and task transitions to completed."""
    tenant_id, client = create_tenant_and_client("async_tenant_alpha")
    pdf_data = make_test_pdf("Asynchronous Background Document Ingestion Content Alpha.")

    res_upload = client.post(
        "/upload",
        files={"file": ("async_alpha.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "async_coll_alpha", "doc_title": "Async Alpha Doc"},
    )

    assert res_upload.status_code == 202
    data = res_upload.json()
    assert "task_id" in data
    assert data["status"] == "processing"
    assert data["collection"] == "async_coll_alpha"

    task_id = data["task_id"]

    # Poll status endpoint
    completed = False
    for _ in range(30):
        res_task = client.get(f"/tasks/{task_id}")
        assert res_task.status_code == 200
        task_info = res_task.json()
        if task_info["status"] == "completed":
            assert task_info["total_chunks"] >= 1
            assert task_info["completed_at"] is not None
            completed = True
            break
        elif task_info["status"] == "failed":
            pytest.fail(f"Task failed unexpectedly: {task_info.get('error_message')}")
        time.sleep(0.1)

    assert completed, "Task did not reach 'completed' status within timeout."


def test_query_after_async_ingest():
    """Verify document ingested asynchronously is retrievable via POST /query."""
    tenant_id, client = create_tenant_and_client("async_tenant_beta")
    pdf_data = make_test_pdf("Unique Token Beta-998877: High throughput asynchronous batch streaming.")

    res_upload = client.post(
        "/upload",
        files={"file": ("beta_doc.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "async_coll_beta", "doc_title": "Beta Streaming Doc"},
    )
    assert res_upload.status_code == 202
    task_id = res_upload.json()["task_id"]

    # Wait for completion
    for _ in range(30):
        res_task = client.get(f"/tasks/{task_id}")
        if res_task.json()["status"] == "completed":
            break
        time.sleep(0.1)

    # Query collection
    res_query = client.post(
        "/query",
        json={
            "question": "What is Beta-998877?",
            "collection_name": "async_coll_beta",
            "enable_web_fallback": False,
        },
    )
    assert res_query.status_code == 200
    data = res_query.json()
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["source"] == "beta_doc.pdf"
    assert data["sources"][0]["doc_title"] == "Beta Streaming Doc"


def test_cross_tenant_task_isolation():
    """Verify Tenant B cannot view or poll Tenant A's async ingestion tasks."""
    tenant_a_id, client_a = create_tenant_and_client("task_victim_tenant")
    tenant_b_id, client_b = create_tenant_and_client("task_attacker_tenant")

    pdf_data = make_test_pdf("Victim Private Task Content.")
    res_upload = client_a.post(
        "/upload",
        files={"file": ("private.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "private_coll"},
    )
    assert res_upload.status_code == 202
    task_id_a = res_upload.json()["task_id"]

    # Attacker tries to poll victim's task ID
    res_attacker = client_b.get(f"/tasks/{task_id_a}")
    assert res_attacker.status_code == 404
    assert f"Ingestion task '{task_id_a}' not found" in res_attacker.json()["detail"]


def test_stale_task_detection():
    """
    Verify in-process crash / timeout staleness check:
    A task created 10 minutes ago in 'processing' state is marked and reported as 'stale'.
    """
    tenant_id, client = create_tenant_and_client("stale_test_tenant")
    db = SessionLocal()
    stale_task_id = generate_uuid()
    try:
        ten_minutes_ago = datetime.datetime.now(timezone.utc) - timedelta(minutes=10)
        stale_task = IngestionTask(
            id=stale_task_id,
            tenant_id=tenant_id,
            collection_name="stale_coll",
            filename="interrupted.pdf",
            status="processing",
            created_at=ten_minutes_ago,
        )
        db.add(stale_task)
        db.commit()
    finally:
        db.close()

    # Poll status -> should detect > 300s elapsed and return 'stale'
    res = client.get(f"/tasks/{stale_task_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "stale"
    assert "Task timed out or was interrupted by a server restart" in data["error_message"]


def test_sync_flag_gated_to_dev_only(monkeypatch):
    """Verify sync=true is only honored when ENV == 'development' and returns 202 in production."""
    tenant, client = create_tenant_and_client("sync_gate_tenant")
    pdf_data = make_test_pdf("Sync Gating Test Content.")

    # 1. Dev Mode (ENV == 'development'): sync=true returns HTTP 200 immediately
    monkeypatch.setattr(main, "ENV", "development")
    res_dev = client.post(
        "/upload?sync=true",
        files={"file": ("sync_dev.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "sync_dev_coll"},
    )
    assert res_dev.status_code == 200
    assert res_dev.json()["status"] == "completed"

    # 2. Production Mode (ENV == 'production'): sync=true is ignored, returns HTTP 202 async
    monkeypatch.setattr(main, "ENV", "production")
    res_prod = client.post(
        "/upload?sync=true",
        files={"file": ("sync_prod.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "sync_prod_coll"},
    )
    assert res_prod.status_code == 202
    assert res_prod.json()["status"] == "processing"


def test_failed_task_handling(monkeypatch):
    """Verify background ingestion failures update task status to 'failed' with error details."""
    tenant, client = create_tenant_and_client("failure_test_tenant")
    pdf_data = make_test_pdf("Will fail intentionally.")

    # Mock ingest_pdf to simulate an unexpected parser failure
    def mock_broken_ingest(*args, **kwargs):
        raise RuntimeError("Simulated corruption in PyMuPDF extraction engine")

    monkeypatch.setattr("tasks.ingest_pdf", mock_broken_ingest)

    res_upload = client.post(
        "/upload",
        files={"file": ("corrupt.pdf", pdf_data, "application/pdf")},
        data={"collection_name": "fail_coll"},
    )
    assert res_upload.status_code == 202
    task_id = res_upload.json()["task_id"]

    # Poll status -> should transition to failed
    for _ in range(30):
        res = client.get(f"/tasks/{task_id}")
        data = res.json()
        if data["status"] == "failed":
            assert "PDF parsing failed" in data["error_message"] or "Simulated corruption in PyMuPDF" in data["error_message"]
            break
        time.sleep(0.1)
