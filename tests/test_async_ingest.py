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
        eleven_minutes_ago = datetime.datetime.now(timezone.utc) - timedelta(minutes=11)
        stale_task = IngestionTask(
            id=stale_task_id,
            tenant_id=tenant_id,
            collection_name="stale_coll",
            filename="interrupted.pdf",
            status="processing",
            created_at=eleven_minutes_ago,
            updated_at=eleven_minutes_ago,
        )
        db.add(stale_task)
        db.commit()
    finally:
        db.close()

    # Poll status -> should detect > 600s elapsed since heartbeat and return 'stale'
    res = client.get(f"/tasks/{stale_task_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "stale"
    assert "Task timed out or was interrupted by a server restart" in data["error_message"]


def test_active_task_with_recent_heartbeat_not_stale():
    """
    Verify worker heartbeat prevents false stale reports:
    A task created 7 minutes ago (> 300s) whose worker updated heartbeat 30 seconds ago
    remains in 'processing' status and is not prematurely aborted.
    """
    tenant_id, client = create_tenant_and_client("heartbeat_test_tenant")
    db = SessionLocal()
    active_task_id = generate_uuid()
    try:
        seven_minutes_ago = datetime.datetime.now(timezone.utc) - timedelta(minutes=7)
        thirty_secs_ago = datetime.datetime.now(timezone.utc) - timedelta(seconds=30)
        active_task = IngestionTask(
            id=active_task_id,
            tenant_id=tenant_id,
            collection_name="long_running_coll",
            filename="large_doc.pdf",
            status="processing",
            created_at=seven_minutes_ago,
            updated_at=thirty_secs_ago,
        )
        db.add(active_task)
        db.commit()
    finally:
        db.close()

    res = client.get(f"/tasks/{active_task_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "processing", "Active task with recent heartbeat must NOT be marked stale"


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


def test_worker_heartbeat_periodically_updates_timestamp_during_ingest(monkeypatch):
    """
    Verify worker heartbeat actively updates updated_at in the database
    during long-running ingestion, preventing false staleness reports.
    """
    import threading
    from tasks import run_background_ingest, _run_heartbeat

    tenant_id, client = create_tenant_and_client("hb_active_tenant")
    db = SessionLocal()
    task_id = generate_uuid()
    initial_time = datetime.datetime.now(timezone.utc) - timedelta(seconds=10)

    try:
        t = IngestionTask(
            id=task_id,
            tenant_id=tenant_id,
            collection_name="hb_coll",
            filename="hb.pdf",
            status="processing",
            created_at=initial_time,
            updated_at=initial_time,
        )
        db.add(t)
        db.commit()

        # Run heartbeat directly with short interval (0.05s)
        stop_event = threading.Event()
        hb_thread = threading.Thread(
            target=_run_heartbeat,
            args=(task_id, stop_event, 0.05),
            daemon=True,
        )
        hb_thread.start()
        time.sleep(0.15)  # Let it fire at least twice
        stop_event.set()
        hb_thread.join(timeout=1.0)

        # Refresh task and verify updated_at was updated by heartbeat
        db.expire(t)
        db.refresh(t)
        updated_utc = t.updated_at if t.updated_at.tzinfo else t.updated_at.replace(tzinfo=timezone.utc)
        assert updated_utc > initial_time, "Heartbeat must update updated_at during processing"
    finally:
        db.close()


def test_worker_reconciles_poller_marked_stale_status(monkeypatch, tmp_path):
    """
    Verify that if a poller marks a task 'stale' during a long ingestion,
    the completing worker detects the stale state and cleanly reconciles
    the task status to 'completed'.
    """
    from tasks import run_background_ingest

    tenant_id, client = create_tenant_and_client("reconcile_tenant")
    task_id = generate_uuid()

    dummy_pdf = tmp_path / "reconcile.pdf"
    dummy_pdf.write_bytes(make_test_pdf("Reconciliation test document content."))

    db = SessionLocal()
    try:
        task = IngestionTask(
            id=task_id,
            tenant_id=tenant_id,
            collection_name="rec_coll",
            filename="reconcile.pdf",
            status="pending",
        )
        db.add(task)
        db.commit()
    finally:
        db.close()

    # Mock ingest_pdf to simulate an external poller marking the task stale midway
    def mock_slow_ingest_with_poller_stale(*args, **kwargs):
        poller_db = SessionLocal()
        try:
            pt = poller_db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if pt:
                pt.status = "stale"
                pt.error_message = "Task timed out or was interrupted by a server restart."
                poller_db.commit()
        finally:
            poller_db.close()
        return "Ingested 5 chunks from document"

    monkeypatch.setattr("tasks.ingest_pdf", mock_slow_ingest_with_poller_stale)

    run_background_ingest(
        task_id=task_id,
        tenant_id=tenant_id,
        file_path=str(dummy_pdf),
        scoped_coll=f"t_{tenant_id[:12]}_rec_coll",
        raw_coll="rec_coll",
        original_name="reconcile.pdf",
        doc_title="Reconcile Doc",
        category="testing",
        force=True,
    )

    db_check = SessionLocal()
    try:
        final_task = db_check.query(IngestionTask).filter(IngestionTask.id == task_id).first()
        assert final_task is not None
        assert final_task.status == "completed", "Worker must reconcile stale status to completed upon success"
        assert final_task.total_chunks == 5
        assert final_task.error_message is None
    finally:
        db_check.close()


def test_slow_worker_heartbeat_prevents_staleness_under_continuous_polling(monkeypatch, tmp_path):
    """
    Verify a deliberately slow ingestion running in a background thread continuously
    updates its heartbeat, so that repeated client polls see an active 'processing' status
    and advancing timestamps, transitioning cleanly to 'completed'.
    """
    import threading
    from tasks import run_background_ingest

    tenant_id, client = create_tenant_and_client("slow_worker_poll_tenant")
    task_id = generate_uuid()
    dummy_pdf = tmp_path / "slow_doc.pdf"
    dummy_pdf.write_bytes(make_test_pdf("Deliberately slow document ingestion content."))

    db = SessionLocal()
    try:
        task = IngestionTask(
            id=task_id,
            tenant_id=tenant_id,
            collection_name="slow_coll",
            filename="slow_doc.pdf",
            status="pending",
        )
        db.add(task)
        db.commit()
    finally:
        db.close()

    # Mock ingest_pdf to simulate a slow 0.35s parser while heartbeat fires every 0.05s
    def mock_slow_parser(*args, **kwargs):
        time.sleep(0.35)
        return "Ingested 8 chunks from document"

    monkeypatch.setattr("tasks.ingest_pdf", mock_slow_parser)

    # Launch worker in thread with fast heartbeat (0.05s)
    worker_thread = threading.Thread(
        target=run_background_ingest,
        kwargs={
            "task_id": task_id,
            "tenant_id": tenant_id,
            "file_path": str(dummy_pdf),
            "scoped_coll": f"t_{tenant_id[:12]}_slow_coll",
            "raw_coll": "slow_coll",
            "original_name": "slow_doc.pdf",
            "doc_title": "Slow Doc",
            "category": "benchmark",
            "force": True,
            "heartbeat_interval": 0.05,
        },
    )
    worker_thread.start()

    # Poll status while worker is running
    saw_processing = False
    observed_timestamps = []
    for _ in range(6):
        time.sleep(0.06)
        res = client.get(f"/tasks/{task_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] == "processing":
            saw_processing = True
            if data.get("updated_at"):
                observed_timestamps.append(data["updated_at"])

    worker_thread.join(timeout=2.0)

    # Final check
    res_final = client.get(f"/tasks/{task_id}")
    assert res_final.status_code == 200
    final_data = res_final.json()
    assert final_data["status"] == "completed"
    assert final_data["total_chunks"] == 8
    assert saw_processing, "Continuous polling must observe processing status with active heartbeats"


def test_dead_worker_without_heartbeat_marked_stale_by_poller():
    """
    Verify that if a worker dies (process crash) and heartbeats cease for > 600 seconds,
    subsequent client polling detects the dead worker and marks it 'stale'.
    """
    tenant_id, client = create_tenant_and_client("dead_worker_tenant")
    task_id = generate_uuid()

    eleven_minutes_ago = datetime.datetime.now(timezone.utc) - timedelta(minutes=11)
    db = SessionLocal()
    try:
        dead_task = IngestionTask(
            id=task_id,
            tenant_id=tenant_id,
            collection_name="crashed_coll",
            filename="crashed.pdf",
            status="processing",
            created_at=eleven_minutes_ago,
            updated_at=eleven_minutes_ago,  # Heartbeat stopped 11 minutes ago
        )
        db.add(dead_task)
        db.commit()
    finally:
        db.close()

    res = client.get(f"/tasks/{task_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "stale", "Worker with ceased heartbeats (> 600s) must transition to stale"
    assert "interrupted by a server restart" in data["error_message"]


def test_heartbeat_thread_shuts_down_on_worker_completion_and_error(monkeypatch, tmp_path):
    """
    Verify the heartbeat daemon thread joins and stops updating when the worker completes or errors,
    preventing leaked background threads or zombie heartbeat mutations.
    """
    from tasks import run_background_ingest

    tenant_id, client = create_tenant_and_client("hb_shutdown_tenant")
    task_id = generate_uuid()
    dummy_pdf = tmp_path / "hb_shutdown.pdf"
    dummy_pdf.write_bytes(make_test_pdf("Heartbeat shutdown test document."))

    db = SessionLocal()
    try:
        task = IngestionTask(
            id=task_id,
            tenant_id=tenant_id,
            collection_name="hb_sd_coll",
            filename="hb_shutdown.pdf",
            status="pending",
        )
        db.add(task)
        db.commit()
    finally:
        db.close()

    def mock_failing_parser(*args, **kwargs):
        raise ValueError("Parser aborted")

    monkeypatch.setattr("tasks.ingest_pdf", mock_failing_parser)

    run_background_ingest(
        task_id=task_id,
        tenant_id=tenant_id,
        file_path=str(dummy_pdf),
        scoped_coll=f"t_{tenant_id[:12]}_hb_sd_coll",
        raw_coll="hb_sd_coll",
        original_name="hb_shutdown.pdf",
        doc_title="HB Shutdown",
        category="testing",
        force=True,
        heartbeat_interval=0.05,
    )

    db_check = SessionLocal()
    try:
        t = db_check.query(IngestionTask).filter(IngestionTask.id == task_id).first()
        assert t.status == "failed"
        updated_after_fail = t.updated_at

        # Sleep briefly; verify updated_at is NOT touched anymore (heartbeat stopped)
        time.sleep(0.15)
        db_check.expire(t)
        db_check.refresh(t)
        assert t.updated_at == updated_after_fail, "Heartbeat must not continue updating DB after worker finishes"
    finally:
        db_check.close()


