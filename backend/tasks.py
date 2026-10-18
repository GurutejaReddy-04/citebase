"""
Asynchronous ingestion worker.

Executes parsing, embedding, and BM25 index generation in-process via FastAPI BackgroundTasks,
avoiding the operational tax of an external task broker at this scale.
"""

import logging
import os
import re
import threading
import time
from typing import Optional

from cache import invalidate_tenant_cache
from database import SessionLocal
from ingest import ingest_pdf
from models import DocumentRecord, IngestionTask, get_utc_now

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 15.0


def _run_heartbeat(task_id: str, stop_event: threading.Event, interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS) -> None:
    """
    Periodically updates updated_at timestamp in IngestionTask to indicate the worker is alive.
    If the worker crashes or the process is killed, heartbeats cease and pollers detect staleness after 600s.
    """
    while not stop_event.wait(timeout=interval_seconds):
        hb_db = SessionLocal()
        try:
            hb_task = hb_db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if hb_task and hb_task.status in ("processing", "stale"):
                hb_task.updated_at = get_utc_now()
                hb_db.commit()
            elif not hb_task or hb_task.status in ("completed", "failed"):
                break
        except Exception as e:
            logger.warning("Heartbeat update failed for task '%s': %s", task_id, e)
        finally:
            hb_db.close()


def run_background_ingest(
    task_id: str,
    tenant_id: str,
    file_path: str,
    scoped_coll: str,
    raw_coll: str,
    original_name: str,
    doc_title: Optional[str],
    category: Optional[str],
    force: bool,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    """
    Background worker executed via FastAPI BackgroundTasks.
    Parses PDF, inserts embeddings into ChromaDB, and updates relational task status.
    """
    db = SessionLocal()
    stop_heartbeat = threading.Event()
    heartbeat_thread: Optional[threading.Thread] = None
    try:
        task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
        if not task:
            logger.error("IngestionTask '%s' not found in database", task_id)
            return

        task.status = "processing"
        task.updated_at = get_utc_now()
        db.commit()

        # Start periodic worker heartbeat daemon
        heartbeat_thread = threading.Thread(
            target=_run_heartbeat,
            args=(task_id, stop_heartbeat, heartbeat_interval),
            daemon=True,
            name=f"heartbeat-{task_id}",
        )
        heartbeat_thread.start()

        logger.info(
            "Starting background ingestion for task '%s' (tenant: %s, file: %s)",
            task_id,
            tenant_id,
            original_name,
        )

        try:
            result_message = ingest_pdf(
                file_path,
                scoped_coll,
                force=force,
                source_name=original_name,
                doc_title=doc_title,
                category=category,
                tenant_id=tenant_id,
            )
        finally:
            stop_heartbeat.set()
            if heartbeat_thread:
                heartbeat_thread.join(timeout=2.0)

        # Parse chunk count from result message e.g. "Ingested 12 chunks..."
        chunk_match = re.search(r"(\d+)\s+chunks", result_message)
        chunk_count = int(chunk_match.group(1)) if chunk_match else 0

        # Save document record
        doc_record = DocumentRecord(
            tenant_id=tenant_id,
            collection_name=raw_coll,
            filename=original_name,
            doc_title=doc_title or original_name,
            category=category or "uncategorized",
            chunk_count=chunk_count,
            file_path=file_path,
            upload_date=time.strftime("%Y-%m-%d"),
        )
        db.add(doc_record)

        # Re-fetch task state to reconcile any concurrent poller modifications
        db.expire(task)
        db.refresh(task)

        # Update task state to completed with state reconciliation
        if task.status == "stale":
            logger.warning(
                "Task '%s' was marked stale by poller, but worker completed successfully; reconciling status to completed.",
                task_id,
            )
        task.status = "completed"
        task.total_chunks = chunk_count
        task.completed_at = get_utc_now()
        task.updated_at = get_utc_now()
        task.error_message = None
        db.commit()

        # Invalidate any cached queries for this tenant since new documents were ingested
        invalidate_tenant_cache(tenant_id)

        logger.info(
            "Background ingestion completed successfully for task '%s' (%d chunks)",
            task_id,
            chunk_count,
        )

    except Exception as e:
        logger.exception("Background ingestion failed for task '%s': %s", task_id, e)
        db.rollback()
        task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
        if task:
            safe_msg = "Ingestion failed. Check server logs for details."
            err_str = str(e).lower()
            if "fitz" in err_str or "pdf" in err_str:
                safe_msg = "PDF parsing failed. The file may be corrupted or password-protected."
            elif "chromadb" in err_str or "chroma" in err_str:
                safe_msg = "Document storage error. Please retry."
            task.status = "failed"
            task.error_message = safe_msg
            task.completed_at = get_utc_now()
            task.updated_at = get_utc_now()
            db.commit()

    finally:
        db.close()
        # Clean up temporary uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up temp upload file: %s", file_path)
            except Exception as e:
                logger.warning("Could not remove temp file %s: %s", file_path, e)
