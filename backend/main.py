"""
FastAPI entry point for CiteBase — Multi-Tenant Production RAG-as-a-Service API.

Features:
- Cryptographic API Key Authentication (256-bit entropy)
- Complete Multi-Tenant Collection Isolation (ChromaDB + BM25)
- Per-Key Rate Limiting via Redis atomic TTL counters (429 with Retry-After)
- Structure-Aware Hybrid Retrieval + Cross-Encoder Reranking
- Real-Time Web Search Fallback and Unified Blending
- Relational Audit Logging (Postgres / SQLite)

Routes:
  POST   /upload                    Ingest PDF into tenant-scoped collection
  POST   /query                     Retrieve chunks and generate cited answer
  GET    /collections               List tenant-isolated collections
  DELETE /collections/{name}        Delete tenant-scoped collection
  POST   /reset                     Reset tenant collections
  GET    /health                    Unauthenticated liveness check
  POST   /admin/api-keys            Create new API key for tenant
  GET    /admin/api-keys            List active API keys for tenant
"""

import hashlib
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename

from auth import (
    TenantContext,
    bootstrap_default_tenant,
    generate_api_key,
    get_current_tenant,
)
from cache import (
    get_cached_response,
    get_redis_client,
    invalidate_tenant_cache,
    set_cached_response,
)
from config import (
    ALLOWED_ORIGINS,
    CHROMA_PATH,
    ENABLE_RERANKER,
    ENABLE_WEB_SEARCH,
    ENV,
    MIN_RELEVANCE_SCORE,
    TOP_K_RESULTS,
    UPLOAD_DIR,
    WEB_SEARCH_FALLBACK_DISTANCE_THRESHOLD,
    WEB_SEARCH_FALLBACK_THRESHOLD,
    WEB_SEARCH_PROVIDER,
)
from database import SessionLocal, get_db, init_db
from exceptions import CollectionNotFoundError, IngestionError, TenantNotFoundError
from chroma import get_chroma_client
from generator import generate_answer
from ingest import ingest_pdf
from models import ApiKey, DocumentRecord, IngestionTask, QueryLog, generate_uuid, get_utc_now
from rate_limiter import check_rate_limit
from reranker import rerank_documents
from retriever import retrieve_context
from tasks import run_background_ingest
from web_search import search_web
from bm25 import delete_bm25_index, get_bm25_file_tenant_id

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# --- lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize DB schemas and bootstrap default tenant on startup."""
    init_db()
    db = SessionLocal()
    try:
        bootstrap_default_tenant(db)
    finally:
        db.close()
    yield
    # Graceful shutdown: Dispose database connection pool
    from database import engine
    engine.dispose()
    logger.info("Database connection pool disposed.")
    logger.info("CiteBase application shutting down. Releasing database connections and resources.")


# --- app ---
app = FastAPI(
    title="CiteBase — Production RAG-as-a-Service API",
    description=(
        "Multi-tenant document intelligence API featuring hybrid search (Dense + BM25), "
        "cross-encoder reranking, Redis caching, async ingestion, and grounded answer synthesis."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# --- custom exception handlers ---
@app.exception_handler(TenantNotFoundError)
def handle_tenant_not_found(request: Request, exc: TenantNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials."})


@app.exception_handler(CollectionNotFoundError)
def handle_collection_not_found(request: Request, exc: CollectionNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"Collection not found: {exc.args[0]}"})


@app.exception_handler(IngestionError)
def handle_ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Ingestion failed. Check logs for details."})


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Response:
    """Record request latency in X-Process-Time response header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


os.makedirs(UPLOAD_DIR, exist_ok=True)

# Canonical ChromaDB collection naming regex (lowercase alphanumeric, hyphen, underscore, 3-45 chars)
_COLLECTION_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{1,43}[a-z0-9]$')


def _validate_collection_name(name: str) -> None:
    """Validate client-supplied collection name syntax."""
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid collection name. Must be 3-45 characters, lowercase alphanumeric with "
                "hyphens (-) or underscores (_), and start/end with a letter or digit."
            ),
        )


# --- schemas ---
class QueryRequest(BaseModel):
    """Query request payload."""
    question:             str = Field(..., min_length=1, max_length=2000)
    collection_name:      Optional[str] = None
    collection_names:     Optional[list[str]] = None
    filters:              Optional[dict[str, Any]] = None
    enable_rerank:        Optional[bool] = None
    enable_web_fallback:  Optional[bool] = None
    web_provider:         Optional[str] = None


class SourceReference(BaseModel):
    page:               int = 1
    source:             str
    score:              float
    source_type:        str = "document"  # "document" or "web"
    url:                Optional[str] = None
    provider:           Optional[str] = None
    collection_name:    Optional[str] = None
    section:            Optional[str] = "General"
    breadcrumb:         Optional[str] = "General"
    doc_title:          Optional[str] = None
    category:           Optional[str] = "uncategorized"
    upload_date:        Optional[str] = None
    chunk_id:           Optional[str] = None
    rerank_score:       Optional[float] = None
    retrieval_channels: Optional[list[str]] = None


class QueryResponse(BaseModel):
    answer:              str
    retrieval_mode:      str = "local_document"  # "local_document", "web_fallback", "blended", "cached"
    fallback_triggered:  bool = False
    sources:             list[SourceReference]
    cached:              bool = False


class CreateApiKeyRequest(BaseModel):
    name:           Optional[str] = Field(default="API Key", min_length=1, max_length=128)
    rate_limit_rpm: Optional[int] = Field(
        default=60,
        ge=1,
        le=10000,
        description="Rate limit in requests per minute (minimum 1, maximum 10,000)",
    )


class ApiKeyResponse(BaseModel):
    id:             str
    key_prefix:     str
    name:           str
    rate_limit_rpm: int
    raw_key:        Optional[str] = None  # Only returned once on creation


class TaskStatusResponse(BaseModel):
    task_id:        str
    status:         str  # pending, processing, completed, failed, stale
    collection:     str
    filename:       str
    total_chunks:   int
    error_message:  Optional[str] = None
    created_at:     str
    updated_at:     Optional[str] = None
    completed_at:   Optional[str] = None


# --- routes ---

@app.post("/upload", summary="Upload PDF for asynchronous ingestion (or sync in dev)")
def upload_document(
    response:        Response,
    background_tasks: BackgroundTasks,
    file:            UploadFile           = File(...),
    collection_name: str                  = Form(...),
    force:           bool                 = Form(False),
    doc_title:       Optional[str]        = Form(None),
    category:        Optional[str]        = Form(None),
    sync:            bool                 = Query(False, description="Run synchronously (development environment only)"),
    tenant:          TenantContext        = Depends(check_rate_limit),
    db:              Session              = Depends(get_db),
) -> dict[str, Any]:
    """
    Upload and ingest a PDF document.
    
    Default (Production):
    - Decoupled asynchronous processing via FastAPI BackgroundTasks.
    - Responds immediately with HTTP 202 Accepted and a `task_id`.
    - Polling status available at `GET /tasks/{task_id}`.
    
    Sync Bypass (Development Only):
    - Honors `sync=true` ONLY when `ENV == 'development'`.
    - Otherwise strictly runs asynchronously to protect server throughput.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw_coll = collection_name.strip()
    _validate_collection_name(raw_coll)

    scoped_coll = tenant.get_scoped_collection_name(raw_coll)

    # Runtime collision check:
    # 1. Intra-tenant collision: verify this physical collection is not already mapped to a different logical collection for this tenant
    existing_other_colls = (
        db.query(DocumentRecord.collection_name)
        .filter(
            DocumentRecord.tenant_id == tenant.tenant_id,
            DocumentRecord.collection_name != raw_coll,
        )
        .distinct()
        .all()
    )
    for (other_coll,) in existing_other_colls:
        if tenant.get_scoped_collection_name(other_coll) == scoped_coll:
            raise HTTPException(
                status_code=409,
                detail=f"Collection name collision: '{raw_coll}' resolves to the same physical collection as existing collection '{other_coll}'.",
            )

    # 2. Cross-tenant collision guard: check if physical collection is already owned by or mapped to another tenant
    chroma_client = get_chroma_client()
    existing_chroma_colls = {c.name: c for c in chroma_client.list_collections()}
    if scoped_coll in existing_chroma_colls:
        coll_meta = existing_chroma_colls[scoped_coll].metadata or {}
        owner_tenant = coll_meta.get("tenant_id")
        if owner_tenant and owner_tenant != tenant.tenant_id:
            raise HTTPException(
                status_code=409,
                detail=f"Cross-tenant collection collision: physical collection for '{raw_coll}' is already owned by another tenant.",
            )

    other_tenant_has_docs = (
        db.query(DocumentRecord)
        .filter(
            DocumentRecord.tenant_id != tenant.tenant_id,
            DocumentRecord.collection_name == raw_coll,
        )
        .first()
    )
    if other_tenant_has_docs and scoped_coll in existing_chroma_colls:
        raise HTTPException(
            status_code=409,
            detail=f"Cross-tenant collection collision: physical collection for '{raw_coll}' is already owned by another tenant.",
        )

    safe_name = secure_filename(file.filename or "unnamed.pdf")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    original_name = safe_name
    task_id = generate_uuid()
    save_path = os.path.join(UPLOAD_DIR, f"{task_id}_{safe_name}")

    if not os.path.abspath(save_path).startswith(os.path.abspath(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Invalid file destination path.")

    MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    total_written = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := file.file.read(8192):
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_BYTES:
                    f.close()
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        logger.exception("Failed to write uploaded file to disk: %s", e)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e

    # 1. Dev-Gated Synchronous Ingestion
    if sync and ENV == "development":
        logger.info(
            "[DEV SYNC] Ingesting '%s' synchronously for tenant '%s'",
            original_name,
            tenant.tenant_name,
        )
        try:
            result = ingest_pdf(
                save_path,
                scoped_coll,
                force=force,
                source_name=original_name,
                doc_title=doc_title,
                category=category,
                tenant_id=tenant.tenant_id,
            )
            doc_record = DocumentRecord(
                tenant_id=tenant.tenant_id,
                collection_name=raw_coll,
                filename=original_name,
                doc_title=doc_title or original_name,
                category=category or "uncategorized",
                file_path=save_path,
                upload_date=time.strftime("%Y-%m-%d"),
            )
            db.add(doc_record)
            db.commit()

            invalidate_tenant_cache(tenant.tenant_id)

            response.status_code = status.HTTP_200_OK
            return {
                "task_id": task_id,
                "status": "completed",
                "message": result,
                "collection": raw_coll,
                "doc_title": doc_title or original_name,
                "category": category or "uncategorized",
                "tenant": tenant.tenant_name,
            }
        except Exception as e:
            logger.exception("Synchronous ingestion failed: %s", e)
            raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    # 2. Production Asynchronous Ingestion (BackgroundTasks)
    task_record = IngestionTask(
        id=task_id,
        tenant_id=tenant.tenant_id,
        collection_name=raw_coll,
        filename=original_name,
        doc_title=doc_title or original_name,
        category=category or "uncategorized",
        status="pending",
    )
    db.add(task_record)
    db.commit()

    background_tasks.add_task(
        run_background_ingest,
        task_id=task_id,
        tenant_id=tenant.tenant_id,
        file_path=save_path,
        scoped_coll=scoped_coll,
        raw_coll=raw_coll,
        original_name=original_name,
        doc_title=doc_title,
        category=category,
        force=force,
    )

    response.status_code = status.HTTP_202_ACCEPTED
    return {
        "task_id": task_id,
        "status": "processing",
        "collection": raw_coll,
        "filename": original_name,
        "message": "File upload accepted. Ingestion running asynchronously in background.",
    }


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, summary="Get status of an async ingestion task")
def get_task_status(
    task_id: str,
    tenant:  TenantContext = Depends(check_rate_limit),
    db:      Session       = Depends(get_db),
) -> TaskStatusResponse:
    """
    Poll the status of an asynchronous document ingestion task.
    Includes in-process timeout staleness detection (> 5 minutes).
    """
    task = (
        db.query(IngestionTask)
        .filter(IngestionTask.id == task_id, IngestionTask.tenant_id == tenant.tenant_id)
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion task '{task_id}' not found.",
        )

    # Heartbeat lease staleness check:
    # 1. Pending tasks: stale if never picked up by worker after 300s (5 min)
    # 2. Processing tasks: stale only if worker heartbeat has ceased for > 600s (10 min)
    now_utc = datetime.now(timezone.utc)
    if task.status == "pending":
        created_utc = task.created_at if task.created_at.tzinfo else task.created_at.replace(tzinfo=timezone.utc)
        if (now_utc - created_utc).total_seconds() > 300.0:
            logger.warning("Task '%s' pending for > 300s without worker pickup; marking as stale", task_id)
            task.status = "stale"
            task.error_message = "Task timed out waiting for worker pickup."
            task.completed_at = get_utc_now()
            task.updated_at = get_utc_now()
            db.commit()
    elif task.status == "processing":
        ref_time = task.updated_at or task.created_at
        ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        if (now_utc - ref_utc).total_seconds() > 600.0:
            logger.warning("Task '%s' processing heartbeat ceased for > 600s; marking as stale", task_id)
            task.status = "stale"
            task.error_message = "Task timed out or was interrupted by a server restart."
            task.completed_at = get_utc_now()
            task.updated_at = get_utc_now()
            db.commit()

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        collection=task.collection_name,
        filename=task.filename,
        total_chunks=task.total_chunks,
        error_message=task.error_message,
        created_at=str(task.created_at),
        updated_at=str(task.updated_at) if task.updated_at else None,
        completed_at=str(task.completed_at) if task.completed_at else None,
    )


@app.get("/tasks", summary="List recent ingestion tasks for authenticated tenant")
def list_tasks(
    limit:  int           = Query(20, ge=1, le=100),
    tenant: TenantContext = Depends(check_rate_limit),
    db:     Session       = Depends(get_db),
) -> list[dict[str, Any]]:
    """List recent ingestion tasks for the calling tenant."""
    tasks = (
        db.query(IngestionTask)
        .filter(IngestionTask.tenant_id == tenant.tenant_id)
        .order_by(IngestionTask.created_at.desc())
        .limit(limit)
        .all()
    )

    # Apply lease-based staleness check to returned tasks
    now_utc = datetime.now(timezone.utc)
    for task in tasks:
        if task.status == "pending":
            created_utc = task.created_at if task.created_at.tzinfo else task.created_at.replace(tzinfo=timezone.utc)
            if (now_utc - created_utc).total_seconds() > 300.0:
                task.status = "stale"
                task.error_message = "Task timed out waiting for worker pickup."
                task.completed_at = get_utc_now()
                task.updated_at = get_utc_now()
        elif task.status == "processing":
            ref_time = task.updated_at or task.created_at
            ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
            if (now_utc - ref_utc).total_seconds() > 600.0:
                task.status = "stale"
                task.error_message = "Task timed out or was interrupted by a server restart."
                task.completed_at = get_utc_now()
                task.updated_at = get_utc_now()
    db.commit()

    return [
        {
            "task_id": t.id,
            "status": t.status,
            "collection": t.collection_name,
            "filename": t.filename,
            "total_chunks": t.total_chunks,
            "error_message": t.error_message,
            "created_at": str(t.created_at),
            "updated_at": str(t.updated_at) if t.updated_at else None,
            "completed_at": str(t.completed_at) if t.completed_at else None,
        }
        for t in tasks
    ]


def _compute_candidate_confidence(cand: dict[str, Any], rerank_enabled: bool) -> float:
    """
    Compute normalized relevance confidence in [0.0, 1.0] for a retrieved candidate:
    - If reranker is active, uses cross-encoder sigmoid score (higher is better).
    - If reranker is disabled:
      - Uses true semantic cosine similarity (1.0 - dense_score) from Chroma vector search.
      - Incorporates multi-channel corroboration when BM25 keyword search also matched,
        provided baseline semantic relevance is non-trivial (dense_sim >= 0.25).
      - Superficial single-token BM25 matches on semantically irrelevant candidates
        (dense_sim < 0.25) cannot unilaterally bypass the fallback threshold.
      - Standalone BM25 matches without dense corroboration are scaled conservatively
        by keyword density (bm25_score / 10.0) rather than assigned a static high score.
      - Falls back to normalized synthetic distance similarity if raw channel metrics are absent.
    """
    if rerank_enabled and cand.get("rerank_score") is not None:
        return float(cand["rerank_score"])

    dense_dist = cand.get("dense_score")
    channels = cand.get("retrieval_channels", [])
    bm25_score = cand.get("bm25_score")

    if dense_dist is not None:
        dense_sim = max(0.0, min(1.0, 1.0 - float(dense_dist)))
        if "bm25" in channels and bm25_score is not None and bm25_score > 0:
            if dense_sim >= 0.25:
                # Award multi-channel corroboration bonus scaled by keyword strength
                bonus = min(0.20, max(0.05, float(bm25_score) / 20.0))
                return min(1.0, dense_sim + bonus)
            # Semantically irrelevant candidates with incidental lexical overlap
            # must not bypass the fallback threshold (0.35)
            return min(0.28, dense_sim + 0.05)
        return dense_sim

    if "bm25" in channels and bm25_score is not None and bm25_score > 0:
        # Standalone sparse match without dense corroboration:
        # Vector search did not return this candidate in its top pool (negligible semantic alignment).
        # Without semantic vector corroboration, an isolated BM25 keyword hit cannot guarantee
        # that the chunk answers the query. To prevent corpus-dependent raw BM25 score spikes
        # from suppressing web fallback on semantically unsupported queries, uncorroborated BM25
        # confidence is capped conservatively below the fallback threshold (0.35).
        return min(0.30, float(bm25_score) / 20.0)

    raw_score = cand.get("score")
    if raw_score is not None:
        return max(0.0, min(1.0, 1.0 - (float(raw_score) / 2.0)))

    return 0.0


@app.post("/query", response_model=QueryResponse, summary="Query tenant collections with hybrid retrieval, reranking, and web fallback")
def query_document(
    request: QueryRequest,
    tenant:  TenantContext = Depends(check_rate_limit),
    db:      Session       = Depends(get_db),
) -> QueryResponse:
    """
    Query documents within the authenticated tenant's isolated collections.
    Supports hybrid BM25 + dense retrieval, cross-encoder reranking, and web search fallback.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    target_collections: list[str] = []
    if request.collection_names and len(request.collection_names) > 0:
        target_collections = [c.strip() for c in request.collection_names if c and c.strip()]
    elif request.collection_name and request.collection_name.strip():
        target_collections = [request.collection_name.strip()]

    if not target_collections:
        raise HTTPException(
            status_code=400,
            detail="At least one collection_name or collection_names must be provided.",
        )

    for coll in target_collections:
        _validate_collection_name(coll)

    t_start = time.perf_counter()

    try:
        rerank_flag = request.enable_rerank if request.enable_rerank is not None else ENABLE_RERANKER
        web_fallback_flag = request.enable_web_fallback if request.enable_web_fallback is not None else ENABLE_WEB_SEARCH
        effective_web_provider = (request.web_provider or WEB_SEARCH_PROVIDER).strip().lower()

        # 0. Check Redis Query Cache
        cached_result = get_cached_response(
            tenant_id=tenant.tenant_id,
            question=request.question,
            collection_names=target_collections,
            filters=request.filters,
            enable_rerank=rerank_flag,
            enable_web_fallback=web_fallback_flag,
            web_provider=effective_web_provider,
        )
        if cached_result:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            query_log = QueryLog(
                tenant_id=tenant.tenant_id,
                api_key_id=tenant.api_key_id,
                question=request.question,
                retrieval_mode="cached",
                fallback_triggered=cached_result.get("fallback_triggered", False),
                sources_count=len(cached_result.get("sources", [])),
                latency_ms=round(latency_ms, 2),
            )
            db.add(query_log)
            db.commit()

            return QueryResponse(
                answer=cached_result["answer"],
                retrieval_mode="cached",
                fallback_triggered=cached_result.get("fallback_triggered", False),
                sources=[SourceReference(**s) for s in cached_result.get("sources", [])],
                cached=True,
            )

        # 1. Resolve scoped internal collection names for the tenant
        # Strict isolation: tenants only access their own collections.
        # Collections owned by another tenant (checked via ChromaDB metadata or relational documents)
        # cannot be accessed even if an engineered or colliding namespace is targeted.
        chroma_client = get_chroma_client()
        existing_colls = {c.name: c for c in chroma_client.list_collections()}

        def is_collection_accessible_to_tenant(phys_coll: str, logical_name: str) -> bool:
            if phys_coll not in existing_colls:
                return False
            coll_meta = existing_colls[phys_coll].metadata or {}
            owner_tenant = coll_meta.get("tenant_id")
            if owner_tenant:
                return owner_tenant == tenant.tenant_id
            # Fail closed: check for foreign tenant documents first.
            # If another tenant has documents under this logical collection name, access must be
            # denied immediately to prevent cross-tenant chunk leakage from shared collections lacking metadata.
            has_other_docs = (
                db.query(DocumentRecord)
                .filter(
                    DocumentRecord.tenant_id != tenant.tenant_id,
                    DocumentRecord.collection_name == logical_name,
                )
                .first()
                is not None
            )
            if has_other_docs:
                return False
            has_own_docs = (
                db.query(DocumentRecord)
                .filter(
                    DocumentRecord.tenant_id == tenant.tenant_id,
                    DocumentRecord.collection_name == logical_name,
                )
                .first()
                is not None
            )
            if has_own_docs:
                return True
            if tenant.is_legacy_tenant:
                return phys_coll.startswith(f"t_{tenant.tenant_id[:12]}_")
            return phys_coll.startswith(f"t_{tenant.normalized_tid}_")

        scoped_collections = []
        for coll_target in target_collections:
            scoped_c = tenant.get_scoped_collection_name(coll_target)
            legacy_c = tenant.get_legacy_scoped_collection_name(coll_target)
            pre_mig_c = tenant.get_pre_migration_scoped_collection_name(coll_target)
            if is_collection_accessible_to_tenant(scoped_c, coll_target):
                scoped_collections.append(scoped_c)
            elif legacy_c and is_collection_accessible_to_tenant(legacy_c, coll_target):
                scoped_collections.append(legacy_c)
            elif pre_mig_c and is_collection_accessible_to_tenant(pre_mig_c, coll_target):
                scoped_collections.append(pre_mig_c)
            elif (
                tenant.tenant_id in ("default_tenant_id", "default_tena")
                and coll_target in existing_colls
                and not coll_target.startswith("t_")
                and not coll_target.startswith("tenant_")
            ):
                scoped_collections.append(coll_target)

        # 2. Local Retrieval across tenant-scoped collections
        local_context = retrieve_context(
            request.question,
            scoped_collections,
            filters=request.filters,
            enable_rerank=rerank_flag,
        )

        # 3. Evaluate fallback trigger
        best_local_confidence = max(
            (_compute_candidate_confidence(c, rerank_flag) for c in local_context),
            default=0.0,
        )
        low_relevance = best_local_confidence < WEB_SEARCH_FALLBACK_THRESHOLD
        score_summary = f"best local confidence: {best_local_confidence:.4f} < threshold {WEB_SEARCH_FALLBACK_THRESHOLD:.4f}"

        trigger_fallback = web_fallback_flag and (len(local_context) == 0 or low_relevance)

        final_context = local_context
        retrieval_mode = "local_document"
        fallback_triggered = False

        if trigger_fallback:
            logger.info(
                "Web fallback triggered for tenant '%s' query '%s' (local chunks: %d, %s)",
                tenant.tenant_name,
                request.question,
                len(local_context),
                score_summary if len(local_context) > 0 else "no local chunks retrieved",
            )
            web_results = search_web(
                query=request.question,
                provider=effective_web_provider,
            )
            if web_results:
                fallback_triggered = True
                if rerank_flag:
                    combined_pool = local_context + web_results
                    reranked_pool, _ = rerank_documents(
                        query=request.question,
                        documents=combined_pool,
                        top_k=TOP_K_RESULTS,
                        min_score=MIN_RELEVANCE_SCORE,
                    )
                    final_context = reranked_pool
                else:
                    final_context = (local_context + web_results)[:TOP_K_RESULTS]

                present_types = {c.get("source_type", "document") for c in final_context}
                if "document" in present_types and "web" in present_types:
                    retrieval_mode = "blended"
                elif "web" in present_types:
                    retrieval_mode = "web_fallback"
                else:
                    retrieval_mode = "local_document"
            else:
                logger.warning("Web search returned no results; proceeding with local context")

        # 4. Generate synthesized cited answer
        answer = generate_answer(request.question, final_context)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 5. Record Query Audit Log
        query_log = QueryLog(
            tenant_id=tenant.tenant_id,
            api_key_id=tenant.api_key_id,
            question=request.question,
            retrieval_mode=retrieval_mode,
            fallback_triggered=fallback_triggered,
            sources_count=len(final_context),
            latency_ms=round(latency_ms, 2),
        )
        db.add(query_log)
        db.commit()

        # Clean collection names on response (strip tenant_id prefix)
        full_prefix = f"t_{tenant.normalized_tid}_"
        legacy_prefix = f"t_{tenant.tenant_id[:12]}_"
        scoped_to_target = {}
        for coll_target in target_collections:
            scoped_to_target[tenant.get_scoped_collection_name(coll_target)] = coll_target
            leg = tenant.get_legacy_scoped_collection_name(coll_target)
            if leg:
                scoped_to_target[leg] = coll_target
            pre = tenant.get_pre_migration_scoped_collection_name(coll_target)
            if pre:
                scoped_to_target[pre] = coll_target

        formatted_sources = []
        for ctx_item in final_context:
            raw_c_name = ctx_item.get("collection_name")
            display_coll = raw_c_name
            if raw_c_name:
                if raw_c_name in scoped_to_target:
                    display_coll = scoped_to_target[raw_c_name]
                elif raw_c_name.startswith(full_prefix):
                    display_coll = raw_c_name[len(full_prefix):]
                elif tenant.is_legacy_tenant and raw_c_name.startswith(legacy_prefix):
                    display_coll = raw_c_name[len(legacy_prefix):]
            formatted_sources.append(
                SourceReference(
                    page=ctx_item.get("page", 1),
                    source=ctx_item.get("source", ""),
                    score=ctx_item.get("score", 0.0),
                    source_type=ctx_item.get("source_type", "document"),
                    url=ctx_item.get("url"),
                    provider=ctx_item.get("provider"),
                    collection_name=display_coll,
                    section=ctx_item.get("section", "General"),
                    breadcrumb=ctx_item.get("breadcrumb", "General"),
                    doc_title=ctx_item.get("doc_title"),
                    category=ctx_item.get("category"),
                    upload_date=ctx_item.get("upload_date"),
                    chunk_id=ctx_item.get("chunk_id"),
                    rerank_score=ctx_item.get("rerank_score"),
                    retrieval_channels=ctx_item.get("retrieval_channels"),
                )
            )

        # 6. Store in Redis Cache
        cache_payload = {
            "answer": answer,
            "retrieval_mode": retrieval_mode,
            "fallback_triggered": fallback_triggered,
            "sources": [s.model_dump() for s in formatted_sources],
        }
        set_cached_response(
            tenant_id=tenant.tenant_id,
            question=request.question,
            collection_names=target_collections,
            response_data=cache_payload,
            filters=request.filters,
            enable_rerank=rerank_flag,
            enable_web_fallback=web_fallback_flag,
            web_provider=effective_web_provider,
        )

        return QueryResponse(
            answer=answer,
            retrieval_mode=retrieval_mode,
            fallback_triggered=fallback_triggered,
            sources=formatted_sources,
            cached=False,
        )

    except Exception as e:
        logger.exception("Query execution failed for tenant '%s': %s", tenant.tenant_name, e)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e


@app.get("/collections", summary="List collections owned by the authenticated tenant")
def list_collections(
    tenant: TenantContext = Depends(check_rate_limit),
    db:     Session       = Depends(get_db),
) -> dict[str, Any]:
    """Return list of collections belonging strictly to the current tenant."""
    try:
        client = get_chroma_client()
        full_prefix = f"t_{tenant.normalized_tid}_"
        legacy_prefix = f"t_{tenant.tenant_id[:12]}_"
        all_chroma_colls = client.list_collections()
        all_colls = [
            c.name for c in all_chroma_colls
            if not ((c.metadata or {}).get("tenant_id") and (c.metadata or {}).get("tenant_id") != tenant.tenant_id)
        ]

        # Relational records store original logical client names (including names 29-45 chars)
        db_docs = (
            db.query(DocumentRecord.collection_name)
            .filter(DocumentRecord.tenant_id == tenant.tenant_id)
            .distinct()
            .all()
        )
        known_logical_names = {r[0] for r in db_docs if r[0]}

        pre_mig_prefix = None
        if len(tenant.tenant_id) > 32:
            old_tid = (
                tenant.tenant_id
                if len(tenant.tenant_id) <= 36
                else f"{tenant.tenant_id[:24]}{hashlib.sha256(tenant.tenant_id.encode('utf-8')).hexdigest()[:8]}"
            )
            pre_mig_prefix = f"t_{old_tid}_"

        tenant_colls: set[str] = set()

        for l_name in known_logical_names:
            scoped_c = tenant.get_scoped_collection_name(l_name)
            legacy_c = tenant.get_legacy_scoped_collection_name(l_name)
            pre_mig_c = tenant.get_pre_migration_scoped_collection_name(l_name)
            if (
                scoped_c in all_colls
                or (legacy_c and legacy_c in all_colls)
                or (pre_mig_c and pre_mig_c in all_colls)
            ):
                tenant_colls.add(l_name)

        for c in all_colls:
            if c.startswith(full_prefix):
                tail = c[len(full_prefix):]
                # If this tail doesn't correspond to a known long name, add directly (skipping internal dotted hashes)
                if not any(tenant.get_scoped_collection_name(l) == c for l in known_logical_names):
                    if "." not in tail:
                        tenant_colls.add(tail)
            elif (
                tenant.is_legacy_tenant
                and c.startswith(legacy_prefix)
            ):
                tail = c[len(legacy_prefix):]
                tenant_colls.add(tail)
            elif (
                pre_mig_prefix
                and c.startswith(pre_mig_prefix)
            ):
                tail = c[len(pre_mig_prefix):]
                if not any(tenant.get_pre_migration_scoped_collection_name(l) == c for l in known_logical_names):
                    if "." not in tail:
                        tenant_colls.add(tail)
            elif (
                tenant.tenant_id in ("default_tenant_id", "default_tena")
                and not c.startswith("t_")
                and not c.startswith("tenant_")
            ):
                tenant_colls.add(c)

        return {"collections": sorted(list(tenant_colls)), "tenant": tenant.tenant_name}
    except Exception as e:
        logger.exception("Failed to list collections for tenant '%s': %s", tenant.tenant_name, e)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e


@app.delete("/collections/{collection_name}", summary="Delete a tenant-owned collection")
def delete_collection(
    collection_name: str,
    tenant:          TenantContext = Depends(check_rate_limit),
    db:              Session       = Depends(get_db),
) -> dict[str, str]:
    """Delete the collection belonging to the authenticated tenant."""
    _validate_collection_name(collection_name)
    scoped_coll = tenant.get_scoped_collection_name(collection_name)
    legacy_coll = tenant.get_legacy_scoped_collection_name(collection_name)
    pre_mig_coll = tenant.get_pre_migration_scoped_collection_name(collection_name)
    try:
        client = get_chroma_client()
        existing_colls = {c.name: c for c in client.list_collections()}
        target_coll = None
        if scoped_coll in existing_colls:
            target_coll = scoped_coll
        elif legacy_coll and legacy_coll in existing_colls:
            target_coll = legacy_coll
        elif pre_mig_coll and pre_mig_coll in existing_colls:
            target_coll = pre_mig_coll

        # Check if any BM25 index file (.json or legacy .pkl) exists on disk
        bm25_candidates = [scoped_coll]
        if legacy_coll:
            bm25_candidates.append(legacy_coll)
        if pre_mig_coll:
            bm25_candidates.append(pre_mig_coll)

        existing_bm25_colls = [
            c for c in bm25_candidates
            if any(
                os.path.exists(os.path.join(CHROMA_PATH, f"{c}_bm25{ext}"))
                for ext in (".json", ".pkl")
            )
        ]
        bm25_exists = len(existing_bm25_colls) > 0

        doc_count = (
            db.query(DocumentRecord)
            .filter(
                DocumentRecord.tenant_id == tenant.tenant_id,
                DocumentRecord.collection_name == collection_name,
            )
            .count()
        )

        # Cross-tenant ownership verification:
        # If target_coll exists, verify it belongs to calling tenant, not another tenant
        if target_coll:
            coll_meta = existing_colls[target_coll].metadata or {}
            owner_tenant = coll_meta.get("tenant_id")
            if owner_tenant and owner_tenant != tenant.tenant_id:
                # Target collection belongs to a different tenant!
                raise CollectionNotFoundError(collection_name)
            if doc_count == 0:
                # Calling tenant has no documents; check if another tenant owns documents here
                other_tenant_has_docs = (
                    db.query(DocumentRecord)
                    .filter(
                        DocumentRecord.tenant_id != tenant.tenant_id,
                        DocumentRecord.collection_name == collection_name,
                    )
                    .first()
                    is not None
                )
                if other_tenant_has_docs:
                    raise CollectionNotFoundError(collection_name)

        # Orphaned BM25 file verification (when target_coll does not exist):
        # Prevent cross-tenant sidecar deletion and unverified deletion
        if not target_coll and bm25_exists:
            for c in existing_bm25_colls:
                file_tid = get_bm25_file_tenant_id(c)
                # If persistent sidecar indicates an explicit foreign owner, reject
                if file_tid and file_tid != tenant.tenant_id:
                    raise CollectionNotFoundError(collection_name)
                # If calling tenant has no relational documents and sidecar does not verify ownership, reject
                if doc_count == 0 and file_tid != tenant.tenant_id:
                    raise CollectionNotFoundError(collection_name)

            if doc_count == 0:
                other_tenant_has_docs = (
                    db.query(DocumentRecord)
                    .filter(
                        DocumentRecord.tenant_id != tenant.tenant_id,
                        DocumentRecord.collection_name == collection_name,
                    )
                    .first()
                    is not None
                )
                if other_tenant_has_docs:
                    raise CollectionNotFoundError(collection_name)

        if not target_coll and not bm25_exists and doc_count == 0:
            raise CollectionNotFoundError(collection_name)

        if target_coll:
            try:
                client.delete_collection(target_coll)
            except ValueError as ve:
                err_text = str(ve).lower()
                # ChromaDB raises ValueError("Collection <name> does not exist.") when collection is absent.
                # Only treat as already removed if specifically matching this condition; propagate any other ValueError.
                if "does not exist" in err_text:
                    logger.info("ChromaDB collection '%s' already removed: %s", target_coll, ve)
                else:
                    logger.exception("ChromaDB delete_collection raised unexpected ValueError for '%s': %s", target_coll, ve)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to delete underlying vector collection '{collection_name}': {ve}. Relational records were preserved.",
                    ) from ve
            except Exception as e:
                logger.exception("Failed to delete physical vector collection '%s': %s", target_coll, e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete underlying vector collection '{collection_name}'. Relational records were preserved.",
                ) from e

        # Delete BM25 sparse index on disk (must propagate failure to prevent retaining plain chunks)
        try:
            for c in bm25_candidates:
                delete_bm25_index(c)
        except Exception as e:
            logger.exception("Failed to delete BM25 index file for collection '%s': %s", collection_name, e)
            raise HTTPException(
                status_code=500,
                detail=f"Incomplete collection deletion: failed to remove persistent BM25 index for '{collection_name}'. Relational records were preserved.",
            ) from e

        # Clean records in Postgres/SQLite only after physical deletion succeeds
        db.query(DocumentRecord).filter(
            DocumentRecord.tenant_id == tenant.tenant_id,
            DocumentRecord.collection_name == collection_name,
        ).delete()
        db.commit()

        invalidate_tenant_cache(tenant.tenant_id)

        logger.info("Deleted collection '%s' (internal: '%s') for tenant '%s'", collection_name, target_coll, tenant.tenant_name)
        return {"message": f"Collection '{collection_name}' deleted."}
    except CollectionNotFoundError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete collection '%s': %s", collection_name, e)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e


@app.post("/reset", summary="Wipe all collections belonging to authenticated tenant")
def reset_tenant_collections(
    tenant: TenantContext = Depends(check_rate_limit),
    db:     Session       = Depends(get_db),
) -> dict[str, str]:
    """Wipe all collections belonging strictly to the calling tenant."""
    try:
        client = get_chroma_client()
        full_prefix = f"t_{tenant.normalized_tid}_"
        legacy_prefix = f"t_{tenant.tenant_id[:12]}_"
        pre_mig_prefix = None
        if len(tenant.tenant_id) > 32:
            old_tid = (
                tenant.tenant_id
                if len(tenant.tenant_id) <= 36
                else f"{tenant.tenant_id[:24]}{hashlib.sha256(tenant.tenant_id.encode('utf-8')).hexdigest()[:8]}"
            )
            pre_mig_prefix = f"t_{old_tid}_"

        # 1. Identify all logical collections owned by this tenant from DB records
        db_logical_records = (
            db.query(DocumentRecord.collection_name)
            .filter(DocumentRecord.tenant_id == tenant.tenant_id)
            .distinct()
            .all()
        )
        tenant_logical_names = {r[0] for r in db_logical_records if r[0]}

        # Map physical collection names to corresponding tenant logical names
        phys_to_logicals: dict[str, set[str]] = {}
        for l_name in tenant_logical_names:
            for cand in [
                tenant.get_scoped_collection_name(l_name),
                tenant.get_legacy_scoped_collection_name(l_name),
                tenant.get_pre_migration_scoped_collection_name(l_name),
            ]:
                if cand:
                    phys_to_logicals.setdefault(cand, set()).add(l_name)

        # 2. Identify physical collections in ChromaDB belonging to this tenant
        all_colls = client.list_collections()
        target_colls: list[Any] = []
        for coll in all_colls:
            coll_meta = coll.metadata or {}
            owner_tenant = coll_meta.get("tenant_id")
            if owner_tenant:
                if owner_tenant == tenant.tenant_id:
                    target_colls.append(coll)
                continue

            # If coll_meta lacks tenant_id, verify prefix and check relational ownership
            matches_prefix = coll.name.startswith(full_prefix) or (
                tenant.is_legacy_tenant and coll.name.startswith(legacy_prefix)
            ) or (
                pre_mig_prefix and coll.name.startswith(pre_mig_prefix)
            )
            if not matches_prefix:
                continue

            # Fail-closed check: if another tenant has documents mapping to this physical collection or logical name, skip
            candidate_logicals = phys_to_logicals.get(coll.name, set())
            has_other_docs = False
            for l_name in candidate_logicals:
                if (
                    db.query(DocumentRecord)
                    .filter(
                        DocumentRecord.tenant_id != tenant.tenant_id,
                        DocumentRecord.collection_name == l_name,
                    )
                    .first()
                    is not None
                ):
                    has_other_docs = True
                    break
            if has_other_docs:
                continue

            target_colls.append(coll)

        failed_physical: list[str] = []
        failed_logicals: set[str] = set()

        # 3. Delete physical ChromaDB collections and their BM25 indexes
        for coll in target_colls:
            p_name = coll.name
            try:
                client.delete_collection(p_name)
            except ValueError as ve:
                if "does not exist" in str(ve).lower():
                    pass
                else:
                    logger.exception("ChromaDB delete_collection error for '%s': %s", p_name, ve)
                    failed_physical.append(p_name)
                    for l in phys_to_logicals.get(p_name, []):
                        failed_logicals.add(l)
            except Exception as e:
                logger.exception("Failed to delete ChromaDB collection '%s': %s", p_name, e)
                failed_physical.append(p_name)
                for l in phys_to_logicals.get(p_name, []):
                    failed_logicals.add(l)

            try:
                delete_bm25_index(p_name)
            except Exception as e:
                logger.exception("Failed to delete BM25 index for collection '%s': %s", p_name, e)
                failed_physical.append(f"{p_name}_bm25")
                for l in phys_to_logicals.get(p_name, []):
                    failed_logicals.add(l)

        # 4. Clean up any remaining BM25 indexes for known tenant logical collections
        for l_name in tenant_logical_names:
            if l_name in failed_logicals:
                continue
            for cand in [
                tenant.get_scoped_collection_name(l_name),
                tenant.get_legacy_scoped_collection_name(l_name),
                tenant.get_pre_migration_scoped_collection_name(l_name),
            ]:
                if cand:
                    try:
                        delete_bm25_index(cand)
                    except Exception as e:
                        logger.exception("Failed to delete BM25 index '%s' for logical '%s': %s", cand, l_name, e)
                        failed_physical.append(f"{cand}_bm25")
                        failed_logicals.add(l_name)

        # 5. Handle failures and relational DocumentRecord cleanup
        if failed_logicals or failed_physical:
            if failed_logicals:
                # Atomically preserve records for failed collections while cleaning succeeded collections
                db.query(DocumentRecord).filter(
                    DocumentRecord.tenant_id == tenant.tenant_id,
                    DocumentRecord.collection_name.not_in(list(failed_logicals)),
                ).delete(synchronize_session=False)
                db.commit()
            invalidate_tenant_cache(tenant.tenant_id)
            raise HTTPException(
                status_code=500,
                detail=f"Incomplete reset: failed to delete physical collections: {', '.join(failed_physical)}. Relational records were preserved for failed collections.",
            )

        # Succeeded completely: delete all relational records for tenant
        db.query(DocumentRecord).filter(DocumentRecord.tenant_id == tenant.tenant_id).delete(synchronize_session=False)
        db.commit()

        invalidate_tenant_cache(tenant.tenant_id)

        logger.info("All collections deleted for tenant '%s'", tenant.tenant_name)
        return {"message": f"All collections for tenant '{tenant.tenant_name}' deleted."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Reset failed for tenant '%s': %s", tenant.tenant_name, e)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again or contact support.") from e


@app.post("/admin/api-keys", response_model=ApiKeyResponse, summary="Create new API key for tenant")
def create_api_key(
    req:    CreateApiKeyRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    db:     Session       = Depends(get_db),
) -> ApiKeyResponse:
    """Generate a new high-entropy 256-bit API key for the calling tenant."""
    rpm = req.rate_limit_rpm if req.rate_limit_rpm is not None else 60
    if rpm < 1:
        raise HTTPException(
            status_code=400,
            detail="Invalid rate_limit_rpm. Must be at least 1 request per minute.",
        )

    raw_key, key_hash, key_prefix = generate_api_key()
    new_key_record = ApiKey(
        tenant_id=tenant.tenant_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=req.name or "API Key",
        rate_limit_rpm=rpm,
        is_active=True,
    )
    db.add(new_key_record)
    db.commit()

    return ApiKeyResponse(
        id=new_key_record.id,
        key_prefix=new_key_record.key_prefix,
        name=new_key_record.name,
        rate_limit_rpm=new_key_record.rate_limit_rpm,
        raw_key=raw_key,  # Returned only once
    )


@app.get("/admin/api-keys", summary="List active API keys for tenant")
def list_api_keys(
    tenant: TenantContext = Depends(get_current_tenant),
    db:     Session       = Depends(get_db),
) -> list[dict[str, Any]]:
    """List metadata of all active API keys for the current tenant."""
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.tenant_id == tenant.tenant_id, ApiKey.is_active.is_(True))
        .all()
    )
    return [
        {
            "id": k.id,
            "key_prefix": k.key_prefix,
            "name": k.name,
            "rate_limit_rpm": k.rate_limit_rpm,
            "created_at": str(k.created_at),
        }
        for k in keys
    ]


@app.get("/health", summary="Liveness check (unauthenticated)")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Check PostgreSQL database, Redis cache, and ChromaDB connectivity."""
    db_ok = False
    cache_ok = False
    chroma_ok = False
    try:
        db_ok = db.execute(text("SELECT 1")).scalar() == 1
    except Exception:
        logger.warning("Database health check failed", exc_info=True)

    try:
        redis_client = get_redis_client()
        cache_ok = bool(redis_client and redis_client.ping())
    except Exception:
        logger.warning("Redis health check failed", exc_info=True)

    try:
        chroma_client = get_chroma_client()
        chroma_client.list_collections()
        chroma_ok = True
    except Exception:
        logger.warning("ChromaDB health check failed", exc_info=True)

    status_str = "ok" if db_ok and cache_ok and chroma_ok else "degraded"
    return {
        "status": status_str,
        "database": db_ok,
        "cache": cache_ok,
        "chromadb": chroma_ok,
    }


# --- static frontend mount ---
_frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.exists(_frontend_dir):
    _frontend_dir = os.path.abspath("frontend")

if os.path.exists(_frontend_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

