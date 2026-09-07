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
    ENABLE_RERANKER,
    ENABLE_WEB_SEARCH,
    ENV,
    MIN_RELEVANCE_SCORE,
    TOP_K_RESULTS,
    UPLOAD_DIR,
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
from bm25 import delete_bm25_index

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

# ChromaDB collection naming regex
_COLLECTION_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$')


def _validate_collection_name(name: str) -> None:
    """Validate client-supplied collection name syntax."""
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid collection name. Must be 3-63 characters, contain only "
                "letters, digits, hyphens (-), or underscores (_), and start/end "
                "with a letter or digit."
            ),
        )


# --- schemas ---
class QueryRequest(BaseModel):
    """Payload schema for hybrid document retrieval and cited QA generation."""
    question:             str = Field(..., min_length=1, max_length=2000)
    collection_name:      Optional[str] = None
    collection_names:     Optional[list[str]] = None
    filters:              Optional[dict[str, Any]] = None
    enable_rerank:        Optional[bool] = None
    enable_web_fallback:  Optional[bool] = None
    web_provider:         Optional[str] = None


class SourceReference(BaseModel):
    """Metadata and citation attributes for a retrieved chunk or web result."""
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
    """Structured response containing synthesized answer, cited sources, and metadata."""
    answer:              str
    retrieval_mode:      str = "local_document"  # "local_document", "web_fallback", "blended", "cached"
    fallback_triggered:  bool = False
    sources:             list[SourceReference]
    cached:              bool = False


class CreateApiKeyRequest(BaseModel):
    """Request schema to provision a new tenant API key."""
    name:           Optional[str] = "API Key"
    rate_limit_rpm: Optional[int] = 60


class ApiKeyResponse(BaseModel):
    """Public API key details with raw secret token included upon initial creation."""
    id:             str
    key_prefix:     str
    name:           str
    rate_limit_rpm: int
    raw_key:        Optional[str] = None  # Only returned once on creation


class TaskStatusResponse(BaseModel):
    """Asynchronous background document ingestion task status and progress schema."""
    task_id:        str
    status:         str  # pending, processing, completed, failed, stale
    collection:     str
    filename:       str
    total_chunks:   int
    error_message:  Optional[str] = None
    created_at:     str
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

    # Staleness check: If stuck in pending/processing for > 5 minutes (300s), mark stale
    if task.status in ("pending", "processing"):
        created_utc = task.created_at
        if created_utc.tzinfo is None:
            created_utc = created_utc.replace(tzinfo=timezone.utc)
        elapsed_seconds = (datetime.now(timezone.utc) - created_utc).total_seconds()

        if elapsed_seconds > 300.0:  # 5 minutes timeout
            logger.warning("Task '%s' exceeded 300s timeout ceiling; marking as stale", task_id)
            task.status = "stale"
            task.error_message = "Task timed out or was interrupted by a server restart."
            task.completed_at = get_utc_now()
            db.commit()

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        collection=task.collection_name,
        filename=task.filename,
        total_chunks=task.total_chunks,
        error_message=task.error_message,
        created_at=str(task.created_at),
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

    # Apply staleness check to returned tasks
    now_utc = datetime.now(timezone.utc)
    for task in tasks:
        if task.status in ("pending", "processing"):
            created_utc = task.created_at
            if created_utc.tzinfo is None:
                created_utc = created_utc.replace(tzinfo=timezone.utc)
            if (now_utc - created_utc).total_seconds() > 300.0:
                task.status = "stale"
                task.error_message = "Task timed out or was interrupted by a server restart."
                task.completed_at = get_utc_now()
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
            "completed_at": str(t.completed_at) if t.completed_at else None,
        }
        for t in tasks
    ]


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

        # 0. Check Redis Query Cache
        cached_result = get_cached_response(
            tenant_id=tenant.tenant_id,
            question=request.question,
            collection_names=target_collections,
            filters=request.filters,
            enable_rerank=rerank_flag,
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
        # Strict isolation: tenants only access their own prefixed collections,
        # while default_tenant_id can also query root demo collections.
        chroma_client = get_chroma_client()
        existing_colls = {c.name for c in chroma_client.list_collections()}

        scoped_collections = []
        for coll_target in target_collections:
            scoped_c = tenant.get_scoped_collection_name(coll_target)
            if scoped_c in existing_colls:
                scoped_collections.append(scoped_c)
            elif (
                tenant.tenant_id == "default_tenant_id"
                and coll_target in existing_colls
                and not coll_target.startswith("t_")
                and not coll_target.startswith("tenant_")
            ):
                scoped_collections.append(coll_target)
            else:
                scoped_collections.append(scoped_c)

        # 2. Local Retrieval across tenant-scoped collections
        local_context = retrieve_context(
            request.question,
            scoped_collections,
            filters=request.filters,
            enable_rerank=rerank_flag,
        )

        # 3. Evaluate fallback trigger
        best_local_score = max((c.get("rerank_score", 1.0) for c in local_context), default=0.0)
        trigger_fallback = web_fallback_flag and (len(local_context) == 0 or best_local_score < WEB_SEARCH_FALLBACK_THRESHOLD)

        final_context = local_context
        retrieval_mode = "local_document"
        fallback_triggered = False

        if trigger_fallback:
            logger.info(
                "Web fallback triggered for tenant '%s' query '%s' (local chunks: %d, score: %.4f < threshold %.4f)",
                tenant.tenant_name,
                request.question,
                len(local_context),
                best_local_score,
                WEB_SEARCH_FALLBACK_THRESHOLD,
            )
            web_results = search_web(
                query=request.question,
                provider=request.web_provider or WEB_SEARCH_PROVIDER,
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
        prefix_to_strip = f"t_{tenant.tenant_id[:12]}_"
        formatted_sources = []
        for ctx_item in final_context:
            raw_c_name = ctx_item.get("collection_name")
            display_coll = raw_c_name.replace(prefix_to_strip, "") if raw_c_name else None
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
def list_collections(tenant: TenantContext = Depends(check_rate_limit)) -> dict[str, Any]:
    """Return list of collections belonging strictly to the current tenant."""
    try:
        client = get_chroma_client()
        tenant_prefix = f"t_{tenant.tenant_id[:12]}_"
        all_colls = [c.name for c in client.list_collections()]

        tenant_colls = []
        for c in all_colls:
            if c.startswith(tenant_prefix):
                tenant_colls.append(c[len(tenant_prefix):])
            elif (
                tenant.tenant_id == "default_tenant_id"
                and not c.startswith("t_")
                and not c.startswith("tenant_")
            ):
                tenant_colls.append(c)

        return {"collections": tenant_colls, "tenant": tenant.tenant_name}
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

    try:
        client = get_chroma_client()
        try:
            client.delete_collection(scoped_coll)
            delete_bm25_index(scoped_coll)

            # Clean records in Postgres/SQLite
            db.query(DocumentRecord).filter(
                DocumentRecord.tenant_id == tenant.tenant_id,
                DocumentRecord.collection_name == collection_name,
            ).delete()
            db.commit()

            invalidate_tenant_cache(tenant.tenant_id)

            logger.info("Deleted scoped collection '%s' for tenant '%s'", scoped_coll, tenant.tenant_name)
            return {"message": f"Collection '{collection_name}' deleted."}
        except ValueError:
            raise CollectionNotFoundError(collection_name)
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
        tenant_prefix = f"t_{tenant.tenant_id[:12]}_"
        for coll in client.list_collections():
            if coll.name.startswith(tenant_prefix):
                try:
                    client.delete_collection(coll.name)
                    delete_bm25_index(coll.name)
                except Exception:
                    logger.warning("Could not delete collection '%s'", coll.name, exc_info=True)

        db.query(DocumentRecord).filter(DocumentRecord.tenant_id == tenant.tenant_id).delete()
        db.commit()

        invalidate_tenant_cache(tenant.tenant_id)

        logger.info("All collections deleted for tenant '%s'", tenant.tenant_name)
        return {"message": f"All collections for tenant '{tenant.tenant_name}' deleted."}

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
    raw_key, key_hash, key_prefix = generate_api_key()
    new_key_record = ApiKey(
        tenant_id=tenant.tenant_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=req.name or "API Key",
        rate_limit_rpm=req.rate_limit_rpm or 60,
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

