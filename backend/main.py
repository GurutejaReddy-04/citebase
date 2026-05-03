"""
FastAPI entry point.

Routes:
  POST   /upload                    ingest a PDF into a named collection
  POST   /query                     retrieve chunks and generate an answer
  GET    /collections               list all collections in ChromaDB
  DELETE /collections/{name}        delete a single collection
  POST   /reset                     wipe all collections (optionally uploaded PDFs too)
  GET    /health                    liveness check
"""

import logging
import os
import re
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import ALLOWED_ORIGINS, UPLOAD_DIR
from db import get_chroma_client
from generator import generate_answer
from ingest import ingest_pdf
from retriever import retrieve_context

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --- app ---
app = FastAPI(
    title="RAG Document Intelligence API",
    description="Upload PDFs and query them with natural language. Answers are grounded in your document with exact page citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ChromaDB collection name rules: 3-63 chars, alphanumeric + hyphens + underscores,
# must start and end with an alphanumeric character.
_COLLECTION_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$')


def _validate_collection_name(name: str) -> None:
    """Raise 400 if the name doesn't satisfy ChromaDB's naming rules."""
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid collection name. Must be 3-63 characters, contain only "
                "letters, digits, hyphens (-), or underscores (_), and start/end "
                "with a letter or digit. Example: 'my-resume' or 'q3_report'."
            ),
        )


# --- schemas ---
class QueryRequest(BaseModel):
    question:        str
    collection_name: str


class SourceReference(BaseModel):
    page:   int
    source: str
    score:  float


class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceReference]


# --- routes ---
@app.post("/upload", summary="Ingest a PDF into a ChromaDB collection")
async def upload_document(
    file:            UploadFile = File(...),
    collection_name: str        = Form(...),
    force:           bool       = Form(False),
):
    """
    Writes to a temp file rather than directly to UPLOAD_DIR to avoid path
    traversal issues. The original filename is preserved as source metadata
    so citations show a readable name rather than the temp path.

    Set force=true to replace previously ingested chunks for the same file.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    original_name = os.path.basename(file.filename)
    _validate_collection_name(collection_name.strip())

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=UPLOAD_DIR)
    os.close(tmp_fd)

    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        logger.info("Saved upload '%s' -> '%s'", original_name, tmp_path)

        result = ingest_pdf(
            tmp_path,
            collection_name.strip(),
            force=force,
            source_name=original_name,
        )
        return {"message": result, "collection": collection_name.strip()}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed for '%s'", original_name)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Removed temp file '%s'", tmp_path)


@app.post("/query", response_model=QueryResponse, summary="Ask a question against an ingested document")
async def query_document(request: QueryRequest):
    """Embed the question, pull top-K chunks from ChromaDB, generate a cited answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    _validate_collection_name(request.collection_name.strip())

    try:
        context = retrieve_context(request.question, request.collection_name)
        answer  = generate_answer(request.question, context)

        return QueryResponse(
            answer=answer,
            sources=[
                SourceReference(page=c["page"], source=c["source"], score=c["score"])
                for c in context
            ],
        )

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections", summary="List all available ChromaDB collections")
def list_collections():
    try:
        client      = get_chroma_client()
        collections = [c.name for c in client.list_collections()]
        return {"collections": collections}
    except Exception as e:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/collections/{collection_name}", summary="Delete a collection and all its embeddings")
async def delete_collection(collection_name: str):
    """Returns 404 if the collection doesn't exist."""
    _validate_collection_name(collection_name)

    try:
        client = get_chroma_client()
        try:
            client.delete_collection(collection_name)
            logger.info("Collection '%s' deleted.", collection_name)
            return {"message": f"Collection '{collection_name}' deleted."}
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete collection '%s'", collection_name)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reset", summary="Wipe all collections and, optionally, uploaded PDFs")
async def reset_system(
    wipe_uploads: bool = Query(False, description="Also delete the uploads directory")
):
    """
    Deletes every collection via the ChromaDB client. Does not touch the
    database files directly. If wipe_uploads=true, the upload directory is
    removed and recreated.
    """
    try:
        client = get_chroma_client()
        for collection in client.list_collections():
            try:
                client.delete_collection(collection.name)
            except Exception:
                logger.warning("Could not delete collection '%s'", collection.name, exc_info=True)

        logger.info("All collections deleted.")

        if wipe_uploads and os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            logger.info("Upload directory reset.")

        return {"message": "All collections deleted. System reset complete."}

    except Exception as e:
        logger.exception("System reset failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}
