"""
PDF ingestion: extract pages with PyMuPDF, split into overlapping chunks,
embed with MiniLM, and persist to ChromaDB.

The `force` flag replaces existing chunks for a file rather than skipping it.
"""

import os
import logging

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from db import get_chroma_client

logger = logging.getLogger(__name__)


def load_pdf(file_path: str, source_name: str = None) -> list[dict]:
    """
    Extract text from each page, skipping blanks.

    Returns a list of {"content", "page", "source"} dicts.
    `source_name` overrides the basename used in metadata (useful when the
    file is a temp path but you want the original name in citations).
    """
    doc = fitz.open(file_path)
    source = source_name or os.path.basename(file_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        pages.append({
            "content": text,
            "page": page_num + 1,
            "source": source,
        })

    doc.close()
    logger.info("Extracted %d non-empty pages from '%s'", len(pages), file_path)
    return pages


def chunk_documents(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Split page text into overlapping chunks.

    Overlap prevents answers from being lost when a sentence straddles a
    chunk boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks, metadatas = [], []

    for page in pages:
        for chunk in splitter.split_text(page["content"]):
            chunks.append(chunk)
            metadatas.append({"page": page["page"], "source": page["source"]})

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks, metadatas


def embed_and_store(
    chunks: list[str],
    metadatas: list[dict],
    collection_name: str,
    source_filename: str,
    force: bool = False,
) -> str:
    """
    Embed chunks with MiniLM and persist them to ChromaDB.

    When `force` is True, existing chunks for this source file are deleted
    first. When False, the file is skipped if chunks already exist.

    Collections are created with cosine distance so similarity scores are
    directly interpretable as 0 (identical) to 2 (opposite).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = get_chroma_client()

    client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    if force:
        existing = vectorstore.get(where={"source": source_filename})
        if existing and existing.get("ids"):
            vectorstore.delete(existing["ids"])
            logger.info(
                "Removed %d existing chunks for '%s' before re-ingestion.",
                len(existing["ids"]),
                source_filename,
            )

    if not force:
        existing = vectorstore.get(where={"source": source_filename})
        if existing and existing.get("ids"):
            msg = (
                f"'{source_filename}' is already ingested in collection "
                f"'{collection_name}' ({len(existing['ids'])} chunks). "
                f"Pass force=True to replace them."
            )
            logger.warning(msg)
            return msg

    # PersistentClient auto-persists, so no explicit .persist() call needed.
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    msg = f"{len(chunks)} chunks stored in collection '{collection_name}'"
    logger.info(msg)
    return msg


def ingest_pdf(
    file_path: str,
    collection_name: str,
    force: bool = False,
    source_name: str = None,
) -> str:
    """Full pipeline: PDF -> chunks -> embeddings -> ChromaDB."""
    effective_source = source_name or os.path.basename(file_path)
    pages = load_pdf(file_path, source_name=effective_source)
    chunks, metadatas = chunk_documents(pages)
    return embed_and_store(chunks, metadatas, collection_name, effective_source, force=force)
