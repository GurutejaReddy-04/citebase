"""
Structure-aware PDF ingestion: extract document layout, headings, and
hierarchical sections with PyMuPDF, split into semantic boundary-aware chunks,
attach contextual breadcrumbs, embed with MiniLM, and persist to ChromaDB.

The `force` flag replaces existing chunks for a file rather than skipping it.
"""

import os
import re
import logging
from collections import Counter
from datetime import datetime, timezone

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    INCLUDE_BREADCRUMBS,
)
from chroma import get_chroma_client

logger = logging.getLogger(__name__)

# Blacklist regex for headers / footers / noise lines / date stamps / version stamps
_NOISE_LINE_REGEX = re.compile(
    r'^(?:'
    r'\d+\s*\|\s*P\s*a\s*g\s*e.*|'                # e.g. "2 | P a g e"
    r'Page\s+\d+(?:\s+of\s+\d+)?|'                 # e.g. "Page 1 of 10"
    r'\d+(?:\s+of\s+\d+)?|'                        # e.g. "1 of 10" or bare page digit
    r'\[\s*\d+\s*\]|'                              # e.g. "[1]"
    r'https?://\S+|www\.\S+|'                      # URLs
    r'\S+@\S+\.\S+|'                               # Emails
    r'.*\.fm\s+Page\s+\d+.*|'                      # Typesetting lines: "03_57_104_final.fm Page 57 Tuesday..."
    r'(?:effective\s+from|version\s+\d+|revised\s+on|published\s+on|academic\s+year|date\s*:).*|' # Date & version stamps
    r'(?:©|\(c\)|copyright|all rights reserved).*' # Copyright notices
    r')$',
    re.IGNORECASE
)

# Trailing words indicating an open-ended/incomplete clause, not a title
_TRAILING_LEADIN_WORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    'as', 'into', 'through', 'during', 'including', 'until', 'against', 'among',
    'and', 'or', 'but', 'so', 'yet', 'nor', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'that', 'which', 'who', 'whom',
    'when', 'where', 'why', 'how', 'if', 'whether', 'because', 'although', 'while'
}


_DATE_RANGE_REGEX = re.compile(
    r'^(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?\d{4}'
    r'(?:\s*[\u2013\u2014\-–—]\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(?:\d{4}|present|current))?$',
    re.IGNORECASE,
)


def _detect_body_font_size(doc: fitz.Document) -> float:
    """Determine the dominant (body) font size across the document."""
    font_sizes = []
    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            # Weight by character count for accurate mode detection
                            font_sizes.extend([round(span.get("size", 10.0), 1)] * len(text))

    if not font_sizes:
        return 10.0
    return Counter(font_sizes).most_common(1)[0][0]


def _detect_header_level(
    text: str,
    max_size: float,
    is_bold: bool,
    body_size: float,
) -> tuple[int | None, str]:
    """
    Detect if text is a section heading and determine its hierarchy level (1-4).

    Returns (level, clean_title) or (None, text) if not a header.
    """
    clean = text.strip()
    if not clean or len(clean) > 90:
        return None, clean

    # Blacklist and date timeline check
    if _NOISE_LINE_REGEX.match(clean) or _DATE_RANGE_REGEX.match(clean):
        return None, clean

    # 1. Markdown syntax (# Title, ## Subtitle)
    if clean.startswith("#"):
        hashes = len(clean) - len(clean.lstrip("#"))
        return min(hashes, 4), clean.lstrip("#").strip()

    # 2. Numbered sections (1. Title, 2.1 Subtitle, 3.1.2 Detail)
    num_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$', clean)
    if num_match and len(clean.split()) <= 10:
        nums = num_match.group(1).split(".")
        if len(num_match.group(2).strip()) > 1:
            level = min(len(nums), 4)
            return level, clean

    # 3. Formal structural keywords (Section 1, Article 4, Appendix A)
    if re.match(r'^(?:SECTION|ARTICLE|CHAPTER|PART|APPENDIX)\s+[0-9IVXLCDM\w]+', clean, re.IGNORECASE) and len(clean.split()) <= 8:
        return 1, clean

    words = clean.split()
    last_word = re.sub(r'[^\w]', '', words[-1].lower()) if words else ''
    if last_word in _TRAILING_LEADIN_WORDS:
        return None, clean

    if clean.endswith((".", ";", ":", ",")):
        return None, clean

    if len(words) > 8:
        return None, clean

    # 4. Font-size based detection
    if max_size >= body_size * 1.35 and (clean.isupper() or clean.istitle() or is_bold):
        return 1, clean
    elif max_size >= body_size * 1.2 and (clean.isupper() or clean.istitle() or is_bold):
        return 2, clean
    elif (max_size >= body_size * 1.1 or is_bold) and (clean.isupper() or clean.istitle()) and len(words) <= 6:
        return 3, clean

    return None, clean


def extract_document_structure(file_path: str, source_name: str = None) -> list[dict]:
    """
    Extract text elements with structural hierarchy, breadcrumbs, and coalesced paragraphs.
    """
    doc = fitz.open(file_path)
    source = source_name or os.path.basename(file_path)
    body_font_size = _detect_body_font_size(doc)
    logger.info("Detected body font size %.1fpt for '%s'", body_font_size, source)

    # Optional: Map PDF bookmarks/TOC if available
    toc = doc.get_toc()  # [[lvl, title, page], ...]
    toc_by_page = {}
    if toc:
        for item in toc:
            lvl, title, pno = item[0], item[1].strip(), item[2]
            toc_by_page.setdefault(pno, []).append((lvl, title))

    elements = []
    active_hierarchy: list[tuple[int, str]] = []  # Stack of (level: int, title: str)

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1

        # Check if page has TOC bookmark entries
        if page_num in toc_by_page:
            for lvl, title in toc_by_page[page_num]:
                while active_hierarchy and active_hierarchy[-1][0] >= lvl:
                    active_hierarchy.pop()
                active_hierarchy.append((lvl, title))

        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        current_paragraphs = []

        def flush_page_body():
            if current_paragraphs:
                full_text = "\n\n".join(p.strip() for p in current_paragraphs if p.strip()).strip()
                if full_text:
                    breadcrumb_titles = [h[1] for h in active_hierarchy]
                    breadcrumb_str = " > ".join(breadcrumb_titles) if breadcrumb_titles else "General"
                    current_section = active_hierarchy[-1][1] if active_hierarchy else "General"

                    elements.append({
                        "content": full_text,
                        "page": page_num,
                        "source": source,
                        "section": current_section,
                        "breadcrumb": breadcrumb_str,
                    })
                current_paragraphs.clear()

        for block in blocks:
            if block.get("type") != 0:
                continue

            block_line_tuples = []
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text or _NOISE_LINE_REGEX.match(line_text):
                    continue
                max_size = max(span.get("size", body_font_size) for span in line.get("spans", []))
                is_bold = any(
                    ("bold" in span.get("font", "").lower() or bool(span.get("flags", 0) & 2))
                    for span in line.get("spans", [])
                )
                block_line_tuples.append((line_text, max_size, is_bold))

            if not block_line_tuples:
                continue

            # Process lines in block
            curr_block_body = []
            for line_text, max_size, is_bold in block_line_tuples:
                level, title = _detect_header_level(line_text, max_size, is_bold, body_font_size)
                if level is not None:
                    if curr_block_body:
                        current_paragraphs.append(" ".join(curr_block_body))
                        curr_block_body.clear()
                    flush_page_body()
                    while active_hierarchy and active_hierarchy[-1][0] >= level:
                        active_hierarchy.pop()
                    active_hierarchy.append((level, title))
                else:
                    curr_block_body.append(line_text)

            if curr_block_body:
                current_paragraphs.append(" ".join(curr_block_body))

        flush_page_body()

    doc.close()
    logger.info("Extracted %d structured elements from '%s'", len(elements), source)
    return elements



def load_pdf(file_path: str, source_name: str = None) -> list[dict]:
    """
    Extract text and layout hierarchy from a PDF document.

    Returns structured elements with section breadcrumbs and page numbers.
    """
    return extract_document_structure(file_path, source_name=source_name)


def chunk_documents(
    elements: list[dict],
    doc_title: str = None,
    category: str = None,
    upload_date: str = None,
) -> tuple[list[str], list[dict]]:
    """
    Split structured document elements into semantic, boundary-aware chunks
    and attach enriched metadata (category, doc_title, upload timestamp, chunk_id).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        keep_separator=True,
    )

    effective_date = upload_date or datetime.now(timezone.utc).isoformat()
    effective_category = (category.strip().lower() if category and category.strip() else "uncategorized")
    effective_title = doc_title.strip() if doc_title and doc_title.strip() else ""

    chunks, metadatas = [], []

    for elem_idx, elem in enumerate(elements):
        text = elem["content"].strip()
        if not text:
            continue

        breadcrumb = elem.get("breadcrumb", "General")
        section = elem.get("section", "General")
        page = elem["page"]
        source = elem["source"]
        title_for_display = effective_title or source

        # If element fits comfortably in chunk_size, keep as a single semantic unit
        sub_texts = [text] if len(text) <= CHUNK_SIZE else splitter.split_text(text)

        for sub_idx, sub_text in enumerate(sub_texts):
            sub_text = sub_text.strip()
            # Clean leading punctuation from separator splitting
            while sub_text and sub_text[0] in ".,;:!? ":
                sub_text = sub_text[1:].strip()

            if len(sub_text) < MIN_CHUNK_SIZE:
                continue

            if INCLUDE_BREADCRUMBS:
                formatted_chunk = f"[Document: {title_for_display} | Section: {breadcrumb} | Page: {page}]\n{sub_text}"
            else:
                formatted_chunk = sub_text

            chunk_id = f"{source}_p{page}_e{elem_idx}_c{sub_idx}"

            chunks.append(formatted_chunk)
            metadatas.append({
                "chunk_id": chunk_id,
                "page": page,
                "source": source,
                "doc_title": effective_title or source,
                "category": effective_category,
                "upload_date": effective_date,
                "section": section,
                "breadcrumb": breadcrumb,
                "char_count": len(sub_text),
            })

    logger.info("Created %d structure-aware enriched chunks from %d elements", len(chunks), len(elements))
    return chunks, metadatas


def embed_and_store(
    chunks: list[str],
    metadatas: list[dict],
    collection_name: str,
    source_filename: str,
    force: bool = False,
) -> str:
    """
    Embed chunks with MiniLM and persist them with enriched metadata to ChromaDB.
    """
    if not chunks:
        return f"No extractable text chunks found in '{source_filename}'."

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

    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    # Rebuild BM25 sparse index for the collection
    try:
        from bm25 import rebuild_bm25_from_chroma
        rebuild_bm25_from_chroma(collection_name)
    except Exception:
        logger.warning("Failed to rebuild BM25 index for collection '%s'", collection_name, exc_info=True)

    msg = f"{len(chunks)} chunks stored in collection '{collection_name}'"
    logger.info(msg)
    return msg


def ingest_pdf(
    file_path: str,
    collection_name: str,
    force: bool = False,
    source_name: str = None,
    doc_title: str = None,
    category: str = None,
) -> str:
    """Full pipeline: PDF -> structured elements -> enriched semantic chunks -> ChromaDB."""
    effective_source = source_name or os.path.basename(file_path)
    elements = load_pdf(file_path, source_name=effective_source)
    chunks, metadatas = chunk_documents(
        elements,
        doc_title=doc_title,
        category=category,
    )
    return embed_and_store(chunks, metadatas, collection_name, effective_source, force=force)

