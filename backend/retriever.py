"""
Semantic retrieval from ChromaDB.

Score is cosine distance: 0 = identical, 2 = opposite. The frontend converts
this to a similarity percentage. Collections must have been created with
hnsw:space=cosine (ingest.py handles that).
"""

import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import EMBEDDING_MODEL, TOP_K_RESULTS
from db import get_chroma_client

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    collection_name: str,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Return the top-k chunks most similar to `query`.

    Each result dict contains:
        content  - chunk text passed into the prompt
        page     - source page number
        source   - original filename
        score    - cosine distance (lower = more relevant)
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = get_chroma_client()

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        logger.warning("No results found in collection '%s'", collection_name)
        return []

    context = [
        {
            "content": doc.page_content,
            "page":    doc.metadata.get("page"),
            "source":  doc.metadata.get("source"),
            "score":   round(score, 4),
        }
        for doc, score in results
    ]

    logger.info(
        "Retrieved %d chunks (best distance: %.4f)",
        len(context),
        context[0]["score"],
    )
    return context
