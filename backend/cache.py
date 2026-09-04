"""
Redis query-response caching layer.

The fastest vector search and LLM generation is the one you never execute.
"""

import hashlib
import json
import logging
import re
from typing import Any, Optional

import redis
from config import CACHE_ENABLED, CACHE_TTL_SECONDS
from redis_client import get_redis_client

logger = logging.getLogger(__name__)


def normalize_question(question: str) -> str:
    """Normalize question text: lowercase, strip punctuation and whitespace."""
    q = question.strip().lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def build_cache_key(
    tenant_id: str,
    question: str,
    collection_names: list[str],
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = True,
) -> str:
    """
    Generate a deterministic, tenant-isolated cache key.
    
    Format:
      rag_cache:{tenant_id}:{32_char_sha256_hash}
      
    Guarantees that different tenants asking identical questions NEVER share cache entries.
    """
    norm_q = normalize_question(question)
    sorted_colls = ",".join(sorted(collection_names))
    filter_repr = json.dumps(filters, sort_keys=True) if filters else "{}"
    
    payload = f"{norm_q}|{sorted_colls}|{filter_repr}|{enable_rerank}"
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    
    return f"rag_cache:{tenant_id}:{payload_hash}"


def get_cached_response(
    tenant_id: str,
    question: str,
    collection_names: list[str],
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Retrieve cached query response from Redis.
    Returns None on cache miss or when caching is disabled.
    """
    if not CACHE_ENABLED:
        return None

    try:
        r = get_redis_client()
        key = build_cache_key(tenant_id, question, collection_names, filters, enable_rerank)
        raw_val = r.get(key)
        if raw_val:
            data = json.loads(raw_val)
            logger.info("Cache HIT for tenant '%s' key '%s'", tenant_id, key)
            return data
        return None
    except Exception as e:
        logger.warning("Cache lookup error for tenant '%s': %s", tenant_id, e)
        return None


def set_cached_response(
    tenant_id: str,
    question: str,
    collection_names: list[str],
    response_data: dict[str, Any],
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = True,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> bool:
    """
    Store query response in Redis with TTL.
    """
    if not CACHE_ENABLED:
        return False

    try:
        r = get_redis_client()
        key = build_cache_key(tenant_id, question, collection_names, filters, enable_rerank)
        payload = json.dumps(response_data)
        r.set(key, payload, ex=ttl_seconds)
        logger.debug("Cache stored for tenant '%s' key '%s' (TTL: %ds)", tenant_id, key, ttl_seconds)
        return True
    except Exception as e:
        logger.warning("Cache write error for tenant '%s': %s", tenant_id, e)
        return False


def invalidate_tenant_cache(tenant_id: str) -> int:
    """
    Invalidate all cached queries belonging to a specific tenant using non-blocking SCAN.
    Called on new document ingestion or collection reset.
    """
    try:
        r = get_redis_client()
        pattern = f"rag_cache:{tenant_id}:*"
        cursor = 0
        deleted_count = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                deleted_count += r.delete(*keys)
            if cursor == 0 or cursor == "0":
                break
        if deleted_count > 0:
            logger.info("Invalidated %d cache entries for tenant '%s'", deleted_count, tenant_id)
        return deleted_count
    except Exception as e:
        logger.warning("Cache invalidation error for tenant '%s': %s", tenant_id, e)
        return 0
