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
from config import CACHE_ENABLED, CACHE_TTL_SECONDS, WEB_SEARCH_PROVIDER
from redis_client import get_redis_client

logger = logging.getLogger(__name__)


def normalize_question(question: str) -> str:
    """
    Normalize question text for deterministic cache key hashing:
    - Lowercase text and collapse consecutive whitespace.
    - Strip trailing sentence punctuation (?, !, .) while preserving internal technical punctuation (e.g. C++, C#, TCP/IP, node.js).
    """
    q = question.strip().lower()
    # Strip terminal sentence punctuation only (trailing ?, !, ., ;, :, quotes)
    q = re.sub(r"[\?\!\.\,\;\:\'\"]+$", "", q).strip()
    # Collapse multiple whitespace characters into single space
    q = re.sub(r"\s+", " ", q)
    return q


def build_cache_key(
    tenant_id: str,
    question: str,
    collection_names: list[str],
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = True,
    enable_web_fallback: bool = False,
    web_provider: Optional[str] = None,
) -> str:
    """
    Generate a deterministic, tenant-isolated cache key.
    
    Format:
      rag_cache:{tenant_id}:{32_char_sha256_hash}
      
    Guarantees that different tenants, execution options, fallback modes,
    or search providers never share cache entries inappropriately.
    """
    norm_q = normalize_question(question)
    sorted_colls = ",".join(sorted(collection_names))
    filter_repr = json.dumps(filters, sort_keys=True) if filters else "{}"
    prov = (web_provider or WEB_SEARCH_PROVIDER).strip().lower()
    
    payload = f"{norm_q}|{sorted_colls}|{filter_repr}|{enable_rerank}|{enable_web_fallback}|{prov}"
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    
    return f"rag_cache:{tenant_id}:{payload_hash}"


def get_cached_response(
    tenant_id: str,
    question: str,
    collection_names: list[str],
    filters: Optional[dict[str, Any]] = None,
    enable_rerank: bool = True,
    enable_web_fallback: bool = False,
    web_provider: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Retrieve cached query response from Redis.
    Returns None on cache miss or when caching is disabled.
    """
    if not CACHE_ENABLED:
        return None

    try:
        r = get_redis_client()
        key = build_cache_key(
            tenant_id,
            question,
            collection_names,
            filters,
            enable_rerank,
            enable_web_fallback,
            web_provider,
        )
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
    enable_web_fallback: bool = False,
    web_provider: Optional[str] = None,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> bool:
    """
    Store query response in Redis with TTL.
    """
    if not CACHE_ENABLED:
        return False

    try:
        r = get_redis_client()
        key = build_cache_key(
            tenant_id,
            question,
            collection_names,
            filters,
            enable_rerank,
            enable_web_fallback,
            web_provider,
        )
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
