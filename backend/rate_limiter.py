"""
Rate limiting via Redis atomic TTL counters.

Enforces per-key request quotas with minute-bucket sliding windows and HTTP 429 Retry-After headers.
"""

import logging
import threading
import time

from fastapi import Depends, HTTPException, status

from auth import TenantContext, get_current_tenant
from config import DEFAULT_RATE_LIMIT_RPM
from redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Simple in-memory fallback dict with lock if Redis is unavailable or fails
_memory_counters: dict[str, tuple[int, int]] = {}
_memory_lock = threading.Lock()


def get_rate_limit_redis_key(limiter_key: str, timestamp: int | None = None) -> str:
    """Construct the exact Redis rate-limit key for a limiter token in the current minute bucket."""
    t = int(time.time()) if timestamp is None else int(timestamp)
    return f"ratelimit:{limiter_key}:{t // 60}"


def check_rate_limit(
    tenant: TenantContext = Depends(get_current_tenant),
) -> TenantContext:
    """
    FastAPI dependency to enforce per-key RPM rate limits.
    
    Raises:
        HTTPException(429) if rate limit is exceeded for the current minute bucket.
    """
    limit = tenant.rate_limit_rpm
    if limit is None or limit <= 0:
        # Defensive fallback: prevent quota bypass via nonpositive numbers
        limit = DEFAULT_RATE_LIMIT_RPM
        tenant.rate_limit_rpm = DEFAULT_RATE_LIMIT_RPM

    now = int(time.time())
    minute_bucket = now // 60
    seconds_left = 60 - (now % 60)
    # Scope counters by unique api_key_id to prevent collision across keys sharing prefix
    limiter_key = tenant.api_key_id or tenant.key_prefix
    key = get_rate_limit_redis_key(limiter_key, now)

    client = get_redis_client()
    current_count = None

    if client is not None:
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 65)
            results = pipe.execute()
            current_count = results[0]
        except Exception as e:
            logger.warning("Redis operation failed (%s); falling back to in-memory rate limiter.", e)
            current_count = None

    if current_count is None:
        # Thread-safe in-memory dict fallback
        with _memory_lock:
            count, bucket = _memory_counters.get(limiter_key, (0, minute_bucket))
            if bucket != minute_bucket:
                count = 1
            else:
                count += 1
            _memory_counters[limiter_key] = (count, minute_bucket)
            current_count = count

    if current_count > limit:
        logger.warning(
            "Rate limit exceeded for tenant '%s' (key: %s) — count: %d, limit: %d rpm",
            tenant.tenant_name,
            tenant.key_prefix,
            current_count,
            limit,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {limit} requests per minute exceeded. Try again in {seconds_left} seconds.",
            headers={
                "Retry-After": str(seconds_left),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(now + seconds_left),
            },
        )

    return tenant
