"""
Rate limiting via Redis atomic TTL counters.

Enforces per-key request quotas with minute-bucket sliding windows and HTTP 429 Retry-After headers.
"""

import logging
import threading
import time

from fastapi import Depends, HTTPException, status

from auth import TenantContext, get_current_tenant
from redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Simple in-memory fallback dict with lock if Redis is unavailable or fails
_memory_counters: dict[str, tuple[int, int]] = {}
_memory_lock = threading.Lock()


def check_rate_limit(
    tenant: TenantContext = Depends(get_current_tenant),
) -> TenantContext:
    """
    FastAPI dependency to enforce per-key RPM rate limits.
    
    Raises:
        HTTPException(429) if rate limit is exceeded for the current minute bucket.
    """
    limit = tenant.rate_limit_rpm
    if limit <= 0:
        return tenant  # Unlimited

    now = int(time.time())
    minute_bucket = now // 60
    seconds_left = 60 - (now % 60)
    key = f"ratelimit:{tenant.key_prefix}:{minute_bucket}"

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
            count, bucket = _memory_counters.get(tenant.key_prefix, (0, minute_bucket))
            if bucket != minute_bucket:
                count = 1
            else:
                count += 1
            _memory_counters[tenant.key_prefix] = (count, minute_bucket)
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
