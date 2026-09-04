"""
Centralized Redis client singleton with graceful fallback to FakeRedis.
Used across cache.py, rate_limiter.py, and health checks.
"""

import logging
import threading
from typing import Optional

import redis
from config import REDIS_URL

logger = logging.getLogger(__name__)

_REDIS_CLIENT: Optional[redis.Redis] = None
_REDIS_LOCK = threading.Lock()


def get_redis_client() -> redis.Redis:
    """
    Return active Redis client singleton.
    Gracefully falls back to fakeredis if standalone Redis server is unreachable.
    Thread-safe double-checked locking.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    with _REDIS_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT

        try:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1.0)
            client.ping()
            logger.info("Connected to Redis server at %s", REDIS_URL)
            _REDIS_CLIENT = client
            return _REDIS_CLIENT
        except Exception as e:
            logger.warning(
                "Redis server unreachable at %s (%s). Falling back to FakeRedis in-memory store.",
                REDIS_URL,
                e,
            )
            import fakeredis
            _REDIS_CLIENT = fakeredis.FakeRedis(decode_responses=True)
            return _REDIS_CLIENT


def reset_redis_client() -> None:
    """Reset the singleton instance (primarily for testing and mock injection)."""
    global _REDIS_CLIENT
    with _REDIS_LOCK:
        _REDIS_CLIENT = None
