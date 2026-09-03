"""
Authentication and tenant resolution.

Hashes API keys with SHA-256 to prevent plaintext credential exposure in persistent storage.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from config import AUTH_ENABLED, BOOTSTRAP_API_KEY, DEFAULT_RATE_LIMIT_RPM, ENV
from database import get_db
from models import ApiKey, Tenant

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class TenantContext:
    """Represents the authenticated tenant context for a request."""
    tenant_id: str
    tenant_name: str
    api_key_id: str
    key_prefix: str
    rate_limit_rpm: int

    def get_scoped_collection_name(self, raw_collection_name: str) -> str:
        """
        Generate the internal, tenant-isolated collection key.
        Guarantees total length <= 63 characters for ChromaDB compliance.
        """
        clean_name = raw_collection_name.strip().lower()
        tenant_prefix = self.tenant_id[:12]
        return f"t_{tenant_prefix}_{clean_name[:45]}"


def hash_api_key(raw_key: str) -> str:
    """Compute deterministic SHA-256 hash of raw API key."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a cryptographically secure, high-entropy API key.
    
    Returns:
        tuple[raw_key, key_hash, key_prefix]:
        - raw_key: The plaintext token (256-bit entropy via secrets.token_urlsafe(32))
        - key_hash: SHA-256 hash stored in DB
        - key_prefix: Truncated prefix (e.g. 'sk_live_a1b2') for logs and identification
    """
    token = secrets.token_urlsafe(32)
    raw_key = f"sk_live_{token}"
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:12]
    return raw_key, key_hash, key_prefix


def extract_api_key_from_request(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Extract API key from X-API-Key or Authorization Bearer header."""
    if header_key:
        return header_key.strip()
    
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    
    return None


def get_current_tenant(
    request: Request,
    api_key: Optional[str] = Depends(extract_api_key_from_request),
    db: Session = Depends(get_db),
) -> TenantContext:
    """
    FastAPI security dependency to authenticate and resolve the tenant context.
    Raises HTTP 401 if key is missing, invalid, or inactive.
    """
    if not AUTH_ENABLED:
        # Development fallback when auth is explicitly disabled
        return TenantContext(
            tenant_id="dev_tenant",
            tenant_name="Development Tenant",
            api_key_id="dev_key_id",
            key_prefix="sk_live_dev",
            rate_limit_rpm=DEFAULT_RATE_LIMIT_RPM,
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide a valid key via 'X-API-Key' or 'Authorization: Bearer <key>' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = hash_api_key(api_key)
    api_key_record = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
        .first()
    )

    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    tenant = db.query(Tenant).filter(Tenant.id == api_key_record.tenant_id, Tenant.is_active.is_(True)).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is suspended or inactive.",
        )

    return TenantContext(
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        api_key_id=api_key_record.id,
        key_prefix=api_key_record.key_prefix,
        rate_limit_rpm=api_key_record.rate_limit_rpm,
    )


def bootstrap_default_tenant(db: Session) -> Optional[str]:
    """
    Bootstrap initial default tenant and API key if database is empty.
    
    Security check: Only prints raw API key to logs when ENV == 'development'.
    """
    tenant_count = db.query(Tenant).count()
    if tenant_count > 0:
        return None

    logger.info("Database is empty. Bootstrapping default initial tenant...")
    
    default_tenant = Tenant(
        name="default_org",
        is_active=True,
    )
    db.add(default_tenant)
    db.flush()

    raw_key = BOOTSTRAP_API_KEY if BOOTSTRAP_API_KEY else f"sk_live_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:12]

    api_key_record = ApiKey(
        tenant_id=default_tenant.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="Default Master API Key",
        rate_limit_rpm=DEFAULT_RATE_LIMIT_RPM,
        is_active=True,
    )
    db.add(api_key_record)
    db.commit()

    if ENV == "development":
        logger.info("==================================================================")
        logger.info("[DEV ONLY] Bootstrapped Default Tenant ID: %s", default_tenant.id)
        logger.info("[DEV ONLY] Bootstrap API Key created with prefix: %s...", raw_key[:12])
        logger.info("==================================================================")
    else:
        logger.info("Default tenant initialized. (Raw key printing suppressed in non-dev environment).")

    return raw_key
