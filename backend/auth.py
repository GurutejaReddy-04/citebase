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

    @property
    def is_legacy_tenant(self) -> bool:
        """
        Only tenants created with legacy <=12-char IDs or seeded default tenants can own legacy collections.
        Modern tenants created with 32-char UUIDs return False to prevent prefix collision hijacking.
        """
        return len(self.tenant_id) <= 12 or self.tenant_id in ("default_tenant_id", "default_tena")

    @property
    def normalized_tid(self) -> str:
        """
        Normalized tenant identifier used in physical collection scoping.
        - Tenant IDs <= 32 chars (standard 32-char hex UUIDs, legacy IDs) are used directly.
        - Long tenant IDs (> 32 chars, up to 64 chars) derive a full 128-bit cryptographic
          namespace token (32 hexadecimal characters of SHA-256), providing full 128-bit
          collision resistance (P < 10^-18).
        """
        if len(self.tenant_id) <= 32:
            return self.tenant_id
        return hashlib.sha256(self.tenant_id.encode("utf-8")).hexdigest()[:32]

    def get_scoped_collection_name(self, raw_collection_name: str) -> str:
        """
        Generate the internal, tenant-isolated collection key.
        - Direct mapping: If 't_{tid}_{clean_name}' <= 63 chars, use directly.
          Because clean_name matches ^[a-z0-9][a-z0-9_-]{1,43}[a-z0-9]$, direct names never contain a dot ('.').
        - Disjoint hash-tailing: If direct length > 63 chars, use 't_{tid}_{prefix}.{name_hash}'.
          The inclusion of a dot ('.') guarantees mathematical disjointness between direct and compressed names,
          preventing any short logical name from colliding with an internal compressed name.
        - Uniqueness Guarantee: Disjointness between uncompressed and compressed classes is mathematically
          proven (empty intersection). Collision resistance among distinct compressed names within a tenant
          is cryptographically guaranteed by 64-bit truncated SHA-256 (probabilistic collision risk < 10^-9
          for 100k collections per tenant). Runtime collision detection is enforced at ingestion boundary.
        - Long Tenant ID Identity: Long tenant IDs (> 32 chars) derive a 128-bit namespace token (32 hex chars),
          unconditionally guaranteeing total physical collection length <= 63 across all tenant IDs up to 64 chars
          and all collection names up to 45 chars.
        """
        clean_name = raw_collection_name.strip()
        tid = self.normalized_tid
        direct = f"t_{tid}_{clean_name}"
        if len(direct) <= 63:
            return direct

        prefix_base = f"t_{tid}_"
        budget = 63 - len(prefix_base) - 1  # 1 char reserved for delimiter '.'
        hash_len = min(16, max(8, budget))
        name_hash = hashlib.sha256(clean_name.encode("utf-8")).hexdigest()[:hash_len]
        prefix_budget = budget - hash_len
        if prefix_budget > 0:
            prefix = clean_name[:prefix_budget].rstrip("_-") or clean_name[:prefix_budget]
            return f"{prefix_base}{prefix}.{name_hash}"
        return f"t_{tid}.{name_hash}"

    def get_legacy_scoped_collection_name(self, raw_collection_name: str) -> Optional[str]:
        """
        Fallback for seeded legacy collections. Returns None for modern 32-char tenants
        to strictly prevent cross-tenant legacy fallback hijacking.
        """
        if not self.is_legacy_tenant:
            return None
        clean_name = raw_collection_name.strip()
        return f"t_{self.tenant_id[:12]}_{clean_name}"

    def get_pre_migration_scoped_collection_name(self, raw_collection_name: str) -> Optional[str]:
        """
        Compatibility resolver for collections created under prior v6 scheme (Oct 2026):
        - Tenant IDs with length 33-36 chars were direct: 't_{tid}_{clean_name}'
        - Tenant IDs > 36 chars used: 't_{tid[:24] + sha256[:8]}_{clean_name}'
        Returns the v6 physical collection name if len(tenant_id) > 32, otherwise None.
        """
        if len(self.tenant_id) <= 32:
            return None
        clean_name = raw_collection_name.strip()
        old_tid = (
            self.tenant_id
            if len(self.tenant_id) <= 36
            else f"{self.tenant_id[:24]}{hashlib.sha256(self.tenant_id.encode('utf-8')).hexdigest()[:8]}"
        )
        direct = f"t_{old_tid}_{clean_name}"
        if len(direct) <= 63:
            return direct
        prefix_base = f"t_{old_tid}_"
        budget = 63 - len(prefix_base) - 1
        hash_len = min(16, max(8, budget))
        name_hash = hashlib.sha256(clean_name.encode("utf-8")).hexdigest()[:hash_len]
        prefix_budget = budget - hash_len
        if prefix_budget > 0:
            prefix = clean_name[:prefix_budget].rstrip("_-") or clean_name[:prefix_budget]
            return f"{prefix_base}{prefix}.{name_hash}"
        return f"t_{old_tid}.{name_hash}"


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
