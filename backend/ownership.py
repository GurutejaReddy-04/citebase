"""
Unified Resource Ownership Resolver for CiteBase.

Enforces a single, consistent, fail-closed tenant ownership model across:
- ChromaDB vector collections
- BM25 index sidecars (.json and legacy .pkl)
- Logical collections across API routes (/upload, /query, /collections, /collections/{name}, /reset)

Core Invariants:
1. Every resource evaluates strictly to ResourceOwnership: OWNED, FOREIGN, or UNKNOWN.
2. UNKNOWN is strictly fail-closed: it must never be exposed, queried, or deleted.
3. Pickle deserialization (pickle.load / pickle.loads) is strictly PROHIBITED during ownership inspection.
4. Absence of evidence of another tenant's ownership is NEVER treated as evidence of caller's ownership.
"""

from enum import Enum
import json
import logging
import os
from typing import Any, Optional, Set

from sqlalchemy.orm import Session

from config import CHROMA_PATH
from models import DocumentRecord
from auth import TenantContext

logger = logging.getLogger(__name__)


class ResourceOwnership(str, Enum):
    OWNED = "OWNED"        # Positively proven to belong to the authenticated tenant
    FOREIGN = "FOREIGN"    # Positively proven to belong to a different tenant
    UNKNOWN = "UNKNOWN"    # Cannot be authoritatively proven -> FAIL CLOSED (treat as unowned/deny)


def get_tenant_candidate_physical_names(tenant: TenantContext, logical_name: str) -> list[str]:
    """
    Return candidate physical collection names in priority order for a given tenant:
    1. Modern scoped name (direct or dot-delimited hash-tailed)
    2. Seeded legacy name (only if tenant.is_legacy_tenant)
    3. Pre-migration v6 scoped name (only if len(tenant.tenant_id) > 32)
    4. Unprefixed name (only if seeded default tenant)
    """
    clean_name = logical_name.strip()
    candidates: list[str] = []

    scoped = tenant.get_scoped_collection_name(clean_name)
    if scoped and scoped not in candidates:
        candidates.append(scoped)

    legacy = tenant.get_legacy_scoped_collection_name(clean_name)
    if legacy and legacy not in candidates:
        candidates.append(legacy)

    pre_mig = tenant.get_pre_migration_scoped_collection_name(clean_name)
    if pre_mig and pre_mig not in candidates:
        candidates.append(pre_mig)

    if tenant.tenant_id in ("default_tenant_id", "default_tena"):
        if (
            clean_name not in candidates
            and not clean_name.startswith("t_")
            and not clean_name.startswith("tenant_")
        ):
            candidates.append(clean_name)

    return candidates


def is_physical_name_claimed_by_other_tenant(
    phys_coll_name: str,
    tenant: TenantContext,
    db: Session,
    candidate_logicals: Optional[Set[str]] = None,
) -> bool:
    """
    Check if any OTHER tenant in DocumentRecord could also resolve to phys_coll_name.
    Only called when inspecting unannotated resources lacking authoritative metadata.
    """
    query = db.query(DocumentRecord.tenant_id, DocumentRecord.collection_name).filter(
        DocumentRecord.tenant_id != tenant.tenant_id
    )
    if candidate_logicals:
        query = query.filter(DocumentRecord.collection_name.in_(list(candidate_logicals)))

    other_records = query.all()
    checked_tenants: dict[str, TenantContext] = {}
    for other_tid, other_coll_name in other_records:
        if other_tid not in checked_tenants:
            checked_tenants[other_tid] = TenantContext(
                tenant_id=other_tid,
                tenant_name="other",
                api_key_id="",
                key_prefix="",
                rate_limit_rpm=60,
            )
        other_ctx = checked_tenants[other_tid]
        if phys_coll_name in get_tenant_candidate_physical_names(other_ctx, other_coll_name):
            return True

    return False


def resolve_chroma_collection_ownership(
    phys_coll_name: str,
    tenant: TenantContext,
    db: Session,
    existing_coll: Optional[Any] = None,
    candidate_logicals: Optional[Set[str]] = None,
) -> ResourceOwnership:
    """
    Determine whether a physical ChromaDB collection belongs to the calling tenant.

    Fail-closed rules:
    - If collection metadata has explicit tenant_id:
        - If tenant_id == tenant.tenant_id -> OWNED
        - Otherwise -> FOREIGN
    - If collection metadata lacks tenant_id (or collection metadata is empty):
        - Caller MUST have positive DocumentRecords for candidate_logicals (has_own).
        - If foreign DocumentRecords map to this exact physical collection -> FOREIGN
        - If neither metadata nor database records prove caller ownership -> UNKNOWN (fail closed / deny)
    - Prefix matching alone NEVER establishes ownership.
    """
    if existing_coll is not None:
        coll_meta = existing_coll.metadata or {}
        owner_tenant = coll_meta.get("tenant_id")
        if owner_tenant:
            if owner_tenant == tenant.tenant_id:
                return ResourceOwnership.OWNED
            return ResourceOwnership.FOREIGN

    # Metadata absent: check relational database for positive proof
    if not candidate_logicals or len(candidate_logicals) == 0:
        return ResourceOwnership.UNKNOWN

    has_own = (
        db.query(DocumentRecord)
        .filter(
            DocumentRecord.tenant_id == tenant.tenant_id,
            DocumentRecord.collection_name.in_(list(candidate_logicals)),
        )
        .first()
        is not None
    )
    if not has_own:
        return ResourceOwnership.UNKNOWN

    if is_physical_name_claimed_by_other_tenant(phys_coll_name, tenant, db, candidate_logicals):
        return ResourceOwnership.FOREIGN

    if has_own:
        return ResourceOwnership.OWNED

    # Seeded default tenant fallback for collections created with default prefix
    if tenant.tenant_id in ("default_tenant_id", "default_tena"):
        if (
            phys_coll_name.startswith("t_default_tenant_id_")
            or phys_coll_name.startswith("t_default_tena_")
            or (
                candidate_logicals
                and any(
                    phys_coll_name in get_tenant_candidate_physical_names(tenant, l)
                    for l in candidate_logicals
                )
            )
        ):
            return ResourceOwnership.OWNED

    return ResourceOwnership.UNKNOWN


def resolve_bm25_file_ownership(
    phys_coll_name: str,
    tenant: TenantContext,
    db: Session,
    candidate_logicals: Optional[Set[str]] = None,
) -> ResourceOwnership:
    """
    Determine whether BM25 index files on disk (.json or .pkl) for phys_coll_name belong to the tenant.

    STRICT SAFETY RULE: NEVER deserializes pickle files.
    - JSON files: Inspect root-level 'tenant_id' or chunk metadatas.
    - Legacy PKL files / unannotated files:
      1. Proven if caller has DocumentRecord rows for candidate_logicals AND no other tenant maps to phys_coll_name.
      2. Proven if modern tenant (128-bit normalized_tid) matches phys_coll_name AND no other tenant maps to it.
      3. Proven if seeded default tenant matches default physical prefix AND no other tenant claims it.
      4. Otherwise -> UNKNOWN (fail closed).
    """
    json_path = os.path.join(CHROMA_PATH, f"{phys_coll_name}_bm25.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tid = data.get("tenant_id")
            if not tid:
                for m in data.get("metadatas", []):
                    if isinstance(m, dict) and m.get("tenant_id"):
                        tid = m["tenant_id"]
                        break
            if tid:
                if str(tid) == tenant.tenant_id:
                    return ResourceOwnership.OWNED
                return ResourceOwnership.FOREIGN
        except Exception:
            pass

    pkl_path = os.path.join(CHROMA_PATH, f"{phys_coll_name}_bm25.pkl")
    if not os.path.exists(pkl_path) and not os.path.exists(json_path):
        return ResourceOwnership.UNKNOWN

    # For legacy .pkl files or JSON lacking explicit tenant_id:
    # 1. Check if caller has positive database proof
    has_own = False
    if candidate_logicals:
        has_own = (
            db.query(DocumentRecord)
            .filter(
                DocumentRecord.tenant_id == tenant.tenant_id,
                DocumentRecord.collection_name.in_(list(candidate_logicals)),
            )
            .first()
            is not None
        )

    # 2. Check if another tenant claims this exact physical collection name
    if is_physical_name_claimed_by_other_tenant(phys_coll_name, tenant, db, candidate_logicals):
        return ResourceOwnership.FOREIGN

    if has_own:
        return ResourceOwnership.OWNED

    # 3. For modern tenants with full 128-bit (32 hex char) unique namespace tokens:
    # If the physical file matches the caller's modern scoped prefix and no foreign tenant claims it:
    if not tenant.is_legacy_tenant and len(tenant.normalized_tid) >= 32:
        if phys_coll_name.startswith(f"t_{tenant.normalized_tid}_"):
            return ResourceOwnership.OWNED

    # 4. For seeded default tenant matching default physical prefix:
    if tenant.tenant_id in ("default_tenant_id", "default_tena"):
        if (
            phys_coll_name.startswith("t_default_tenant_id_")
            or phys_coll_name.startswith("t_default_tena_")
            or (
                candidate_logicals
                and any(
                    phys_coll_name in get_tenant_candidate_physical_names(tenant, l)
                    for l in candidate_logicals
                )
            )
        ):
            return ResourceOwnership.OWNED

    return ResourceOwnership.UNKNOWN


def resolve_owned_collection(
    logical_name: str,
    tenant: TenantContext,
    db: Session,
    chroma_client: Any,
    existing_colls: Optional[dict[str, Any]] = None,
) -> tuple[Optional[str], ResourceOwnership]:
    """
    Find physical collection for logical_name that is authoritatively OWNED by tenant.
    Returns (physical_name, ResourceOwnership).
    If a matching collection exists but is FOREIGN, returns (physical_name, FOREIGN).
    If no candidate matches or candidate is UNKNOWN, returns (None, UNKNOWN).
    """
    if existing_colls is None:
        try:
            existing_colls = {c.name: c for c in chroma_client.list_collections()}
        except Exception:
            existing_colls = {}

    candidates = get_tenant_candidate_physical_names(tenant, logical_name)

    # 1. Check ChromaDB physical candidates
    for cand in candidates:
        if cand in existing_colls:
            ownership = resolve_chroma_collection_ownership(
                phys_coll_name=cand,
                tenant=tenant,
                db=db,
                existing_coll=existing_colls[cand],
                candidate_logicals={logical_name},
            )
            if ownership == ResourceOwnership.OWNED:
                return cand, ResourceOwnership.OWNED
            elif ownership == ResourceOwnership.FOREIGN:
                return cand, ResourceOwnership.FOREIGN

    # 2. Check on-disk BM25 candidates (when Chroma vector collection is absent or deleted)
    for cand in candidates:
        json_p = os.path.join(CHROMA_PATH, f"{cand}_bm25.json")
        pkl_p = os.path.join(CHROMA_PATH, f"{cand}_bm25.pkl")
        if os.path.exists(json_p) or os.path.exists(pkl_p):
            ownership = resolve_bm25_file_ownership(
                phys_coll_name=cand,
                tenant=tenant,
                db=db,
                candidate_logicals={logical_name},
            )
            if ownership == ResourceOwnership.OWNED:
                return cand, ResourceOwnership.OWNED
            elif ownership == ResourceOwnership.FOREIGN:
                return cand, ResourceOwnership.FOREIGN

    # 3. Check relational database records for logical collection name
    has_own = (
        db.query(DocumentRecord)
        .filter(
            DocumentRecord.tenant_id == tenant.tenant_id,
            DocumentRecord.collection_name == logical_name,
        )
        .first()
        is not None
    )
    if has_own:
        scoped = tenant.get_scoped_collection_name(logical_name)
        if not is_physical_name_claimed_by_other_tenant(scoped, tenant, db, {logical_name}):
            return scoped, ResourceOwnership.OWNED

    return None, ResourceOwnership.UNKNOWN
