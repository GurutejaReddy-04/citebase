"""
SQLAlchemy ORM Models for CiteBase Multi-Tenant RAG-as-a-Service Architecture.
Defines schemas for tenants, users, api_keys, documents, and query audit logs.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


def generate_uuid() -> str:
    """Generate a clean 12-char alphanumeric UUID."""
    return uuid.uuid4().hex[:12]


def get_utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class Tenant(Base):
    """Organization entity owning collections, API keys, and query logs."""
    __tablename__ = "tenants"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("DocumentRecord", back_populates="tenant", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="tenant", cascade="all, delete-orphan")
    ingestion_tasks = relationship("IngestionTask", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    """Individual user belonging to a tenant organization."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(32), default="member", nullable=False)  # admin, member
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    api_keys = relationship("ApiKey", back_populates="user")


class ApiKey(Base):
    """SHA-256 hashed API key with rate limiting and tenant binding."""
    __tablename__ = "api_keys"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256 hash of secret key
    key_prefix = Column(String(16), nullable=False, index=True)  # e.g., 'sk_live_a1b2' for identification
    name = Column(String(128), default="Default Key", nullable=False)
    rate_limit_rpm = Column(Integer, default=60, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")
    user = relationship("User", back_populates="api_keys")
    query_logs = relationship("QueryLog", back_populates="api_key")


class DocumentRecord(Base):
    """Record of an ingested document within a tenant's collection."""
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    collection_name = Column(String(128), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    doc_title = Column(String(255), nullable=True)
    category = Column(String(64), nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    file_path = Column(String(512), nullable=False)
    upload_date = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    tenant = relationship("Tenant", back_populates="documents")


class QueryLog(Base):
    """Audit log entry for a query executed against tenant collections."""
    __tablename__ = "query_logs"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    api_key_id = Column(String(64), ForeignKey("api_keys.id"), nullable=True, index=True)
    question = Column(Text, nullable=False)
    retrieval_mode = Column(String(32), nullable=False)  # local_document, blended, web_fallback
    fallback_triggered = Column(Boolean, default=False, nullable=False)
    sources_count = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    tenant = relationship("Tenant", back_populates="query_logs")
    api_key = relationship("ApiKey", back_populates="query_logs")


class IngestionTask(Base):
    """Tracks async PDF ingestion job status, progress, and errors."""
    __tablename__ = "ingestion_tasks"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    collection_name = Column(String(128), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    doc_title = Column(String(255), nullable=True)
    category = Column(String(64), nullable=True)
    status = Column(String(32), default="pending", nullable=False, index=True)  # pending, processing, completed, failed, stale
    total_chunks = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="ingestion_tasks")

