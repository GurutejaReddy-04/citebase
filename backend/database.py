"""
SQLAlchemy database engine and session management.

Supports PostgreSQL in containerized production and SQLite for low-overhead local testing.
"""

import logging
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Handle SQLite vs PostgreSQL connection options
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    # Ensure data directory exists for SQLite
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if db_path.startswith("./"):
        db_path = db_path[2:]
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database tables if they do not exist and ensure schema migrations."""
    from sqlalchemy import text
    import models  # Ensure all model classes are imported before table creation
    Base.metadata.create_all(bind=engine)

    # Lightweight schema migration for SQLite and existing databases
    try:
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                res = conn.execute(text("PRAGMA table_info(ingestion_tasks);")).fetchall()
                col_names = [row[1] for row in res]
                if col_names and "updated_at" not in col_names:
                    conn.execute(text("ALTER TABLE ingestion_tasks ADD COLUMN updated_at DATETIME;"))
            elif engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE IF EXISTS ingestion_tasks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))
    except Exception as e:
        logger.exception("Failed to execute automatic schema migration for ingestion_tasks: %s", e)
        raise RuntimeError(f"Database schema migration failed: {e}") from e
