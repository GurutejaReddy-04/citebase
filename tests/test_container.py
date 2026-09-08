"""
Test Suite for Phase 11: Containerization & Docker Orchestration Configuration.

Validates:
1. Dockerfile presence, healthcheck definition, python base, and entrypoint
2. .dockerignore rules
3. docker-compose.yml structure: services (api, postgres, redis), volumes, healthchecks, dependency links
4. Multi-tenant ChromaDB persistence guarantee (no pgvector migration, local volume persistence)
"""

import os
import re
import yaml
import pytest


def test_dockerfile_configuration():
    """Verify Dockerfile specifications."""
    dockerfile_path = "Dockerfile"
    assert os.path.exists(dockerfile_path), "Dockerfile does not exist"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Base image check
    assert "FROM python:3.10-slim" in content
    # Healthcheck check
    assert "HEALTHCHECK" in content
    assert "curl -f http://localhost:8000/health" in content
    # Expose port
    assert "EXPOSE 8000" in content
    # ASGI entrypoint
    assert 'CMD ["uvicorn", "main:app"' in content


def test_dockerignore_configuration():
    """Verify .dockerignore exclusions."""
    dockerignore_path = ".dockerignore"
    assert os.path.exists(dockerignore_path), ".dockerignore does not exist"

    with open(dockerignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".git" in content
    assert "__pycache__" in content
    assert ".pytest_cache" in content


def test_docker_compose_structure():
    """Verify docker-compose.yml service topology and volume orchestration."""
    compose_path = "docker-compose.yml"
    assert os.path.exists(compose_path), "docker-compose.yml does not exist"

    with open(compose_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    services = config.get("services", {})
    assert "postgres" in services, "PostgreSQL service missing from docker-compose"
    assert "redis" in services, "Redis service missing from docker-compose"
    assert "api" in services, "API service missing from docker-compose"

    # Verify PostgreSQL config
    pg = services["postgres"]
    assert "postgres:16-alpine" in pg["image"]
    assert "healthcheck" in pg
    assert any("pg_isready" in cmd for cmd in pg["healthcheck"]["test"])

    # Verify Redis config
    red = services["redis"]
    assert "redis:7-alpine" in red["image"]
    assert "healthcheck" in red
    assert "redis-cli" in red["healthcheck"]["test"]

    # Verify API config & dependency on healthchecks
    api = services["api"]
    assert api["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert api["depends_on"]["redis"]["condition"] == "service_healthy"

    # Verify ChromaDB volume preservation (NO pgvector migration)
    volumes = config.get("volumes", {})
    assert "chroma_data" in volumes
    assert "postgres_data" in volumes
    assert "redis_data" in volumes
    assert "uploaded_docs" in volumes
    assert "bm25_data" in volumes
