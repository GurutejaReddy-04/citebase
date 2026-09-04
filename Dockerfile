# Multi-Tenant Production RAG-as-a-Service API
# Python 3.10 slim base image for optimal footprint and security

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    TRANSFORMERS_CACHE=/app/data/hf_cache \
    HF_HOME=/app/data/hf_cache

WORKDIR /app

# Install system build + runtime dependencies, build wheels, then remove build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build toolchain (not needed at runtime) to shrink image
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN useradd --create-home --shell /bin/bash appuser

# Create application data directories with correct ownership
RUN mkdir -p /app/chroma_db \
             /app/data/uploaded_docs \
             /app/data/bm25_indices \
             /app/data/hf_cache \
    && chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser frontend/ /app/frontend/

# Switch to non-root user for runtime security
USER appuser

# Expose API port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production uvicorn ASGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
