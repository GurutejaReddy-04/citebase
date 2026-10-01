"""
Reproducible Cache Latency and Speedup Benchmark.

Experimentally derives and measures:
1. Cold Query Latency (Cache Miss): Full retrieval, cross-encoder reranking, and generation pipeline.
2. Hot Query Latency (Cache Hit): Direct retrieval from Redis cache by deterministic SHA-256 tenant key.
3. Observed Speedup Ratio: Calculated as Cold P50 / Hot P50.

Usage:
    python scripts/benchmark_cache.py [--iterations N] [--coll-name NAME]
"""

import argparse
import logging
import os
import sys
import time
import uuid

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from fastapi.testclient import TestClient
import main
from main import app
from database import SessionLocal, init_db
from models import ApiKey, Tenant
from auth import generate_api_key
from cache import get_redis_client

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cache_benchmark")


def make_benchmark_pdf(text: str = "Benchmark Sample Text for Latency Derivation") -> bytes:
    """Generate in-memory valid PDF containing text using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def setup_benchmark_tenant(client: TestClient) -> str:
    """Initialize a dedicated benchmark tenant and return its API key."""
    init_db()
    db = SessionLocal()
    suffix = uuid.uuid4().hex[:6]
    tenant_name = f"bench_tenant_{suffix}"
    try:
        tenant = Tenant(name=tenant_name, is_active=True)
        db.add(tenant)
        db.flush()

        raw_key, key_hash, key_prefix = generate_api_key()
        api_key_rec = ApiKey(
            tenant_id=tenant.id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=f"Key for {tenant_name}",
            rate_limit_rpm=10000,
            is_active=True,
        )
        db.add(api_key_rec)
        db.commit()
        return raw_key
    finally:
        db.close()


def run_benchmark(iterations: int = 5, use_live_llm: bool = False):
    print("=" * 70)
    print("CiteBase Cache Performance & Speedup Benchmark")
    print(f"Sampling iterations per mode: {iterations}")
    print("=" * 70)

    # Use simulated LLM generation latency by default for deterministic offline benchmarking
    if not use_live_llm:
        import generator
        mock_gen = lambda q, ctx: (
            time.sleep(0.15) or f"Synthesized benchmark answer for '{q}' from {len(ctx)} sources [1]."
        )
        main.generate_answer = mock_gen
        generator.generate_answer = mock_gen

    client = TestClient(app)
    api_key = setup_benchmark_tenant(client)
    client.headers.update({"X-API-Key": api_key})

    # Ingest benchmark document
    coll_name = f"bench_coll_{uuid.uuid4().hex[:6]}"
    pdf_content = (
        "Enterprise document caching architecture provides dramatic latency reduction "
        "by storing deterministic query responses partitioned by tenant boundaries. "
        "Cached responses skip vector index lookups, sparse keyword scanning, and cross-encoder inference."
    )
    pdf_bytes = make_benchmark_pdf(pdf_content)

    print(f"Ingesting benchmark document into collection '{coll_name}'...")
    res_up = client.post(
        "/upload?sync=true",
        files={"file": ("bench.pdf", pdf_bytes, "application/pdf")},
        data={"collection_name": coll_name, "doc_title": "Cache Benchmark Doc"},
    )
    if res_up.status_code != 200:
        print(f"Error during ingestion: {res_up.text}")
        sys.exit(1)

    query_payload = {
        "question": "How does enterprise document caching reduce latency across vector systems?",
        "collection_name": coll_name,
        "enable_web_fallback": False,
    }

    # Verify Redis connectivity
    r = get_redis_client()
    redis_available = r is not None and r.ping()
    print(f"Redis cache connection: {'CONNECTED' if redis_available else 'IN-MEMORY / UNAVAILABLE'}")

    cold_latencies = []
    hot_latencies = []

    print("\nMeasuring cold queries (cache bypass)...")
    for i in range(iterations):
        # Flush or bypass cache to ensure cold miss
        if r:
            r.flushdb()
        t0 = time.perf_counter()
        res = client.post("/query", json=query_payload)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200, f"Query failed: {res.text}"
        data = res.json()
        cold_latencies.append(lat_ms)
        print(f"  [Cold {i+1}/{iterations}] Latency: {lat_ms:.2f} ms (cached: {data.get('cached')})")

    # Prime cache for hot measurements
    res_prime = client.post("/query", json=query_payload)
    assert res_prime.status_code == 200

    print("\nMeasuring hot queries (cache hits)...")
    for i in range(iterations):
        t0 = time.perf_counter()
        res = client.post("/query", json=query_payload)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        data = res.json()
        hot_latencies.append(lat_ms)
        print(f"  [Hot {i+1}/{iterations}] Latency: {lat_ms:.2f} ms (cached: {data.get('cached')})")

    # Calculate statistics
    cold_sorted = sorted(cold_latencies)
    hot_sorted = sorted(hot_latencies)

    cold_p50 = cold_sorted[len(cold_sorted) // 2]
    hot_p50 = hot_sorted[len(hot_sorted) // 2]

    speedup = (cold_p50 / hot_p50) if hot_p50 > 0 else 1.0

    print("\n" + "=" * 70)
    print("EXPERIMENTAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Cold P50 Latency: {cold_p50:.2f} ms (Min: {cold_sorted[0]:.2f} ms, Max: {cold_sorted[-1]:.2f} ms)")
    print(f"Hot P50 Latency:  {hot_p50:.2f} ms (Min: {hot_sorted[0]:.2f} ms, Max: {hot_sorted[-1]:.2f} ms)")
    print(f"Measured Speedup Ratio (Cold P50 / Hot P50): {speedup:.1f}x")
    print("=" * 70)

    # Clean up test collection
    client.delete(f"/collections/{coll_name}")
    print(f"Cleaned up test collection '{coll_name}'.")

    return {
        "cold_p50": cold_p50,
        "hot_p50": hot_p50,
        "speedup": speedup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CiteBase Cache Benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of samples per state")
    parser.add_argument("--use-live-llm", action="store_true", help="Execute against live Gemini API instead of deterministic offline simulation")
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, use_live_llm=args.use_live_llm)
