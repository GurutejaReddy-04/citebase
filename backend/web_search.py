"""
Web search fallback provider.

When local documents provide insufficient signal, queries fall back to the live web
rather than letting the LLM invent facts.
"""

import logging
import requests
from typing import Any, Optional

from config import (
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_PROVIDER,
    TAVILY_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


def search_duckduckgo(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search DuckDuckGo using the ddgs library."""
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))

        results = []
        for r in raw_results:
            title = r.get("title", "").strip()
            url = r.get("href", "").strip()
            snippet = r.get("body", "").strip()
            if not snippet:
                continue
            results.append({
                "title": title or "Web Source",
                "url": url,
                "content": snippet,
                "source_type": "web",
                "provider": "duckduckgo",
                "doc_title": f"Web: {title}" if title else "Web Source",
                "source": url,
                "page": 1,
                "section": "Web Search",
                "breadcrumb": "Web Search",
                "score": 0.0,
                "rerank_score": None,  # Populated by cross-encoder when enabled
            })
        logger.info("DuckDuckGo search returned %d results for query: %s", len(results), query)
        return results
    except Exception as e:
        logger.warning("DuckDuckGo search failed for query '%s': %s", query, e)
        return []


def search_tavily(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Search Tavily REST API.
    Fully implemented alternative behind config flag / API key.
    """
    effective_key = api_key or TAVILY_API_KEY
    if not effective_key:
        logger.warning("Tavily API key not set. Falling back to DuckDuckGo search.")
        return search_duckduckgo(query, max_results=max_results)

    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": effective_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.warning("Tavily API returned status %d: %s. Falling back to DuckDuckGo.", resp.status_code, resp.text)
            return search_duckduckgo(query, max_results=max_results)

        data = resp.json()
        raw_results = data.get("results", [])

        results = []
        for r in raw_results:
            title = r.get("title", "").strip()
            url_link = r.get("url", "").strip()
            content = r.get("content", "").strip()
            if not content:
                continue
            results.append({
                "title": title or "Web Source",
                "url": url_link,
                "content": content,
                "source_type": "web",
                "provider": "tavily",
                "doc_title": f"Web: {title}" if title else "Web Source",
                "source": url_link,
                "page": 1,
                "section": "Web Search",
                "breadcrumb": "Web Search",
                "score": 0.0,
                "rerank_score": round(float(r.get("score", 1.0)), 4),
            })
        logger.info("Tavily search returned %d results for query: %s", len(results), query)
        return results
    except Exception as e:
        logger.warning("Tavily search failed for query '%s': %s. Falling back to DuckDuckGo.", query, e)
        return search_duckduckgo(query, max_results=max_results)


def search_web(
    query: str,
    provider: Optional[str] = None,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """
    Dispatch web search to configured provider ('duckduckgo' or 'tavily').
    """
    if not ENABLE_WEB_SEARCH:
        logger.info("Web search is disabled in config.")
        return []

    active_provider = (provider or WEB_SEARCH_PROVIDER).lower()
    if active_provider == "tavily":
        return search_tavily(query, max_results=max_results)
    return search_duckduckgo(query, max_results=max_results)
