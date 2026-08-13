"""SearXNG search provider — self-hosted meta-search engine.

SearXNG is a free, open-source meta-search engine that aggregates results from
multiple search engines (Google, Bing, DuckDuckGo, etc.) without tracking.
It can be self-hosted (one Docker container) or accessed via public instances.

This is the recommended search provider for:
  - Institutions (Path B): self-host a SearXNG instance on the university server.
    Free, unlimited queries, private (queries don't go to a commercial API),
    and configurable to search specific academic engines (Google Scholar, etc.).
  - China/restricted networks: configure SearXNG with accessible engines
    (Baidu, Bing China) to work behind the GFW.
  - Path C (rented GPU): run SearXNG alongside the app on the rented server.

Setup (self-hosted):
  docker run -d -p 8080:8080 searxng/searxng
  Then set SEARXNG_URL=http://localhost:8080 in .env

The SearXNG API is simple:
  GET /search?q=<query>&format=json
  Returns JSON with results: [{url, title, content, ...}, ...]
"""

import logging
from typing import Optional

import httpx

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class SearXNGSearch(SearchProvider):
    """SearXNG meta-search provider.

    Queries a SearXNG instance (self-hosted or public) for web results.
    The instance aggregates from multiple search engines — more comprehensive
    than any single API, and free/unlimited when self-hosted.
    """

    def __init__(self, instance_url: str):
        # Normalize URL (remove trailing slash)
        self._url = instance_url.rstrip("/")

    @property
    def name(self) -> str:
        return "SearXNG"

    def search(self, query: str, num_results: int = 10,
               engines: Optional[str] = None) -> list[SearchResult]:
        """Search via SearXNG instance.

        Args:
            query: The search query string.
            num_results: Max results to return.
            engines: Optional comma-separated engine names to query (e.g.,
                "google scholar" or "google"). If None, uses all configured
                engines. Use this for cascading: try "google scholar" first,
                then "google" as fallback.
        """
        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if engines:
            params["engines"] = engines
        try:
            resp = httpx.get(
                f"{self._url}/search",
                params=params,
                headers={"Accept": "application/json"},
                timeout=15,
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("SearXNG search failed for '%s': %s", query[:60], e)
            return []

        results = []
        for item in data.get("results", [])[:num_results]:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content", "")
            is_pdf = url.lower().endswith(".pdf") or "pdf" in item.get("mimetype", "").lower()
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        return results
