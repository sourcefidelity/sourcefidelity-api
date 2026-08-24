"""SearXNG search provider — self-hosted meta-search engine.

SearXNG is a free, open-source metasearch application. Its upstream engines
are not unlimited: they may impose CAPTCHAs, rate limits, access blocks and
terms even when the SearXNG instance itself is healthy.
It can be self-hosted (one Docker container) or accessed via public instances.

This is the recommended search provider for:
  - Institutions (Path B): self-host a SearXNG instance on the university server.
    Self-hosted and private at the aggregation layer, with configurable native
    academic engines. Upstream availability must be measured and disclosed.
  - Restricted networks: configure SearXNG with accessible engines
    (for example Crossref, CORE and arXiv) reachable from the deployment.
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
    The instance aggregates multiple engines, but does not remove their limits.
    """

    def __init__(self, instance_url: str):
        # Normalize URL (remove trailing slash)
        self._url = instance_url.rstrip("/")
        self.last_unresponsive_engines: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "SearXNG"

    def search(self, query: str, num_results: int = 10,
               engines: Optional[str] = None) -> list[SearchResult]:
        """Search via SearXNG instance.

        Args:
            query: The search query string.
            num_results: Max results to return.
            engines: Optional comma-separated engine names to query. If None,
                uses the instance defaults.
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
            self.last_unresponsive_engines = [
                tuple(item[:2]) for item in data.get("unresponsive_engines", [])
                if isinstance(item, list) and len(item) >= 2
            ]
        except Exception as e:
            self.last_unresponsive_engines = [(engines or "default", type(e).__name__)]
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
