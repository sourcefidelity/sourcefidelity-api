"""Exa search provider — neural/semantic search API (formerly Metaphor).

Exa uses neural search (not keyword-based) — understands the MEANING of the
query, not just keywords. Particularly good for academic content: searching
"the impact of anime on Japanese cultural diplomacy" finds papers about that
TOPIC, not just papers containing those keywords.

API: POST https://api.exa.ai/search
Headers: x-api-key: ...
Body: {"query": "...", "numResults": 10, "type": "auto"}
Response: {"results": [{"title", "url", "id"}]}

Free tier available. Also supports "find similar" (search by URL) which could
be useful for finding OA copies of known papers.
"""

import logging

import httpx

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

_EXA_URL = "https://api.exa.ai/search"


class ExaSearch(SearchProvider):
    """Exa neural search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Exa"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Exa neural search API."""
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "numResults": min(num_results, 10),
            "type": "auto",  # auto = let Exa decide keyword vs neural
        }
        try:
            resp = httpx.post(_EXA_URL, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Exa search failed for '%s': %s", query[:60], e)
            return []

        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            # Exa doesn't always provide snippets in the search response;
            # use the text field if available
            snippet = item.get("text", item.get("summary", ""))[:300]
            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        return results
