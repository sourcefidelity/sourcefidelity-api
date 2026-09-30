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
import time

import httpx
from app.services.retrieval_deadline import remaining

from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_EXA_URL = "https://api.exa.ai/search"
# No `category` is sent. Measured 2026-09-25 on the labelled corpus, Exa's
# "publication" category found 13 of 16 real articles against 15 of 16
# without it (2 lost, none gained), so the filter narrows rather than helps.

# Documented account-level refusals (exa.ai/docs error codes, 2026-09-25):
# 401 invalid key, 402 credits exhausted or budget exceeded. Neither clears
# within a run, so the adapter stops calling. 403 is not here: it can be
# per-request content moderation. 503 is documented as over capacity, not
# billed and independent of request rate, so it is retried once.
_CIRCUIT_STATUSES = frozenset({401, 402})
_RETRY_STATUSES = frozenset({503})


class ExaSearch(SearchProvider):
    """Exa neural search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._circuit_status: str | None = None

    @property
    def name(self) -> str:
        return "Exa"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Exa neural search API."""
        if self._circuit_status is not None:
            self.last_status = self._circuit_status
            self.last_cost_usd = None
            return []
        self.last_status = "started"
        self.last_cost_usd = None
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            # The base price covers up to 10 results; each one above is billed.
            "numResults": min(num_results, 10),
            "type": "auto",  # auto = let Exa decide keyword vs neural
        }
        try:
            resp = httpx.post(_EXA_URL, headers=headers, json=payload, timeout=remaining(15))
            if resp.status_code in _RETRY_STATUSES:
                time.sleep(1.0)
                resp = httpx.post(_EXA_URL, headers=headers, json=payload, timeout=remaining(15))
            resp.raise_for_status()
            data = resp.json()
            items = data["results"]
            if not isinstance(items, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("url"), str)
                or not item["url"] or not isinstance(item.get("title", ""), str)
                or not isinstance(item.get("text", item.get("summary", "")), str)
                for item in items
            ):
                raise ValueError("invalid search result schema")
            cost = (data.get("costDollars") or {}).get("total")
            if type(cost) in (int, float) and cost >= 0:
                self.last_cost_usd = cost
        except Exception as e:
            self.last_status = classify_search_failure(e)
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in _CIRCUIT_STATUSES:
                self._circuit_status = self.last_status
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "Exa search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        results = []
        for item in items[:num_results]:
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
        self.last_status = "completed"
        return results
