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
import re
from typing import Optional

import httpx

from app.config import settings
from app.services.retrieval.provider_runtime import (
    ProviderPolicy,
    ProviderRequestPacer,
    provider_policy,
)
from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)


def classify_searxng_failure(reason: str) -> str:
    """Normalize an upstream-engine reason without retaining arbitrary text."""
    normalized = re.sub(r"[^a-z0-9]+", " ", reason.lower()).strip()
    if "captcha" in normalized or "challenge" in normalized:
        return "captcha"
    if "too many" in normalized or "rate limit" in normalized or "429" in normalized:
        return "rate_limited"
    if any(value in normalized for value in ("access denied", "forbidden", "403")):
        return "access_restricted"
    if "timeout" in normalized or "timed out" in normalized:
        return "timeout"
    if any(value in normalized for value in ("parse", "invalid", "decode")):
        return "response_invalid"
    return "operational_failure"


def _aggregate_failure_status(failures: list[dict[str, str]]) -> str:
    """Return the most actionable status represented by one SearXNG response."""
    categories = {item["category"] for item in failures}
    for category in (
        "captcha",
        "rate_limited",
        "access_restricted",
        "timeout",
        "response_invalid",
        "operational_failure",
    ):
        if category in categories:
            return category
    return "completed"


class SearXNGSearch(SearchProvider):
    """SearXNG meta-search provider.

    Queries a SearXNG instance (self-hosted or public) for web results.
    The instance aggregates multiple engines, but does not remove their limits.
    """

    def __init__(
        self,
        instance_url: str,
        *,
        request_timeout_seconds: float = 15.0,
        engine_timeout_seconds: float | None = None,
    ):
        # Normalize URL (remove trailing slash)
        self._url = instance_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._engine_timeout_seconds = engine_timeout_seconds
        self.last_unresponsive_engines: list[tuple[str, str]] = []
        self.last_failure_reasons: list[dict[str, str]] = []
        self.last_pacing_seconds = 0.0
        self._pacer = ProviderRequestPacer(settings.PROVIDER_HEALTH_STATE_PATH)
        self._min_interval = provider_policy(
            "searxng", ProviderPolicy(min_interval_seconds=1.0)
        ).min_interval_seconds

    @property
    def name(self) -> str:
        return "SearXNG"

    def search(
        self,
        query: str,
        num_results: int = 10,
        engines: Optional[str] = None,
        *,
        pageno: int = 1,
    ) -> list[SearchResult]:
        """Search via SearXNG instance.

        Args:
            query: The search query string.
            num_results: Max results to return.
            engines: Optional comma-separated engine names to query. If None,
                uses the instance defaults.
        """
        self.last_status = "started"
        self.last_failure_reasons = []
        self.last_pacing_seconds = 0.0
        params = {
            "q": query,
            "format": "json",
            "pageno": max(1, pageno),
        }
        if engines:
            params["engines"] = engines
        if self._engine_timeout_seconds is not None:
            params["timeout_limit"] = self._engine_timeout_seconds
        try:
            # Share pacing across retrievers, workers and recovery probes.
            # Instance identifiers are hashed; no query is persisted. Pace
            # the whole instance so overlapping engine groups share the gate.
            with self._pacer.request(
                self._url,
                min_interval=self._min_interval,
                max_wait=self._request_timeout_seconds,
            ) as waited:
                self.last_pacing_seconds = waited
                resp = httpx.get(
                    f"{self._url}/search",
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=self._request_timeout_seconds,
                    follow_redirects=True,
                )
            resp.raise_for_status()
            data = resp.json()
            self.last_unresponsive_engines = [
                tuple(item[:2]) for item in data.get("unresponsive_engines", [])
                if isinstance(item, list) and len(item) >= 2
            ]
            self.last_failure_reasons = [
                {
                    "engine": str(engine),
                    "category": classify_searxng_failure(str(reason)),
                }
                for engine, reason in self.last_unresponsive_engines
            ]
        except Exception as e:
            self.last_status = classify_search_failure(e)
            self.last_unresponsive_engines = [(engines or "default", type(e).__name__)]
            self.last_failure_reasons = [
                {
                    "engine": engines or "default",
                    "category": self.last_status,
                }
            ]
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "SearXNG search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
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
        self.last_status = (
            _aggregate_failure_status(self.last_failure_reasons)
            if not results and self.last_failure_reasons
            else "completed"
        )
        return results
