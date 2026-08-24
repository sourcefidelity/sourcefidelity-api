"""Semantic Scholar retrieval adapter.

Rate limit: 1 request per second (cumulative across all endpoints) with an API
key. Without a key, 100 requests / 5 minutes. This adapter enforces the
1-req/sec limit via a module-level throttle shared across all calls, so
multiple S2 requests in a resolution chain don't exceed it.
"""

import copy
import logging
import threading
import time

import httpx

from app.config import settings
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalSource, RetrievalResult
from app.services.retrieval.provider_runtime import (
    ProviderHealthStore,
    ProviderPolicy,
    provider_policy,
)

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Minimum seconds between S2 API request starts. The documented key limit is
# 1/sec; a wider margin accommodates network/server clock and arrival jitter.
_DEFAULT_POLICY = ProviderPolicy(
    timeout_seconds=30.0,
    min_interval_seconds=2.0,
    batch_size=5,
    max_batches=5,
    retry_delays_seconds=(15.0, 60.0),
    cooldown_seconds=900,
    max_cooldown_seconds=86400,
)

# Module-level throttle state: timestamp of the last S2 request.
_last_request_time: float = 0.0
_throttle_lock = threading.Lock()


def _throttle(min_interval: float) -> None:
    """Serialize calls and keep every S2 request start at least 1.05s apart."""
    global _last_request_time
    with _throttle_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_time = time.monotonic()


class SemanticScholarRetriever(RetrievalSource):
    name = "semantic_scholar"
    capabilities = frozenset({"doi", "batch_doi", "metadata", "abstract", "oa_link"})
    documentation_url = "https://api.semanticscholar.org/api-docs/graph"
    default_policy = _DEFAULT_POLICY
    deferred = True

    def __init__(self, health_store: ProviderHealthStore | None = None) -> None:
        self.policy = provider_policy(self.name, self.default_policy)
        self._health_store = health_store or ProviderHealthStore()
        self._rate_limited = False
        self._doi_cache: dict[str, RetrievalResult] = {}
        self.provider_metrics = {
            "calls": 0,
            "batch_calls": 0,
            "batch_items": 0,
            "batch_cache_hits": 0,
            "rate_limit_retries": 0,
            "rate_limited": 0,
            "circuit_skips": 0,
            "cooldown_skips": 0,
            "cooldown_seconds": 0,
            "network_errors": 0,
        }

    def _request(
        self,
        method: str,
        url: str,
        params: dict,
        json_body: dict | None = None,
    ) -> "httpx.Response | None":
        """Call S2 once; open a run-level circuit after a 429 response."""
        if self._rate_limited:
            self.provider_metrics["circuit_skips"] += 1
            return None
        if self._health_store.cooldown_remaining(self.name) > 0:
            self.provider_metrics["cooldown_skips"] += 1
            return None
        _throttle(self.policy.min_interval_seconds)
        self.provider_metrics["calls"] += 1
        try:
            resp = httpx.request(
                method, url, headers=self._headers(), params=params,
                json=json_body, timeout=self.policy.timeout_seconds,
            )
        except httpx.HTTPError:
            self.provider_metrics["network_errors"] += 1
            raise
        for delay in self.policy.retry_delays_seconds:
            if resp.status_code != 429:
                break
            self.provider_metrics["rate_limit_retries"] += 1
            time.sleep(delay)
            _throttle(self.policy.min_interval_seconds)
            self.provider_metrics["calls"] += 1
            try:
                resp = httpx.request(
                    method, url, headers=self._headers(), params=params,
                    json=json_body, timeout=self.policy.timeout_seconds,
                )
            except httpx.HTTPError:
                self.provider_metrics["network_errors"] += 1
                raise
        if resp.status_code == 429:
            self._rate_limited = True
            self.provider_metrics["rate_limited"] += 1
            cooldown = self._health_store.record_rate_limit(self.name, self.policy)
            self.provider_metrics["cooldown_seconds"] = cooldown
            logger.warning(
                "Semantic Scholar remained rate limited; disabling it for this "
                "run and starting a %ss cross-process cooldown", cooldown
            )
            return None
        if resp.is_success:
            self._health_store.record_success(self.name)
        return resp

    def _headers(self) -> dict:
        headers = {"User-Agent": f"SourceFidelity/{settings.APP_VERSION}"}
        if settings.S2_API_KEY:
            headers["x-api-key"] = settings.S2_API_KEY
        return headers

    def _fields(self) -> str:
        return "paperId,title,year,authors,externalIds,openAccessPdf"

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        normalized = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized.lower()

    def prefetch_dois(self, dois: list[str]) -> int:
        """Populate the per-run DOI cache using S2's 500-ID batch endpoint."""
        normalized = list(dict.fromkeys(
            self._normalize_doi(doi) for doi in dois if doi.strip()
        ))
        missing = [doi for doi in normalized if doi not in self._doi_cache]
        prefetched = 0
        batch_limit = (
            len(missing) if self.policy.max_batches == 0
            else self.policy.batch_size * self.policy.max_batches
        )
        for start in range(0, min(len(missing), batch_limit), self.policy.batch_size):
            chunk = missing[start:start + self.policy.batch_size]
            try:
                resp = self._request(
                    "POST",
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/batch",
                    {"fields": self._fields()},
                    {"ids": [f"DOI:{doi}" for doi in chunk]},
                )
                if resp is None:
                    break
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) != len(chunk):
                    raise ValueError("Unexpected Semantic Scholar batch response shape")
                self.provider_metrics["batch_calls"] += 1
                self.provider_metrics["batch_items"] += len(chunk)
                for doi, paper in zip(chunk, payload):
                    if paper:
                        self._doi_cache[doi] = self._parse_paper(paper)
                        prefetched += 1
                    else:
                        self._doi_cache[doi] = RetrievalResult(
                            source_name=self.name, success=False, error="Not found"
                        )
            except Exception as exc:
                logger.warning("S2 DOI batch prefetch failed: %s", exc)
                break
        return prefetched

    def search_by_doi(self, doi: str) -> RetrievalResult:
        normalized_doi = self._normalize_doi(doi)
        if normalized_doi in self._doi_cache:
            self.provider_metrics["batch_cache_hits"] += 1
            return copy.deepcopy(self._doi_cache[normalized_doi])

        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="DOI not prefetched; individual Semantic Scholar fallback disabled",
        )

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="Title search disabled; Semantic Scholar is a DOI-only OA supplement",
        )

    def _parse_paper(self, data: dict) -> RetrievalResult:
        external = data.get("externalIds", {}) or {}
        doi = external.get("DOI")

        authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")]
        year = str(data.get("year")) if data.get("year") else "n.d."
        title = data.get("title", "")

        oa = data.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
        locations = [
            AcquisitionLocation(
                url=pdf_url,
                provider=self.name,
                media_type="application/pdf",
                representation_kind=RepresentationKind.PDF,
                license=oa.get("license"),
                access_type="open_access",
                is_best=True,
            )
        ] if pdf_url else []

        abstract = data.get("abstract") or None

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=pdf_url,
            locations=locations,
            abstract=abstract,
            metadata=data,
        )
