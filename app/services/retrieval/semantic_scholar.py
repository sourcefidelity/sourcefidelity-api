"""Semantic Scholar retrieval adapter.

Rate limit: 1 request per second (cumulative across all endpoints) with an API
key. Without a key, 100 requests / 5 minutes. This adapter enforces the
1-req/sec limit via a module-level throttle shared across all calls, so
multiple S2 requests in a resolution chain don't exceed it.
"""

import copy
import logging
import re
import threading
import time

import httpx

from app.services.processing_metrics import record_provider_request
from app.config import secret_value, settings
from app.log_safety import safe_exception_code
from app.services.retrieval.base import SCHOLARLY_PAPER_KINDS, OBSERVED_AUTHOR_LIMIT, AcquisitionLocation, RepresentationKind, RetrievalSource, RetrievalResult
from app.services.retrieval import shared_pacing
from app.services.retrieval.provider_runtime import (
    ProviderHealthStore,
    ProviderPolicy,
    provider_policy,
)

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Minimum seconds between S2 request starts across every worker process. The
# documented key limit is 1 request per second across all endpoints; the margin
# absorbs arrival jitter. It used to be 2.0s per process, which four worker
# processes turned into about two a second on one key (2026-09-25).
#
# DOIs go through the batch endpoint, which accepts 500 ids. Chunks of 5 capped
# at 5 batches left every DOI after the 25th unlooked-up ("not prefetched") on a
# long reference list; a malformed id still splits its chunk (see
# _prefetch_doi_chunk), so a larger chunk costs nothing on error.
_DEFAULT_POLICY = ProviderPolicy(
    timeout_seconds=30.0,
    min_interval_seconds=1.1,
    batch_size=100,
    max_batches=0,
    # Measured 2026-09-25 on the labelled corpus: with the key, about half of
    # all calls are refused with 429 at random, at 1.1s, 2s or 3s spacing
    # alike, so it is gateway load-shedding, not our rate. Short retries
    # recover them: 33% succeed first time, 70% by the second attempt, 90% by
    # the fourth and 97.7% within six. The former 15s/60s waits turned each
    # refusal into a minute-long stall.
    retry_delays_seconds=(1.5, 2.0, 3.0, 4.0, 5.0),
    cooldown_seconds=900,
    # A day-long cooldown turned a burst of our own over-limit calls into a
    # day without Semantic Scholar; an hour is enough for a real outage.
    max_cooldown_seconds=3600,
)
# Without shared pacing (no Redis) each worker process paces alone; assume the
# worker's four processes and space each one's calls four times as far apart.
_UNSHARED_INTERVAL_MULTIPLIER = 4
_MAX_RETRY_AFTER_SECONDS = 60.0
# Calls in a row that stayed refused after every retry before the run stops
# calling and a cross-process cooldown starts. One exhausted call is expected
# about once in 40 under the measured load-shedding and is only that
# reference's failed search, never an outage.
_EXHAUSTED_CALLS_BEFORE_CIRCUIT = 3

# Module-level throttle state: timestamp of the last S2 request.
_last_request_time: float = 0.0
_throttle_lock = threading.Lock()


def _throttle(min_interval: float) -> None:
    """Keep S2 request starts `min_interval` apart across every worker process."""
    global _last_request_time
    with _throttle_lock:
        shared_wait = shared_pacing.reserve_start("semantic_scholar", min_interval)
        local_interval = min_interval if shared_wait is not None else min_interval * _UNSHARED_INTERVAL_MULTIPLIER
        now = time.monotonic()
        wait = max(0.0, local_interval - (now - _last_request_time), shared_wait or 0.0)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _retry_after_seconds(resp: "httpx.Response") -> float | None:
    """A bounded Retry-After (seconds) the provider sent with a 429, if any."""
    try:
        value = float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
    return value if 0 < value <= _MAX_RETRY_AFTER_SECONDS else None


class SemanticScholarRetriever(RetrievalSource):
    # Unchanged from the previous behaviour, which derived this from its
    # `deferred` flag. Recorded here as its own decision rather than a
    # side effect of how it batches DOI lookups.
    required_for_search_completion = False
    name = "semantic_scholar"
    # A paper index; it does not catalogue books.
    supported_source_kinds = SCHOLARLY_PAPER_KINDS
    capabilities = frozenset({"doi", "batch_doi", "metadata", "abstract", "oa_link",
                              "metadata_only_search", "title_author"})
    documentation_url = "https://api.semanticscholar.org/api-docs/graph"
    default_policy = _DEFAULT_POLICY
    deferred = True

    def __init__(self, health_store: ProviderHealthStore | None = None) -> None:
        self.policy = provider_policy(self.name, self.default_policy)
        self._health_store = health_store or ProviderHealthStore()
        self._rate_limited = False
        self._exhausted_calls = 0
        self._doi_cache: dict[str, RetrievalResult] = {}
        self._doi_failures: dict[str, RetrievalResult] = {}
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
            "batch_failures": 0,
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
        record_provider_request("semantic_scholar")
        try:
            resp = httpx.request(
                method, url, headers=self._headers(), params=params,
                json=json_body, timeout=self.policy.timeout_seconds,
            )
        except httpx.HTTPError:
            self.provider_metrics["network_errors"] += 1
            raise
        self._record_http_status(resp.status_code)
        for delay in self.policy.retry_delays_seconds:
            if resp.status_code != 429:
                break
            self.provider_metrics["rate_limit_retries"] += 1
            # Follow the provider's own instruction when it gives one, and hold
            # every other worker back for the same time.
            instructed = _retry_after_seconds(resp)
            if instructed is not None:
                delay = instructed
            shared_pacing.defer("semantic_scholar", delay)
            time.sleep(delay)
            _throttle(self.policy.min_interval_seconds)
            self.provider_metrics["calls"] += 1
            record_provider_request("semantic_scholar")
            try:
                resp = httpx.request(
                    method, url, headers=self._headers(), params=params,
                    json=json_body, timeout=self.policy.timeout_seconds,
                )
            except httpx.HTTPError:
                self.provider_metrics["network_errors"] += 1
                raise
            self._record_http_status(resp.status_code)
        if resp.status_code == 429:
            self._exhausted_calls += 1
            self.provider_metrics["exhausted_calls"] = self.provider_metrics.get("exhausted_calls", 0) + 1
            if self._exhausted_calls < _EXHAUSTED_CALLS_BEFORE_CIRCUIT:
                return resp   # this call failed as rate limited; the next may not
            self._rate_limited = True
            self.provider_metrics["rate_limited"] += 1
            cooldown = self._health_store.record_rate_limit(self.name, self.policy)
            self.provider_metrics["cooldown_seconds"] = cooldown
            logger.warning(
                "Semantic Scholar remained rate limited; disabling it for this "
                "run and starting a %ss cross-process cooldown", cooldown
            )
            return None
        self._exhausted_calls = 0
        if resp.is_success:
            self._health_store.record_success(self.name)
        return resp

    def _record_http_status(self, status: int) -> None:
        # Numeric counters only: no URLs, headers, response text or identifiers.
        key = f"http_status:{int(status)}"
        self.provider_metrics[key] = self.provider_metrics.get(key, 0) + 1

    def _record_prefetch_failure(self, chunk: list[str], reason: str) -> None:
        self.provider_metrics['batch_failures'] += 1
        key = f'batch_failure:{reason}'
        self.provider_metrics[key] = self.provider_metrics.get(key, 0) + 1
        for doi in chunk:
            if doi not in self._doi_cache:
                self._doi_failures[doi] = RetrievalResult(
                    source_name=self.name, success=False,
                    error=f'Semantic Scholar prefetch failed: {reason}',
                    metadata={'prefetch_diagnostic': {'outcome': reason}})

    def _headers(self) -> dict:
        headers = {"User-Agent": f"SourceFidelity/{settings.APP_VERSION}"}
        if settings.S2_API_KEY:
            headers["x-api-key"] = secret_value(settings.S2_API_KEY)
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
                try:
                    chunk_count, terminal = self._prefetch_doi_chunk(chunk)
                except httpx.HTTPStatusError as exc:
                    # One batch carries every DOI of the paper, so a batch that
                    # stayed refused gets one more retry round (measured random
                    # 429s, 2026-09-25) before its DOIs are given up.
                    if exc.response.status_code != 429 or self._rate_limited:
                        raise
                    self.provider_metrics["batch_retry_rounds"] = self.provider_metrics.get("batch_retry_rounds", 0) + 1
                    chunk_count, terminal = self._prefetch_doi_chunk(chunk)
                prefetched += chunk_count
                if terminal:
                    break
            except Exception as exc:
                reason = (f'http_{exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError)
                          else 'network_error' if isinstance(exc, httpx.HTTPError)
                          else 'response_invalid' if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError))
                          else 'unexpected_failure')
                self._record_prefetch_failure(chunk, reason)
                logger.warning(
                    "S2 DOI batch prefetch failed (type=%s)", type(exc).__name__
                )
                break
        return prefetched

    def _prefetch_doi_chunk(self, chunk: list[str]) -> tuple[int, bool]:
        """Prefetch one chunk; isolate a malformed identifier after batch 400.

        Semantic Scholar rejects the complete request when any supplied paper
        identifier is malformed. Recursively splitting the configured
        five-item chunk preserves valid DOI results without enabling broad
        individual fallback. A singleton 400 is cached as unavailable.
        """
        resp = self._request(
            "POST",
            f"{SEMANTIC_SCHOLAR_BASE}/paper/batch",
            {"fields": self._fields()},
            {"ids": [f"DOI:{doi}" for doi in chunk]},
        )
        if resp is None:
            self._record_prefetch_failure(chunk, 'rate_limited' if self._rate_limited else 'cooldown_skipped')
            return 0, True
        if resp.status_code == 400:
            if len(chunk) > 1:
                middle = len(chunk) // 2
                left_count, left_terminal = self._prefetch_doi_chunk(chunk[:middle])
                if left_terminal:
                    return left_count, True
                right_count, right_terminal = self._prefetch_doi_chunk(chunk[middle:])
                return left_count + right_count, right_terminal
            self._doi_cache[chunk[0]] = RetrievalResult(
                source_name=self.name,
                success=False,
                error="Semantic Scholar rejected this DOI identifier",
                metadata={'prefetch_diagnostic': {'outcome': 'identifier_rejected'}},
            )
            return 0, False
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) != len(chunk):
            raise ValueError("Unexpected Semantic Scholar batch response shape")
        self.provider_metrics["batch_calls"] += 1
        self.provider_metrics["batch_items"] += len(chunk)
        prefetched = 0
        for doi, paper in zip(chunk, payload):
            if paper:
                self._doi_cache[doi] = self._parse_paper(paper)
                prefetched += 1
            else:
                self._doi_cache[doi] = RetrievalResult(
                    source_name=self.name, success=False, error="Not found"
                )
        return prefetched, False

    def _circuit_error(self) -> str:
        """Report a declined call as a skip, never as an outcome of a request.

        Mirrors the CORE adapter: no request is made once the circuit or
        cooldown is open, so the trace must not describe the result in terms
        that the outcome classifier reads as a timeout or a failure to match.
        """
        from app.services.retrieval.core import PROVIDER_SKIPPED_ERROR

        state = "rate limited (429)" if self._rate_limited else "in cooldown"
        return f"{PROVIDER_SKIPPED_ERROR}: Semantic Scholar {state}"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        normalized_doi = self._normalize_doi(doi)
        if normalized_doi in self._doi_cache:
            self.provider_metrics["batch_cache_hits"] += 1
            return copy.deepcopy(self._doi_cache[normalized_doi])

        if normalized_doi in self._doi_failures:
            return copy.deepcopy(self._doi_failures[normalized_doi])

        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="DOI not prefetched; individual Semantic Scholar fallback disabled",
        )

    def paper_by_id(self, paper_id: str) -> RetrievalResult:
        """The registered record for one Semantic Scholar paper ID (40 hex characters).

        Used to read a submitted semanticscholar.org paper link whose page
        answers automated requests with a challenge (2026-10-02).
        """
        if not re.fullmatch(r"[0-9a-f]{40}", str(paper_id or "")):
            return RetrievalResult(source_name=self.name, success=False, error="invalid paper id")
        resp = self._request("GET", f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}", {"fields": self._fields()})
        if resp is None or resp.status_code != 200:
            return RetrievalResult(source_name=self.name, success=False,
                                   error=self._circuit_error() if resp is None else f"HTTP {resp.status_code}")
        try:
            return self._parse_paper(resp.json())
        except ValueError:
            return RetrievalResult(source_name=self.name, success=False, error="response_invalid")

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        """Search by title alone, deliberately ignoring the supplied author.

        Measured 2026-09-21 against this API: adding the author surname to the
        query collapses recall even for works the index certainly holds — Khan's
        "The separation of platforms and commerce" returns 8 results by title
        and 0 with "Khan" appended. The relevance ranker treats the extra token
        as evidence against the match rather than for it, so the author is used
        only by the caller's own field comparison afterwards, never in the query.

        A returned title is a lead, not an identity. The caller compares fields
        and dispositions unrelated topical hits itself.
        """
        # The documented search treats a hyphenated term as matching nothing
        # ("Hyphenated query terms yield no matches (replace it with space)"),
        # so "COVID-19" or "self-efficacy" would report an empty index.
        cleaned = " ".join(re.sub(r"(?<=\w)[-‐‑–](?=\w)", " ", str(title or "")).split())
        if not cleaned:
            return RetrievalResult(
                source_name=self.name, success=False, error="No title supplied",
            )
        try:
            resp = self._request(
                "GET",
                f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                {
                    "query": cleaned[:300],
                    "limit": 5,
                    "fields": "title,authors,year,externalIds,abstract,openAccessPdf",
                },
            )
        except httpx.HTTPError as exc:
            return RetrievalResult(
                source_name=self.name, success=False,
                error=f"semantic_scholar_title_search:{type(exc).__name__}",
            )
        if resp is None:
            return RetrievalResult(
                source_name=self.name, success=False, error=self._circuit_error(),
            )
        if resp.status_code != 200:
            return RetrievalResult(
                source_name=self.name, success=False,
                error=f"Semantic Scholar title search HTTP {resp.status_code}",
            )
        try:
            payload = resp.json()
        except ValueError:
            return RetrievalResult(
                source_name=self.name, success=False, error="response_invalid",
            )
        rows = payload.get("data") or []
        if not rows:
            return RetrievalResult(
                source_name=self.name, success=False, error="No results",
                metadata={"identity_search_result_count": 0},
            )
        result = self._parse_paper(rows[0])
        result.metadata = {
            **(result.metadata or {}),
            "identity_search_result_count": len(rows),
        }
        return result

    def _parse_paper(self, data: dict) -> RetrievalResult:
        external = data.get("externalIds", {}) or {}
        doi = external.get("DOI")

        # Large-collaboration records carry hundreds of names; a relevance search
        # can surface one for any query. Only the leading authors take part in
        # identity comparison, and the bounded field they feed rejects more.
        authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")][:OBSERVED_AUTHOR_LIMIT]
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
