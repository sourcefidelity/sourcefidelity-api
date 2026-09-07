"""CORE retrieval adapter.

CORE (https://core.ac.uk) aggregates full-text content from 10,000+
repositories. Requires a free API key (CORE_API_KEY).
"""

import copy
from datetime import datetime, timezone
import logging
import re
import threading
import time

import httpx

from app.config import settings
from app.log_safety import safe_exception_code
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalSource, RetrievalResult
from app.services.retrieval.provider_runtime import (
    ProviderHealthStore,
    ProviderPolicy,
    provider_policy,
)

logger = logging.getLogger(__name__)

CORE_BASE = "https://api.core.ac.uk/v3"
# Trailing slash matters: without it, /v3/search/outputs 301-redirects to the
# slashed form, costing an extra round-trip on every call.
SEARCH_URL = f"{CORE_BASE}/search/outputs/"

# CORE v3 uses token-based limits whose actual values are returned in
# X-RateLimit-* response headers. Request cost and key tier vary, so the
# adapter records those headers and honors a bounded retry on 429 rather than
# relying on the retired v2 quota table. Independently, live testing found a
# per-key concurrency constraint: when 2+ requests are in-flight at once, the
#      server stalls one until ~15s and our read timeout fires. Measured:
#      the same 4 queries sequential = all succeed 1.8-2.6s; 4-concurrent =
#      1 ReadTimeout @ 21s.
# So we need BOTH rate-limiting AND full serialization. Holding the lock for
# the whole request (not just the start) makes CORE calls strictly serial; the
# min-interval wait inside the lock keeps us under the per-window quota.
_CORE_MAX_RETRY_WAIT_SECONDS = 15.0
_core_lock = threading.Lock()
_core_last_request = 0.0
_core_next_allowed_request = 0.0


def _retry_wait_seconds(
    resp: "httpx.Response",
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse CORE's relative or absolute next-request instruction."""
    value = resp.headers.get("X-RateLimit-Retry-After") or resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if retry_at.tzinfo is None:
            return None
        current = now or datetime.now(timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())
    except ValueError:
        return None


def _core_request(
    url: str,
    params: dict,
    timeout: float = 15,
    headers: dict | None = None,
    min_interval: float = 10.0,
) -> "httpx.Response":
    """Make a serialized CORE request with shared response-directed pacing.

    CORE sends a future X-RateLimit-Retry-After timestamp on successful
    responses as well as on 429s. A short instruction is shared by every
    retriever in this process; the configured interval remains the fallback.
    Longer instructions are left to the existing bounded 429 circuit rather
    than making an application worker sleep without limit.
    """
    global _core_last_request, _core_next_allowed_request
    with _core_lock:
        now = time.monotonic()
        wait = max(
            0.0,
            min_interval - (now - _core_last_request),
            _core_next_allowed_request - now,
        )
        if wait > 0:
            time.sleep(wait)
        _core_last_request = time.monotonic()
        resp = httpx.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        directed_wait = _retry_wait_seconds(resp)
        if directed_wait is not None and directed_wait <= _CORE_MAX_RETRY_WAIT_SECONDS:
            _core_next_allowed_request = max(
                _core_next_allowed_request,
                time.monotonic() + directed_wait,
            )
        return resp


def _build_title_query(title: str, author: str | None) -> str:
    """Build a CORE title search query using the syntax that actually works.

    Empirically validated (Aug 12): CORE's `q` default-field AND semantics work
    well, but `title:"..."` (quoted phrase under the title: field) broadens
    catastrophically — `title:"New Media Giants"` returns ~7.5M hits and times
    out under load, while the default-field form `New Media Giants Croteau`
    returns 2 hits. Quotes appear to disable the field filter rather than
    enforce a phrase. So: use the default field, AND the significant title
    tokens, and append the author surname when available for triangulation.
    """
    toks = [w for w in re.split(r"[^A-Za-z0-9]+", title) if len(w) >= 4]
    parts = toks[:6]  # cap to avoid over-constraining short titles
    if author:
        surname = author.split(",")[0].strip()
        if surname:
            parts.append(surname)
    return " ".join(parts) if parts else title



class CoreRetriever(RetrievalSource):
    name = "core"
    capabilities = frozenset({"doi", "batch_doi", "title_author", "metadata", "oa_link"})
    documentation_url = "https://api.core.ac.uk/docs/v3"
    default_policy = ProviderPolicy(
        timeout_seconds=8.0,
        min_interval_seconds=10.0,
        batch_size=5,
        cooldown_seconds=300,
        max_consecutive_failures=2,
        max_timeouts_per_run=3,
    )

    def __init__(self, health_store: ProviderHealthStore | None = None) -> None:
        self.policy = provider_policy(self.name, self.default_policy)
        self._health_store = health_store or ProviderHealthStore()
        self._rate_limited = False
        self._timeout_circuit_open = False
        self._consecutive_timeouts = 0
        self._doi_cache: dict[str, RetrievalResult] = {}
        self.provider_metrics = {
            "calls": 0,
            "grouped_doi_calls": 0,
            "grouped_doi_items": 0,
            "grouped_doi_cache_hits": 0,
            "rate_limit_retries": 0,
            "rate_limited": 0,
            "circuit_skips": 0,
            "timeouts": 0,
            "network_errors": 0,
            "timeout_circuit_opens": 0,
            "cooldown_skips": 0,
            "cooldown_seconds": 0,
            "rate_limit_limit": None,
            "rate_limit_remaining": None,
        }

    def _record_rate_limit_headers(self, resp: httpx.Response) -> None:
        for header, metric in (
            ("X-RateLimit-Limit", "rate_limit_limit"),
            ("X-RateLimit-Remaining", "rate_limit_remaining"),
        ):
            value = resp.headers.get(header)
            if value is not None:
                try:
                    self.provider_metrics[metric] = int(value)
                except ValueError:
                    logger.debug("CORE returned non-integer %s: %s", header, value)

    @staticmethod
    def _retry_wait_seconds(resp: httpx.Response) -> float | None:
        return _retry_wait_seconds(resp)

    def _request(self, params: dict) -> "httpx.Response | None":
        """Call CORE once with bounded timeout and rate-limit circuits."""
        if self._rate_limited or self._timeout_circuit_open:
            self.provider_metrics["circuit_skips"] += 1
            return None
        if self._health_store.cooldown_remaining(self.name) > 0:
            self.provider_metrics["cooldown_skips"] += 1
            return None
        resp = self._send(params)
        self._record_rate_limit_headers(resp)
        if resp.status_code == 429:
            wait = self._retry_wait_seconds(resp)
            if wait is not None and wait <= _CORE_MAX_RETRY_WAIT_SECONDS:
                self.provider_metrics["rate_limit_retries"] += 1
                # _core_request already published this response-directed wait
                # to the shared pacing gate; retry through that gate rather
                # than sleeping independently and racing another caller.
                resp = self._send(params)
                self._record_rate_limit_headers(resp)
            if resp.status_code != 429:
                return resp
            self._rate_limited = True
            self.provider_metrics["rate_limited"] += 1
            logger.warning("CORE rate limited; disabling CORE for the remainder of this run")
            return None
        return resp

    def _send(self, params: dict) -> httpx.Response:
        """Perform one serialized request and update timeout health state."""
        self.provider_metrics["calls"] += 1
        try:
            resp = _core_request(
                SEARCH_URL,
                params,
                headers=self._headers(),
                timeout=self.policy.timeout_seconds,
                min_interval=self.policy.min_interval_seconds,
            )
        except httpx.TimeoutException:
            self.provider_metrics["timeouts"] += 1
            self._consecutive_timeouts += 1
            cumulative_limit_reached = bool(
                self.policy.max_timeouts_per_run
                and self.provider_metrics["timeouts"]
                >= self.policy.max_timeouts_per_run
            )
            if (
                self._consecutive_timeouts >= self.policy.max_consecutive_failures
                or cumulative_limit_reached
            ):
                self._timeout_circuit_open = True
                self.provider_metrics["timeout_circuit_opens"] += 1
                cooldown = self._health_store.record_timeout(self.name, self.policy)
                self.provider_metrics["cooldown_seconds"] = cooldown
                logger.warning(
                    "CORE timeout budget reached (%s consecutive, %s total); "
                    "disabling it for this run and starting a %ss "
                    "cross-process cooldown",
                    self._consecutive_timeouts,
                    self.provider_metrics["timeouts"],
                    cooldown,
                )
            raise
        except httpx.HTTPError:
            self.provider_metrics["network_errors"] += 1
            raise
        self._consecutive_timeouts = 0
        if resp.status_code != 429:
            self._health_store.record_success(self.name)
        return resp

    def _circuit_error(self) -> str:
        if self._timeout_circuit_open or self.provider_metrics["cooldown_skips"]:
            return "CORE timeout circuit/cooldown is open"
        return "Rate limited (429); circuit open"

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        normalized = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized.lower()

    def prefetch_dois(self, dois: list[str]) -> int:
        """Prefetch DOI outputs with small v3 Boolean queries and cache misses."""
        if not settings.CORE_API_KEY:
            return 0
        normalized = list(dict.fromkeys(
            self._normalize_doi(doi) for doi in dois if doi.strip()
        ))
        missing = [doi for doi in normalized if doi not in self._doi_cache]
        prefetched = 0
        for start in range(0, len(missing), self.policy.batch_size):
            chunk = missing[start:start + self.policy.batch_size]
            try:
                query = " OR ".join(f"doi:{doi}" for doi in chunk)
                resp = self._request({"q": query, "limit": len(chunk)})
                if resp is None:
                    break
                resp.raise_for_status()
                results = resp.json().get("results", [])
                found: dict[str, RetrievalResult] = {}
                for output in results:
                    result = self._parse_output(output)
                    if result.doi:
                        found[self._normalize_doi(result.doi)] = result
                for doi in chunk:
                    if doi in found:
                        self._doi_cache[doi] = found[doi]
                        prefetched += 1
                    else:
                        self._doi_cache[doi] = RetrievalResult(
                            source_name=self.name, success=False, error="No results"
                        )
                self.provider_metrics["grouped_doi_calls"] += 1
                self.provider_metrics["grouped_doi_items"] += len(chunk)
            except Exception as exc:
                logger.warning(
                    "CORE grouped DOI prefetch failed (type=%s)",
                    type(exc).__name__,
                )
                break
        return prefetched

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.CORE_API_KEY}",
            "User-Agent": f"SourceFidelity/{settings.APP_VERSION}",
        }

    def search_by_doi(self, doi: str) -> RetrievalResult:
        if not settings.CORE_API_KEY:
            return RetrievalResult(
                source_name=self.name, success=False, error="No CORE_API_KEY configured"
            )
        normalized_doi = self._normalize_doi(doi)
        if normalized_doi in self._doi_cache:
            self.provider_metrics["grouped_doi_cache_hits"] += 1
            return copy.deepcopy(self._doi_cache[normalized_doi])
        try:
            # v3 moved DOI lookup into the `q` DSL: `doi` as a bare query param
            # is silently IGNORED, which degrades to an unfiltered ~482M-result
            # query — slow under load (read timeouts) AND returns the wrong
            # paper. `q=doi:<doi>` is the correct v3 form (returns the right
            # paper, ~1-2s, single-digit hits).
            params = {"q": f"doi:{doi}", "limit": 1}
            resp = self._request(params)
            if resp is None:
                return RetrievalResult(
                    source_name=self.name,
                    success=False,
                    error=self._circuit_error(),
                )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return RetrievalResult(source_name=self.name, success=False, error="No results")
            return self._parse_output(results[0])
        except Exception as e:
            error = safe_exception_code(e)
            logger.warning("CORE DOI search failed (type=%s)", type(e).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        if not settings.CORE_API_KEY:
            return RetrievalResult(
                source_name=self.name, success=False, error="No CORE_API_KEY configured"
            )
        try:
            # Use the query form that actually works on v3 (see _build_title_query):
            # default-field AND of significant title words + author surname.
            # The old `title:"{title}"` form broadened to millions of hits and
            # timed out under load.
            q = _build_title_query(title, author)
            params = {"q": q, "limit": 5}
            resp = self._request(params)
            if resp is None:
                return RetrievalResult(
                    source_name=self.name,
                    success=False,
                    error=self._circuit_error(),
                )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return RetrievalResult(source_name=self.name, success=False, error="No results")

            from app.services.relevance import score_relevance

            for output in results:
                result = self._parse_output(output)
                matched_title = result.title or ""
                matched_authors = result.authors or []
                rel = score_relevance(title, matched_title, author, matched_authors)
                if rel.is_relevant:
                    return result
                logger.debug(
                    "CORE match rejected: %s", rel.detail[:100],
                )

            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {len(results)} results were keyword coincidences)",
            )
        except Exception as e:
            error = safe_exception_code(e)
            logger.warning("CORE title search failed (type=%s)", type(e).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def _parse_output(self, data: dict) -> RetrievalResult:
        doi = data.get("doi")
        title = data.get("title", "")
        year = str(data.get("yearPublished")) if data.get("yearPublished") else "n.d."

        raw_authors = data.get("authors", []) or []
        authors = []
        for a in raw_authors:
            if isinstance(a, dict):
                name = a.get("name", "")
            else:
                name = str(a)
            if name:
                authors.append(name)

        # CORE may provide a download URL under either key
        source_urls = data.get("sourceFulltextUrls") or []
        candidate_urls = [data.get("downloadUrl"), data.get("fullTextIdentifier"), *source_urls]
        locations = []
        seen = set()
        for candidate_url in candidate_urls:
            if not candidate_url or candidate_url in seen:
                continue
            seen.add(candidate_url)
            is_pdf = ".pdf" in candidate_url.lower() or candidate_url == data.get("downloadUrl")
            locations.append(
                AcquisitionLocation(
                    url=candidate_url,
                    provider=self.name,
                    media_type="application/pdf" if is_pdf else None,
                    representation_kind=RepresentationKind.PDF if is_pdf else None,
                    host_type="repository",
                    access_type="open_access",
                    is_best=not locations,
                )
            )
        download_url = locations[0].url if locations else None

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=download_url,
            locations=locations,
            metadata=data,
        )
