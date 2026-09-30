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

from app.services.processing_metrics import record_provider_request
from app.config import secret_value, settings
from app.log_safety import safe_exception_code
from app.services.retrieval.base import SCHOLARLY_PAPER_KINDS, AcquisitionLocation, RepresentationKind, RetrievalSource, RetrievalResult
from app.services.retrieval import shared_pacing
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
# interval wait inside the lock keeps us under the per-window quota.
#
# Pacing follows the key's own quota (2026-09-25). CORE reports the requests
# allowed per window (one minute: the documented tiers are 10 or 25 a minute)
# in X-RateLimit-Limit, and puts a future X-RateLimit-Retry-After timestamp on
# EVERY response, about 9s ahead, even with the whole quota remaining. The
# documentation says that header matters only after the limit is hit. Obeying
# it on every call, on top of a fixed 10s interval, held CORE to about six
# calls a minute; it is now obeyed only when X-RateLimit-Remaining is 0 or on
# a 429, and the interval is derived from the reported limit.
_CORE_MAX_RETRY_WAIT_SECONDS = 15.0
_CORE_WINDOW_SECONDS = 60.0
_core_lock = threading.Lock()
_core_last_request = 0.0
_core_next_allowed_request = 0.0
_core_quota_interval: float | None = None


def _header_int(resp: "httpx.Response", name: str) -> int | None:
    try:
        return int(resp.headers.get(name))
    except (TypeError, ValueError):
        return None


def _quota_interval(limit: int | None) -> float | None:
    """Seconds between calls that stay inside the reported per-window quota."""
    if not limit or limit <= 0:
        return None
    return _CORE_WINDOW_SECONDS / limit * 1.05   # a small margin for clock skew


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


class CoreBusy(Exception):
    """Another worker process held CORE's one in-flight slot for too long."""


# How long to wait for another process's CORE request before declining.
_CORE_SHARED_WAIT_SECONDS = 60.0


def _core_request(
    url: str,
    params: dict,
    timeout: float = 15,
    headers: dict | None = None,
    min_interval: float = 10.0,
) -> "httpx.Response":
    """Make a serialized CORE request with shared, quota-derived pacing.

    The interval between calls comes from the key's reported quota once one
    response has been seen; `min_interval` is the fallback before that. A
    Retry-After instruction is obeyed only when the quota is exhausted or the
    call was refused; a short one is shared by every retriever in this process,
    and longer ones are left to the bounded 429 circuit rather than making an
    application worker sleep without limit.

    The thread lock serializes this process; `shared_pacing` serializes and
    paces every worker process on the same key, because CORE stalls a second
    in-flight request and the quota belongs to the key. Without Redis the
    process-local pacing alone applies, as before.
    """
    global _core_last_request, _core_next_allowed_request, _core_quota_interval
    with _core_lock, shared_pacing.exclusive(
            "core", hold_seconds=timeout + 10, wait_seconds=_CORE_SHARED_WAIT_SECONDS) as held:
        if held is False:
            raise CoreBusy("CORE busy in another worker")
        now = time.monotonic()
        interval = _core_quota_interval if _core_quota_interval is not None else min_interval
        shared_wait = shared_pacing.reserve_start("core", interval) or 0.0
        wait = max(
            0.0,
            interval - (now - _core_last_request),
            _core_next_allowed_request - now,
            shared_wait,
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
        reported = _quota_interval(_header_int(resp, "X-RateLimit-Limit"))
        if reported is not None:
            _core_quota_interval = reported
        exhausted = resp.status_code == 429 or _header_int(resp, "X-RateLimit-Remaining") == 0
        directed_wait = _retry_wait_seconds(resp) if exhausted else None
        if directed_wait is not None and directed_wait <= _CORE_MAX_RETRY_WAIT_SECONDS:
            _core_next_allowed_request = max(
                _core_next_allowed_request,
                time.monotonic() + directed_wait,
            )
            shared_pacing.defer("core", directed_wait)
        return resp


# Punctuation other than in-word apostrophes and hyphens is removed from the
# quoted phrase: a colon inside it (a subtitle separator) made CORE answer
# HTTP 500 (measured 2026-09-25), and the phrase match ignores punctuation.
_PHRASE_UNSAFE = re.compile(r"[^\w\s'’-]")
_MAX_TITLE_WORDS = 6


def _author_clause(author: str | None) -> str:
    """`authors:<surname>`, quoted when the surname has more than one word."""
    surname = (author or "").split(",")[0].strip()
    surname = " ".join(re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", surname))
    if not surname:
        return ""
    return f'authors:"{surname}"' if " " in surname else f"authors:{surname}"


def _title_queries(title: str, author: str | None) -> list[str]:
    """CORE title queries, most exact first (measured 2026-09-25).

    Without a field name CORE searches every field, full text included, and
    matches ANY of the words: the former default-field query matched 11-31
    million records for well-known titles, ran slowly enough to hit our read
    timeout, and put keyword coincidences ("... Reproducibility Project") in
    the five results kept, so the relevance filter then rejected them all. A
    quoted title under the title field is an exact phrase and returned only
    the right record; the earlier note that it "broadens catastrophically" no
    longer holds. A phrase misses a slightly misquoted title, so the fallback
    requires each significant title word in the title field instead, which
    still found the right record for a misquoted title.
    """
    author_clause = _author_clause(author)
    phrase = " ".join(_PHRASE_UNSAFE.sub(" ", title or "").split())
    words = [w for w in re.findall(r"\w+", title or "") if len(w) >= 4 and not w.isdigit()]
    queries = []
    if phrase:
        queries.append(" AND ".join(part for part in (f'title:"{phrase}"', author_clause) if part))
    if len(words) >= 2 or (words and author_clause):
        clauses = [f"title:{w}" for w in words[:_MAX_TITLE_WORDS]]
        queries.append(" AND ".join(clauses + ([author_clause] if author_clause else [])))
    return list(dict.fromkeys(queries))


# A call the app declined to make. Distinct from any outcome of a real request.
PROVIDER_SKIPPED_ERROR = "provider_call_skipped"


class CoreRetriever(RetrievalSource):
    # Broad open-access aggregator covering repositories the indexes miss. It
    # augments Crossref and OpenAlex, the main article sources; a failed CORE
    # search does not make a search incomplete (owner decision 2026-09-29).
    required_for_search_completion = False
    name = "core"
    # Open-access repositories: papers, reports and theses, not books.
    supported_source_kinds = SCHOLARLY_PAPER_KINDS
    capabilities = frozenset({"doi", "batch_doi", "title_author", "metadata", "oa_link", "metadata_only_search"})
    documentation_url = "https://api.core.ac.uk/docs/v3"
    default_policy = ProviderPolicy(
        # Measured 2026-09-21 by running Stardom at both values, with the former
        # unfielded query (see _title_queries): CORE answered a
        # title/author search in 6.7-10.4s (n=4, median 8.1s); an 8.0s budget
        # sits inside that distribution, so half the calls time out, three
        # timeouts open the circuit and the rest of the run is skipped.
        # The paired runs cost CORE 80.8s at 8.0s against 120.2s at 20.0s — a
        # difference of 39 seconds, which buys a required corroboration route
        # that actually answers. The wall-time regression first blamed on this
        # setting was mostly web-search variance, not CORE.
        # The labelled-corpus run of 2026-09-25 measured the fielded queries at
        # 1.6s median and 4.7s p95 per call, with no timeouts; 20s stays as a
        # hang bound only. The same run showed CORE's quota is a 150-request
        # bucket (X-RateLimit-Remaining fell to 29 with no 429) whose
        # retry-after moves about 4-5s ahead per request, which `_core_request`
        # obeys once the bucket is empty. A 10s fallback pace before the first
        # quota header made four concurrent processes wait up to 64s for
        # nothing; 1s is the fallback now.
        timeout_seconds=20.0,
        min_interval_seconds=1.0,
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
            "busy_skips": 0,
        }
        self._last_skip: str | None = None

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
        try:
            resp = self._send(params)
        except CoreBusy:
            # Not a CORE failure and not a timeout: our own workers were busy.
            self.provider_metrics["busy_skips"] += 1
            self._last_skip = "busy"
            return None
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
        sent = True
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
        except CoreBusy:
            sent = False   # declined before any request left this worker
            raise
        finally:
            if sent:
                self.provider_metrics["calls"] += 1
                record_provider_request("core")
        self._consecutive_timeouts = 0
        if resp.status_code != 429:
            self._health_store.record_success(self.name)
        return resp

    def _circuit_error(self) -> str:
        """Report a skipped call as a skip, never as a timeout.

        No request is made once the circuit or cooldown is open, so the trace
        must not record a timeout the app never waited for. The marker below is
        classified as `cooldown_skipped`; the word "timeout" is deliberately
        absent, because the outcome classifier matches error text.
        """
        if self._last_skip == "busy" and not (self._timeout_circuit_open or self._rate_limited):
            self._last_skip = None
            return f"{PROVIDER_SKIPPED_ERROR}: CORE busy in another worker"
        if self._timeout_circuit_open or self.provider_metrics["cooldown_skips"]:
            return f"{PROVIDER_SKIPPED_ERROR}: CORE circuit open after repeated slow responses"
        return f"{PROVIDER_SKIPPED_ERROR}: CORE rate limited (429)"

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
            "Authorization": f"Bearer {secret_value(settings.CORE_API_KEY)}",
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
            from app.services.relevance import score_relevance
            from app.services.reference_review_scope import screen_metadata
            # Exact title phrase first; the word-by-word title query only when
            # the phrase found nothing relevant (see _title_queries).
            outputs: list[dict] = []
            seen: set = set()
            for q in _title_queries(title, author):
                resp = self._request({"q": q, "limit": 5})
                if resp is None:
                    if outputs:
                        break   # judge what was already returned
                    return RetrievalResult(
                        source_name=self.name,
                        success=False,
                        error=self._circuit_error(),
                    )
                resp.raise_for_status()
                results = resp.json().get("results")
                if not isinstance(results, list) or any(not isinstance(output, dict) for output in results):
                    raise ValueError("Invalid CORE result list")
                fresh = [output for output in results
                         if (output.get("id") or output.get("doi") or output.get("title")) not in seen]
                seen.update(output.get("id") or output.get("doi") or output.get("title") for output in fresh)
                outputs.extend(fresh)
                if any(score_relevance(title, parsed.title or "", author, parsed.authors or []).is_relevant
                       for parsed in map(self._parse_output, fresh)):
                    break
            if not outputs:
                return RetrievalResult(source_name=self.name, success=False, error="No results",
                    metadata={"identity_search_reason_code": "metadata_empty_response_unqualified"})

            parsed_results = [self._parse_output(output) for output in outputs]
            review_screen = screen_metadata(title, author, parsed_results)
            for result in parsed_results:
                result.metadata = {**(result.metadata or {}), 'bounded_review_screen': review_screen}
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
                error=f"No relevant match (top {len(outputs)} results were keyword coincidences)",
                metadata={"identity_search_result_count": len(outputs),
                          "bounded_review_screen": review_screen,
                          "identity_search_reason_code": "metadata_candidates_filtered"},
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
