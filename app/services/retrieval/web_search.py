"""Final-stage, format-neutral web candidate discovery adapter.

After the academic-DB chain (OpenAlex, Crossref, etc.) fails to find a source,
this adapter searches the public web for article pages and supported full-text
representations. Acquisition and identity validation remain in the shared
resolver so one search result cannot bypass the ordinary evidence gates.

Source-access neutrality (§3.5): the app verifies against whatever it finds.
Does NOT access Sci-Hub or pirated copies — only results from the search
provider, which indexes public web pages. If a search result points to an
author homepage or institutional repository, that's legitimate OA access.

Search results are merged across a bounded query ladder, deterministically
ranked and returned as several locations. An LLM is not allowed to select one
link irrevocably before acquisition.
"""

import logging
import re
from collections import Counter
from typing import Callable, Optional
from urllib.parse import unquote

import httpx

from app.log_safety import private_value_id

from app.config import settings
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
)
from app.services.retrieval.provider_runtime import (
    ProviderHealthStore,
    ProviderPolicy,
    provider_policy,
)
from app.services.search import get_search_provider
from app.services.safe_fetch import safe_fetch_bytes
from app.services.search.base import SearchProvider

logger = logging.getLogger(__name__)

# Max locations to expose per search-provider tier. Shared acquisition still
# validates identity, type and completeness; retaining five prevents two weak
# landing pages from silently displacing a strong lower-ranked PDF.
_MAX_DOWNLOAD_ATTEMPTS = 5
_DOWNLOAD_TIMEOUT_SECONDS = 10

# Max PDF size to accept (50MB — same as student-URL limit)
_MAX_PDF_SIZE = 50 * 1024 * 1024
_MAX_SEARCH_TITLE_CHARS = 240
_MAX_SEARCH_AUTHOR_CHARS = 60

_SEARXNG_DEFAULT_POLICY = ProviderPolicy(
    timeout_seconds=15.0,
    cooldown_seconds=180,
    max_cooldown_seconds=3600,
    max_consecutive_failures=3,
)


def _provider_search_outcome(provider: object, results: list) -> str:
    if results:
        return "results"
    status = getattr(provider, "last_status", None)
    if status in {
        "operational_failure",
        "access_restricted",
        "rate_limited",
        "captcha",
        "timeout",
        "response_invalid",
    }:
        return status
    return "no_results"


class WebSearchRetriever(RetrievalSource):
    """Find ranked public-web locations after structured retrieval fails.

    Uses exact identifiers and bibliographic queries, then adds a PDF-specific
    query only when the earlier results do not yield enough distinct locations.

    The configured primary provider runs first. Optional bounded escalation
    providers (normally Tavily then Exa) run only when it produces no results.
    Exact queries are cached for this retriever so repeated canonical works do not
    consume another external search. Unofficial HTML scrapers are never added
    as an implicit fallback.
    """

    name = "web_search"
    capabilities = frozenset({"doi", "title_author", "locations", "web_discovery"})

    def __init__(
        self,
        search_provider: Optional[SearchProvider] = None,
        *,
        searx_engine_groups: tuple[str, ...] | None = None,
        searx_timeout_retries: int | None = None,
        searx_timeout_circuit_threshold: int | None = None,
        health_store: ProviderHealthStore | None = None,
        on_provider_recovered: Callable[[str], None] | None = None,
    ):
        self._search = search_provider or get_search_provider()
        self._escalation: list[SearchProvider] = []
        for name in settings.SEARCH_ESCALATION_PROVIDERS.split(","):
            name = name.strip().lower()
            if not name or (self._search and name == self._search.name.lower()):
                continue
            provider = get_search_provider(name)
            if provider:
                self._escalation.append(provider)
        self._query_cache: dict[str, list] = {}
        self._query_attempts: dict[str, list[dict]] = {}
        self._suspended_searx_groups: set[str] = set()
        self._suspended_search_providers: set[str] = set()
        self._searx_timeout_failures: Counter[str] = Counter()
        self._provider_timeout_failures: Counter[str] = Counter()
        self._health_store = health_store or ProviderHealthStore()
        self._searx_policy = provider_policy("searxng", _SEARXNG_DEFAULT_POLICY)
        self.recovered_provider_keys: set[str] = set()
        self._recovery_notifications: set[str] = set()
        self._on_provider_recovered = (
            on_provider_recovered or _schedule_provider_recovery
        )
        self._searx_engine_groups = searx_engine_groups
        self._searx_timeout_retries = (
            settings.SEARXNG_TIMEOUT_RETRIES
            if searx_timeout_retries is None
            else max(0, searx_timeout_retries)
        )
        self._searx_timeout_circuit_threshold = (
            settings.SEARXNG_TIMEOUT_CIRCUIT_THRESHOLD
            if searx_timeout_circuit_threshold is None
            else max(1, searx_timeout_circuit_threshold)
        )
        self.search_metrics: Counter[str] = Counter()
        self._escalation_limits = self._parse_escalation_limits(
            settings.SEARCH_ESCALATION_MAX_CALLS
        )
        if not self._search:
            logger.info("WebSearchRetriever: no primary search provider configured")

    @staticmethod
    def _parse_escalation_limits(value: str) -> dict[str, int]:
        """Parse ``provider:count`` limits, ignoring malformed entries safely."""
        limits: dict[str, int] = {}
        for item in value.split(","):
            name, separator, raw_limit = item.strip().partition(":")
            if not separator or not name:
                continue
            try:
                limits[name.lower()] = max(0, int(raw_limit))
            except ValueError:
                logger.warning("Ignoring invalid search escalation limit: %s", item)
        return limits

    def search_by_doi(self, doi: str) -> RetrievalResult:
        """Generate several locations from exact-DOI queries."""
        return self._search_for_locations(
            queries=[f'"{doi}"', f'"{doi}" full text', f'"{doi}" filetype:pdf'],
            doi=doi,
            title=None,
        )

    def search_by_title_author(
        self,
        title: str,
        author: str | None = None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Generate locations from exact and normalized bibliographic queries."""
        exact = _exact_bibliographic_query(title, author, year)
        normalized = " ".join(_significant_tokens(title))
        queries = [exact, f"{exact} filetype:pdf"]
        if normalized and normalized.lower() != title.lower():
            queries.append(
                " ".join(
                    part
                    for part in (
                        normalized[:_MAX_SEARCH_TITLE_CHARS],
                        _search_author_hint(author),
                        year or "",
                    )
                    if part
                )
            )
        return self._search_for_locations(
            queries=queries,
            doi=None,
            title=title,
            author=author,
            year=year,
        )

    def search_after_failed_candidates(
        self,
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        tried_providers: set[str],
    ) -> RetrievalResult:
        """Ask the next configured search provider after acquisition rejects a tier.

        Ordinary provider escalation is triggered by no search results. This
        separate bounded path is triggered only after the resolver has tried
        and rejected every candidate returned by an earlier provider. It keeps
        search-result relevance distinct from representation identity and
        completeness validation.
        """
        queries: list[str] = []
        if doi:
            queries.extend((f'"{doi}" filetype:pdf', f'"{doi}" full text'))
        if title:
            exact = _exact_bibliographic_query(title, author, year)
            queries.extend((f"{exact} filetype:pdf", exact))

        all_attempts: list[dict] = []
        for provider in self._escalation:
            provider_name = provider.name.lower()
            if provider_name in tried_providers:
                continue
            candidates: dict[str, tuple[object, str, str]] = {}
            queries_run: list[str] = []
            provider_attempts: list[dict] = []
            for query in queries:
                if not query or query in queries_run:
                    continue
                call_metric = f"provider_calls:{provider_name}"
                limit = self._escalation_limits.get(provider_name, 0)
                if self.search_metrics[call_metric] >= limit:
                    self.search_metrics[f"budget_skips:{provider_name}"] += 1
                    provider_attempts.append(
                        {
                            "provider": provider_name,
                            "query": query,
                            "outcome": "budget_skipped",
                            "result_count": 0,
                        }
                    )
                    break
                queries_run.append(query)
                self.search_metrics[call_metric] += 1
                results = provider.search(query, num_results=10)
                provider_attempts.append(
                    {
                        "provider": provider_name,
                        "query": query,
                        "outcome": _provider_search_outcome(provider, results),
                        "result_count": len(results),
                    }
                )
                for candidate in results:
                    candidates.setdefault(
                        candidate.url, (candidate, query, provider_name)
                    )
                if len(candidates) >= _MAX_DOWNLOAD_ATTEMPTS and len(queries_run) >= 2:
                    break
            all_attempts.extend(provider_attempts)
            if candidates:
                logger.info(
                    "Search escalated to %s after prior candidates failed validation",
                    provider.name,
                )
                return self._locations_result(
                    candidates,
                    queries_run=queries_run,
                    doi=doi,
                    title=title,
                    author=author,
                    year=year,
                    search_attempts=all_attempts,
                    metadata={
                        "escalation_reason": "prior_candidates_failed_validation",
                    },
                )
            tried_providers.add(provider_name)

        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="no untried search provider returned candidates",
            metadata={
                "search_attempts": all_attempts,
                "escalation_reason": "prior_candidates_failed_validation",
            },
        )

    def _search_for_locations(
        self,
        queries: list[str],
        doi: Optional[str],
        title: Optional[str],
        author: Optional[str] = None,
        year: Optional[str] = None,
    ) -> RetrievalResult:
        """Merge, rank and expose a bounded candidate set for shared acquisition."""
        candidates: dict[str, tuple[object, str, str]] = {}
        queries_run: list[str] = []
        for query in queries:
            if not query or query in queries_run:
                continue
            queries_run.append(query)
            for candidate in self._run_search(query):
                candidates.setdefault(
                    candidate.url,
                    (candidate, query, self._result_provider_for_query(query)),
                )
            if len(candidates) >= _MAX_DOWNLOAD_ATTEMPTS and len(queries_run) >= 2:
                break

        search_attempts = [
            attempt
            for query in queries_run
            for attempt in getattr(self, "_query_attempts", {}).get(query, [])
        ]
        if not candidates:
            return RetrievalResult(
                source_name=self.name,
                success=False,
                error="no search results",
                metadata={
                    "queries_run": queries_run,
                    "candidate_count": 0,
                    "search_attempts": search_attempts,
                },
            )

        return self._locations_result(
            candidates,
            queries_run=queries_run,
            doi=doi,
            title=title,
            author=author,
            year=year,
            search_attempts=search_attempts,
        )

    def _locations_result(
        self,
        candidates: dict[str, tuple[object, str, str]],
        *,
        queries_run: list[str],
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        search_attempts: list[dict],
        metadata: dict | None = None,
    ) -> RetrievalResult:
        """Rank a provider tier and expose bounded typed acquisition locations."""
        ranked = sorted(
            candidates.values(),
            key=lambda item: _candidate_score(item[0], doi, title, author, year),
            reverse=True,
        )[:_MAX_DOWNLOAD_ATTEMPTS]
        locations: list[AcquisitionLocation] = []
        for index, (candidate, query, search_provider) in enumerate(ranked):
            is_pdf = candidate.is_pdf or _looks_like_pdf(candidate.url)
            locations.append(
                AcquisitionLocation(
                    url=candidate.url,
                    provider=self.name,
                    media_type="application/pdf" if is_pdf else "text/html",
                    representation_kind=(
                        RepresentationKind.PDF if is_pdf else RepresentationKind.HTML
                    ),
                    is_best=index == 0,
                    metadata={
                        "search_query": query,
                        "search_provider": search_provider,
                        "search_engine_group": self._result_engine_group_for_query(
                            query
                        ),
                        "search_title": candidate.title,
                        "search_snippet": candidate.snippet,
                        "deterministic_score": _candidate_score(
                            candidate, doi, title, author, year
                        ),
                        "may_contain_full_text": not is_pdf,
                    },
                )
            )
        logger.info(
            "Web discovery produced %d ranked locations from %d quer%s",
            len(locations),
            len(queries_run),
            "y" if len(queries_run) == 1 else "ies",
        )
        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            full_text_url=locations[0].url,
            locations=locations,
            metadata={
                **(metadata or {}),
                "queries_run": queries_run,
                "candidate_count": len(candidates),
                "search_attempts": search_attempts,
            },
        )

    def _result_provider_for_query(self, query: str) -> str:
        """Return the provider that produced the retained results for a query."""
        attempts = getattr(self, "_query_attempts", {}).get(query, [])
        for attempt in reversed(attempts):
            if attempt.get("outcome") == "results":
                return str(attempt.get("provider") or "unknown").lower()
        return "unknown"

    def _result_engine_group_for_query(self, query: str) -> str | None:
        """Return the exact SearXNG engine group that supplied retained results."""
        attempts = getattr(self, "_query_attempts", {}).get(query, [])
        for attempt in reversed(attempts):
            if attempt.get("outcome") != "results":
                continue
            if str(attempt.get("provider") or "").lower() != "searxng":
                return None
            value = str(attempt.get("engines") or "").strip()
            return value or None
        return None

    def _run_search(self, query: str) -> list:
        """Run a cached primary search, then bounded configured escalation.

        OpenAlex already incorporates Unpaywall OA locations, including arXiv
        preprints linked to works. Crossref and CORE also run as direct
        retrieval adapters. SearXNG therefore does not repeat those academic
        APIs; it is an opportunistic public-web fallback only.
        """
        if query in self._query_cache:
            self.search_metrics["query_cache_hits"] += 1
            return self._query_cache[query]

        from app.services.search.searxng import SearXNGSearch

        query_attempts: list[dict] = []
        if not hasattr(self, "_query_attempts"):
            self._query_attempts = {}

        if isinstance(self._search, SearXNGSearch):
            # Upstream engines can fail independently of the self-hosted
            # SearXNG service. A failed group is suspended for the rest of the
            # run to avoid repeated CAPTCHA, access-denial and timeout traffic.
            # The ordered groups are deployment-configurable because upstream
            # reachability differs by region.
            results: list = []
            configured_groups = getattr(self, "_searx_engine_groups", None)
            engine_groups = list(configured_groups or ()) or [
                item.strip()
                for item in settings.SEARXNG_ENGINE_GROUPS.split(";")
                if item.strip()
            ]
            for engines in engine_groups:
                if engines in self._suspended_searx_groups:
                    query_attempts.append(
                        {
                            "provider": "searxng",
                            "engines": engines,
                            "query": query,
                            "outcome": "cooldown_skipped",
                            "result_count": 0,
                        }
                    )
                    continue
                provider_key = _searx_provider_key(engines)
                cooldown_remaining = self._health_store.cooldown_remaining(provider_key)
                if cooldown_remaining > 0:
                    self.search_metrics[f"cooldown_skips:{provider_key}"] += 1
                    self._suspended_searx_groups.add(engines)
                    query_attempts.append(
                        {
                            "provider": "searxng",
                            "engines": engines,
                            "query": query,
                            "outcome": "cooldown_skipped",
                            "result_count": 0,
                            "cooldown_remaining_seconds": round(cooldown_remaining),
                        }
                    )
                    continue
                if (
                    self._health_store.incident(provider_key) is not None
                    and not self._health_store.claim_recovery_probe(provider_key)
                ):
                    self.search_metrics[f"probe_lease_skips:{provider_key}"] += 1
                    self._suspended_searx_groups.add(engines)
                    query_attempts.append(
                        {
                            "provider": "searxng",
                            "engines": engines,
                            "query": query,
                            "outcome": "recovery_probe_in_progress",
                            "result_count": 0,
                        }
                    )
                    continue
                timeout_retries = getattr(self, "_searx_timeout_retries", 0)
                for attempt_number in range(1, timeout_retries + 2):
                    self.search_metrics[f"provider_calls:searxng:{engines}"] += 1
                    results = self._search.search(
                        query,
                        num_results=10,
                        engines=engines,
                    )
                    outcome = _provider_search_outcome(self._search, results)
                    failures = list(
                        getattr(self._search, "last_failure_reasons", []) or []
                    )
                    query_attempts.append(
                        {
                            "provider": "searxng",
                            "engines": engines,
                            "query": query,
                            "page_number": 1,
                            "attempt_number": attempt_number,
                            "outcome": outcome,
                            "result_count": len(results),
                            "failure_reasons": failures,
                            "unresponsive_engines": [
                                item[0]
                                for item in self._search.last_unresponsive_engines
                            ],
                        }
                    )
                    if results:
                        self._searx_timeout_failures[engines] = 0
                        if self._health_store.record_success(provider_key):
                            self.recovered_provider_keys.add(provider_key)
                            self.search_metrics[f"recoveries:{provider_key}"] += 1
                            self._notify_recovery("searxng")
                        self._query_attempts[query] = query_attempts
                        self._query_cache[query] = results
                        return results

                    failure_categories = {
                        str(item.get("category") or "") for item in failures
                    }
                    hard_failure = bool(
                        failure_categories
                        & {"captcha", "access_restricted", "rate_limited"}
                    )
                    if hard_failure:
                        self._suspended_searx_groups.add(engines)
                        hard_status = next(
                            category
                            for category in (
                                "captcha",
                                "rate_limited",
                                "access_restricted",
                            )
                            if category in failure_categories
                        )
                        cooldown = self._health_store.record_unavailable(
                            provider_key,
                            self._searx_policy,
                            status=hard_status,
                        )
                        self.search_metrics[f"cooldown_seconds:{provider_key}"] = cooldown
                        break
                    if outcome != "timeout":
                        if outcome == "no_results" and self._health_store.record_success(
                            provider_key
                        ):
                            self.recovered_provider_keys.add(provider_key)
                            self.search_metrics[f"recoveries:{provider_key}"] += 1
                            self._notify_recovery("searxng")
                        break
                    if attempt_number <= timeout_retries:
                        self.search_metrics[f"timeout_retries:searxng:{engines}"] += 1
                        continue
                    self._searx_timeout_failures[engines] += 1
                    threshold = getattr(
                        self, "_searx_timeout_circuit_threshold", 1
                    )
                    if self._searx_timeout_failures[engines] >= threshold:
                        self._suspended_searx_groups.add(engines)
                        cooldown = self._health_store.record_timeout(
                            provider_key, self._searx_policy
                        )
                        self.search_metrics[f"cooldown_seconds:{provider_key}"] = cooldown
                    break
        elif self._search:
            provider_name = str(self._search.name).strip().casefold()
            health_managed = provider_name == "duckduckgo"
            if health_managed and provider_name in self._suspended_search_providers:
                results = []
                outcome = "cooldown_skipped"
            elif health_managed and self._health_store.cooldown_remaining(provider_name) > 0:
                self._suspended_search_providers.add(provider_name)
                self.search_metrics[f"cooldown_skips:{provider_name}"] += 1
                results = []
                outcome = "cooldown_skipped"
            elif (
                health_managed
                and self._health_store.incident(provider_name) is not None
                and not self._health_store.claim_recovery_probe(provider_name)
            ):
                self._suspended_search_providers.add(provider_name)
                self.search_metrics[f"probe_lease_skips:{provider_name}"] += 1
                results = []
                outcome = "recovery_probe_in_progress"
            else:
                self.search_metrics[f"provider_calls:{provider_name}"] += 1
                results = self._search.search(query, num_results=10)
                outcome = _provider_search_outcome(self._search, results)
                if health_managed:
                    if outcome in {"results", "no_results"}:
                        self._provider_timeout_failures[provider_name] = 0
                        if self._health_store.record_success(provider_name):
                            self.recovered_provider_keys.add(provider_name)
                            self.search_metrics[f"recoveries:{provider_name}"] += 1
                            self._notify_recovery(provider_name)
                    elif outcome in {"captcha", "access_restricted", "rate_limited"}:
                        self._suspended_search_providers.add(provider_name)
                        cooldown = self._health_store.record_unavailable(
                            provider_name,
                            provider_policy(provider_name, _SEARXNG_DEFAULT_POLICY),
                            status=outcome,
                        )
                        self.search_metrics[f"cooldown_seconds:{provider_name}"] = cooldown
                    elif outcome == "timeout":
                        self._provider_timeout_failures[provider_name] += 1
                        if self._provider_timeout_failures[provider_name] >= 3:
                            self._suspended_search_providers.add(provider_name)
                            cooldown = self._health_store.record_timeout(
                                provider_name,
                                provider_policy(provider_name, _SEARXNG_DEFAULT_POLICY),
                            )
                            self.search_metrics[
                                f"cooldown_seconds:{provider_name}"
                            ] = cooldown
            query_attempts.append(
                {
                    "provider": provider_name,
                    "query": query,
                    "outcome": outcome,
                    "result_count": len(results),
                }
            )
            if results:
                self._query_attempts[query] = query_attempts
                self._query_cache[query] = results
                return results

        for provider in self._escalation:
            provider_name = provider.name.lower()
            call_metric = f"provider_calls:{provider_name}"
            limit = self._escalation_limits.get(provider_name, 0)
            if self.search_metrics[call_metric] >= limit:
                self.search_metrics[f"budget_skips:{provider_name}"] += 1
                query_attempts.append(
                    {
                        "provider": provider_name,
                        "query": query,
                        "outcome": "budget_skipped",
                        "result_count": 0,
                    }
                )
                continue
            self.search_metrics[call_metric] += 1
            results = provider.search(query, num_results=10)
            query_attempts.append(
                {
                    "provider": provider_name,
                    "query": query,
                    "outcome": _provider_search_outcome(provider, results),
                    "result_count": len(results),
                }
            )
            if results:
                logger.info("Search escalated to %s", provider.name)
                self._query_attempts[query] = query_attempts
                self._query_cache[query] = results
                return results

        self._query_attempts[query] = query_attempts
        self._query_cache[query] = []
        return []

    def _notify_recovery(self, provider: str) -> None:
        """Emit one bounded refresh event per provider and retriever run."""
        notifications = getattr(self, "_recovery_notifications", set())
        if provider in notifications:
            return
        notifications.add(provider)
        self._recovery_notifications = notifications
        try:
            self._on_provider_recovered(provider)
        except Exception as exc:
            # Search success remains usable even if the maintenance queue is
            # temporarily unavailable; paper checkpoints retain dependencies.
            logger.warning(
                "Could not schedule recovered-provider refresh for %s: %s",
                provider,
                type(exc).__name__,
            )

    def _try_download_pdf(self, url: str) -> Optional[bytes]:
        """Download a PDF from a URL with SSRF + size-cap + magic-byte checks.

        The URL comes from search-provider results (arbitrary web pages), so it
        is treated as untrusted: safe_fetch rejects non-public hosts and aborts
        downloads past _MAX_PDF_SIZE before they can exhaust memory.
        """
        try:
            data = safe_fetch_bytes(
                url,
                max_bytes=_MAX_PDF_SIZE,
                accept_content_types=("application/pdf",),
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                max_meta_refreshes=1,
            )
            if not data[:5] == b"%PDF-":
                logger.debug(
                    "URL returned non-PDF content: %s",
                    private_value_id("url", url),
                )
                return None
            return data
        except Exception as e:
            logger.debug(
                "PDF download failed for %s: %s",
                private_value_id("url", url),
                type(e).__name__,
            )
            return None


def _significant_tokens(value: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if len(token) >= 3
    ]


def _exact_bibliographic_query(
    title: str,
    author: str | None = None,
    year: str | None = None,
) -> str:
    """Build a bounded exact query without serializing a full author list."""
    normalized_title = " ".join(title.split())
    if len(normalized_title) > _MAX_SEARCH_TITLE_CHARS:
        normalized_title = normalized_title[:_MAX_SEARCH_TITLE_CHARS].rsplit(
            " ", 1
        )[0]
    parts = [f'"{normalized_title}"']
    author_hint = _search_author_hint(author)
    if author_hint:
        parts.append(author_hint)
    if year:
        parts.append(year.strip()[:12])
    return " ".join(parts)


def _search_author_hint(author: str | None) -> str:
    if not author:
        return ""
    normalized = " ".join(author.split())
    first_author = re.split(
        r"\s+(?:and|&)\s+|;", normalized, maxsplit=1, flags=re.I
    )[0]
    if "," in first_author:
        first_author = first_author.split(",", 1)[0]
    if len(first_author) > _MAX_SEARCH_AUTHOR_CHARS:
        first_author = first_author[:_MAX_SEARCH_AUTHOR_CHARS].rsplit(" ", 1)[0]
    return first_author.strip()


def _searx_provider_key(engines: str) -> str:
    normalized = ",".join(
        item.strip().casefold() for item in engines.split(",") if item.strip()
    )
    return f"searxng:{normalized or 'default'}"


def _schedule_provider_recovery(provider: str) -> None:
    # Lazy import avoids coupling the retrieval module to Celery initialization.
    from app.tasks.provider_recovery import schedule_provider_recovery

    schedule_provider_recovery(provider)


def _looks_like_pdf(url: str) -> bool:
    path = unquote(url).split("?", 1)[0].lower()
    return path.endswith(".pdf") or ".pdf/" in path


def _candidate_score(candidate, doi: str | None, title: str | None,
                     author: str | None, year: str | None) -> int:
    """Rank only on inspectable search evidence; acquisition still validates identity."""
    haystack = " ".join((candidate.url, candidate.title, candidate.snippet)).lower()
    score = 30 if candidate.is_pdf or _looks_like_pdf(candidate.url) else 0
    if doi and doi.lower() in unquote(haystack):
        score += 100
    if title:
        expected = set(_significant_tokens(title))
        observed = set(_significant_tokens(haystack))
        if expected:
            score += round(60 * len(expected & observed) / len(expected))
    if author:
        surname = _significant_tokens(author.split(",", 1)[0])
        if surname and surname[-1] in haystack:
            score += 15
    if year and year in haystack:
        score += 8
    if any(marker in haystack for marker in ("repository", ".edu/", ".ac.", "journal", "doi.org")):
        score += 8
    if any(marker in haystack for marker in ("login", "catalog record", "search results")):
        score -= 25
    return score
