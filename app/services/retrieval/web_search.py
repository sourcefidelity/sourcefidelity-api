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

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import logging
import re
import threading
import time
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
from app.services.retrieval.provider_gate import ProviderBusy
from app.services.safe_fetch import safe_fetch_bytes
from app.services.search.base import SearchProvider
from app.services.search.base import classify_search_failure
from app.services.search.policy import API_FIRST_SEARCH_POLICY, required_api_providers
from app.services.search.transient import BRAVE_TRANSIENT_POLICY
from app.services.candidate_budget import inspection_capacity
from app.services.search.candidate_ranking import candidate_score, POLICY as RANKING_POLICY, PURPOSE

logger = logging.getLogger(__name__)

# Max locations to expose per search-provider tier. Shared acquisition still
# validates identity, type and completeness. Representation format is not identity.
_MAX_DOWNLOAD_ATTEMPTS = 5
CAPACITY_POLICY = "search-inspection-capacity-v4"
_DOWNLOAD_TIMEOUT_SECONDS = 10

# Max PDF size to accept (50MB — same as student-URL limit)
_MAX_PDF_SIZE = 50 * 1024 * 1024
_MAX_SEARCH_TITLE_CHARS = 240
_MAX_SEARCH_AUTHOR_CHARS = 60


def _query_inspection_capacity(query, provider=None):
    capacity = inspection_capacity(_MAX_DOWNLOAD_ATTEMPTS)
    if PURPOSE.get() == "identity":
        from app.services.identity_landing import remaining_inspections as html_remaining
        from app.services.identity_pdf import remaining_inspections as pdf_remaining
        # A PDF-only query requires a file slot. General queries require an
        # HTML slot: do not spend on catalog/article pages we cannot inspect.
        html_slots = html_remaining()
        # Search providers do not guarantee a PDF response to filetype:pdf.
        # Never buy more results once no HTML inspection is possible. Retain
        # remaining PDF slots for files returned while inspection was viable.
        capacity = min(capacity, html_slots,
                       pdf_remaining() if "filetype:pdf" in query else html_slots)
        if provider == "brave" and html_slots > 1:
            # Reserve one shared HTML inspection for the next required tier.
            capacity = min(capacity, html_slots - 1)
    return capacity

# Direct providers (selected on the configured path) whose cooldown,
# incident and recovery-probe state this retriever manages itself. Brave and
# Exa are not here: the API-first policy governs them. Empty since the only
# member was removed (2026-10-19); the mechanism stays for a provider that
# meets account-level limits its own bounded retry cannot clear.
_HEALTH_MANAGED_DIRECT_PROVIDERS: frozenset[str] = frozenset()

# Outcomes of a real Brave or Exa request that mean the provider itself
# failed. Kept aligned with the curable outcomes in
# `search_completion` blocking providers: an incident is opened
# exactly when the failure could be what holds a search incomplete.
_REQUIRED_API_INCIDENT_OUTCOMES = frozenset({
    "operational_failure", "timeout", "rate_limited", "response_invalid",
    "captcha", "access_restricted",
})
# A failure that clears within this long is a blip, not an outage, and does
# not report a recovery. Without it, on 2026-09-23 a Brave route that was only
# intermittently reachable reported a recovery on every fail-then-succeed flip:
# 95 recovery requeues in 40 minutes, 14 jobs re-run in the first minute, and
# every re-run job eligible to be re-run again at the next flip. A provider
# that is mostly down but occasionally answers therefore never triggers
# re-runs, which is right: re-running would mostly fail again.
_REQUIRED_API_MIN_OUTAGE_SECONDS = 600


def _incident_age_seconds(incident: dict) -> float:
    """How long an incident has been open; 0 when that cannot be read."""
    try:
        opened = datetime.fromisoformat(str(incident.get("incident_opened_at")))
    except (TypeError, ValueError):
        return 0.0
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - opened).total_seconds())


# Stored with the incident for diagnostics; the API-first path does not
# enforce it, so it never delays a search.
_REQUIRED_API_HEALTH_POLICY = ProviderPolicy(
    timeout_seconds=15.0,
    cooldown_seconds=300,
    max_cooldown_seconds=3600,
)

_DIRECT_PROVIDER_DEFAULT_POLICY = ProviderPolicy(
    timeout_seconds=15.0,
    cooldown_seconds=180,
    max_cooldown_seconds=3600,
    max_consecutive_failures=3,
)


# Calls each required API provider has made for the reference being resolved.
# Set per reference by `reference_search_scope`; None outside one, where only
# the job-wide ceiling applies.
_REFERENCE_PROVIDER_CALLS: ContextVar[Counter | None] = ContextVar(
    "web_search_reference_provider_calls", default=None)


@contextmanager
def reference_search_scope():
    """Count required-API calls for one reference (the per-reference floor)."""
    token = _REFERENCE_PROVIDER_CALLS.set(Counter())
    try:
        yield
    finally:
        _REFERENCE_PROVIDER_CALLS.reset(token)


def reference_search_scoped(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with reference_search_scope():
            return function(*args, **kwargs)
    return scoped


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


def _limits_over_defaults(name: str) -> dict[str, int]:
    """A `provider:count` setting over its code default, provider by provider
    (owner decision 2026-10-19): a provider the configured value does not name
    keeps its default rather than getting no calls. Only an explicit 0 turns a
    provider off."""
    from app.config import Settings
    default = Settings.model_fields[name].default or ""
    return {**WebSearchRetriever._parse_escalation_limits(default),
            **WebSearchRetriever._parse_escalation_limits(getattr(settings, name) or "")}


class WebSearchRetriever(RetrievalSource):
    """Find ranked public-web locations after structured retrieval fails.

    Uses exact identifiers and bibliographic queries, then adds a PDF-specific
    query only when the earlier results do not yield enough distinct locations.

    API-first runs the required API providers, followed by optional Tavily.
    The legacy configured cascade remains explicitly selectable.
    Escalation follows empty searches or rejection of a nonempty candidate tier.
    Exact queries are cached for this retriever so repeated canonical works do not
    consume another external search. Unofficial HTML scrapers are never added
    as an implicit fallback.
    """
    # The bounded-web route is required by the API-first policy and is
    # already passed `required=True` explicitly at its call site.
    required_for_search_completion = True

    name = "web_search"
    capabilities = frozenset({"doi", "title_author", "locations", "web_discovery"})

    def __init__(
        self,
        search_provider: Optional[SearchProvider] = None,
        *,
        health_store: ProviderHealthStore | None = None,
        on_provider_recovered: Callable[[str], None] | None = None,
    ):
        self.policy_version = settings.SEARCH_POLICY_VERSION
        self._policy_providers = {}
        self._policy_query_cache = {}
        if self.policy_version == API_FIRST_SEARCH_POLICY and search_provider is None:
            names = [*required_api_providers()]
            fallback = getattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "none")
            if fallback == "tavily":
                names.append("tavily")
            self._policy_providers = {name: get_search_provider(name) for name in names}
        self._search = search_provider or (next(iter(self._policy_providers.values()))
                                           if self._policy_providers else get_search_provider())
        self._escalation: list[SearchProvider] = []
        for name in ([] if self._policy_providers else settings.SEARCH_ESCALATION_PROVIDERS.split(",")):
            name = name.strip().lower()
            if not name or (self._search and name == self._search.name.lower()):
                continue
            provider = get_search_provider(name)
            if provider:
                self._escalation.append(provider)
        self._query_cache: dict[str, list] = {}
        self._query_attempts: dict[str, list[dict]] = {}
        self._suspended_search_providers: set[str] = set()
        self._provider_timeout_failures: Counter[str] = Counter()
        self._health_store = health_store or ProviderHealthStore()
        self.recovered_provider_keys: set[str] = set()
        self._recovery_notifications: set[str] = set()
        self._on_provider_recovered = (
            on_provider_recovered or _schedule_provider_recovery
        )
        self.search_metrics: Counter[str] = Counter()
        self._escalation_limits = _limits_over_defaults("SEARCH_ESCALATION_MAX_CALLS")
        self._hard_call_limits = _limits_over_defaults("WEB_SEARCH_HARD_MAX_CALLS")
        self._per_reference_floor = _limits_over_defaults("WEB_SEARCH_PER_REFERENCE_FLOOR")
        if not self._search:
            logger.info("WebSearchRetriever: no primary search provider configured")

    def search_reference(self, *, doi, title, author, year) -> RetrievalResult:
        """Finish a provider's bounded bibliographic ladder before escalating.

        Returned locations are candidates only. The resolver calls the next
        tier after the shared acquisition/validation gates reject this tier.
        """
        queries = []
        if doi:
            queries.append(f'"{doi}"')
        if title:
            queries.append(_exact_bibliographic_query(title, author, year))
        if queries:
            queries.append(f"{queries[-1]} filetype:pdf")
        return self._search_policy_locations(
            queries=queries, doi=doi, title=title, author=author, year=year,
            tried_providers=set(),
        )

    def _search_policy_locations(self, *, queries, doi, title, author, year, tried_providers):
        all_attempts = []
        for name, provider in self._policy_providers.items():
            if name in tried_providers:
                continue
            candidates = {}
            queries_run = []
            # `filetype:` is a Google/Brave operator. Exa and Tavily document
            # no such operator, so the variant is a paid call on literal text.
            tier_queries = [q for q in dict.fromkeys(queries)
                            if name in _FILETYPE_OPERATOR_PROVIDERS or "filetype:" not in q]
            if PURPOSE.get() == "identity":
                available_queries = [q for q in tier_queries if _query_inspection_capacity(q, name)]
                # Keep one zero-capacity attempt to record required incompletion.
                tier_queries = available_queries or tier_queries[:1]
            title_query = _exact_bibliographic_query(title, author, year) if title else None
            recovery_query = title_query if doi and title_query in tier_queries else None
            if recovery_query and inspection_capacity(_MAX_DOWNLOAD_ATTEMPTS) == 1:
                # Direct/structured DOI checks precede this tier. With one slot,
                # retain an opportunity to recover an incorrectly supplied DOI.
                tier_queries.remove(recovery_query)
                tier_queries.insert(0, recovery_query)
            for query in tier_queries:
                capacity = _query_inspection_capacity(query, name)
                if candidates and len(candidates) >= capacity:
                    break
                slots = max(0, capacity - len(candidates))
                if (recovery_query and recovery_query not in queries_run
                        and query == f'"{doi}"' and slots > 1):
                    # Do not fill every inspection slot with identifier hits
                    # before searching the independently supplied title.
                    slots -= 1
                results, attempts = self._run_policy_query(
                    name, provider, query, num_results=slots)
                queries_run.append(query)
                all_attempts.extend(attempts)
                for candidate in results:
                    candidates.setdefault(candidate.url, (candidate, query, name))
                if any(a["outcome"] not in {"results", "no_results"} for a in attempts):
                    # No immediate retry of an operationally failed API tier.
                    break
                if len(candidates) >= capacity:
                    break
            if candidates:
                return self._locations_result(
                    candidates, queries_run=queries_run, doi=doi, title=title,
                    author=author, year=year, search_attempts=all_attempts,
                    metadata={"search_policy_version": self.policy_version},
                )
            tried_providers.add(name)
        return RetrievalResult(source_name=self.name, success=False, error="no search results",
            metadata={"search_policy_version": self.policy_version, "search_attempts": all_attempts})

    def _run_policy_query(self, name, provider, query, num_results=None):
        from app.services.retrieval_deadline import expired
        if expired():
            # The reference's own budget ran out; the provider was never called
            # (`provider_calls: 0`). Recorded as `timeout` until 2026-09-24,
            # which named Brave or Exa as the cause and so as a re-run trigger.
            # `budget_skipped` keeps the search incomplete without blaming them.
            return [], [{"provider": name, "query": query, "outcome": "budget_skipped",
                         "required": name in required_api_providers(), "provider_calls": 0,
                         "result_count": 0, "latency_seconds": 0.0, "cost_usd": None,
                         "reason_code": "reference_elapsed_budget_timeout"}]
        num_results = min(inspection_capacity(_MAX_DOWNLOAD_ATTEMPTS),
                          _MAX_DOWNLOAD_ATTEMPTS if num_results is None else num_results)
        if num_results <= 0:
            return [], [{"provider": name, "query": query, "outcome": "budget_skipped",
                         "required": name in required_api_providers(), "provider_calls": 0,
                         "result_count": 0, "latency_seconds": 0.0, "cost_usd": None,
                         "reason_code": "candidate_inspection_capacity_exhausted"}]
        key = (name, query, num_results)
        if key in self._policy_query_cache:
            self.search_metrics["query_cache_hits"] += 1
            results, attempts = self._policy_query_cache[key]
            return results, [{**a, "cache_hit": True, "provider_calls": 0, "latency_seconds": 0.0,
                              "cost_usd": None} for a in attempts]
        started = time.monotonic()
        metric = f"provider_calls:{name}"
        outcome = None
        reason = None
        results = []
        calls = 0
        if name == "brave" and not (settings.BRAVE_SEARCH_TRANSIENT_ENABLED or settings.BRAVE_SEARCH_RETENTION_PERMITTED):
            outcome, reason = "access_restricted", "search_retention_permission_unconfirmed"
        elif provider is None:
            outcome, reason = "operational_failure", "provider_not_configured"
        elif not self._reserve_policy_call(name):
            outcome, reason = "budget_skipped", "configured_call_ceiling"
            self.search_metrics[f"budget_skips:{name}"] += 1
        elif (daily := self._reserve_daily_call(name))[0]:
            self._release_policy_call(name)
            outcome, reason = "budget_skipped", daily[0]
            self.search_metrics[f"budget_skips:{name}"] += 1
        else:
            from app.services.retrieval import provider_gate
            calls = 1
            try:
                with provider_gate.turn(name, interval=provider_gate.interval_for(name)):
                    results = provider.search(query, num_results=num_results)
                # Only adapters that explicitly supply an application-owned
                # bounded reason expose it; never copy response/error bodies.
                provider_reason = getattr(provider, "last_reason_code", None)
                if isinstance(provider_reason, str) and re.fullmatch(r"[A-Za-z0-9_]{1,100}", provider_reason):
                    reason = provider_reason
                outcome = _provider_search_outcome(provider, results)
                if not results and getattr(provider, "last_status", None) != "completed" and outcome == "no_results":
                    outcome = "operational_failure"
            except provider_gate.ProviderBusy:
                # Never sent (provider-gate-v1): not a call and not a provider failure.
                calls, results = 0, []
                outcome, reason = "budget_skipped", provider_gate.ProviderBusy.reason_code
                self.search_metrics[f"budget_skips:{name}"] += 1
                self._release_policy_call(name)
                from app.services.search import search_budget
                search_budget.release(daily[1])
            except Exception as exc:
                outcome = classify_search_failure(exc)
            finally:
                from app.services.processing_metrics import record_search_usage
                record_search_usage(name, calls, getattr(provider, 'last_cost_usd', None) if calls else None)
        elapsed = time.monotonic() - started
        self.search_metrics[f"latency_seconds:{name}"] += elapsed
        if calls:
            self._track_required_api_health(name, outcome)
        attempt = {"provider": name, "query": query, "outcome": outcome,
                   "result_count": len(results), "required": name in required_api_providers(),
                   "provider_calls": calls, "latency_seconds": elapsed,
                   "cost_usd": getattr(provider, "last_cost_usd", None) if calls else None,
                   "credits_remaining": None, "reason_code": reason}
        # No Brave response survives a query/tier in the retriever instance,
        # and a call the gate held back may be made later in the run.
        if (name != "brave" or not settings.BRAVE_SEARCH_TRANSIENT_ENABLED) and reason != ProviderBusy.reason_code:
            self._policy_query_cache[key] = results, [attempt]
        return results, [attempt]

    def share_run_budget_with(self, other: "WebSearchRetriever") -> None:
        """Count calls against `other`'s per-run ceilings (one paper run's
        worker threads each hold their own retriever; `SourceResolver.for_worker`)."""
        self.search_metrics = other.search_metrics
        self.__dict__["_call_lock_instance"] = other._call_lock()

    def _call_lock(self) -> threading.Lock:
        return self.__dict__.setdefault("_call_lock_instance", threading.Lock())

    def _reserve_policy_call(self, name: str) -> bool:
        """Check the per-run ceilings and count the call in one locked step,
        so references searched at once cannot both take the last call."""
        with self._call_lock():
            if not self._policy_call_permitted(name):
                return False
            metric = f"provider_calls:{name}"
            if self.search_metrics[metric] >= self._escalation_limits.get(name, 0):
                self.search_metrics[f"floor_calls:{name}"] += 1
            self.search_metrics[metric] += 1
            reference_calls = _REFERENCE_PROVIDER_CALLS.get()
            if reference_calls is not None:
                reference_calls[name] += 1
            return True

    def _release_policy_call(self, name: str) -> None:
        """Give back a call `_reserve_policy_call` counted but that was not made."""
        with self._call_lock():
            metric = f"provider_calls:{name}"
            self.search_metrics[metric] -= 1
            if self.search_metrics[metric] >= self._escalation_limits.get(name, 0):
                self.search_metrics[f"floor_calls:{name}"] -= 1
            reference_calls = _REFERENCE_PROVIDER_CALLS.get()
            if reference_calls is not None:
                reference_calls[name] -= 1

    @staticmethod
    def _reserve_daily_call(name: str) -> tuple[str | None, list[str]]:
        """The deployment's and the checking scope's daily budgets (search-daily-budget-v1)."""
        from app.services.search import search_budget, search_memo
        context = search_memo.active_context()
        scope = (context.scope_type, context.scope_id) if context is not None else None
        return search_budget.reserve(name, scope)

    def _policy_call_permitted(self, name: str) -> bool:
        """Job-wide ceiling first, then a bounded per-reference floor.

        The job-wide ceiling (`SEARCH_ESCALATION_MAX_CALLS`) was spent first
        come, first served: on 2026-09-29 a paper's later references found Exa
        already `budget_skipped`, so their required search could never
        complete. Each reference now keeps `WEB_SEARCH_PER_REFERENCE_FLOOR`
        calls to each required API provider after the ceiling is spent, up to
        the absolute cap `WEB_SEARCH_HARD_MAX_CALLS` (never below the ceiling).

        Worst case per paper run (one retriever per run) is the hard cap:
        with the defaults brave:120 and exa:80, 120 x $0.005 (Brave list
        price, `BRAVE_USD_PER_REQUEST`) + 80 x $0.007 (Exa `auto` search, up
        to 10 results, exa.ai/docs/reference/pricing, checked 2026-09-29)
        = about $1.16. A targeted re-run is a new run with its own cap.
        """
        calls = self.search_metrics[f"provider_calls:{name}"]
        ceiling = self._escalation_limits.get(name, 0)
        if calls < ceiling:
            return True
        if name not in required_api_providers() or ceiling <= 0:
            # A provider the operator gave no ceiling stays off.
            return False
        hard_cap = max(ceiling, getattr(self, "_hard_call_limits", {}).get(name, 0))
        if calls >= hard_cap:
            return False
        reference_calls = _REFERENCE_PROVIDER_CALLS.get()
        if reference_calls is None:
            return False
        return reference_calls[name] < getattr(self, "_per_reference_floor", {}).get(name, 0)

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
        if getattr(self, "_policy_providers", None):
            return self.search_reference(doi=doi, title=None, author=None, year=None)
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
        if getattr(self, "_policy_providers", None):
            return self.search_reference(doi=None, title=title, author=author, year=year)
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
        if getattr(self, "_policy_providers", None):
            queries = []
            if doi:
                queries.append(f'"{doi}"')
            if title:
                queries.append(_exact_bibliographic_query(title, author, year))
            if queries:
                queries.append(f"{queries[-1]} filetype:pdf")
            return self._search_policy_locations(
                queries=queries, doi=doi, title=title, author=author, year=year,
                tried_providers=tried_providers,
            )
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
                try:
                    results = provider.search(query, num_results=10)
                finally:
                    # Legacy escalation calls are billable too; record them,
                    # including failures, like the current policy path.
                    from app.services.processing_metrics import record_search_usage
                    record_search_usage(provider_name, 1, getattr(provider, 'last_cost_usd', None))
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
        all_ranked = sorted(
            candidates.values(),
            key=lambda item: _candidate_score(item[0], doi, title, author, year),
            reverse=True,
        )
        ranked = []
        format_remaining = None
        if PURPOSE.get() == "identity":
            from app.services.identity_landing import remaining_inspections as html_remaining
            from app.services.identity_pdf import remaining_inspections as pdf_remaining
            format_remaining = {True: pdf_remaining(), False: html_remaining()}
        for item in all_ranked:
            pdf = bool(item[0].is_pdf or _looks_like_pdf(item[0].url))
            if len(ranked) >= inspection_capacity(_MAX_DOWNLOAD_ATTEMPTS):
                break
            if format_remaining is not None:
                if not format_remaining[pdf]:
                    continue
                format_remaining[pdf] -= 1
            ranked.append(item)
        selected_urls = {item[0].url for item in ranked}
        from app.services.reference_review_scope import scope, input_hash
        from collections import Counter
        def review_scope(candidate):
            if doi and doi.casefold() in candidate.url.casefold():
                return 'material'
            return scope(title or '', candidate.title, author or '')
        transient = (settings.BRAVE_SEARCH_TRANSIENT_ENABLED
                     and all(item[2] == "brave" for item in all_ranked))
        from app.services.search.candidate_audit import auditable_provider
        search_ranks = {item[0].url: rank for rank, item in enumerate(all_ranked, start=1)}
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
                        "search_engine_group": next((a.get("engines") for a in search_attempts
                            if a.get("query") == query and a.get("provider") == search_provider
                            and a.get("outcome") == "results"), None),
                        "search_title": candidate.title,
                        "bounded_review_scope": review_scope(candidate),
                        "search_snippet": candidate.snippet,
                        "deterministic_score": _candidate_score(
                            candidate, doi, title, author, year
                        ),
                        "may_contain_full_text": not is_pdf,
                        # Development candidate audit only; never Brave.
                        **({"search_rank": search_ranks[candidate.url]}
                           if not transient and auditable_provider(search_provider) else {}),
                    },
                )
            )
        logger.info(
            "Web discovery produced %d ranked locations from %d quer%s",
            len(locations),
            len(queries_run),
            "y" if len(queries_run) == 1 else "ies",
        )
        if transient:
            # Ranking uses the response only within this function. Acquisition
            # receives a transient URL, never copied title/snippet/rank fields.
            for location in locations:
                decision = location.metadata['bounded_review_scope']
                location.metadata.clear()
                location.metadata["search_provider"] = "brave"
                location.metadata['bounded_review_scope'] = decision
            metadata = {**(metadata or {}), "search_retention_policy": BRAVE_TRANSIENT_POLICY,
                        "bounded_review_input_sha256": input_hash(title or '', author or ''),
                        "transient_unselected_review_counts": dict(Counter(
                            review_scope(item[0]) for item in all_ranked if item[0].url not in selected_urls)),
                        "transient_unselected_count": len(all_ranked) - len(ranked)}
        return RetrievalResult(
            source_name=self.name,
            success=bool(locations),
            doi=doi,
            title=title,
            full_text_url=locations[0].url if locations else None,
            locations=locations,
            metadata={
                **(metadata or {}),
                "candidate_ranking_policy": RANKING_POLICY,
                "candidate_ranking_purpose": PURPOSE.get(),
                "queries_run": queries_run,
                "candidate_count": len(candidates),
                "search_attempts": search_attempts,
                "candidate_dispositions": [
                    {"url": candidate.url, "candidate_title": candidate.title,
                     "discovery_provider": provider, "rank": index + 1,
                     "outcome": "not_attempted", "reason_code": "bounded_location_limit",
                     **({"search_query": query, "search_rank": index + 1}
                        if auditable_provider(provider) else {})}
                    for index, (candidate, query, provider) in enumerate(all_ranked)
                    if candidate.url not in selected_urls and not transient
                ],
            },
        )

    def _result_provider_for_query(self, query: str) -> str:
        """Return the provider that produced the retained results for a query."""
        attempts = getattr(self, "_query_attempts", {}).get(query, [])
        for attempt in reversed(attempts):
            if attempt.get("outcome") == "results":
                return str(attempt.get("provider") or "unknown").lower()
        return "unknown"

    def _run_search(self, query: str) -> list:
        """Run a cached primary search, then bounded configured escalation.

        OpenAlex already incorporates Unpaywall OA locations, including arXiv
        preprints linked to works. Crossref and CORE also run as direct
        retrieval adapters. The configured web provider therefore does not
        repeat those academic APIs; it is an opportunistic public-web fallback.
        """
        if query in self._query_cache:
            self.search_metrics["query_cache_hits"] += 1
            return self._query_cache[query]

        query_attempts: list[dict] = []
        if not hasattr(self, "_query_attempts"):
            self._query_attempts = {}

        if self._search:
            provider_name = str(self._search.name).strip().casefold()
            health_managed = provider_name in _HEALTH_MANAGED_DIRECT_PROVIDERS
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
                try:
                    results = self._search.search(query, num_results=10)
                finally:
                    from app.services.processing_metrics import record_search_usage
                    record_search_usage(provider_name, 1, getattr(self._search, 'last_cost_usd', None))
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
                            provider_policy(provider_name, _DIRECT_PROVIDER_DEFAULT_POLICY),
                            status=outcome,
                        )
                        self.search_metrics[f"cooldown_seconds:{provider_name}"] = cooldown
                    elif outcome == "timeout":
                        self._provider_timeout_failures[provider_name] += 1
                        if self._provider_timeout_failures[provider_name] >= 3:
                            self._suspended_search_providers.add(provider_name)
                            cooldown = self._health_store.record_timeout(
                                provider_name,
                                provider_policy(provider_name, _DIRECT_PROVIDER_DEFAULT_POLICY),
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
            try:
                results = provider.search(query, num_results=10)
            finally:
                from app.services.processing_metrics import record_search_usage
                record_search_usage(provider_name, 1, getattr(provider, 'last_cost_usd', None))
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

    def _track_required_api_health(self, name: str, outcome: str | None) -> None:
        """Open an incident when Brave or Exa fails, and report its recovery.

        Until 2026-09-24 the required web APIs had no health record at all, so
        a reference held incomplete by a Brave or Exa outage could never be
        re-run automatically. This is **passive** -- it observes calls a paper was making
        anyway and adds no request of its own, so it costs nothing. An active
        probe would detect recovery sooner but spends paid API calls on a
        schedule, and is not authorised.

        It records the incident only; nothing in the API-first path consults
        the cooldown, so what gets searched is unchanged. Called only after a
        real request (`calls == 1`), so our own ceilings, retention
        permission, missing configuration and elapsed budget never open one.
        """
        if name not in required_api_providers():
            return
        if outcome in _REQUIRED_API_INCIDENT_OUTCOMES:
            self._health_store.record_unavailable(
                name, provider_policy(name, _REQUIRED_API_HEALTH_POLICY),
                status=str(outcome))
            self.search_metrics[f"incidents:{name}"] += 1
        elif outcome in {"results", "no_results"}:
            incident = self._health_store.incident(name)
            if not isinstance(incident, dict):
                return
            lasted = _incident_age_seconds(incident)
            # Close the incident either way; report a recovery only when an
            # outage actually lasted. Only a real `True` counts, so a stub or
            # Mock store cannot schedule re-runs of real papers.
            if (self._health_store.record_success(name) is True
                    and lasted >= _REQUIRED_API_MIN_OUTAGE_SECONDS):
                self.recovered_provider_keys.add(name)
                self.search_metrics[f"recoveries:{name}"] += 1
                self._notify_recovery(name)
            else:
                self.search_metrics[f"blips:{name}"] += 1

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
                usage_label="file download",
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


# Engines whose query language supports `filetype:pdf`.
_FILETYPE_OPERATOR_PROVIDERS = frozenset({"brave"})


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


def _schedule_provider_recovery(provider: str) -> None:
    # Lazy import avoids coupling the retrieval module to Celery initialization.
    from app.tasks.provider_recovery import schedule_provider_recovery

    schedule_provider_recovery(provider)


def _looks_like_pdf(url: str) -> bool:
    path = unquote(url).split("?", 1)[0].lower()
    return path.endswith(".pdf") or ".pdf/" in path


def _candidate_score(candidate, doi: str | None, title: str | None,
                     author: str | None, year: str | None) -> int:
    return candidate_score(candidate, doi, title, author, year)
