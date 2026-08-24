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
from typing import Optional
from urllib.parse import unquote

import httpx

from app.config import settings
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
)
from app.services.search import get_search_provider
from app.services.safe_fetch import safe_fetch_bytes
from app.services.search.base import SearchProvider

logger = logging.getLogger(__name__)

# Max PDFs to try downloading per source (each download is a network call;
# try a few candidates in case the first is a wrong paper or a dead link)
_MAX_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_TIMEOUT_SECONDS = 10

# Max PDF size to accept (50MB — same as student-URL limit)
_MAX_PDF_SIZE = 50 * 1024 * 1024


class WebSearchRetriever(RetrievalSource):
    """Find ranked public-web locations after structured retrieval fails.

    Uses exact identifiers and bibliographic queries, then adds a PDF-specific
    query only when the earlier results do not yield enough distinct locations.

    The configured primary provider runs first. Optional bounded escalation
    providers (normally Tavily then Exa) run only when it produces no results.
    Exact queries are cached for the process so repeated canonical works do not
    consume another external search. Unofficial HTML scrapers are never added
    as an implicit fallback.
    """

    name = "web_search"
    capabilities = frozenset({"doi", "title_author", "locations", "web_discovery"})

    def __init__(self, search_provider: Optional[SearchProvider] = None):
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
        exact_parts = [f'"{title}"']
        if author:
            exact_parts.append(author)
        if year:
            exact_parts.append(year)
        exact = " ".join(exact_parts)
        normalized = " ".join(_significant_tokens(title))
        queries = [exact, f"{exact} filetype:pdf"]
        if normalized and normalized.lower() != title.lower():
            queries.append(" ".join(part for part in (normalized, author or "", year or "") if part))
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
            exact = " ".join(
                part for part in (f'"{title}"', author or "", year or "") if part
            )
            queries.extend((f"{exact} filetype:pdf", exact))

        all_attempts: list[dict] = []
        for provider in self._escalation:
            provider_name = provider.name.lower()
            if provider_name in tried_providers:
                continue
            candidates: dict[str, tuple[object, str]] = {}
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
                        "outcome": "results" if results else "no_results",
                        "result_count": len(results),
                    }
                )
                for candidate in results:
                    candidates.setdefault(candidate.url, (candidate, query))
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
        """Merge, rank and expose the best three candidates for shared acquisition."""
        candidates: dict[str, tuple[object, str]] = {}
        queries_run: list[str] = []
        for query in queries:
            if not query or query in queries_run:
                continue
            queries_run.append(query)
            for candidate in self._run_search(query):
                candidates.setdefault(candidate.url, (candidate, query))
            if len(candidates) >= _MAX_DOWNLOAD_ATTEMPTS and len(queries_run) >= 2:
                break

        if not candidates:
            return RetrievalResult(source_name=self.name, success=False, error="no search results")

        search_attempts = [
            attempt
            for query in queries_run
            for attempt in getattr(self, "_query_attempts", {}).get(query, [])
        ]
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
        candidates: dict[str, tuple[object, str]],
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
        for index, (candidate, query) in enumerate(ranked):
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
            # These are public-web scrapers, not supported APIs. A failed group
            # is suspended for the rest of the run to avoid repeated CAPTCHA,
            # 403 and 429 traffic. Bing is excluded because Microsoft retired
            # its supported Search APIs; scraping the consumer site is not a
            # production substitute.
            results: list = []
            for engines in ("mojeek,qwant", "startpage"):
                if engines in self._suspended_searx_groups:
                    continue
                self.search_metrics[f"provider_calls:searxng:{engines}"] += 1
                results = self._search.search(query, num_results=10, engines=engines)
                query_attempts.append(
                    {
                        "provider": "searxng",
                        "engines": engines,
                        "query": query,
                        "outcome": "results" if results else "no_results",
                        "result_count": len(results),
                        "unresponsive_engines": [
                            item[0] for item in self._search.last_unresponsive_engines
                        ],
                    }
                )
                if results:
                    self._query_attempts[query] = query_attempts
                    self._query_cache[query] = results
                    return results
                if self._search.last_unresponsive_engines:
                    self._suspended_searx_groups.add(engines)
        elif self._search:
            self.search_metrics[f"provider_calls:{self._search.name.lower()}"] += 1
            results = self._search.search(query, num_results=10)
            query_attempts.append(
                {
                    "provider": self._search.name.lower(),
                    "query": query,
                    "outcome": "results" if results else "no_results",
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
                    "outcome": "results" if results else "no_results",
                    "result_count": len(results),
                }
            )
            if results:
                logger.info("Search escalated to %s for '%s'", provider.name, query[:60])
                self._query_attempts[query] = query_attempts
                self._query_cache[query] = results
                return results

        self._query_attempts[query] = query_attempts
        self._query_cache[query] = []
        return []

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
                logger.debug("URL returned non-PDF content: %s", url[:60])
                return None
            return data
        except Exception as e:
            logger.debug("PDF download failed for %s: %s", url[:60], e)
            return None


def _significant_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", value) if len(token) >= 3]


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
