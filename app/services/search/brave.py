"""Brave Search API provider.

Brave Search is an independent search engine (not a Google/Bing proxy) with
its own index. It uses an official metered API and is available where
consumer-search scraping is unstable or unsupported.

API: https://api.search.brave.com/res/v1/web/search
Pricing and monthly credits change independently of the application; consult
Brave's current primary pricing documentation before deployment.

Good for personal users (Path A/C) who don't want to self-host SearXNG.
"""

import logging
import unicodedata

import httpx

from app.config import settings
from app.services.retrieval_deadline import remaining

from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_RESPONSE_CONTRACT = "brave-web-response-v3"
# Request settings (2026-09-25, from Brave's API reference):
# - `spellcheck` defaults on and "if enabled, modified query is always used for
#   search": an unusual or invented title was silently searched as a corrected
#   phrase, so a completed search was not a search of the title we sent.
# - `text_decorations` defaults on and puts highlight markers in the strings we
#   compare against the reference.
# - `result_filter` limits the response to the sections we read. News is kept
#   because students cite news sites; its results follow the web results.
_REQUEST_SETTINGS = {"spellcheck": "false", "text_decorations": "false", "result_filter": "web,news"}


class _ResponseContractError(ValueError):
    """Application-owned reason codes, never provider response contents."""


def _section_results(section: object, name: str) -> list[dict]:
    """Validated results of one response section (web or news)."""
    if not isinstance(section, dict) or section.get("type", "search" if name == "web" else "news") not in {"search", "news"}:
        raise _ResponseContractError(f"brave_web_v2_invalid_{name}")
    results = section.get("results")
    if not isinstance(results, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("url"), str) or not item["url"].strip()
        or not isinstance(item.get("title"), str)
        or (item.get("description") is not None and not isinstance(item["description"], str))
        for item in results
    ):
        raise _ResponseContractError(f"brave_web_v2_invalid_{name}_results")
    return results


def _echo_key(text: object) -> str | None:
    """Brave's echo of our query, compared without case or spacing.

    Measured 2026-09-25: for a query containing Cyrillic, Brave echoed the
    query lowercased; an exact comparison rejected that valid response and the
    reference lost its Brave search.
    """
    if not isinstance(text, str):
        return None
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def _web_results(data: object, query: str) -> tuple[list[dict], str]:
    # Brave's successful envelope identifies the response and original query;
    # the web section and descriptions are nullable. Missing web.results in a
    # PRESENT web object is malformed, not the documented empty-web response.
    if not isinstance(data, dict) or data.get("type") != "search":
        raise _ResponseContractError("brave_web_v2_invalid_envelope")
    original = data.get("query")
    if (not isinstance(original, dict)
            or _echo_key(original.get("original")) != _echo_key(query)
            or any(data.get(key) is not None for key in ("error", "errors"))):
        raise _ResponseContractError("brave_web_v2_invalid_query_envelope")
    # Should not occur with spellcheck off. If Brave still rewrote the query,
    # its results are leads but an empty answer is not a search of our text.
    altered = original.get("altered") not in (None, "", query)
    web, news = data.get("web"), data.get("news")
    results = _section_results(web, "web") if web is not None else []
    news_results = _section_results(news, "news") if news is not None else []
    seen = {item["url"] for item in results}
    results = results + [item for item in news_results if item["url"] not in seen]
    if not results and original.get("more_results_available") is True:
        raise _ResponseContractError("brave_web_v2_inconsistent_empty_page")
    if altered and not results:
        raise _ResponseContractError("brave_web_v3_query_altered_empty")
    reason = "brave_web_v2_results" if results else "brave_web_v2_empty_web"
    return results, ("brave_web_v3_query_altered" if altered else reason)


def _request_cost(served: bool) -> float | None:
    """List price of one served request; None when unpriced or not billed."""
    price = settings.BRAVE_USD_PER_REQUEST
    return float(price) if served and price is not None else None


class BraveSearch(SearchProvider):
    """Brave Search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        # The lower-cased provider name is the configured budget/health key.
        return "Brave"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Brave Search API."""
        self.last_status = "started"
        self.last_reason_code = None
        self.last_cost_usd = None
        served = False
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(num_results, 20),
            **_REQUEST_SETTINGS,
        }
        try:
            resp = httpx.get(_BRAVE_URL, headers=headers, params=params, timeout=remaining(15))
            resp.raise_for_status()
            # A request Brave served is billed whatever its content; an error
            # status is assumed unbilled (the pricing page does not say).
            served = True
            data = resp.json()
            web_results, self.last_reason_code = _web_results(data, query)
        except Exception as e:
            self.last_cost_usd = _request_cost(served)
            self.last_status = classify_search_failure(e)
            query_sha256, failure = safe_search_failure_log(query, e)
            self.last_reason_code = str(e) if isinstance(e, _ResponseContractError) else failure
            logger.warning(
                "Brave search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        self.last_cost_usd = _request_cost(served)
        results = []
        # Brave returns results in web.results
        for item in web_results[:num_results]:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("description") or ""
            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        self.last_status = "completed"
        return results
