"""Bright Data SERP API provider.

A contracted service that returns parsed search-engine results, rather than a
scraper this application operates. That distinction is the reason it is here:
web results are used as negative evidence -- "this search completed and found
nothing" is what licenses a fabrication finding about a student's reference --
so the route producing them has to be one whose terms permit the use and whose
empty answers mean the engine found nothing, not that a limit was reached.

API: POST https://api.brightdata.com/request
The request names a zone (created in the Bright Data console) and the engine
URL to fetch. `brd_json=1` asks Bright Data to parse the page and return
structured results instead of HTML, so this adapter never parses engine markup.

Account credits, quotas and terms change independently of this application;
consult Bright Data's current primary documentation before deployment.
"""

import logging
import time
from urllib.parse import quote_plus

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

_BRIGHTDATA_URL = "https://api.brightdata.com/request"
BRIGHTDATA_RESPONSE_CONTRACT = "brightdata-serp-v1"

# `num`/`count` are rejected by the SERP API and stripped with a warning
# header, so the engine's own default page size is requested and the caller's
# limit is applied to the parsed results instead.
# The documented quickstart sends interface language and result region with
# every query. Without them the engine infers both, so identical queries
# can return different result sets from one call to the next.
_ENGINE_SEARCH_URL = {
    "google": "https://www.google.com/search?q={query}&hl={hl}&gl={gl}&brd_json=1",
    "bing": "https://www.bing.com/search?q={query}&setlang={hl}&cc={gl}&brd_json=1",
}

# Failed requests are not billed ("pay only for successful delivery"), so a
# bounded retry costs nothing and addresses the failure actually observed:
# a transport 200 carrying an upstream 502. Measured over a full paper,
# 42 queries completed 13 times without one.
_UNBILLED_RETRY_ATTEMPTS = 3
# Gateway codes only: the request arrived and their collection failed, so the
# same request may succeed moments later. A 429 is deliberately absent.
# Measured on 2026-09-23 over the same 42-query paper: retrying rate limits
# in-search took completion *down* from 88.1% to 78.6% and turned four 429s
# into eight. The limit is account-level and sustained, so a pause measured in
# seconds cannot clear it, and the faster responses only raised the request
# rate into it. A 429 now surfaces immediately as `rate_limited`, which
# suspends the provider through the health store -- the mechanism that can
# actually wait the limit out.
_RETRYABLE_STATUS = frozenset({502, 503, 504})
_RETRYABLE_UPSTREAM = frozenset(str(code) for code in _RETRYABLE_STATUS)
# Pauses between gateway retries, per attempt, bounded by the budget below.
_RETRY_BACKOFF_SECONDS = (1.0, 3.0)

# Bright Data answers with transport 200 even when the upstream fetch failed,
# reporting the real outcome in this header. Trusting the transport status
# would turn a failed fetch into "the search completed and found nothing",
# which is the one misreading this application cannot afford.
_UPSTREAM_STATUS_HEADER = "x-brd-status-code"


class _ResponseContractError(ValueError):
    """Application-owned reason codes, never provider response contents."""

    def __init__(self, reason: str, *, operational_status: str | None = None):
        super().__init__(reason)
        # What the failure *was*, when the reason code names a real upstream
        # status. Bright Data answers a rate limit with transport 200 and the
        # true code in a header, so without this a 429 is classified
        # `response_invalid` -- a malformed body -- and the caller's cooldown
        # and incident handling never engages on the one failure it exists for.
        self.operational_status = operational_status


# The upstream code decides what the failure was, not what the body looked
# like. These are the statuses the caller acts on differently.
_UPSTREAM_OPERATIONAL_STATUS = {
    "429": "rate_limited", "432": "rate_limited",
    "401": "access_restricted", "403": "access_restricted",
}


def _organic_results(data: object) -> tuple[list[dict], str]:
    """Validate the parsed envelope and return its organic results.

    A malformed body is an operational failure, never an empty search: the
    difference decides whether the application may treat the answer as
    evidence that nothing was found.
    """
    if not isinstance(data, dict):
        raise _ResponseContractError("brightdata_serp_v1_invalid_envelope")
    if data.get("error") or data.get("errors"):
        raise _ResponseContractError("brightdata_serp_v1_error_envelope")
    # Bright Data's parsers have used both keys for the organic block.
    results = data.get("organic")
    if results is None:
        results = data.get("results")
    if results is None:
        # A page with genuinely no matches omits the organic block entirely and
        # reports the count instead. That is a certified empty search and may
        # stand as evidence that nothing was found; a page missing the block
        # for any other reason may not, and is an operational failure.
        general = data.get("general")
        stated = general.get("results_cnt") if isinstance(general, dict) else None
        if stated == 0:
            return [], "brightdata_serp_v1_zero_results"
        raise _ResponseContractError("brightdata_serp_v1_missing_organic")
    if not isinstance(results, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("link") or item.get("url"), str)
        or not (item.get("link") or item.get("url")).strip()
        for item in results
    ):
        raise _ResponseContractError("brightdata_serp_v1_invalid_results")
    return results, (
        "brightdata_serp_v1_results" if results else "brightdata_serp_v1_empty"
    )


class BrightDataSearch(SearchProvider):
    """Bright Data SERP API implementation."""

    def __init__(
        self,
        api_token: str,
        zone: str,
        *,
        engine: str = "google",
        timeout_seconds: float = 30.0,
        language: str = "en",
        region: str = "us",
    ):
        self._api_token = api_token
        self._zone = zone
        self._engine = engine if engine in _ENGINE_SEARCH_URL else "google"
        self._timeout_seconds = timeout_seconds
        self._language = language
        self._region = region

    @property
    def name(self) -> str:
        # The lower-cased provider name is the configured budget/health key.
        return "BrightData"

    def _collect(self, body: dict) -> tuple[list, str]:
        """One search, retried while the upstream fetch fails.

        A retry is free: Bright Data bills successful delivery only, and the
        observed failure is a transport 200 carrying an upstream 502 -- the
        request reached them and their own collection failed. Without this,
        a full paper completed 13 of 42 queries. Contract failures are not
        retried: a malformed body will not become well-formed.
        """
        # The configured timeout is the budget for the whole search, retries
        # and pauses included. Spending it per attempt instead would make
        # three attempts cost three times the number in the configuration,
        # which is not what the setting says and not what a caller waiting on
        # it expects.
        started = time.monotonic()
        resp: httpx.Response | None = None
        upstream = ""
        for attempt in range(1, _UNBILLED_RETRY_ATTEMPTS + 1):
            budget_left = self._timeout_seconds - (time.monotonic() - started)
            if budget_left <= 0:
                break
            resp = httpx.post(
                _BRIGHTDATA_URL,
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=min(remaining(self._timeout_seconds), budget_left),
            )
            upstream = (resp.headers.get(_UPSTREAM_STATUS_HEADER) or "").strip()
            retryable = (resp.status_code in _RETRYABLE_STATUS
                         or upstream in _RETRYABLE_UPSTREAM
                         or (resp.status_code == 200 and not resp.content))
            if retryable and attempt < _UNBILLED_RETRY_ATTEMPTS:
                pause = _RETRY_BACKOFF_SECONDS[attempt - 1]
                # Never let a retry outlive the request deadline: a search that
                # returns after the caller has given up is a cost with no
                # answer. Stop and report the failure it already has.
                left = self._timeout_seconds - (time.monotonic() - started)
                if min(remaining(self._timeout_seconds), left) <= pause:
                    break
                self.last_reason_code = (
                    f"brightdata_serp_v1_retry_{upstream or resp.status_code}")
                time.sleep(pause)
                continue
            resp.raise_for_status()
            if upstream and not upstream.startswith("2"):
                raise _ResponseContractError(
                    f"brightdata_serp_v1_upstream_status_{upstream}",
                    operational_status=_UPSTREAM_OPERATIONAL_STATUS.get(upstream))
            if not resp.content:
                raise _ResponseContractError("brightdata_serp_v1_empty_body")
            # Delivered: Bright Data bills this request even if its content
            # then fails our parser. Retried failures above are not billed.
            self._delivered = True
            return _organic_results(resp.json())
        # Reached only by the deadline break above; every other path returned
        # or raised. Report the failure actually observed, not a generic one.
        if resp is None:
            # The budget was gone before a request was even sent.
            raise _ResponseContractError("brightdata_serp_v1_no_budget_remaining")
        code = upstream or str(resp.status_code)
        raise _ResponseContractError(
            f"brightdata_serp_v1_upstream_status_{upstream}" if upstream
            else f"brightdata_serp_v1_status_{resp.status_code}",
            operational_status=_UPSTREAM_OPERATIONAL_STATUS.get(code))

    def _request_cost(self) -> float | None:
        price = settings.BRIGHTDATA_USD_PER_REQUEST
        return float(price) if getattr(self, "_delivered", False) and price is not None else None

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via the Bright Data SERP API."""
        self.last_status = "started"
        self.last_reason_code = None
        # Cost is the configured per-request price of a delivered request;
        # without a price it stays unknown, never zero.
        self.last_cost_usd = None
        self._delivered = False
        target = _ENGINE_SEARCH_URL[self._engine].format(
            query=quote_plus(query), hl=self._language, gl=self._region)
        body = {
                    "zone": self._zone,
                    "url": target,
                    "format": "raw",
                    # Never let the provider rewrite the query. Bright Data's
                    # optional optimization adjusts "some queries ... while
                    # others are sent unchanged", and its behaviour may evolve.
                    # This application records a query hash beside its result
                    # and may treat a completed empty search as evidence that a
                    # reference was not found, so the query executed has to be
                    # the query recorded. Sent explicitly so a console default
                    # cannot silently change what was searched.
                    "search_rewrite": False,
        }
        if self._region:
            # Pin the collection region instead of letting the zone
            # auto-select one, so repeated queries are comparable.
            body["country"] = self._region
        try:
            organic, self.last_reason_code = self._collect(body)
        except Exception as e:
            self.last_cost_usd = self._request_cost()
            self.last_status = (
                getattr(e, "operational_status", None) or classify_search_failure(e))
            query_sha256, failure = safe_search_failure_log(query, e)
            self.last_reason_code = (
                str(e) if isinstance(e, _ResponseContractError) else failure
            )
            logger.warning(
                "Bright Data search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        self.last_cost_usd = self._request_cost()
        results = []
        for item in organic[:num_results]:
            url = (item.get("link") or item.get("url") or "").strip()
            title = item.get("title") or ""
            snippet = item.get("description") or item.get("snippet") or ""
            if url:
                results.append(SearchResult(
                    url=url,
                    title=title if isinstance(title, str) else "",
                    snippet=snippet if isinstance(snippet, str) else "",
                    is_pdf=url.lower().endswith(".pdf"),
                ))
        self.last_status = "completed"
        return results
