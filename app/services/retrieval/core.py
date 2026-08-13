"""CORE retrieval adapter.

CORE (https://core.ac.uk) aggregates full-text content from 10,000+
repositories. Requires a free API key (CORE_API_KEY).
"""

import logging
import re
import threading
import time

import httpx

from app.config import settings
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

CORE_BASE = "https://api.core.ac.uk/v3"
# Trailing slash matters: without it, /v3/search/outputs 301-redirects to the
# slashed form, costing an extra round-trip on every call.
SEARCH_URL = f"{CORE_BASE}/search/outputs/"

# CORE enforces two limits that bite under concurrent access:
#  (1) ~5-10 single requests per 10-second window per key
#      (https://core.ac.uk/documentation/api) — exceeded → 429.
#  (2) a per-key CONCURRENCY cap: when 2+ requests are in-flight at once, the
#      server stalls one until ~15s and our read timeout fires. Measured:
#      the same 4 queries sequential = all succeed 1.8-2.6s; 4-concurrent =
#      1 ReadTimeout @ 21s.
# So we need BOTH rate-limiting AND full serialization. Holding the lock for
# the whole request (not just the start) makes CORE calls strictly serial; the
# min-interval wait inside the lock keeps us under the per-window quota.
_CORE_MIN_INTERVAL = 1.2  # seconds between request starts (~8/10s, under the limit)
_core_lock = threading.Lock()
_core_last_request = 0.0


def _core_request(url: str, params: dict, timeout: float = 15) -> "httpx.Response":
    """Make a CORE API request: rate-limited AND fully serialized (thread-safe).

    Acquires the global lock, waits out the min-interval since the last request,
    then performs the call before releasing — so no two CORE requests ever
    overlap and the per-window quota is respected.
    """
    global _core_last_request
    with _core_lock:
        now = time.monotonic()
        wait = _CORE_MIN_INTERVAL - (now - _core_last_request)
        if wait > 0:
            time.sleep(wait)
        _core_last_request = time.monotonic()
        return httpx.get(url, params=params, timeout=timeout, follow_redirects=True)


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
        try:
            # v3 moved DOI lookup into the `q` DSL: `doi` as a bare query param
            # is silently IGNORED, which degrades to an unfiltered ~482M-result
            # query — slow under load (read timeouts) AND returns the wrong
            # paper. `q=doi:<doi>` is the correct v3 form (returns the right
            # paper, ~1-2s, single-digit hits).
            params = {"q": f"doi:{doi}", "limit": 1}
            resp = _core_request(SEARCH_URL, params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return RetrievalResult(source_name=self.name, success=False, error="No results")
            return self._parse_output(results[0])
        except Exception as e:
            logger.warning("CORE DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

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
            resp = _core_request(SEARCH_URL, params)
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
            logger.warning("CORE title search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

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
        download_url = data.get("downloadUrl") or (source_urls[0] if source_urls else None)

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=download_url,
            metadata=data,
        )
