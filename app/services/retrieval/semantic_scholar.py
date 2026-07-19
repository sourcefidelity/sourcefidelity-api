"""Semantic Scholar retrieval adapter.

Rate limit: 1 request per second (cumulative across all endpoints) with an API
key. Without a key, 100 requests / 5 minutes. This adapter enforces the
1-req/sec limit via a module-level throttle shared across all calls, so
multiple S2 requests in a resolution chain don't exceed it.
"""

import logging
import time

import httpx

from app.config import settings
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Minimum seconds between S2 API requests (limit is 1/sec; 0.05 margin).
_MIN_INTERVAL = 1.05

# Module-level throttle state: timestamp of the last S2 request.
_last_request_time: float = 0.0


def _throttle() -> None:
    """Block until at least _MIN_INTERVAL has elapsed since the last S2 request."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


class SemanticScholarRetriever(RetrievalSource):
    name = "semantic_scholar"

    def _headers(self) -> dict:
        headers = {"User-Agent": f"SourceFidelity/{settings.APP_VERSION}"}
        if settings.S2_API_KEY:
            headers["x-api-key"] = settings.S2_API_KEY
        return headers

    def _fields(self) -> str:
        return "title,year,authors,externalIds,openAccessPdf,journal,publicationTypes,abstract"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{doi}"
        params = {"fields": self._fields()}
        try:
            _throttle()
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15)
            if resp.status_code == 404:
                return RetrievalResult(source_name=self.name, success=False, error="Not found")
            resp.raise_for_status()
            data = resp.json()
            return self._parse_paper(data)
        except Exception as e:
            logger.warning("S2 DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        try:
            # Use the regular search endpoint (returns multiple results) rather
            # than /match (single result), so we can apply the relevance filter
            # and reject keyword-coincidence matches.
            q = title
            if author:
                q += f" {author}"
            url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
            params = {"query": q, "fields": self._fields(), "limit": 5}
            _throttle()
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15)
            if resp.status_code == 429:
                return RetrievalResult(source_name=self.name, success=False, error="Rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data") or []
            if not papers:
                return RetrievalResult(source_name=self.name, success=False, error="No match")

            from app.services.relevance import score_relevance

            for paper in papers:
                result = self._parse_paper(paper)
                matched_title = result.title or ""
                matched_authors = result.authors or []
                rel = score_relevance(title, matched_title, author, matched_authors)
                if rel.is_relevant:
                    return result
                logger.debug(
                    "S2 match rejected: %s", rel.detail[:100],
                )

            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {len(papers)} results were keyword coincidences)",
            )
        except Exception as e:
            logger.warning("S2 title search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def _parse_paper(self, data: dict) -> RetrievalResult:
        external = data.get("externalIds", {}) or {}
        doi = external.get("DOI")

        authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")]
        year = str(data.get("year")) if data.get("year") else "n.d."
        title = data.get("title", "")

        oa = data.get("openAccessPdf") or {}
        pdf_url = oa.get("url")

        abstract = data.get("abstract") or None

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=pdf_url,
            abstract=abstract,
            metadata=data,
        )
