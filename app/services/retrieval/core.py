"""CORE retrieval adapter.

CORE (https://core.ac.uk) aggregates full-text content from 10,000+
repositories. Requires a free API key (CORE_API_KEY).
"""

import logging

import httpx

from app.config import settings
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

CORE_BASE = "https://api.core.ac.uk/v3"


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
            url = f"{CORE_BASE}/search/outputs"
            params = {"doi": doi, "limit": 1}
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15, follow_redirects=True)
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
            q = f'title:"{title}"'
            url = f"{CORE_BASE}/search/outputs"
            # Fetch a few candidates and apply a relevance filter, same as
            # OpenAlex — CORE also returns keyword-coincidence matches.
            params = {"q": q, "limit": 5}
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15, follow_redirects=True)
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
