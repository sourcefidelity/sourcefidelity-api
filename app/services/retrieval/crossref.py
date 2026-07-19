"""Crossref retrieval adapter (metadata only).

Crossref returns rich bibliographic metadata but NO full text. Used as a
last resort for metadata verification, and for book lookups (editor vs
author roles -> monograph vs edited-collection detection).
"""

import logging

import httpx

from app.config import settings
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"


class CrossrefRetriever(RetrievalSource):
    name = "crossref"

    def _headers(self) -> dict:
        email = settings.CROSSREF_EMAIL or settings.OPENALEX_EMAIL or "support@sourcefidelity.org"
        return {"User-Agent": f"SourceFidelity/{settings.APP_VERSION} (mailto:{email})"}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        url = f"{CROSSREF_BASE}/works/{doi}"
        try:
            resp = httpx.get(url, headers=self._headers(), timeout=15)
            if resp.status_code == 404:
                return RetrievalResult(source_name=self.name, success=False, error="Not found")
            resp.raise_for_status()
            data = resp.json()
            return self._parse_message(data.get("message", {}))
        except Exception as e:
            logger.warning("Crossref DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        try:
            params: dict = {"query.title": title, "rows": 1}
            if author:
                params["query.author"] = author
            url = f"{CROSSREF_BASE}/works"
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                return RetrievalResult(source_name=self.name, success=False, error="No results")
            return self._parse_message(items[0])
        except Exception as e:
            logger.warning("Crossref title search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def search_by_isbn(self, isbn: str) -> RetrievalResult:
        """Search for a book by ISBN.

        Returns contributor roles (editor vs author) for monograph vs
        edited-collection detection.
        """
        url = f"{CROSSREF_BASE}/works"
        params = {"filter": f"isbn:{isbn}", "rows": 1}
        try:
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                return RetrievalResult(source_name=self.name, success=False, error="Not found")
            return self._parse_message(items[0])
        except Exception as e:
            logger.warning("Crossref ISBN search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def _parse_message(self, msg: dict) -> RetrievalResult:
        """Parse a Crossref work message."""
        doi = msg.get("DOI")

        title = ""
        titles = msg.get("title", []) or []
        if titles:
            title = titles[0]

        year = "n.d."
        issued = msg.get("issued", {}) or {}
        date_parts = issued.get("date-parts", [[0]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = str(date_parts[0][0])

        authors = []
        for author in msg.get("author", []) or []:
            given = author.get("given", "")
            family = author.get("family", "")
            if given or family:
                authors.append(f"{given} {family}".strip())

        publisher = msg.get("publisher", "")

        # Contributor roles (for monograph vs edited collection detection)
        editors = []
        for ed in msg.get("editor", []) or []:
            given = ed.get("given", "")
            family = ed.get("family", "")
            if given or family:
                editors.append(f"{given} {family}".strip())

        # Extract abstract — Crossref stores it as JATS XML, strip the tags.
        abstract = _strip_jats_xml(msg.get("abstract"))

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=None,  # Crossref is metadata-only
            abstract=abstract,
            metadata={
                "message": msg,
                "editors": editors,
                "publisher": publisher,
            },
        )


def _strip_jats_xml(raw: str | None) -> str | None:
    """Strip JATS XML tags from a Crossref abstract.

    Crossref abstracts look like:
        <jats:title>Abstract</jats:title><jats:p>Text here...</jats:p>
    """
    if not raw:
        return None
    import re
    # Remove all XML tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None
