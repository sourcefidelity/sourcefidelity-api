"""Elsevier Article Retrieval adapter.

Retrieves article text and metadata from Elsevier (ScienceDirect) via the
Article Retrieval API. This adapter solves the Elsevier gap: ScienceDirect
uses PII (not DOI) in its web URLs, which blocked the publisher-PDF constructor.
The Article Retrieval API accepts DOI directly.

Access levels (entitlement model):
  - API key alone:      full text for OA articles; metadata + abstract for all
  - API key + insttoken: full text for paywalled articles (if institution subscribes)

Two-step workflow:
  1. Fetch view=META_ABS → metadata + abstract + openaccess flag
  2. If OA or insttoken configured → fetch view=FULL → complete article text (XML)

The API returns XML (JATS-style), not PDF. We strip the XML tags to get clean
text, reusing the same _strip_jats_xml approach as the Crossref adapter.

Auth: X-ELS-APIKey header (required) + X-ELS-Insttoken header (optional).
Rate limits: per-API-key, ~weekly quota. Fine for per-article verification.
"""

import logging
import re

import httpx

from app.config import settings
from app.services.relevance import score_relevance
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

ELSEVIER_BASE = "https://api.elsevier.com/content/article"


class ElsevierRetriever(RetrievalSource):
    """Retrieves articles from Elsevier via the Article Retrieval API."""

    name = "elsevier"

    def _headers(self) -> dict:
        headers = {
            "X-ELS-APIKey": settings.ELSEVIER_API_KEY,
            "Accept": "application/json",
            "User-Agent": "SourceFidelity/0.1.0 (source-verification-bot)",
        }
        if settings.ELSEVIER_INST_TOKEN:
            headers["X-ELS-Insttoken"] = settings.ELSEVIER_INST_TOKEN
        return headers

    def search_by_doi(self, doi: str) -> RetrievalResult:
        if not settings.ELSEVIER_API_KEY:
            return RetrievalResult(
                source_name=self.name, success=False, error="No ELSEVIER_API_KEY configured"
            )

        try:
            # Step 1: Fetch metadata + abstract to check OA status
            resp = httpx.get(
                f"{ELSEVIER_BASE}/doi/{doi}",
                headers=self._headers(),
                params={"view": "META_ABS"},
                timeout=20,
            )
            if resp.status_code == 401:
                return RetrievalResult(source_name=self.name, success=False, error="Unauthorized (invalid API key)")
            if resp.status_code == 404:
                return RetrievalResult(source_name=self.name, success=False, error="Not found in Elsevier")
            resp.raise_for_status()

            data = resp.json()
            coredata = data.get("full-text-retrieval-response", {}).get("coredata", {})
            if not coredata:
                # Some responses nest differently
                coredata = data.get("coredata", data)

            is_oa = (
                coredata.get("openaccessArticle") is True
                or str(coredata.get("openaccess", "")).lower() in ("true", "1", "yes")
            )

            # Parse metadata
            title = coredata.get("dc:title") or coredata.get("title", "")
            doi_found = coredata.get("prism:doi") or coredata.get("doi") or doi
            year_raw = coredata.get("prism:coverDate") or coredata.get("publicationDate", "")
            year = year_raw[:4] if year_raw else "n.d."

            # Authors
            authors = []
            for a in coredata.get("dc:creator", []):
                name = a.get("$", "") if isinstance(a, dict) else str(a)
                if name:
                    authors.append(name)

            # Abstract
            abstract_raw = coredata.get("dc:description") or coredata.get("abstract", "")
            abstract = _strip_xml_tags(abstract_raw) if abstract_raw else None

            # Step 2: Attempt full text.
            # - OA articles: always entitled (API key alone).
            # - Paywalled articles: on a campus network, IP grants access
            #   (no insttoken needed). Off-campus without insttoken, the API
            #   returns 401/403 and we fall back to abstract.
            # So we always TRY the FULL view and let the API decide entitlement.
            full_text = self._fetch_full_text(doi)
            if full_text:
                logger.info(
                    "Retrieved Elsevier FULL text for %s (oa=%s, insttoken=%s)",
                    doi, is_oa, bool(settings.ELSEVIER_INST_TOKEN),
                )

            return RetrievalResult(
                source_name=self.name,
                success=True,
                doi=doi_found,
                title=title,
                year=year,
                authors=authors,
                abstract=abstract,
                full_text=full_text.encode("utf-8") if full_text else None,
                metadata={
                    "coredata": coredata,
                    "is_oa": is_oa,
                    "pii": coredata.get("pii"),
                },
            )
        except httpx.HTTPStatusError as e:
            logger.warning("Elsevier DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e)[:100])
        except Exception as e:
            logger.warning("Elsevier DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e)[:100])

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        # Elsevier doesn't have a clean title-search endpoint for the Article
        # Retrieval API (the ScienceDirect Search API is separate and requires
        # its own entitlement). For title-based lookup, defer to other sources
        # (OpenAlex/CORE/S2 can find Elsevier articles by title, then this
        # adapter retrieves them by DOI).
        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="Elsevier title search not supported (use DOI via OpenAlex/CORE first)",
        )

    def _fetch_full_text(self, doi: str) -> str | None:
        """Fetch the full text (view=FULL) for an article.

        Returns clean text (XML stripped), or None if not entitled.
        """
        try:
            resp = httpx.get(
                f"{ELSEVIER_BASE}/doi/{doi}",
                headers=self._headers(),
                params={"view": "FULL", "httpAccept": "text/xml"},
                timeout=30,
            )
            if resp.status_code in (401, 403):
                logger.info("Elsevier FULL view not entitled for %s (status %d)", doi, resp.status_code)
                return None
            resp.raise_for_status()

            # The response is XML; extract the article body text
            text = _extract_article_text(resp.text)
            return text if text and len(text) > 100 else None
        except Exception as e:
            logger.debug("Elsevier full text fetch failed for %s: %s", doi, e)
            return None


def _strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_article_text(xml: str) -> str:
    """Extract readable article text from Elsevier's XML response.

    Elsevier's FULL view returns JATS-style XML. The main body is typically in
    <ce:para> or <body> tags. We strip all tags and return the text.
    """
    # Remove XML declaration and DOCTYPE
    text = re.sub(r"<\?xml[^>]*\?>", "", xml)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)

    # Try to extract just the body (between <body> tags if present)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", text, re.DOTALL | re.IGNORECASE)
    if body_match:
        text = body_match.group(1)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
