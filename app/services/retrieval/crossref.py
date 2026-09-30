"""Crossref retrieval adapter (metadata only).

Crossref returns rich bibliographic metadata but NO full text. Used as a
last resort for metadata verification, and for book lookups (editor vs
author roles -> monograph vs edited-collection detection).
"""

import html
import logging

import httpx

from app.config import settings
from app.log_safety import safe_exception_code
from app.services.doi_cache import doi_request_segment
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
)
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"


class CrossrefRetriever(RetrievalSource):
    # A principal scholarly index. Its silence is part of what makes
    # "not found" mean anything, so its failure must block that claim.
    required_for_search_completion = True
    name = "crossref"
    capabilities = frozenset({"doi", "title_author", "isbn", "metadata", "abstract", "locations", "metadata_only_search"})
    documentation_url = "https://api.crossref.org/swagger-ui/index.html"
    default_policy = ProviderPolicy(timeout_seconds=15.0)

    def __init__(self) -> None:
        self.policy = provider_policy(self.name, self.default_policy)
        self.provider_metrics = {
            "calls": 0,
            "successes": 0,
            "not_found": 0,
            "client_errors": 0,
            "server_errors": 0,
            "network_errors": 0,
        }

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        self.provider_metrics["calls"] += 1
        record_provider_request("crossref")
        try:
            resp = httpx.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.policy.timeout_seconds,
            )
        except httpx.HTTPError:
            self.provider_metrics["network_errors"] += 1
            raise
        if resp.status_code == 404:
            self.provider_metrics["not_found"] += 1
        elif 400 <= resp.status_code < 500:
            self.provider_metrics["client_errors"] += 1
        elif resp.status_code >= 500:
            self.provider_metrics["server_errors"] += 1
        else:
            self.provider_metrics["successes"] += 1
        return resp

    def _headers(self) -> dict:
        email = settings.CROSSREF_EMAIL or settings.OPENALEX_EMAIL or "support@sourcefidelity.org"
        return {"User-Agent": f"SourceFidelity/{settings.APP_VERSION} (mailto:{email})"}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        segment = doi_request_segment(str(doi or ""))
        if segment is None:
            return RetrievalResult(source_name=self.name, success=False, error="No DOI supplied")
        url = f"{CROSSREF_BASE}/works/{segment}"
        try:
            resp = self._get(url)
            if resp.status_code == 404:
                return RetrievalResult(source_name=self.name, success=False, error="Not found",
                    metadata={"identifier_check": "not_registered", "identity_search_result_count": 0})
            resp.raise_for_status()
            data = resp.json()
            return self._parse_message(data.get("message", {}))
        except Exception as e:
            error = safe_exception_code(e)
            logger.warning("Crossref DOI search failed (type=%s)", type(e).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        try:
            params: dict = {"query.title": title, "rows": 1}
            if author:
                params["query.author"] = author
            url = f"{CROSSREF_BASE}/works"
            resp = self._get(url, params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items")
            if not isinstance(items, list):
                raise ValueError("Invalid Crossref result list")
            if not items:
                return RetrievalResult(source_name=self.name, success=False, error="No results",
                    metadata={"identity_search_result_count": 0} if data.get("status") == "ok" else {})
            from app.services.reference_review_scope import screen_metadata
            results = [self._parse_message(item) for item in items]
            result = results[0]
            result.metadata['bounded_review_screen'] = screen_metadata(title, author, results)
            result.metadata['identity_search_result_count'] = len(items)
            return result
        except Exception as e:
            error = safe_exception_code(e)
            logger.warning("Crossref title search failed (type=%s)", type(e).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def search_by_isbn(self, isbn: str) -> RetrievalResult:
        """Search for a book by ISBN.

        Returns contributor roles (editor vs author) for monograph vs
        edited-collection detection.
        """
        url = f"{CROSSREF_BASE}/works"
        params = {"filter": f"isbn:{isbn}", "rows": 1}
        try:
            resp = self._get(url, params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                return RetrievalResult(source_name=self.name, success=False, error="Not found")
            return self._parse_message(items[0])
        except Exception as e:
            error = safe_exception_code(e)
            logger.warning("Crossref ISBN search failed (type=%s)", type(e).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def _parse_message(self, msg: dict) -> RetrievalResult:
        """Parse a Crossref work message."""
        doi = msg.get("DOI")

        title = ""
        titles = msg.get("title", []) or []
        if titles:
            title = html.unescape(str(titles[0]))

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
        locations = _parse_full_text_links(msg)

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=locations[0].url if locations else None,
            locations=locations,
            abstract=abstract,
            metadata={
                "message": msg,
                "editors": editors,
                "publisher": publisher,
                "work_type": msg.get("type"),
                "page": msg.get("page"),
                # The journal, volume and issue identify an article the
                # way the publisher identifies a book. They were in the
                # retained message but never surfaced, so every observed
                # container and volume was empty and the comparison
                # resolved `unknown` for want of a located value.
                "container_title": _first_container(msg),
                "volume": _scalar(msg.get("volume")),
                "issue": _scalar(msg.get("issue")),
            },
        )


def _scalar(value) -> str:
    """Crossref scalars only; a list or object is not a volume."""
    return value.strip() if isinstance(value, str) and len(value) <= 80 else ""


def _first_container(msg: dict) -> str:
    """The deposited container title, when there is exactly one.

    Crossref deposits a list. More than one entry means the record is
    ambiguous about what contains the work, which is not a value to
    compare a student's reference against.
    """
    titles = msg.get("container-title")
    if isinstance(titles, list) and len(titles) == 1:
        return html.unescape(_scalar(titles[0]))[:1000]
    return ""


def _parse_full_text_links(msg: dict) -> list[AcquisitionLocation]:
    """Preserve Crossref-deposited full-text/TDM URLs and their use metadata."""
    locations: list[AcquisitionLocation] = []
    seen: set[str] = set()
    licenses = [
        item.get("URL") for item in (msg.get("license") or [])
        if isinstance(item, dict) and item.get("URL")
    ]
    for link in msg.get("link") or []:
        url = link.get("URL") if isinstance(link, dict) else None
        if not url or url in seen:
            continue
        seen.add(url)
        media_type = (link.get("content-type") or "").lower() or None
        if media_type and "pdf" in media_type:
            kind = RepresentationKind.PDF
        elif media_type and "xml" in media_type:
            kind = RepresentationKind.XML
        elif media_type and ("html" in media_type or "xhtml" in media_type):
            kind = RepresentationKind.HTML
        elif media_type and "text/plain" in media_type:
            kind = RepresentationKind.PLAIN_TEXT
        else:
            kind = None
        locations.append(
            AcquisitionLocation(
                url=url,
                provider="crossref",
                media_type=media_type,
                representation_kind=kind,
                version=link.get("content-version"),
                license=licenses[0] if licenses else None,
                intended_application=link.get("intended-application"),
                access_type="tdm" if link.get("intended-application") == "text-mining" else None,
                metadata={"licenses": licenses},
            )
        )
    return locations


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
