"""Project Gutenberg retrieval through its official OPDS catalog."""

from io import BytesIO
import logging
import re
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup
import httpx

from app.log_safety import safe_exception_code
from app.services.relevance import extract_surnames, score_relevance
from app.services.retrieval.base import (
    BOOK_SOURCE_KINDS,
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
    SourceRepresentation,
)
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

GUTENBERG_BASE = "https://www.gutenberg.org"
GUTENBERG_SEARCH = f"{GUTENBERG_BASE}/ebooks/search.opds/"
_HEADERS = {
    "User-Agent": "SourceFidelity/0.1 (https://github.com/sourcefidelity/sourcefidelity-api)"
}
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dcterms": "http://purl.org/dc/terms/",
}
_ACQUISITION_REL = "http://opds-spec.org/acquisition"
_START_MARKER = re.compile(
    r"\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*{3}", re.IGNORECASE
)
_END_MARKER = re.compile(
    r"\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*{3}", re.IGNORECASE
)


class GutenbergRetriever(RetrievalSource):
    """Retrieve public-domain-in-the-USA editions from Gutenberg's OPDS feed."""
    # Unchanged for now. A public-domain archive failing says nothing about
    # a modern book, which argues for False; the kind and year bounds now
    # keep it from being consulted for one at all.
    required_for_search_completion = True
    # Public-domain book texts. The year bound lives in the resolver.
    supported_source_kinds = BOOK_SOURCE_KINDS

    name = "gutenberg"
    capabilities = frozenset({"title_search", "full_text", "public_domain"})
    documentation_url = "https://www.gutenberg.org/ebooks/offline_catalogs.html"

    def __init__(self) -> None:
        self.policy = provider_policy(
            self.name,
            ProviderPolicy(timeout_seconds=30, min_interval_seconds=2),
        )

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name, success=False, error="Gutenberg does not use DOIs"
        )

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        if not self.policy.enabled:
            return RetrievalResult(source_name=self.name, success=False, error="Provider disabled")
        query = title
        surname = (extract_surnames(author) or [""])[0] if author else ""
        if surname:
            query = f"{title} {surname}"
        try:
            record_provider_request("gutenberg")
            response = httpx.get(
                GUTENBERG_SEARCH,
                params={"query": query},
                headers=_HEADERS,
                timeout=self.policy.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            candidates = _parse_search_feed(response.content)
            for candidate in candidates[:5]:
                relevance = score_relevance(
                    title, candidate["title"], author, candidate["authors"]
                )
                if not relevance.is_relevant:
                    continue
                return self._fetch_edition(candidate["item_url"])
            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {min(5, len(candidates))} results rejected)",
            )
        except Exception as exc:
            error = safe_exception_code(exc)
            logger.warning("Gutenberg OPDS search failed (type=%s)", type(exc).__name__)
            return RetrievalResult(source_name=self.name, success=False, error=error)

    def _fetch_edition(self, item_url: str) -> RetrievalResult:
        record_provider_request("gutenberg")
        response = httpx.get(
            item_url,
            headers=_HEADERS,
            timeout=self.policy.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        edition = _parse_item_feed(response.content)
        if not edition:
            return RetrievalResult(
                source_name=self.name, success=False, error="No downloadable OPDS edition"
            )
        return RetrievalResult(
            source_name=self.name,
            success=True,
            title=edition["title"],
            authors=edition["authors"],
            year=edition["published_year"],
            full_text_url=edition["acquisition_url"],
            locations=[
                AcquisitionLocation(
                    url=edition["acquisition_url"],
                    provider=self.name,
                    media_type="application/epub+zip",
                    representation_kind=RepresentationKind.EPUB,
                    license=edition["rights"],
                    access_type="public_domain",
                    is_best=True,
                )
            ],
            metadata={
                "catalog_url": item_url,
                "rights": edition["rights"],
                "license_class": "public_domain",
                "rights_jurisdiction": "USA",
                "format": edition["format"],
            },
        )

    def download_full_text(self, result: RetrievalResult) -> RetrievalResult:
        """Download the OPDS-advertised EPUB and extract its readable text."""
        if not result.full_text_url:
            return result
        try:
            record_provider_request("gutenberg")
            response = httpx.get(
                result.full_text_url,
                headers=_HEADERS,
                timeout=max(60, self.policy.timeout_seconds),
                follow_redirects=True,
            )
            response.raise_for_status()
            clean = _extract_epub_text(response.content)
            if len(clean) < 200:
                raise ValueError("Downloaded edition did not contain enough readable text")
            result.set_representation(
                SourceRepresentation(
                    kind=RepresentationKind.PLAIN_TEXT,
                    media_type="text/plain",
                    content=clean.encode("utf-8"),
                    source_url=result.full_text_url,
                    original_kind=RepresentationKind.EPUB,
                    charset="utf-8",
                    completeness="complete",
                )
            )
        except Exception as exc:
            logger.warning("Gutenberg acquisition failed (type=%s)", type(exc).__name__)
            result.success = False
            result.error = safe_exception_code(exc)
        return result


def _parse_search_feed(payload: bytes) -> list[dict]:
    root = ET.fromstring(payload)
    candidates: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        title = entry.findtext("atom:title", default="", namespaces=_NS).strip()
        authors = [
            node.text.strip()
            for node in entry.findall("atom:author/atom:name", _NS)
            if node.text and node.text.strip()
        ]
        if not authors:
            content = entry.findtext("atom:content", default="", namespaces=_NS).strip()
            if content and not re.fullmatch(r"\d+ downloads?", content, re.IGNORECASE):
                authors = [part.strip() for part in content.split(" and ") if part.strip()]
        link = next(
            (
                node.get("href")
                for node in entry.findall("atom:link", _NS)
                if node.get("rel") == "subsection"
            ),
            None,
        )
        if title and link:
            candidates.append(
                {"title": title, "authors": authors, "item_url": urljoin(GUTENBERG_BASE, link)}
            )
    return candidates


def _parse_item_feed(payload: bytes) -> dict | None:
    root = ET.fromstring(payload)
    for entry in root.findall("atom:entry", _NS):
        rights = entry.findtext("atom:rights", default="", namespaces=_NS).strip()
        acquisitions = [
            node
            for node in entry.findall("atom:link", _NS)
            if node.get("rel") == _ACQUISITION_REL
        ]
        preferred = next(
            (
                link for link in acquisitions
                if link.get("type") == "application/epub+zip"
                and "noimages" in (link.get("href") or "")
            ),
            next((link for link in acquisitions if link.get("type") == "application/epub+zip"), None),
        )
        if preferred is None:
            continue
        published = entry.findtext("atom:published", default="", namespaces=_NS)
        return {
            "title": entry.findtext("atom:title", default="", namespaces=_NS).strip(),
            "authors": [
                node.text.strip()
                for node in entry.findall("atom:author/atom:name", _NS)
                if node.text and node.text.strip()
            ],
            "published_year": published[:4] or None,
            "rights": rights,
            "format": preferred.get("type"),
            "acquisition_url": urljoin(GUTENBERG_BASE, preferred.get("href") or ""),
        }
    return None


def _extract_epub_text(payload: bytes) -> str:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            for name in archive.namelist():
                if not name.lower().endswith((".xhtml", ".html", ".htm")):
                    continue
                soup = BeautifulSoup(archive.read(name), "xml")
                for node in soup(["script", "style", "nav"]):
                    node.decompose()
                text = soup.get_text("\n", strip=True)
                if text:
                    parts.append(text)
    except BadZipFile as exc:
        raise ValueError("Gutenberg acquisition was not a valid EPUB") from exc
    return _strip_boilerplate("\n\n".join(parts))


def _strip_boilerplate(raw_text: str) -> str:
    start_match = _START_MARKER.search(raw_text)
    end_match = _END_MARKER.search(raw_text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)
    return raw_text[start_idx:end_idx].strip()
