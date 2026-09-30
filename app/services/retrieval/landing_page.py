"""Format-neutral discovery and extraction from scholarly landing pages."""

import json
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app.services.retrieval.base import AcquisitionLocation, RepresentationKind


_META_LOCATIONS = {
    "citation_pdf_url": (RepresentationKind.PDF, "application/pdf"),
    "citation_fulltext_html_url": (RepresentationKind.HTML, "text/html"),
    "citation_xml_url": (RepresentationKind.XML, "application/xml"),
}


def discover_scholarly_locations(
    page_html: str,
    page_url: str,
    provider: str = "landing_page",
) -> list[AcquisitionLocation]:
    """Extract reviewed standard full-text signals without guessing URLs."""
    soup = BeautifulSoup(page_html, "lxml")
    locations: list[AcquisitionLocation] = []
    seen: set[str] = set()

    def add(url: str | None, kind: RepresentationKind | None, media_type: str | None,
            signal: str) -> None:
        if not url:
            return
        absolute = urljoin(page_url, url.strip())
        if absolute in seen:
            return
        seen.add(absolute)
        locations.append(
            AcquisitionLocation(
                url=absolute,
                provider=provider,
                representation_kind=kind,
                media_type=media_type,
                landing_page_url=page_url,
                metadata={"discovery_signal": signal},
            )
        )

    for name, (kind, media_type) in _META_LOCATIONS.items():
        node = soup.find("meta", attrs={"name": lambda value: value and value.lower() == name})
        add(node.get("content") if node else None, kind, media_type, name)

    for link in soup.find_all("link", href=True):
        rel = {str(value).lower() for value in (link.get("rel") or [])}
        media_type = (link.get("type") or "").lower()
        if "alternate" not in rel and "enclosure" not in rel:
            continue
        kind = _kind_for_media_type(media_type)
        if kind:
            add(link["href"], kind, media_type, "link_rel")

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
        except (TypeError, json.JSONDecodeError):
            continue
        for item in payload if isinstance(payload, list) else [payload]:
            if not isinstance(item, dict):
                continue
            encoding = item.get("encoding") or item.get("associatedMedia") or []
            for candidate in encoding if isinstance(encoding, list) else [encoding]:
                if not isinstance(candidate, dict):
                    continue
                media_type = candidate.get("encodingFormat") or candidate.get("fileFormat")
                add(
                    candidate.get("contentUrl") or candidate.get("url"),
                    _kind_for_media_type(media_type),
                    media_type,
                    "json_ld",
                )

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = anchor.get_text(" ", strip=True).lower()
        try:
            pdf_path = urlsplit(href).path.lower().endswith(".pdf")
        except ValueError:
            continue
        if pdf_path or "download pdf" in text or text in {"pdf", "view pdf"}:
            add(href, RepresentationKind.PDF, "application/pdf", "anchor_pdf")
    return locations


def _kind_for_media_type(media_type: str | None) -> RepresentationKind | None:
    value = (media_type or "").lower()
    # oEmbed XML/JSON describes an embeddable card for a page. It is metadata,
    # not the scholarly work represented by that page.
    if "oembed" in value or value in {"application/rss+xml", "application/atom+xml", "application/rdf+xml"}:
        return None
    if "pdf" in value:
        return RepresentationKind.PDF
    if "html" in value:
        return RepresentationKind.HTML
    if "xml" in value:
        return RepresentationKind.XML
    if "epub" in value:
        return RepresentationKind.EPUB
    if "text/plain" in value:
        return RepresentationKind.PLAIN_TEXT
    return None
