"""Qualify an operator-supplied saved HTML page as a source representation.

A saved page is evidence the operator already holds, not a publisher response.
This module therefore performs no network access at all: it parses the supplied
bytes, applies the same observed-identity, work-type and article-coverage checks
the ordinary web path uses, and yields the extracted article text. Nothing here
grants admission; the caller applies the existing scope, retention and
acceptance rules.

The raw HTML is never stored, served or rendered. Only the extracted plain text
becomes a representation, so embedded scripts, frames and remote resources are
never executed or fetched. Observed active-content markers are recorded as
inspectable evidence rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from app.services.retrieval.base import RetrievalResult
from app.services.source_type import (
    SourceKindAssessment,
    classify_content_source_kind,
    classify_html_source_kind,
    compare_source_kinds,
)
from app.services.web_completeness import (
    assess_web_completeness,
    extract_complete_article_body,
)
from app.services.web_source_metadata import extract_web_source_metadata

SUPPLIED_HTML_QUALIFICATION_VERSION = "supplied-html-intake-v1"

# Markers recorded for inspection. Their presence is expected on ordinary
# publisher pages and is not a rejection reason, because the supplied bytes are
# parsed offline and only the extracted text is retained.
_ACTIVE_CONTENT_MARKERS = (
    (re.compile(r"<script\b", re.I), "script element"),
    (re.compile(r"<iframe\b", re.I), "iframe element"),
    (re.compile(r"<(?:object|embed|applet)\b", re.I), "embedded object"),
    (re.compile(r"\son[a-z]{3,20}\s*=", re.I), "inline event handler"),
    (re.compile(r"javascript:", re.I), "javascript URL"),
    (re.compile(r"<meta[^>]+http-equiv\s*=\s*['\"]?refresh", re.I), "meta refresh"),
)

_HTML_DOCUMENT_RE = re.compile(rb"<\s*(?:!doctype\s+html|html|head|body)\b", re.I)


class SuppliedHtmlRejected(ValueError):
    """The supplied page does not qualify as the cited source."""

    def __init__(self, reason_code: str, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.evidence = evidence or {}


@dataclass(frozen=True)
class SuppliedHtmlQualification:
    """Inspectable result of qualifying one supplied page."""

    article_text: str
    observed: dict
    completeness: dict
    identity_confidence: str
    identity_comparisons: list[dict]
    observed_source_kind: str
    observed_source_kind_confidence: str
    source_kind_verdict: str
    html_sha256: str
    extracted_sha256: str
    active_content_markers: tuple[str, ...] = field(default_factory=tuple)

    def as_evidence(self) -> dict:
        """Everything a reviewer needs to re-check this admission."""
        return {
            "version": SUPPLIED_HTML_QUALIFICATION_VERSION,
            "supplied_document": True,
            "network_access": "none",
            "retained_representation": "extracted_plain_text",
            "html_sha256": self.html_sha256,
            "extracted_sha256": self.extracted_sha256,
            "web_identity": self.observed,
            "web_completeness": self.completeness,
            "identity_confidence": self.identity_confidence,
            "identity_comparisons": self.identity_comparisons,
            "observed_source_kind": self.observed_source_kind,
            "observed_source_kind_confidence": self.observed_source_kind_confidence,
            "source_kind_verdict": self.source_kind_verdict,
            "active_content_markers": list(self.active_content_markers),
        }


def looks_like_html_document(content: bytes, media_type: str | None = None) -> bool:
    """Recognize an HTML upload from its own bytes, not only its declared type."""
    if content[:5].lower() == b"%pdf-":
        return False
    if _HTML_DOCUMENT_RE.search(content[:65_536]):
        return True
    return bool(media_type and media_type.split(";")[0].strip().casefold()
                in {"text/html", "application/xhtml+xml"})


def qualify_supplied_html(
    content: bytes,
    *,
    expected_title: str | None = None,
    expected_author: str | None = None,
    expected_year: str | None = None,
    expected_doi: str | None = None,
    expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    reference_url: str | None = None,
) -> SuppliedHtmlQualification:
    """Apply the ordinary observed-identity and coverage checks offline.

    ``reference_url`` is only a parsing hint for relative metadata. It is never
    fetched and never recorded as the representation's source, because the
    supplied bytes are not a response from that address.
    """
    import trafilatura

    if not looks_like_html_document(content):
        raise SuppliedHtmlRejected(
            "not_an_html_document", "Supplied file is not an HTML document"
        )
    html_sha256 = hashlib.sha256(content).hexdigest()
    page_html = content.decode("utf-8", errors="replace")
    markers = tuple(
        label for pattern, label in _ACTIVE_CONTENT_MARKERS if pattern.search(page_html)
    )

    observed = extract_web_source_metadata(page_html, reference_url or "")
    page_text = trafilatura.extract(
        page_html,
        include_comments=False,
        include_links=False,
        include_tables=False,
        favor_recall=True,
    )
    if not page_text or len(page_text) < 100:
        raise SuppliedHtmlRejected(
            "readable_text_unavailable",
            "Supplied page has no readable article text",
            {"html_sha256": html_sha256},
        )
    page_text = extract_complete_article_body(page_html, page_text)
    completeness = assess_web_completeness(page_html, page_text)

    structured_kind = classify_html_source_kind(page_html, reference_url or "")
    content_kind = classify_content_source_kind(page_text, source_url=reference_url)
    observed_kind = (
        content_kind if content_kind.confidence == "high" else structured_kind
    )
    kind_compatibility = compare_source_kinds(expected_source_kind, observed_kind)
    # An operator supplying the file asserts that it is the cited work, so
    # require positive type confirmation rather than mere absence of conflict.
    if expected_source_kind.is_known and kind_compatibility.verdict != "compatible":
        raise SuppliedHtmlRejected(
            "source_kind_unconfirmed",
            "Supplied page work type does not confirm the cited reference — "
            + kind_compatibility.reason,
            {
                "html_sha256": html_sha256,
                "expected_source_kind": expected_source_kind.kind,
                "observed_source_kind": observed_kind.kind,
            },
        )

    # The shared identity standard, unchanged: a supplied file earns no
    # weaker comparison than a fetched page.
    from app.services.source_resolver import _web_identity_comparison

    probe = RetrievalResult(
        source_name="supplied_html",
        success=True,
        title=observed["title"],
        authors=observed["authors"],
        year=observed["year"],
        doi=observed["doi"],
    )
    identity, comparisons, confirmed = _web_identity_comparison(
        probe,
        title=expected_title,
        author=expected_author,
        year=expected_year,
        doi=expected_doi,
    )
    comparison_records = [c.model_dump(mode="json") for c in comparisons]
    if identity.has_material_conflict:
        raise SuppliedHtmlRejected(
            "bibliographic_fields_conflict",
            "Supplied page bibliographic fields conflict with the citation",
            {
                "html_sha256": html_sha256,
                "identity_comparisons": comparison_records,
                "conflicting_fields": [
                    c.field_name for c in comparisons
                    if c.outcome == "material_conflict"
                ],
            },
        )
    if not confirmed:
        raise SuppliedHtmlRejected(
            "bibliographic_identity_unconfirmed",
            "Supplied page bibliographic identity remains unconfirmed",
            {
                "html_sha256": html_sha256,
                "identity_comparisons": comparison_records,
            },
        )
    if completeness["verdict"] != "complete":
        raise SuppliedHtmlRejected(
            "article_body_incomplete",
            "Supplied page does not expose a complete article body",
            {"html_sha256": html_sha256, "web_completeness": completeness},
        )

    return SuppliedHtmlQualification(
        article_text=page_text,
        observed=observed,
        completeness=completeness,
        identity_confidence="high",
        identity_comparisons=comparison_records,
        observed_source_kind=observed_kind.kind,
        observed_source_kind_confidence=observed_kind.confidence,
        source_kind_verdict=kind_compatibility.verdict,
        html_sha256=html_sha256,
        extracted_sha256=hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
        active_content_markers=markers,
    )
