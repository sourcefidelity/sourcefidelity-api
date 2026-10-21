"""Authorized pages: the one way to read a source's text (authorized-pages-v1).

Owner decisions 2026-10-21 (architecture review candidate 5, ARCHITECTURE §5).
Every reader of source text asks this module for the source's pages instead of
extracting them itself. One fixed order of checks runs on every open:

1. derivative provenance (`_validate_derivative_provenance`);
2. the source's bytes against its content hash;
3. with an artifact: the source against the artifact's source identity, then
   the artifact's extraction version and extracted-text hash, which binds page
   order, labels and text, so a changed page label is also a mismatch.

A source passing them is extracted once and its pages are shared by every
later reader of the same source; a source failing one gives one typed reason
and its text is not used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from app.services import verification_evidence as ve

VERSION = "authorized-pages-v1"

UnavailableReason = Literal[
    "derivative_provenance_invalid",
    "authorized_source_mismatch",
    "stale_source_extraction",
    "source_extraction_mismatch",
]


@dataclass(frozen=True)
class AuthorizedPages:
    source: ve.AuthorizedRepresentation
    pages: tuple
    limitations: tuple[str, ...]
    extracted_text_sha256: str


@dataclass(frozen=True)
class PagesUnavailable:
    reason_code: UnavailableReason
    detail: str


def open_authorized_pages(source: ve.AuthorizedRepresentation,
                          artifact: ve.VerificationEvidenceArtifact | None = None,
                          ) -> AuthorizedPages | PagesUnavailable:
    """The source's authorized pages, or why its text may not be used.

    Without an artifact (the first build, which creates the source identity,
    and readers that run before it) only the source's own checks apply.
    """
    opened = _opened(source)
    if isinstance(opened, PagesUnavailable) or artifact is None:
        return opened
    if not ve._source_matches_artifact(source, artifact):
        return PagesUnavailable("authorized_source_mismatch",
                                "The source does not match the artifact's source identity")
    coverage = artifact.coverage
    if coverage.extraction_version != ve.EXTRACTION_VERSION:
        return PagesUnavailable("stale_source_extraction",
                                "The artifact was built with another extraction version")
    if coverage.extracted_text_sha256 != opened.extracted_text_sha256:
        return PagesUnavailable("source_extraction_mismatch",
                                "The source's pages differ from the pages the artifact was built from")
    return opened


@lru_cache(maxsize=8)
def _opened(source: ve.AuthorizedRepresentation) -> AuthorizedPages | PagesUnavailable:
    try:
        ve._validate_derivative_provenance(source)
    except ve.EvidenceAuthorizationError as exc:
        return PagesUnavailable("derivative_provenance_invalid", str(exc))
    if hashlib.sha256(source.content).hexdigest() != source.content_sha256:
        return PagesUnavailable("authorized_source_mismatch",
                                "The source's bytes do not match its content hash")
    pages, limitations = ve._extract_pages(source)
    return AuthorizedPages(source=source, pages=tuple(pages), limitations=tuple(limitations),
                           extracted_text_sha256=ve._extracted_pages_sha256(pages))


def clear_cache() -> None:
    """Forget extracted pages (tests that replace the extractor)."""
    _opened.cache_clear()
