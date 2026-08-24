"""Bounded book-identifier extraction and source-kind normalization.

ISBNs are evidence candidates, not assertions.  Books commonly print several
identifiers for hardback, paperback, electronic, series, or cited editions.
Every candidate is checksum-validated and retains where it was observed so an
edition-metadata adapter can corroborate it against the cited work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Literal

import fitz

from app.services.source_type import (
    document_kind_for_source_kind as _shared_document_kind_for_source_kind,
    normalize_source_kind as _shared_normalize_source_kind,
)


CompletenessDocumentKind = Literal["article", "book", "chapter", "unknown"]

_ISBN_LABEL_RE = re.compile(
    r"\b(?:e[- ]?isbn|isbn(?:[- ]?1[03])?)\s*[:=]?\s*"
    r"((?:97[89][\s.-]?)?\d(?:[\dXx][\s.-]?){8,16})",
    re.IGNORECASE,
)
_ISBN_13_RE = re.compile(
    r"(?<!\d)(97[89](?:[\s.-]?\d){10})(?!\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ISBNCandidate:
    value: str
    canonical_isbn13: str | None
    occurrences: int
    labelled_occurrences: int
    page_indexes: tuple[int, ...]
    score: int


@dataclass(frozen=True)
class ISBNExtractionReport:
    candidates: tuple[ISBNCandidate, ...]
    inspected_pages: int
    text_available: bool
    content_sha256: str


def normalize_source_kind(value: str) -> str:
    """Normalize a supported Zotero-like source kind or raise ValueError."""
    return _shared_normalize_source_kind(value, strict=True)


def document_kind_for_source_kind(value: str) -> CompletenessDocumentKind:
    normalize_source_kind(value)
    return _shared_document_kind_for_source_kind(value)  # type: ignore[return-value]


def content_sha256(content: bytes) -> str:
    """Stable representation identity used instead of mutable filenames."""
    return hashlib.sha256(content).hexdigest()


def normalize_isbn(value: str) -> str | None:
    normalized = re.sub(r"[^0-9X]", "", value.upper())
    return normalized if is_valid_isbn(normalized) else None


def is_valid_isbn(value: str) -> bool:
    normalized = re.sub(r"[^0-9X]", "", value.upper())
    if len(normalized) == 10:
        if "X" in normalized[:-1]:
            return False
        return (
            sum(
                (10 - index) * (10 if character == "X" else int(character))
                for index, character in enumerate(normalized)
            )
            % 11
            == 0
        )
    if len(normalized) == 13 and normalized.isdigit() and normalized[:3] in {"978", "979"}:
        check = sum(
            (1 if index % 2 == 0 else 3) * int(character)
            for index, character in enumerate(normalized[:12])
        )
        return (10 - check % 10) % 10 == int(normalized[-1])
    return False


def isbn10_to_isbn13(value: str) -> str | None:
    normalized = normalize_isbn(value)
    if normalized is None:
        return None
    if len(normalized) == 13:
        return normalized
    body = "978" + normalized[:9]
    check = sum(
        (1 if index % 2 == 0 else 3) * int(character)
        for index, character in enumerate(body)
    )
    return body + str((10 - check % 10) % 10)


def _sample_page_indexes(page_count: int, limit: int) -> list[int]:
    if page_count <= limit:
        return list(range(page_count))
    front = list(range(min(18, page_count)))
    back = list(range(max(0, page_count - 6), page_count))
    return sorted(set((front + back)[:limit]))


def extract_isbn_candidates(
    pdf_bytes: bytes,
    *,
    page_limit: int = 24,
) -> ISBNExtractionReport:
    """Extract checksum-valid ISBN candidates from bounded PDF text.

    Unlabelled ISBN-10-like digit sequences are deliberately ignored because
    page text contains many unrelated ten-digit identifiers.  Unlabelled
    ISBN-13 candidates must start with the ISBN/EAN book prefixes 978/979.
    """
    digest = content_sha256(pdf_bytes)
    observations: dict[str, dict[str, object]] = {}
    inspected = 0
    text_available = False

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            indexes = _sample_page_indexes(len(document), page_limit)
            metadata_text = " ".join(
                str(value) for value in (document.metadata or {}).values() if value
            )
            sources = [(-1, metadata_text)]
            sources.extend((index, document[index].get_text()) for index in indexes)
            inspected = len(indexes)
        finally:
            document.close()
    except Exception:
        return ISBNExtractionReport((), 0, False, digest)

    for page_index, text in sources:
        if text.strip():
            text_available = True
        labelled_spans: set[tuple[int, int]] = set()
        for match in _ISBN_LABEL_RE.finditer(text):
            normalized = normalize_isbn(match.group(1))
            if normalized is None:
                continue
            labelled_spans.add(match.span(1))
            record = observations.setdefault(
                normalized,
                {"count": 0, "labelled": 0, "pages": set()},
            )
            record["count"] = int(record["count"]) + 1
            record["labelled"] = int(record["labelled"]) + 1
            if page_index >= 0:
                record["pages"].add(page_index)  # type: ignore[union-attr]

        for match in _ISBN_13_RE.finditer(text):
            if any(start <= match.start(1) and match.end(1) <= end for start, end in labelled_spans):
                continue
            normalized = normalize_isbn(match.group(1))
            if normalized is None:
                continue
            record = observations.setdefault(
                normalized,
                {"count": 0, "labelled": 0, "pages": set()},
            )
            record["count"] = int(record["count"]) + 1
            if page_index >= 0:
                record["pages"].add(page_index)  # type: ignore[union-attr]

    candidates: list[ISBNCandidate] = []
    for value, record in observations.items():
        pages = tuple(sorted(record["pages"]))  # type: ignore[arg-type]
        occurrences = int(record["count"])
        labelled = int(record["labelled"])
        early_bonus = 2 if any(page < 12 for page in pages) else 0
        score = labelled * 5 + min(occurrences, 3) + early_bonus
        candidates.append(
            ISBNCandidate(
                value=value,
                canonical_isbn13=isbn10_to_isbn13(value),
                occurrences=occurrences,
                labelled_occurrences=labelled,
                page_indexes=pages,
                score=score,
            )
        )

    # Collapse equivalent ISBN-10/ISBN-13 observations onto the strongest
    # representative while preserving independent edition candidates.
    grouped: dict[str, ISBNCandidate] = {}
    for candidate in candidates:
        key = candidate.canonical_isbn13 or candidate.value
        previous = grouped.get(key)
        if previous is None or (
            candidate.score,
            len(candidate.value),
        ) > (previous.score, len(previous.value)):
            grouped[key] = candidate

    ordered = sorted(
        grouped.values(),
        key=lambda candidate: (
            candidate.score,
            candidate.labelled_occurrences,
            candidate.occurrences,
            len(candidate.value),
        ),
        reverse=True,
    )
    return ISBNExtractionReport(tuple(ordered), inspected, text_available, digest)
