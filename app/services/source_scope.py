"""Conservative cited-component scope checks for calibration and verification.

Bibliographic identity does not prove that an acquired representation contains
the cited component.  In particular, a publisher preview of a monograph may
contain only roman-numbered front matter while a reference identifies an
Arabic-numbered chapter/page range.  These helpers expose that mismatch
without inferring absence when page evidence is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


_CITED_PAGE_RANGE = re.compile(
    r"\b(?:pp?\.?|pages?)\s*(?P<start>\d{1,5})\s*[-–—]\s*(?P<end>\d{1,5})\b",
    re.IGNORECASE,
)
_PAGE_TRANSITION = re.compile(
    r"[←↞]\s*(?P<left>[ivxlcdm]+|\d{1,5})\s*\|\s*"
    r"(?P<right>[ivxlcdm]+|\d{1,5})\s*[→↠]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceScopeAssessment:
    status: Literal["adequate", "inadequate", "unresolved", "not_required"]
    reason_code: str
    expected_page_range: tuple[int, int] | None
    observed_arabic_pages: tuple[int, ...] = ()
    observed_roman_front_matter: bool = False


def cited_page_range(reference_text: str) -> tuple[int, int] | None:
    """Return one bounded bibliographic page range without guessing."""
    matches = list(_CITED_PAGE_RANGE.finditer(reference_text or ""))
    if len(matches) != 1:
        return None
    start = int(matches[0].group("start"))
    end = int(matches[0].group("end"))
    if start <= 0 or end < start or end - start > 2_000:
        return None
    return start, end


def assess_cited_component_scope(
    reference_text: str,
    representation_text: str,
    *,
    passage_page_labels: list[str | None] | None = None,
) -> SourceScopeAssessment:
    """Check only whether visible page evidence reaches the cited component.

    ``unresolved`` is deliberately common.  It means the representation may be
    useful for an explicit positive statement, but it must not enter a human
    packet that purports to review the cited page/component unless a separate
    reviewer has established the scope manually.
    """
    expected = cited_page_range(reference_text)
    if expected is None:
        return SourceScopeAssessment(
            status="not_required",
            reason_code="no_single_cited_page_range",
            expected_page_range=None,
        )

    arabic_pages: set[int] = set()
    roman_front_matter = False
    for match in _PAGE_TRANSITION.finditer(representation_text or ""):
        for value in (match.group("left"), match.group("right")):
            if value.isdigit():
                arabic_pages.add(int(value))
            else:
                roman_front_matter = True
    for label in passage_page_labels or []:
        if not label:
            continue
        arabic_pages.update(int(value) for value in re.findall(r"\d+", label))
        if re.fullmatch(r"\s*[ivxlcdm]+\s*", label, re.IGNORECASE):
            roman_front_matter = True

    start, end = expected
    if any(start <= page <= end for page in arabic_pages):
        return SourceScopeAssessment(
            status="adequate",
            reason_code="cited_page_range_observed",
            expected_page_range=expected,
            observed_arabic_pages=tuple(sorted(arabic_pages)),
            observed_roman_front_matter=roman_front_matter,
        )
    if roman_front_matter and not arabic_pages:
        return SourceScopeAssessment(
            status="inadequate",
            reason_code="front_matter_only_cited_page_range_absent",
            expected_page_range=expected,
            observed_roman_front_matter=True,
        )
    if arabic_pages and all(page < start or page > end for page in arabic_pages):
        return SourceScopeAssessment(
            status="inadequate",
            reason_code="observed_pages_outside_cited_range",
            expected_page_range=expected,
            observed_arabic_pages=tuple(sorted(arabic_pages)),
            observed_roman_front_matter=roman_front_matter,
        )
    return SourceScopeAssessment(
        status="unresolved",
        reason_code="cited_page_range_not_observable",
        expected_page_range=expected,
        observed_roman_front_matter=roman_front_matter,
    )
