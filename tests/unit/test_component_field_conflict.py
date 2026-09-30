"""A disagreeing journal, volume or issue is recorded as a disagreement.

Until 2026-09-23 the component-field comparator was binary: equal after
normalization gave `agreement`, and everything else gave `unknown`. So
`material_conflict` was unreachable for these fields and
`bibliographic_field_conflicts` could not fire -- measured, it had produced
nothing across 146 stored reports and 15,142 candidates.

The worked case is a real one. A reference gave Khan's "The separation of
platforms and commerce" as Harvard Law Review 131(5); the article is in the
Columbia Law Review, volume 119 (2019) -- an open-access article. Every part
of that is checkable, and none of it was being reported.
"""
import pytest

from app.services.reference_discovery import (
    _MATERIAL_CONFLICT_FIELDS,
    _abbreviation_compatible,
    _field_materially_conflicts,
)


@pytest.mark.parametrize("submitted,observed", [
    ("Harvard Law Review", "Columbia Law Review"),
    ("Harvard Law Review", "Yale Law Journal"),
])
def test_the_title_comparison_separates_different_journals(submitted, observed) -> None:
    """The comparison is sound; it is the reporting that is withheld.

    `container_title` is excluded from `_MATERIAL_CONFLICT_FIELDS` for one
    measured reason recorded below, so this exercises the comparison itself.
    """
    assert _abbreviation_compatible(submitted, observed) is False


@pytest.mark.parametrize("submitted,observed", [
    # Every one of these occurs in the stored corpus and is a writing or
    # indexing difference, not a disagreement about the journal.
    ("Journal of Education for Teaching",
     "Journal of Education for Teaching: International Research"),
    ("Computers & Education", "Computers &amp; Education"),
    ("Marvels & Tales", "Marvels &amp;amp; Tales"),
    ("Libri & Liberi", "Libri et Liberi"),
    ("Journal of Popular Film & Television", "Journal of Popular Film and Television"),
    ("J. Econ. Perspect.", "Journal of Economic Perspectives"),
])
def test_a_formatting_or_abbreviation_variant_is_not_a_conflict(submitted, observed) -> None:
    assert _abbreviation_compatible(submitted, observed) is True
    assert _field_materially_conflicts("container_title", submitted, observed) is False


def test_volume_and_issue_compare_by_number() -> None:
    assert _field_materially_conflicts("volume", "131", "119") is True
    assert _field_materially_conflicts("issue", "5", "4") is True
    assert _field_materially_conflicts("volume", "Vol. 131", "131") is False


def test_a_missing_value_is_never_a_conflict() -> None:
    """A provider that records no volume has not contradicted the reference."""
    assert _field_materially_conflicts("volume", "131", "") is False
    assert _field_materially_conflicts("issue", "", "4") is False


def test_a_translated_journal_title_is_why_container_title_is_withheld() -> None:
    """Measured: one firing across 3,687 stored candidates, and it was wrong.

    A reference gave "Economics and Law" where Crossref records "Ekonomia i
    Prawo" -- the same journal under its English and Polish titles. No string
    rule separates that from a genuinely different journal, and the provider
    recorded only one of the pair, so comparing against every listed title
    does not rescue it either. Khan's reference is still reported, through
    volume 131 against 119.
    """
    assert "container_title" not in _MATERIAL_CONFLICT_FIELDS
    assert _field_materially_conflicts(
        "container_title", "Economics and Law", "Ekonomia i Prawo") is False
    assert _field_materially_conflicts("volume", "131", "119") is True


def test_pages_and_publisher_cannot_produce_a_conflict() -> None:
    """Measured false positives, and the defect is in the inputs.

    "102305" is an Elsevier article number, not a page range; "Unpublished" is
    a DataCite placeholder; and the publisher field sometimes holds a
    mis-parsed journal string. Reporting any of these would tell a reader a
    correct reference is wrong.
    """
    assert "pages" not in _MATERIAL_CONFLICT_FIELDS
    assert "publisher" not in _MATERIAL_CONFLICT_FIELDS
    assert _MATERIAL_CONFLICT_FIELDS == {"volume", "issue"}
    assert _field_materially_conflicts("pages", "1-11", "102305") is False
    assert _field_materially_conflicts("publisher", "British Council", "Unpublished") is False
