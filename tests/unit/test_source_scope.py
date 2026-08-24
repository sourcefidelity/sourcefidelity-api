"""Cited-component scope regressions."""

from app.services.source_scope import (
    assess_cited_component_scope,
    cited_page_range,
)


def test_reference_page_range_is_extracted_without_entering_the_title_contract():
    reference = "Hardy, J. (2010). Cross-media promotion (pp. 3-32). New York: Peter Lang."

    assert cited_page_range(reference) == (3, 32)


def test_roman_front_matter_cannot_stand_in_for_cited_arabic_pages():
    reference = "Hardy, J. (2010). Cross-media promotion (pp. 3-32). New York: Peter Lang."
    preview = (
        "Foreword. But, as Hardy points out, media companies have brands to sell. "
        "← xi | xii → Serving as our guide, Hardy is careful. "
        "Matthew P. McAllister ← xiii | xiv →"
    )

    result = assess_cited_component_scope(reference, preview)

    assert result.status == "inadequate"
    assert result.reason_code == "front_matter_only_cited_page_range_absent"
    assert result.expected_page_range == (3, 32)


def test_observed_page_inside_cited_range_is_adequate():
    result = assess_cited_component_scope(
        "Example (2020). Chapter title (pp. 3–32). Publisher.",
        "Chapter text ← 9 | 10 → continues.",
    )

    assert result.status == "adequate"
    assert result.observed_arabic_pages == (9, 10)


def test_missing_page_evidence_abstains_instead_of_claiming_scope():
    result = assess_cited_component_scope(
        "Example (2020). Chapter title (pp. 3–32). Publisher.",
        "A generic publisher description without page coordinates.",
    )

    assert result.status == "unresolved"
    assert result.reason_code == "cited_page_range_not_observable"
