"""Disjoint locators must not become their first page or an invented range."""

import pytest

from app.services.verification_evidence import _page_locator_values


@pytest.mark.parametrize(("locator", "expected"), [
    ("19, 21", {19, 21}),
    ("pp. 19, 21–23", {19, 21, 22, 23}),
    ("3–5, 8, 11-12", {3, 4, 5, 8, 11, 12}),
    ("p. 19", {19}),
    ("pages 34–35, 82", {34, 35, 82}),
    ("19,", set()),
    ("19, unknown", set()),
    ("21–19", set()),
    ("0", set()),
    ("1–99999", set()),
    ("19; 21", set()),
])
def test_page_locator_values(locator, expected):
    assert _page_locator_values(locator) == expected


def test_sentence_recovery_preserves_the_complete_page_list():
    from app.services.paper_extraction import (
        CitationMarkerCensusEntry, _recover_linked_sentence_citations,
    )
    marker = "(United Artists, 1927, pp. 19, 21)"
    text = f"The campaign makes this claim {marker}."
    census = CitationMarkerCensusEntry(
        marker_id="marker-1", text=marker, passage_start=text.index(marker),
        passage_end=text.index(marker) + len(marker), reference_ids=["ref-1"],
        link_status="linked", marker_type="parenthetical", paragraph_index=0,
        member_count=1,
    )
    result = _recover_linked_sentence_citations([], [census], text)
    assert len(result) == 1
    assert result[0].page_number == "19, 21"
    assert _page_locator_values(result[0].page_number) == {19, 21}
