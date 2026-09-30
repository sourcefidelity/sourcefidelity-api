"""Exact markers survive initials and repeated attribution in one sentence."""
from types import SimpleNamespace

import pytest

from app.services.citation_extractor import _extract_attributed_text_and_index
from app.services.sentence_splitter import split_sentences
from app.services.schemas import CitationMarkerMember
from app.services.verification_evidence import ClaimEvidence, _source_binding
from app.services.verification_report import ReportAuthorizationError, _validate_source_binding


def test_attributed_span_contains_complete_marker_across_author_initial():
    text = "Article 1: Smith, J. (2018). A history of reporting."
    marker = "Smith, J. (2018)"
    start = text.index(marker)
    span, _, left, right = _extract_attributed_text_and_index(
        text, start, start + len(marker), split_sentences(text), "narrative",
    )
    assert marker in span
    assert left <= start < start + len(marker) <= right
    assert span == text[left:right]
    assert "A history of reporting" not in span


@pytest.mark.parametrize("compound", [False, True])
def test_repeated_markers_for_same_source_are_not_ambiguous_source_identity(compound):
    text = "Smith (2020) describes a shared result (Smith, 2020)."
    refs = ["ref-smith"]
    marker_texts = [("Smith (2020)", "ref-smith"), ("(Smith, 2020)", "ref-smith")]
    if compound:
        text = "Smith (2020) and Jones (2021) describe a shared result (Smith, 2020)."
        refs.append("ref-jones")
        marker_texts.insert(1, ("Jones (2021)", "ref-jones"))
    markers = [CitationMarkerMember(text=m,local_start=text.index(m),
        local_end=text.index(m)+len(m),reference_ids=[rid],marker_type="parenthetical" if m.startswith("(") else "narrative")
        for m,rid in marker_texts]
    claim = ClaimEvidence(claim_id="repeated",paper_version_id="paper",text=text,
        reference_ids=refs,citation_marker="(Smith, 2020)",citation_markers=markers,
        passage_start=0,passage_end=len(text))
    before = claim.model_dump_json()
    binding = _source_binding(claim,active_reference_id="ref-smith",cited_author_label="Smith")
    _validate_source_binding(SimpleNamespace(claim=claim,source_binding=binding))
    assert binding.marker_text == "(Smith, 2020)"
    assert claim.model_dump_json() == before
    if compound:
        other = _source_binding(claim,active_reference_id="ref-jones",cited_author_label="Jones")
        _validate_source_binding(SimpleNamespace(claim=claim,source_binding=other))
        assert other.marker_text == "Jones (2021)"


def test_repeated_marker_recovery_does_not_accept_nonexact_member():
    text = "Smith (2020) states this (Smith, 2020)."
    markers = [CitationMarkerMember(text=m,local_start=s,local_end=s+len(m),
        reference_ids=["ref-smith"],marker_type="parenthetical")
        for m,s in [("Smith (2020)",0),("(Smith, 2020)",1)]]
    claim = ClaimEvidence(claim_id="invalid",paper_version_id="paper",text=text,
        reference_ids=["ref-smith"],citation_marker="Smith (2020)",citation_markers=markers,
        passage_start=0,passage_end=len(text))
    with pytest.raises(ValueError,match="not exact"):
        _source_binding(claim,active_reference_id="ref-smith",cited_author_label="Smith")


def test_duplicate_records_at_one_marker_are_still_rejected():
    text = "Smith (2020) states this."
    marker = CitationMarkerMember(text="Smith (2020)",local_start=0,local_end=12,
        reference_ids=["ref-smith"],marker_type="narrative")
    claim = ClaimEvidence(claim_id="duplicate",paper_version_id="paper",text=text,
        reference_ids=["ref-smith"],citation_marker=marker.text,citation_markers=[marker,marker],
        passage_start=0,passage_end=len(text))
    binding = _source_binding(claim,active_reference_id="ref-smith",cited_author_label="Smith")
    with pytest.raises(ReportAuthorizationError):
        _validate_source_binding(SimpleNamespace(claim=claim,source_binding=binding))
