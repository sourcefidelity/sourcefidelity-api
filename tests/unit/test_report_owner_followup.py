"""Summary traceability, media actions, citation boundaries and edition guards."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services.evidence_report import (
    _build_role_summaries, _render_role_summaries, _member_accepts_upload,
    _eligible_display_passages, _render_reference_panel_template,
)
from app.services.citation_extractor import _extract_attributed_text_and_index
from app.services.sentence_splitter import split_sentences
from app.services.reference_discovery import (
    ExpectedBibliographicFields, build_reference_discovery_candidate,
    is_edition_sensitive_reference,
)
from app.services.retrieval.base import RetrievalResult
from app.services.verification_evidence import passage_role_from_text


def test_single_missing_entry_is_a_summary_priority_with_citation_location():
    citations = [{"members": []} for _ in range(4)] + [{"members": [], "missing_reference_members": ["Writer, 2020"]}]
    summary = _build_role_summaries(citations=citations, overview={}, pervasive_hanging_indent=False)
    for audience in ("student", "instructor"):
        text = " ".join(summary[audience]["academic_practice"])
        assert "citation 5" in text and "Writer, 2020" in text
        html = _render_role_summaries(summary, audience=audience)
        assert "<ol>" not in html and "<ul>" in html


def test_repeated_issues_take_priority_over_one_off_within_category():
    indirect = {"best_evidence": {"evidence_role": "representation_of_other_work"}}
    citations = [{"members": [indirect]}, {"members": [indirect]},
                 {"members": [], "missing_reference_members": ["Writer, 2020"]}]
    summary = _build_role_summaries(citations=citations, overview={}, pervasive_hanging_indent=False)
    assert "2 citations" in " ".join(summary["instructor"]["academic_practice"])
    assert "Writer" not in " ".join(summary["instructor"]["academic_practice"])


def test_report_summary_requires_paper_flags_except_hanging_indent():
    citations=[{"members": [], "missing_reference_members": ["Writer, 2020"]}]
    summary=_build_role_summaries(citations=citations,overview={},pervasive_hanging_indent=True,
                                 reference_practice=[],require_paper_flags=True)
    assert not summary['student']['academic_practice']
    assert summary['student']['reference_formatting']
    citations[0]['paper_location']={'localization_level':'exact_rectangle','rectangles':[{'page_index':0}]}
    summary=_build_role_summaries(citations=citations,overview={},pervasive_hanging_indent=False,
                                 reference_practice=[],require_paper_flags=True)
    assert 'Writer' in ' '.join(summary['student']['academic_practice'])


@pytest.mark.parametrize("kind,raw,allowed", [
    ("traditional_media", "Director (1946). Title [Film].", False),
    ("unknown", "Director (1946). Title [Film].", False),
    ("video", "Lecture video", False),
    ("monograph", "Writer (2020). Book.", True),
])
def test_media_does_not_offer_unsupported_upload(kind, raw, allowed):
    assert _member_accepts_upload({"coverage_level": "unavailable", "source": {"source_kind": kind, "raw_reference": raw}}) is allowed


@pytest.mark.parametrize("punctuation", ["!", "?", "."])
def test_parenthetical_after_closed_quotation_keeps_complete_sentence(punctuation):
    sentence = f'The review announced “This is a remarkable result{punctuation}” (Writer, 2020).'
    text = "An unrelated earlier sentence. " + sentence
    start = text.index("(Writer")
    result, _, left, right = _extract_attributed_text_and_index(text, start, text.index(")", start)+1, split_sentences(text), "parenthetical")
    assert result == sentence and text[left:right] == sentence


def test_ordinary_previous_sentence_is_not_inherited():
    text = 'A separate earlier claim. Another claim (Writer, 2020).'
    start = text.index('(Writer')
    result, *_ = _extract_attributed_text_and_index(text,start,text.index(')',start)+1,split_sentences(text),'parenthetical')
    assert result == 'Another claim (Writer, 2020).'


def test_publisher_conformance_statement_not_presented_as_book_evidence():
    text = 'The eBook was built with accessibility in mind. A VPAT for WCAG 2.1 AA is available upon request.'
    passages = [{"passage_id": "metadata", "excerpt": text}]
    original = deepcopy(passages)
    assert passage_role_from_text(text) == "publication_metadata"
    assert _eligible_display_passages(passages, {"status": "not_assessed"}) == []
    assert _eligible_display_passages(passages, {"status": "not_assessed"}, preferred_passage_ids=["metadata"]) == passages
    assert passages == original
    scholarly = 'We evaluated accessibility against WCAG 2.1. The study compared reader performance and conformance reports.'
    assert passage_role_from_text(scholarly) != "publication_metadata"
    assert _eligible_display_passages([{"excerpt": scholarly}], None)


@pytest.mark.parametrize("year", ["1983", "2013"])
def test_book_year_mismatch_requires_same_edition_not_numeric_tolerance(year):
    candidate = build_reference_discovery_candidate(expected=ExpectedBibliographicFields(
        title="A sufficiently specific book title", authors=["Writer"],year="1984", source_kind="monograph"),
        result=RetrievalResult(source_name="catalog",success=True,title="A sufficiently specific book title",authors=["Writer"],year=year),
        provider="catalog",attempt_id="attempt")
    comparison = next(c for c in candidate.comparisons if c.field_name == 'year')
    assert comparison.outcome == 'unknown' and comparison.reason_code == 'book_edition_year_unresolved'


def test_exact_isbn_keeps_one_year_difference_inspectable():
    candidate = build_reference_discovery_candidate(expected=ExpectedBibliographicFields(
        title="A book title",authors=["Writer"],year="1984",source_kind="monograph",isbn="0385292651"),
        result=RetrievalResult(source_name="catalog",success=True,title="A book title",authors=["Writer"],year="1983",metadata={"isbn":"9780385292658"}),
        provider="catalog",attempt_id="attempt")
    assert next(c for c in candidate.comparisons if c.field_name=='year').outcome=='material_conflict'
    assert candidate.observed.isbn=='9780385292658'


def test_legacy_book_url_is_not_itself_book_type_evidence():
    expected=ExpectedBibliographicFields(source_kind='webpage')
    assert is_edition_sensitive_reference(expected,'Writer (1984). A book. New York: Small Publisher. Retrieved from https://example.org/book')
    assert not is_edition_sensitive_reference(expected,'Writer (1984). An article. https://archive.org/details/article')


def test_duplicate_reference_panel_names_both_works():
    peers=[dict(author='Writer',year='2020',title=title,raw_reference=f'Writer (2020). {title}.') for title in ('First work','Second work')]
    html=_render_reference_panel_template(dict(source=peers[0],finding='Distinguish these references.',related_references=peers,finding_type='duplicate_citation_key'),1)
    assert 'First work' in html and 'Second work' in html
