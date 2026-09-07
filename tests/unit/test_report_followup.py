"""Source-bound follow-up report, quotation, and presentation regressions."""

from types import SimpleNamespace

import pytest

from app.services.evidence_report import (
    _build_role_summaries, _reference_retrieval_counts, _render_member,
    _inline_locator, _check_sentence, _positive_quote_check, _unavailable_member,
    citation_tones,
)
from app.services.verification_evidence import _quotation_match


def test_reference_counts_are_unique_and_exclude_unavailable_from_limited():
    citations = [{"members": [
        {"reference_id": "a", "coverage_level": "abstract_only"},
        {"reference_id": "b", "coverage_level": "partial_text"},
        {"reference_id": "c", "coverage_level": "unavailable"},
    ]}, {"members": [{"reference_id": "a", "coverage_level": "full_text"}]}]
    assert _reference_retrieval_counts(citations, 4) == {
        "verified_full_text_sources": 1, "abstract_or_limited_sources": 1,
        "unavailable_sources": 2,
    }


def test_summary_names_affected_citations_without_inventing_missing_details():
    partial = {"best_evidence": {"evidence_note": "The abstract addresses only part of the attributed statement."}}
    citations = [{"members": [partial]}, {"members": [partial]},
                 {"members": [{"coverage_level": "full_text", "relevance_status": "no_connection"}]},
                 {"members": [{"coverage_level": "full_text", "relevance_status": "not_assessed"}]},
                 {"members": [{"coverage_level": "unavailable"}]}]
    summary = _build_role_summaries(citations=citations, overview={}, pervasive_hanging_indent=False)
    for audience in ("student", "instructor"):
        text = " ".join(summary[audience]["evidence"])
        assert "citations 1, 2" in text
        assert "For 1 citation," in text
        assert "citation 3" in text
        assert "evidence is absent" in text
        assert "displayed evidence" not in text
        assert "qualifier, scope, actor" not in text


def test_half_full_half_unavailable_and_two_thirds_full_have_exact_shares():
    full = {"coverage_level": "full_text", "relevance_status": "connected"}
    missing = {"coverage_level": "unavailable"}
    assert citation_tones({"members": [missing, full]}) == ["evidence_available", "not_assessed"]
    assert citation_tones({"members": [full, missing, full]}) == ["evidence_available", "evidence_available", "not_assessed"]


def test_reference_precedes_evidence_and_locator_is_inline():
    member = {"reference_id": "a", "coverage_level": "full_text", "source": {
        "author": "Researcher", "year": "2020", "title": "Research study", "raw_reference": "Researcher (2020). Research study."},
        "best_evidence": {"text": "An inspectable source passage.", "locator": "Page 5"}}
    html = _render_member(member, grouped=True)
    assert html.index('class="full-reference"') < html.index('class="source-excerpt"')
    assert '(p. 5)</span></blockquote>' in html
    assert '<p class="locator">' not in html
    assert _inline_locator({"locator": "PDF page 2"}).endswith('(PDF p. 2)</span>')
    assert _inline_locator({"locator": "Page 5", "evidence_kind": "abstract"}) == ""
    assert _check_sentence("Quotation", {"label": "Quotation cannot be checked."}) == "Quotation cannot be checked."


def test_missing_source_does_not_emit_an_unassessed_quotation_message():
    reference = SimpleNamespace(reference_id='a', raw_ref='Author (2020). Study.', author='Author', year='2020', title='Study', doi=None, url=None)
    claim = SimpleNamespace(text='“The complete quotation” (Author, 2020).', reference_ids=['a'])
    member = _unavailable_member({}, reference, claim)
    assert member['show_quotation_check'] is False


@pytest.mark.parametrize('relevance', ['', 'not_relevant', 'uncertain'])
def test_unassessed_abstract_is_complete_and_availability_is_teal(relevance):
    from app.services.evidence_report import member_tone
    reference = SimpleNamespace(reference_id='a', raw_ref='Author (2020). Study.', author='Author', year='2020', title='Study', doi=None, url=None)
    abstract = 'The book describes a performer and a complicated career. Later illness ended the career. Childhood and family relationships are also discussed.'
    claim = SimpleNamespace(text='The manager demanded changes in voice and appearance (Author, 2020).', reference_ids=['a'])
    member = _unavailable_member({'abstract_available':True, 'abstract_evidence':{'text':abstract}, 'abstract_relevance':{'relevance':relevance}}, reference, claim)
    assert member['best_evidence']['display_text'] == abstract
    assert member['relevance_status'] == 'not_assessed'
    assert member_tone(member) == 'limited_evidence'


def test_quotation_and_locator_labels_omit_implementation_detail():
    from app.services.evidence_report import _reason_label
    result = _positive_quote_check(['A complete quotation'], ['A complete\nquotation'])
    assert result['label'] == 'Complete quoted wording matches the source.'
    assert result['matches'][0]['method'] != 'literal'
    assert _check_sentence('Locator', {'label':_reason_label('located_span_matches_supplied_locator')}) == 'Provided page/paragraph number matches the retrieved source'


def test_missing_member_is_named_and_flagged_without_inventing_reference():
    from app.services.evidence_report import _missing_reference_members, citation_tones
    from app.services.citation_extractor import extract_citations
    from app.services.schemas import ParsedReference
    text='The account mentions both sources (Author, 2020; Missing, 2020).'
    citations=extract_citations(text,[ParsedReference(reference_id='a',author='Author',year='2020',citation_key='Author2020')],format_hint='apa',use_llm_boundaries=False)
    members=_missing_reference_members(SimpleNamespace(citations=citations),0,len(text))
    assert members == ['Missing, 2020']
    assert citation_tones({'members':[], 'missing_reference_members':members}) == ['attention']


@pytest.mark.parametrize('marker', ['...', '. . .', '[...]', '[ . . . ]', '…'])
def test_marked_omissions_match_unchanged_wording_without_a_practice_failure(marker):
    source = 'The independent research team observed several important outcomes in the study. Careful follow-up established significant differences across all three treatment groups.'
    quote = 'The independent research team observed ' + marker + ' significant differences across all three treatment groups.'
    match = _quotation_match(source, quote)
    assert match and match[2] == 'ellipsis_normalized'
    check = _positive_quote_check([quote], [source])
    assert not check['attention']
    assert 'meaning has not been automatically verified' in check['label']


def test_bracket_expansion_and_mixed_linewrap_hyphens_are_recognized():
    source = 'The AI-enabled system supported cultivating independent learners across several different classrooms. Students used MT and employed diverse strategies to address various language-related challenges.'
    quotes = ['The AI-\nenabled system supported cultivat-\ning independent learners across several different classrooms.',
              'Students used [machine translation] and employed diverse strategies to address vari-\nous language-related challenges.']
    check = _positive_quote_check(quotes, [source])
    assert check['status'] == 'complete' and not check['attention']
    assert len(check['matches']) == 2
    assert check['matches'][1]['method'] == 'marked_editorial_match'


def test_brackets_do_not_hide_unmarked_changes_or_manufacture_semantic_approval():
    source = 'The independent research team found that all participants improved after completing the supervised program.'
    altered = 'The independent research team [recently] found that no participants improved after completing the supervised program.'
    assert _quotation_match(source, altered) is None
    # A marked negation is located as an editorial change, not endorsed.
    marked = 'The independent research team found that [no] participants improved after completing the supervised program.'
    check = _positive_quote_check([marked], [source])
    assert check and 'meaning has not been automatically verified' in check['label']
    assert _quotation_match(source, '[anything] participants improved') is None


def test_unmarked_lexical_hyphen_is_not_silently_removed():
    assert _quotation_match('Spanishspeaking students', 'Spanish-speaking students') is None


def test_unmatched_short_editorial_quote_abstains_instead_of_flagging():
    from app.services.verification_evidence import _academic_practice_checks
    source = SimpleNamespace(text_quality='digital', derivation_method=None, completeness_verdict='complete', edition_or_version=None)
    claim = SimpleNamespace(claim_type='quotation', text='“[The students] improved.”', page_locator=None)
    quotation, _ = _academic_practice_checks(claim=claim, source=source, pages=[], passages=[])
    assert quotation.status == 'not_assessable'
    assert quotation.outcome == 'marked_editorial_changes_require_review'
