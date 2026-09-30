from app.services.evidence_report import summary_text
"""Source-bound follow-up report, quotation, and presentation regressions."""

from types import SimpleNamespace

import pytest

from app.services.evidence_report import (
    _build_report_summary, _reference_retrieval_counts, _render_member,
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


def test_relevance_gate_notices_are_no_longer_summarized():
    partial = {"coverage_level":"full_text", "relevance_status":"connected", "best_evidence": {"evidence_note": "The passage addresses only part of the attributed statement."}}
    citations = [{"members": [partial]}, {"members": [partial]},
                 {"members": [{"coverage_level": "full_text", "relevance_status": "no_connection"}]},
                 {"members": [{"coverage_level": "full_text", "relevance_status": "not_assessed"}]},
                 {"members": [{"coverage_level": "unavailable"}]}]
    summary = _build_report_summary(citations=citations, overview={}, pervasive_hanging_indent=False)
    # Owner request 2026-09-29: these came from the retired relevance-gate display.
    text = " ".join(map(summary_text, summary["evidence"]))
    assert "For 1 citation," not in text and "only part of the attributed statement" not in text
    assert "displayed evidence" not in text
    assert "qualifier, scope, actor" not in text


def test_half_full_half_unavailable_and_two_thirds_full_have_exact_shares():
    full = {"coverage_level": "full_text", "relevance_status": "connected"}
    missing = {"coverage_level": "unavailable"}
    assert citation_tones({"members": [missing, full]}) == ["evidence_available", "not_assessed"]
    assert citation_tones({"members": [full, missing, full]}) == ["evidence_available", "evidence_available", "not_assessed"]


def test_evidence_precedes_reference_and_locator_is_inline():
    member = {"reference_id": "a", "coverage_level": "full_text", "source": {
        "author": "Researcher", "year": "2020", "title": "Research study", "raw_reference": "Researcher (2020). Research study."},
        "evidence_sentences": [{"key": "4:1:9", "text": "An inspectable source sentence.", "page": "5"}],
        "best_evidence": {"text": "An inspectable source passage.", "locator": "Page 5"}}
    html = _render_member(member, grouped=True)
    assert html.index('class="evidence-disclosure"') < html.index('class="full-reference"')
    assert '<span class="ev-page">p. 5</span> <q>An inspectable source sentence.</q>' in html
    assert 'An inspectable source passage.' not in html
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


def test_grouped_missing_members_preserve_independent_census_findings():
    from app.services.evidence_report import _missing_reference_members
    def marker(text, **changes):
        values=dict(text=text,link_status='missing_reference',reference_ids=[],
                    candidate_reference_ids=[],marker_type='parenthetical',member_count=1,
                    passage_start=10,passage_end=30)
        values.update(changes)
        return SimpleNamespace(**values)
    extraction=SimpleNamespace(citations=[],citation_marker_census=[
        marker('(First, 1938)'),marker('(Second & Third, 1941)'),
        marker('(Uncertain, 1940)',candidate_reference_ids=['candidate']),
        marker('(Elsewhere, 1940)',passage_start=101,passage_end=120),
        marker('(1946)'),marker('(First, 1938)')])
    assert _missing_reference_members(extraction,0,100) == ['First, 1938','Second & Third, 1941']


def test_narrative_missing_member_is_in_academic_summary_and_marker_targets():
    from app.services.evidence_report import _missing_reference_members, _build_report_summary
    from app.services.report_member_navigation import missing_reference_targets
    text='Absent (2020) describes the relationship.'
    import hashlib
    citations=[SimpleNamespace(marker_member='Absent (2020)',marker_type='narrative',
        citation_marker='Absent (2020)',is_secondary=False,link_status='missing_reference',
        candidate_reference_ids=[],passage_start=0,passage_end=len(text))]
    assert _missing_reference_members(SimpleNamespace(citations=citations),0,len(text)) == []
    finding=SimpleNamespace(finding_type='missing_reference_entry',passage_start=0,passage_end=len(text),
        marker_text_sha256=hashlib.sha256(b'Absent (2020)').hexdigest(),candidate_reference_ids=[])
    extraction=SimpleNamespace(citations=citations,reference_consistency=SimpleNamespace(findings=[finding]))
    missing=_missing_reference_members(extraction,0,len(text))
    assert missing == ['Absent, 2020']
    citation={'missing_reference_members':missing,'citation_marker':'Absent (2020)',
              'paper_location':{'rectangles':[dict(page_index=0,x0=0,y0=0,x1=200,y1=20)]}}
    assert missing_reference_targets(citation,{0:[(0,0,40,20,'Absent'),(45,0,85,20,'(2020)')]})
    summaries=_build_report_summary(citations=[citation],overview={},pervasive_hanging_indent=False)
    assert any('Absent, 2020' in row for row in map(summary_text, summaries['academic_practice']))
    citations[0].is_secondary=True
    assert _missing_reference_members(extraction,0,len(text)) == []
    citations[0].is_secondary=False
    finding.marker_text_sha256='0'*64
    assert _missing_reference_members(extraction,0,len(text)) == []


@pytest.mark.parametrize('marker', ['...', '. . .', '[...]', '[ . . . ]', '…'])
def test_marked_omissions_match_unchanged_wording_without_a_practice_failure(marker):
    source = 'The independent research team observed several important outcomes in the study. Careful follow-up established significant differences across all three treatment groups.'
    quote = 'The independent research team observed ' + marker + ' significant differences across all three treatment groups.'
    match = _quotation_match(source, quote)
    assert match and match[2] == 'ellipsis_normalized'
    check = _positive_quote_check([quote], [source])
    assert not check['attention']
    # No coaching sentence (owner decision 2026-09-30).
    assert check['label'] == 'Unchanged quotation wording matches the source around the marked brackets or omissions.'


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
    assert check and check['label'] == 'Unchanged quotation wording matches the source around the marked brackets or omissions.'
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
