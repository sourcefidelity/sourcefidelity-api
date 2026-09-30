"""Owner report corrections: broad topic, compact diagnostics, orange fills."""
import hashlib
from app.services.report_layers import topical_mismatch
from app.services.evidence_report import _render_reference_panel_template, render_evidence_report_html
from app.services.passage_relevance import assess_abstract_relevance
from app.services.verification_evidence import ClaimEvidence, ClaimContextSegment
from app.services.source_resolver import _web_link_differences
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('relation,warning',[('overlapping',False),('uncertain',False),('disjoint',True)])
def test_scope_requires_disjoint_topics_and_supplies_context(monkeypatch,relation,warning):
    claim=ClaimEvidence(claim_id='c',paper_version_id='p',text='Her Hollywood work concerned stereotypes.',
        antecedent_context=[ClaimContextSegment(context_index=0,distance_before=1,text='The actor worked in Hollywood.',paper_start=0,paper_end=35)])
    abstract='The book describes the actor’s career.'
    def respond(system,prompt,**kwargs):
        assert 'biography' in system and 'BROAD TOPIC' in system
        assert 'The actor worked in Hollywood.' in prompt and 'Actor biography' in prompt
        return dict(assessments=[dict(passage_id='abstract',relevance='not_relevant',confidence='high',
            evidence_role='methods_or_background',rationale='No interview details.')],
            scope=dict(relevance='apparent_mismatch',confidence='high',discrepancy='different_subject',
                abstract_span=abstract,claim_span=claim.text,rationale='Compared broad topics.',topic_relation=relation,
                broad_subject_relation='incompatible' if warning else 'compatible',
                plausible_connection='absent' if warning else 'present',subject_comparison='Mocked broad-subject comparison.'))
    monkeypatch.setattr('app.services.passage_relevance.chat_completion_json',respond)
    result=assess_abstract_relevance(claim,abstract,source_title='Actor biography')
    assert result['scope_assessment']['attention'] is warning
    member=dict(coverage_level='abstract_only',best_evidence=dict(text=abstract),
        reference_identity=dict(status='confirmed'),abstract_relevance=result)
    assert topical_mismatch(member,dict(student_text=claim.text)) is warning
    result['scope_assessment'].pop('scope_policy_version')
    assert not topical_mismatch(member,dict(student_text=claim.text))


def test_submitted_link_panel_has_no_operational_disclosure():
    finding=dict(finding_type='submitted_link_issue',finding='The submitted link returned a missing or removed page when checked.',
        source=dict(raw_reference='A book. https://example.org/book'),submitted_link_observations=[{}])
    html=_render_reference_panel_template(finding,1)
    assert 'missing or removed' in html
    assert '<details' not in html and 'Submitted link check' not in html


def test_key_is_two_lines_with_each_mark_on_its_own_label():
    # Owner layout 2026-09-28.
    from bs4 import BeautifulSoup
    html=render_evidence_report_html(dict(title='Example',citation_format='APA',citations=[],paper_surface={}),csp_nonce='feedback-test-nonce')
    key=BeautifulSoup(html,'html.parser').select_one('#active-key')
    lines=[' '.join(line.get_text(' ').split()) for line in key.select('.key-line')]
    assert lines==['Judgment: Supported Qualified or Mixed Contradicts Insufficient evidence LLM Undecided Not judged',
                   'Issues: Academic-Practice Citation/Reference Issue Unverifiable Reference Source Record Conflict Submitted-Link']
    assert key.select_one('.key-practice').get_text()=='Academic-Practice'
    assert key.select_one('.key-reference').get_text()=='Citation/Reference Issue'
    assert key.select_one('.key-unverified').get_text()=='Unverifiable Reference'
    assert key.select_one('.key-record').get_text()=='Source Record Conflict'
    assert key.select_one('.key-link i.submitted-link-key') is not None
    assert not key.select('.reference-key,.relevance-key,.practice-key,.unverified-key,.difference-key,.badge-key')
    assert 'orange underline' not in html
    assert '.reference-practice-overlay .mark-relevance{fill:#ef82ba;' in html


def test_only_conflicting_fields_are_explained():
    comparisons=[SimpleNamespace(field_name=f,outcome=o) for f,o in [('title','agreement'),('year','material_conflict'),('author','unknown')]]
    result=SimpleNamespace(title='A title',authors=[],year='2016',doi=None)
    assert _web_link_differences(comparisons,result,title='A title',author='Someone',year='2017',doi=None)==[
        dict(field='year',submitted='2017',destination='2016')]


@pytest.mark.parametrize('label,expected',[('Second edition',True),('Third edition',False),('',False),('Second edition\nThird edition',False),('A review of the second edition',False)])
def test_accepted_edition_label_does_not_infer_equivalence(label,expected):
    import fitz
    from app.services.source_resolver import _accepted_copy_has_edition_label
    with fitz.open() as doc:
        doc.new_page().insert_text((40,50),label)
        assert _accepted_copy_has_edition_label(doc.tobytes(),'Some book (2nd ed.)') is expected
