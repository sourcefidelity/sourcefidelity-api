from app.services.evidence_report import summary_text
from copy import deepcopy
import hashlib
import pytest
from bs4 import BeautifulSoup

from app.services.report_layers import topical_mismatch
from app.services.evidence_report import _build_report_summary, _render_panel_template, render_evidence_report_html
from app.services.passage_relevance import assess_abstract_relevance
from app.services.verification_evidence import ClaimEvidence
from app.services.retrieval.landing_page import discover_scholarly_locations
from app.services.highlight_priority import prioritize_svg_highlights


def bound_member():
    text='The experiment measures asbestos fibre strength.'
    claim='Narrative archetypes organize a fictional storyworld.'
    hashes=dict(abstract_sha256=hashlib.sha256(text.encode()).hexdigest(),
                claim_sha256=hashlib.sha256(claim.encode()).hexdigest())
    scope=dict(status='complete',scope_policy_version='abstract-topic-v4',relevance='apparent_mismatch',
        topic_relation='disjoint',broad_subject_relation='incompatible',plausible_connection='absent',
        subject_comparison='Material strength and narrative structure are different subjects.',
        confidence='high',discrepancy='different_subject',attention=True,
        abstract_span='asbestos fibre strength',claim_span='Narrative archetypes',
        rationale='The source concerns material strength, not narrative structure.',**hashes)
    member=dict(coverage_level='abstract_only',best_evidence=dict(text=text),
        source=dict(raw_reference='Writer (2020). Material strength.',author='Writer',year='2020',title='Material strength'),
        reference_identity=dict(status='confirmed'),
        abstract_relevance=dict(status='complete',scope_assessment=scope,**hashes))
    return member,dict(student_text=claim,members=[member])


@pytest.mark.parametrize('field,value', [('scope_policy_version','abstract-topic-v3'),
    ('broad_subject_relation','compatible'),('broad_subject_relation','uncertain'),
    ('plausible_connection','present'),('plausible_connection','uncertain'),('subject_comparison','')])
def test_old_or_compatible_scope_does_not_authorize_pink(field,value):
    member,citation=bound_member()
    assert topical_mismatch(member,citation)
    member['abstract_relevance']['scope_assessment'][field]=value
    assert not topical_mismatch(member,citation)


def test_scope_producer_requires_new_checks_not_just_model_mismatch(monkeypatch):
    member,citation=bound_member()
    scope=deepcopy(member['abstract_relevance']['scope_assessment'])
    for key in ('status','scope_policy_version','attention','abstract_sha256','claim_sha256'):
        scope.pop(key)
    response=dict(assessments=[dict(passage_id='abstract',relevance='not_relevant',
        evidence_role='source_own_claim_or_finding',confidence='high',rationale='Different subjects.')],scope=scope)
    monkeypatch.setattr('app.services.passage_relevance.chat_completion_json',lambda *a,**k:response)
    claim=ClaimEvidence(claim_id='c',paper_version_id='p',text=citation['student_text'])
    assess=lambda:assess_abstract_relevance(claim,member['best_evidence']['text'])['scope_assessment']
    assert assess()['attention']
    scope['plausible_connection']='present'
    assert not assess()['attention']
    scope.pop('broad_subject_relation')
    assert assess()['status']=='not_assessed'


def test_both_audiences_summarize_and_window_heading_identifies_mismatch():
    member,citation=bound_member()
    result=_build_report_summary(citations=[citation],overview={},pervasive_hanging_indent=False)
    assert any('1 source is possibly not related to the citation' in row for row in map(summary_text, result['evidence']))
    html=_render_panel_template(citation,1)
    soup=BeautifulSoup(html,'html.parser')
    assert BeautifulSoup(str(soup.select_one('h3')),'html.parser').get_text()=='Not Judged - Abstract Retrieved – possible topical mismatch'
    assert soup.select_one('h3 mark.topical')
    result=_build_report_summary(citations=[citation],overview={},pervasive_hanging_indent=False,require_paper_flags=True)
    assert not result['evidence']


def test_geometric_priority_is_pink_below_formatting():
    html=render_evidence_report_html(dict(title='Test',citation_format='APA',citations=[],paper_surface={}),csp_nonce='test-pink-policy-nonce')
    # The key no longer lists the pink mark (owner layout 2026-09-28); How to read explains it.
    assert 'pink highlights indicate a possible topical mismatch' in html
    assert 'Gold outlines' not in html and 'gold outline:' not in html
    marks=''.join(f'<rect class="{name}" x="0" y="0" width="10" height="10"/>'
                  for name in ('source-highlight','mark-relevance','reference-formatting-hit','academic-highlight'))
    soup=BeautifulSoup(prioritize_svg_highlights(marks),'html.parser')
    assert [r['class'] for r in soup.select('rect')]==[['academic-highlight']]


def test_feeds_are_not_article_xml_and_view_pdf_is_retained():
    html='''<link rel="alternate" type="application/rss+xml" href="/feed/">
    <link rel="alternate" type="application/atom+xml" href="/comments/">
    <link rel="alternate" type="application/xml" href="/article.xml">
    <a href="http://[malformed">View PDF</a>
    <a href="/download?id=1">View PDF</a><a href="/paper.pdf?download=1">Read</a>'''
    locations=discover_scholarly_locations(html,'https://journal.example/article')
    assert [l.url for l in locations]==['https://journal.example/article.xml',
        'https://journal.example/download?id=1','https://journal.example/paper.pdf?download=1']
