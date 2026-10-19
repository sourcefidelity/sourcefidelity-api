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
    assert BeautifulSoup(str(soup.select_one('h3')),'html.parser').get_text()=='Not Judged - Abstract Retrieved – Possible Topical Mismatch'
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


def matched_member():
    member, citation = bound_member()
    scope = member['abstract_relevance']['scope_assessment']
    scope.update(relevance='generally_relevant', attention=False, topic_relation='overlapping',
                 broad_subject_relation='compatible', plausible_connection='present')
    return member, citation


def test_a_generally_relevant_comparison_is_a_topical_match_under_the_mismatch_conditions():
    from app.services.report_layers import topical_match
    member, citation = matched_member()
    assert topical_match(member, citation) and not topical_mismatch(member, citation)
    member['abstract_relevance']['scope_assessment']['relevance'] = 'uncertain'
    assert not topical_match(member, citation)
    for change in ({'reference_identity': {'status': 'possible_match'}}, {'coverage_level': 'full_text'}):
        other, cited = matched_member()
        other.update(change)
        assert not topical_match(other, cited)
    other, cited = matched_member()
    cited['members'] = [other, {}, {}]   # a statement naming three sources shows neither
    assert not topical_match(other, cited)
    other, cited = matched_member()
    cited['student_text'] += ' Changed.'  # no longer the statement assessed
    assert not topical_match(other, cited)


def test_the_window_names_the_topic_in_words():
    from app.services.evidence_report import _member_label
    member, citation = matched_member()
    assert _member_label(member, citation).endswith('Abstract Retrieved</span></span> – Topical Match')
    mismatch, cited = bound_member()
    assert 'Topical Match' not in _member_label(mismatch, cited)


def test_a_possible_match_agreeing_on_title_and_author_carries_topic_labels():
    from app.services.report_layers import topical_match
    member, citation = matched_member()
    member['reference_identity'] = {'status': 'possible_match', 'title_and_author_agree': True}
    assert topical_match(member, citation)
    member['reference_identity'] = {'status': 'possible_match', 'title_and_author_agree': False}
    assert not topical_match(member, citation)


def test_a_clause_citation_is_checked_against_the_whole_statement_assessed():
    from app.services.report_layers import topical_match
    member, citation = matched_member()
    clause = {'student_text': 'and a clause of it (Writer, 2020).', 'members': [member],
              'clause_of': {'student_text': citation['student_text']}}
    assert topical_match(member, clause)
    mismatch, cited = bound_member()
    assert topical_mismatch(mismatch, {'student_text': 'a clause.', 'members': [mismatch],
                                       'clause_of': {'student_text': cited['student_text']}})


def test_the_text_of_an_identified_single_authored_book_carries_topic_labels():
    from app.services.report_layers import topical_match
    member, citation = matched_member()
    member['reference_identity'] = {'status': 'search_incomplete'}
    assert not topical_match(member, citation)
    member.update(located_book={'title': 'Material strength'}, text_is_located_book=True)
    assert topical_match(member, citation)
    member['text_is_located_book'] = False   # text from elsewhere stays unlabelled
    assert not topical_match(member, citation)


def test_an_unsettled_comparison_reads_topical_relevance_uncertain():
    from app.services.evidence_report import _member_label, _render_panel_template
    from app.services.report_layers import topical_uncertain
    member, citation = matched_member()
    scope = member['abstract_relevance']['scope_assessment']
    scope['relevance'] = 'uncertain'
    member['abstract_scope_note'] = {'note': 'Both concern material strength.'}
    assert topical_uncertain(member, citation)
    assert _member_label(member, citation).endswith(' – Topical Relevance Uncertain')
    html = _render_panel_template(citation, 1)
    assert 'Both concern material strength. Read the source to judge whether it supports the statement.' in html
    scope['relevance'] = 'generally_relevant'   # a match keeps its note without the sentence
    html = _render_panel_template(citation, 1)
    assert 'Read the source to judge' not in html and 'Topical Match' in html
