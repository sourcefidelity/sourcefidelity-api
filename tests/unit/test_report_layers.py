"""Independent presentation layers cannot manufacture assessments."""
from copy import deepcopy
import hashlib

import fitz
import pytest

from app.services.report_layers import member_layers, topical_mismatch, member_marks
from app.services.report_member_navigation import availability_tone
from app.services.evidence_report import render_evidence_report_html
from app.services.report_export import _render_pdf, _TONE_COLORS


def mismatch():
    text, claim = 'The study concerns fish.', 'The study concerns birds.'
    hashes = {'abstract_sha256': hashlib.sha256(text.encode()).hexdigest(), 'claim_sha256': hashlib.sha256(claim.encode()).hexdigest()}
    scope = {**hashes, 'status': 'complete', 'relevance': 'apparent_mismatch', 'attention': True,
             'scope_policy_version': 'abstract-topic-v4', 'topic_relation': 'disjoint',
             'broad_subject_relation': 'incompatible', 'plausible_connection': 'absent', 'subject_comparison': 'Different broad subjects with no plausible connection.',
             'confidence': 'high', 'discrepancy': 'different_subject', 'abstract_span': 'fish', 'claim_span': 'birds', 'rationale': 'Different subjects.'}
    return {'coverage_level': 'abstract_only', 'best_evidence': {'text': text},
            'reference_identity': {'status': 'confirmed'},
            'abstract_relevance': {**hashes, 'status': 'complete', 'scope_assessment': scope}}, {'student_text': claim}


def test_bound_topical_mismatch_has_separate_layer_without_changing_coverage():
    member, citation = mismatch()
    before = deepcopy(member)
    assert member_layers(member, citation) == {'relevance'}
    assert availability_tone(member) == 'limited_evidence'
    assert member == before
    assert 'mark-relevance' in member_marks(member, citation, dict(x0=10, y0=10, x1=30, y1=20))


@pytest.mark.parametrize('break_binding', ['coverage', 'text', 'claim', 'truncated', 'rationale', 'type', 'span'])
def test_mismatch_abstains_when_evidence_is_removed_or_binding_fails(break_binding):
    member, citation = mismatch()
    if break_binding == 'coverage': member['coverage_level'] = 'unavailable'
    elif break_binding == 'text': member['best_evidence']['text'] = ''
    elif break_binding == 'claim': citation['student_text'] += ' Changed.'
    elif break_binding == 'truncated': member['best_evidence']['excerpt_truncated'] = True
    else:
        key = {'rationale': 'rationale', 'type': 'discrepancy', 'span': 'abstract_span'}[break_binding]
        member['abstract_relevance']['scope_assessment'][key] = ''
    assert not topical_mismatch(member, citation)


@pytest.mark.parametrize('coverage', ['full_text', 'partial_text', 'abstract_only', 'unavailable'])
def test_missing_passage_or_legacy_boolean_is_not_topical_mismatch(coverage):
    member = {'coverage_level': coverage, 'relevance_status': 'no_connection', 'abstract_scope_attention': True}
    assert member_layers(member, {'student_text': 'Claim'}) == set()


def test_three_display_controls_and_distinct_available_text_states():
    view = {'title': 'Synthetic report', 'citation_format': 'APA', 'citations': [], 'paper_surface': {}}
    html = render_evidence_report_html(view, csp_nonce='six-layer-test-nonce')
    # Owner decision 2026-09-27: Sources and Judgment only; no Paper layout.
    assert 'id="paper-layout"' not in html and 'id="sources-layout"' not in html and 'id="judgment-layout"' not in html
    assert 'id="judgment-layout"' not in html and 'id="layer-judgment"' not in html
    for name in ['evidence', 'relevance', 'reference', 'practice', 'copying']:
        assert f'id="layer-{name}"' not in html
    assert "name !== changed" not in html
    tones = {availability_tone({'coverage_level': level}) for level in ['full_text', 'partial_text', 'abstract_only', 'unavailable']}
    assert len(tones) == 4
    assert len({_TONE_COLORS[tone] for tone in tones}) == 4
    from bs4 import BeautifulSoup
    parsed = BeautifulSoup(html, 'html.parser')
    bar = parsed.select_one('#report-layout > .workspace-bar')
    assert bar.select_one('.evidence-controls [data-report-step]')
    assert len(bar.select('[data-report-step]')) == 2
    # The window and splitter slide together as one pane beside the paper.
    assert parsed.select_one('#report-layout > .side-pane > .evidence-column > #evidence-panel')
    assert parsed.select_one('#report-layout > .side-pane > #report-splitter')
    assert not parsed.select('header #paper-layout')


def test_quotation_and_locator_issues_are_not_relevance_or_judgment():
    member = {'coverage_level': 'full_text', 'show_quotation_check': True,
              'quotation_check': {'attention': True, 'status': 'complete'}, 'show_locator_check': True,
              'locator_check': {'attention': True, 'status': 'complete'}}
    # A locator issue is Academic Practice, yellow on the citation (2026-10-03).
    from app.services.report_layers import locator_attention
    assert member_layers(member, {}) == {'practice'} and locator_attention(member)
    member['coverage_level'] = 'unavailable'
    assert member_layers(member, {}) == set()


def test_simultaneous_marks_do_not_paint_over_the_source_fill():
    member, citation = mismatch()
    member.update(source={'year': '2020'}, reference_findings=[{'finding_type':'duplicate_citation_key'}],
                  show_quotation_check=True, quotation_check={'attention':True, 'status':'complete'})
    target = dict(x0=10, y0=10, x1=80, y1=20)
    markup = member_marks(member, citation, target, [(50,10,70,20,'2020')])
    assert 'mark-relevance" x="8.500"' in markup
    assert 'indicator-reference' in markup
    assert 'indicator-practice' in markup
    assert 'mark-practice' not in markup  # Wording geometry lives on the citation, not its source marker.
    assert '<rect class="layer-mark mark-reference"' not in markup
    assert '<rect class="layer-indicator indicator-reference"' in markup
    assert '<line class="layer-indicator indicator-reference"' not in markup
    assert '<circle class="layer-indicator indicator-reference"' not in markup


def test_partial_text_pdf_has_mauve_fill_and_neutral_span():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72,72), 'A claim (Writer, 2020).')
        content = doc.tobytes()
    citation = {'citation_marker': '(Writer, 2020)', 'members': [{'coverage_level': 'partial_text', 'source': {'author': 'Writer', 'year': '2020'}}],
                'paper_location': {'rectangles': [{'page_index': 0, 'x0': 70, 'y0': 50, 'x1': 300, 'y1': 85}]}}
    output, counts = _render_pdf(content, citations=[citation], reference_practice=[], export_binding='synthetic')
    with fitz.open(stream=output, filetype='pdf') as pdf:
        fills = [d['fill'] for d in pdf[0].get_drawings() if d['fill']]
        assert any(all(abs(a-b)<.002 for a,b in zip(fill,_TONE_COLORS['partial_evidence'])) for fill in fills)
    assert counts['citation_underlines'] == counts['source_member_highlights'] == 1
