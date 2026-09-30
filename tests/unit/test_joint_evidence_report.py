"""Joint selection is optional, bound, and presentation-only."""
from copy import deepcopy
import hashlib

import pytest

from app.services.evidence_report import (
    _joint_report_selection, _member_evidence_contexts, _render_member,
)
from app.services.report_export import _portable_member_html


def inputs():
    passages = {key: {'passage_id': key, 'excerpt': text,
        'passage_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        for key, text in [('a', 'The bridge closed. The road was diverted.'),
                          ('b', 'Residents moved. Temporary homes were supplied.') ]}
    package = {'source_identity': {'content_sha256': 'source'},
               'coverage': {'extracted_text_sha256': 'extraction'},
               'source_binding': {'reference_id': 'r'}}
    payload = {**deepcopy(package), 'authoritative_evidence_package': package,
        'joint_selection_context': {'passage_ids': ['a', 'b'], 'claim': {'claim_id': 'c'}},
        'passage_relevance_gate': {'status': 'complete'},
        'joint_evidence_selection': {'status': 'complete', 'source_title': 'Study',
            'version': 'joint-test', 'selected': [
                {'passage_id': 'a', 'reason': 'primary', 'source_span': 'The bridge closed.'},
                {'passage_id': 'b', 'reason': 'distinct_aspect', 'source_span': 'Residents moved.'}]}}
    return payload, passages


def project(payload, passages, preferred=()):
    return _joint_report_selection(payload, 'The bridge closed and residents moved.',
        passages, list(passages.values()), preferred, [passages['a']], {'version': 'legacy'}, source_title='Study')


def test_projection_calls_validator_with_exact_originals_and_saved_context(monkeypatch):
    from app.services import joint_evidence_selection as joint
    payload, passages = inputs()
    before = deepcopy((payload, passages))
    seen = {}
    def validate(receipt, **kwargs):
        seen.update(kwargs)
        return True
    monkeypatch.setattr(joint, 'validate_joint_projection', validate)
    passages['slice'] = {'passage_id': 'slice', 'parent_passage_id': 'a', 'excerpt': 'bridge'}
    selected, spans, trace = project(payload, passages)
    assert [p['passage_id'] for p in selected] == ['a', 'b']
    assert spans == {'a': 'The bridge closed.', 'b': 'Residents moved.'}
    assert seen['passages'] == {k: p['excerpt'] for k, p in before[1].items()}
    assert seen['claim_context'] == payload['joint_selection_context']
    assert trace['joint_selection_status'] == 'applied'
    assert payload == before[0]


@pytest.mark.parametrize('defect', ['invalid', 'title', 'member', 'hash', 'missing_id', 'extra_id', 'ineligible', 'protected'])
def test_bad_receipt_or_protected_conflict_falls_back(monkeypatch, defect):
    from app.services import joint_evidence_selection as joint
    payload, passages = inputs()
    monkeypatch.setattr(joint, 'validate_joint_projection', lambda *a, **k: defect != 'invalid')
    if defect == 'title': payload['joint_evidence_selection']['source_title'] = 'Different study'
    if defect == 'member': payload['source_binding']['reference_id'] = 'other'
    if defect == 'hash': passages['a']['excerpt'] += ' Changed.'
    if defect == 'missing_id': payload['joint_selection_context']['passage_ids'] = ['a']
    if defect == 'extra_id': payload['joint_selection_context']['passage_ids'].append('missing')
    if defect == 'ineligible': payload['joint_evidence_selection']['selected'][1]['passage_id'] = 'unknown'
    selected, spans, trace = project(payload, passages, preferred=['b'] if defect == 'protected' else [])
    assert selected == [passages['a']] and spans is None
    assert trace['joint_selection_status'] == 'fallback'
    if defect == 'protected': assert trace['joint_selection_reason'] == 'joint_protected_priority_conflict'


def member(count=3):
    extracts = [{'passage_id': str(i), 'text': f'Extract {i}. More context {i}.',
                 'display_text': f'Extract {i}.', 'context_text': f'Extract {i}. More context {i}.',
                 'locator': f'Page {i+1}'} for i in range(count)]
    return {'source': {'author': 'A', 'year': '2024', 'title': 'Study', 'raw_reference': 'Reference'},
        'best_evidence': extracts[0], 'additional_evidence': extracts[1:],
        'evidence_extracts': extracts, 'display_selection': {'joint_selection_status': 'applied'}}


@pytest.mark.parametrize('count', [1, 2, 3])
def test_conditional_short_extracts_one_heading_and_bounded_context(count):
    value = member(count)
    before = deepcopy(value)
    html = _render_member(value)
    # The window shows only GLM-selected sentences now (owner decision
    # 2026-09-28); the relevance gate's extracts remain in the PDF.
    assert '<blockquote' not in html and 'Additional evidence and context' not in html
    pdf_html = _portable_member_html(value)
    assert pdf_html.count('<blockquote') == 2 * count
    assert pdf_html.count('Additional evidence and context') == 1
    assert 'Full context for the selected excerpt' not in pdf_html
    assert value == before


def test_no_joint_receipt_keeps_one_historical_primary():
    value = member(3)
    value.pop('display_selection')
    assert '<blockquote' not in _render_member(value)
    assert _portable_member_html(value).count('<blockquote') == 4


def test_context_omits_equal_shorter_or_repeated_text_and_caps_three():
    value = member(4)
    assert len(_member_evidence_contexts(value)) == 3
    value['evidence_extracts'][0]['context_text'] = 'Extract 0.'
    value['evidence_extracts'][0]['text'] = 'Extract 0.'
    value['evidence_extracts'][1]['context_text'] = 'Extract'
    assert len(_member_evidence_contexts(value)) == 1
    value['evidence_extracts'][2]['display_text'] = '<script>unsafe</script>'
    assert '<script>unsafe' not in _render_member(value)
    assert '&lt;script&gt;unsafe' in _portable_member_html(value)


def test_real_payload_validator_roundtrip_and_moved_receipt_rejection(monkeypatch):
    import json
    from types import SimpleNamespace
    from app.services import joint_evidence_selection as joint
    from app.services.verification_evidence import CandidatePassageRelevanceEvidence, PassageRelevanceGateEvidence
    from app.services.verification_report import build_inspectable_report_payload
    from app.services.evidence_report import _available_member
    from app.services.schemas import ParsedReference
    from tests.unit.test_passage_relevance import _artifact
    artifact = _artifact()
    artifact.passage_relevance = PassageRelevanceGateEvidence(status='complete', assessments=[
        CandidatePassageRelevanceEvidence(passage_id=p.passage_id, relevance='partially_relevant',
            confidence='high', evidence_role='source_own_claim_or_finding',
            assessed_text_offset_end=len(p.text), assessed_text_sha256=hashlib.sha256(p.text.encode()).hexdigest())
        for p in artifact.passages])
    def response(system, prompt, **kwargs):
        rows = json.loads(prompt)['candidates'][:2]
        return {'selected': [{'passage_id': p['passage_id'], 'reason': 'primary' if i == 0 else 'distinct_aspect',
                             'claim_token_ranges': [[0, 2]] if i == 0 else [[3, 5]],
                             'source_sentence_ids': ['s000']} for i, p in enumerate(rows)]}
    monkeypatch.setattr(joint, 'chat_completion_json', response)
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title='Study')
    assert artifact.joint_evidence_selection.status == 'complete'
    payload = json.loads(json.dumps(build_inspectable_report_payload(artifact)))
    package = payload['authoritative_evidence_package']
    member_binding = {k: package[k] for k in ('package_id', 'package_sha256', 'claim_id')}
    member_binding['reference_id'] = artifact.source_binding.reference_id
    reference = ParsedReference(reference_id=artifact.source_binding.reference_id,
                                raw_ref='Smith (2020). Study.', title='Study', author='Smith', year='2020')
    before = deepcopy(payload)
    view = _available_member(member_binding, SimpleNamespace(id='synthetic-record', report_payload=payload), reference, artifact.claim)
    assert view['display_selection']['joint_selection_status'] == 'applied'
    assert len(view['evidence_extracts']) == 2
    assert payload == before
    payload['joint_selection_context']['claim']['reference_ids'] = ['another-member']
    fallback = _available_member(member_binding, SimpleNamespace(id='synthetic-record', report_payload=payload), reference, artifact.claim)
    assert fallback['display_selection']['joint_selection_status'] == 'fallback'
    assert 'evidence_extracts' not in fallback
