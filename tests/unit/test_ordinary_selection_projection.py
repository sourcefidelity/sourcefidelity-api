"""Ordinary obligation assessment through selection and report serialization.

All transport is mocked; these assertions establish binding, not usefulness.
"""
import hashlib
import json

import pytest

from app.services import passage_relevance as gate, joint_evidence_selection as joint
from app.services.evidence_report import _joint_report_selection
from app.services.verification_report import build_inspectable_report_payload
from app.services.verification_evidence import (
    CitationSourceBinding, EvidenceObligation, EvidenceObligationSet,
)
from tests.unit.test_passage_relevance import _artifact


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('mode', ['selected', 'empty', 'failure', 'malformed'])
def test_ordinary_obligation_to_report(monkeypatch, mode):
    artifact = _artifact()
    marker = artifact.claim.citation_marker
    start = artifact.claim.text.index(marker)
    artifact.source_binding = CitationSourceBinding(reference_id='reference-1',
        cited_author_label='Smith', marker_text=marker,
        marker_local_start=start, marker_local_end=start + len(marker))
    target = 'Licensing supports competition'
    artifact.evidence_obligations = EvidenceObligationSet(status='complete', obligations=[
        EvidenceObligation(obligation_id='primary', obligation_type='exact_factual_assertion',
            reference_id='reference-1', target_text=target, target_text_sha256=sha(target),
            original_text_sha256=sha(target), derivation_method='exact_source_attributed_text',
            aggregate_scope='not_aggregate', accuracy_judgment_allowed=True,
            coverage_judgment_allowed=False)])
    original = [p.model_dump() for p in artifact.passages]

    def assess(system, prompt, **kwargs):
        if mode == 'failure':
            raise RuntimeError('Mock transport failure')
        if mode == 'malformed':
            return {'assessments': []}
        return {'assessments': [dict(passage_id=p['passage_id'], relevance='relevant',
            confidence='high', evidence_role='source_own_claim_or_finding')
            for p in json.loads(prompt)['passages']]}

    def select(system, prompt, **kwargs):
        assert mode in ('selected', 'empty')
        if mode == 'empty':
            return {'selected': []}
        return {'selected': [dict(passage_id=json.loads(prompt)['candidates'][0]['passage_id'],
            reason='primary', claim_token_ranges=[[0, 2]], source_sentence_ids=['s000'])]}

    monkeypatch.setattr(gate, 'chat_completion_json', assess)
    monkeypatch.setattr(joint, 'chat_completion_json', select)
    artifact = gate.apply_passage_relevance_gate(artifact, source_title='Study')
    assert len(artifact.passage_relevance.obligation_findings) == 1
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title='Study')
    receipt = artifact.joint_evidence_selection
    payload = json.loads(json.dumps(build_inspectable_report_payload(artifact)))
    passages = {p.passage_id: dict(passage_id=p.passage_id, excerpt=p.text,
        passage_text_sha256=sha(p.text)) for p in artifact.passages}
    baseline = list(passages.values())[:1]
    args = (payload, artifact.claim.text, passages, list(passages.values()), [], baseline, {})
    selected, spans, trace = _joint_report_selection(*args, source_title='Study')
    assert [p.model_dump() for p in artifact.passages] == original
    if mode in ('failure', 'malformed'):
        assert receipt.status == 'not_assessed' and receipt.call_count == 0
        assert trace['joint_selection_status'] == 'fallback'
        assert selected == baseline and spans is None
    else:
        assert receipt.status == 'complete' and receipt.target_text_sha256 == sha(target)
        assert trace['joint_selection_status'] == 'applied'
        assert len(selected) == int(mode == 'selected')
        if selected:
            assert spans[selected[0]['passage_id']] == receipt.selected[0].source_span
        artifact.source_binding.reference_id = 'other-member'
        assert not joint.validate_joint_selection(artifact, receipt)
