"""Round-trip joint selection through the actual report payload boundary."""
import hashlib

from tests.unit.test_joint_evidence_selection import case
from app.services.joint_evidence_selection import select_joint_evidence
from app.services.verification_report import build_inspectable_report_payload
from app.services.evidence_report import _joint_report_selection


def test_real_receipt_survives_report_payload_and_rejects_changed_source(case):
    artifact, _ = case
    artifact.joint_evidence_selection = select_joint_evidence(artifact, source_title='Study')
    assert artifact.joint_evidence_selection.status == 'complete'
    payload = build_inspectable_report_payload(artifact)
    passages = {p.passage_id: dict(passage_id=p.passage_id, excerpt=p.text,
        passage_text_sha256=hashlib.sha256(p.text.encode()).hexdigest()) for p in artifact.passages}
    baseline = list(passages.values())[:1]
    args = (payload, artifact.claim.text, passages, list(passages.values()), [], baseline, {})
    selected, spans, trace = _joint_report_selection(*args, source_title='Study')
    assert trace['joint_selection_status'] == 'applied'
    assert spans[selected[0]['passage_id']] == artifact.joint_evidence_selection.selected[0].source_span
    selected[0]['excerpt'] += ' Altered.'
    _, spans, trace = _joint_report_selection(*args, source_title='Study')
    assert spans is None and trace['joint_selection_status'] == 'fallback'
