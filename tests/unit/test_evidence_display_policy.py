"""Display-policy baseline controls, not semantic model acceptance.

Use the real selector binding, saved payload and both renderers. The mocked
choices test plumbing only; they do not certify the model's usefulness labels.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.services import joint_evidence_selection as joint
from app.services.evidence_report import _available_member, _render_member
from app.services.report_export import _portable_member_html
from app.services.schemas import ParsedReference
from app.services.verification_evidence import (
    CandidatePassageRelevanceEvidence, PassageRelevanceGateEvidence,
)
from app.services.verification_report import build_inspectable_report_payload
from tests.unit.test_passage_relevance import _artifact


def prepared():
    artifact = _artifact()
    artifact.passage_relevance = PassageRelevanceGateEvidence(
        status="complete", assessments=[CandidatePassageRelevanceEvidence(
            passage_id=p.passage_id, relevance="partially_relevant",
            confidence="high", evidence_role="source_own_claim_or_finding",
            assessed_text_offset_end=len(p.text),
            assessed_text_sha256=hashlib.sha256(p.text.encode()).hexdigest(),
        ) for p in artifact.passages])
    return artifact


def project(artifact):
    payload = json.loads(json.dumps(build_inspectable_report_payload(artifact)))
    before = deepcopy(payload)
    package = payload["authoritative_evidence_package"]
    binding = {k: package[k] for k in ("package_id", "package_sha256", "claim_id")}
    binding["reference_id"] = artifact.source_binding.reference_id
    reference = ParsedReference(reference_id=binding["reference_id"],
        raw_ref="Smith (2020). Study. https://example.org/study", title="Study",
        author="Smith", year="2020", url="https://example.org/study")
    view = _available_member(binding, SimpleNamespace(id="policy-control", report_payload=payload),
                             reference, artifact.claim)
    assert payload == before
    return payload, view


def test_valid_empty_survives_serialization_and_both_renderers(monkeypatch):
    artifact = prepared()
    original_passages = deepcopy(artifact.passages)
    monkeypatch.setattr(joint, "chat_completion_json", lambda *a, **k: {"selected": []})
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    assert joint.validate_joint_selection(artifact, artifact.joint_evidence_selection)
    payload, view = project(artifact)
    assert view["display_selection"]["joint_selection_status"] == "applied"
    assert view["best_evidence"] is None
    assert view["evidence_extracts"] == view["additional_evidence"] == []
    assert len(payload["authoritative_evidence_package"]["passages"]) == len(original_passages)
    assert artifact.passages == original_passages
    for html in (_render_member(view), _portable_member_html(view)):
        assert "<blockquote" not in html
        assert "https://example.org/study" in html
        assert "Study" in html


@pytest.mark.parametrize("failure", ["timeout", "invalid", "stale", "budget"])
def test_failure_is_not_a_completed_empty_selection(monkeypatch, failure):
    artifact = prepared()
    before = deepcopy(artifact.passages)
    def response(*a, **k):
        if failure == "timeout":
            raise TimeoutError("private provider detail")
        return {"selected": []} if failure != "invalid" else {"unexpected": "private"}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    if failure == "budget":
        monkeypatch.setattr(joint.settings, "VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS", 1)
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    if failure == "stale":
        artifact.joint_evidence_selection.input_fingerprint = "stale"
    assert not joint.validate_joint_selection(artifact, artifact.joint_evidence_selection)
    _, view = project(artifact)
    assert view["display_selection"]["joint_selection_status"] == "fallback"
    assert "evidence_extracts" not in view
    assert view["best_evidence"] is not None
    assert artifact.passages == before
    assert "private provider detail" not in artifact.joint_evidence_selection.model_dump_json()


@pytest.mark.parametrize("label", ["relevant", "partially_relevant"])
def test_partial_evidence_eligible_without_support_or_complete_claim(monkeypatch, label):
    artifact = prepared()
    for assessment in artifact.passage_relevance.assessments:
        assessment.relevance = label
    def response(system, prompt, **kwargs):
        p = json.loads(prompt)["candidates"][0]
        return {"selected": [{"passage_id": p["passage_id"], "reason": "primary",
                              "claim_token_ranges": [[0, 2]], "source_sentence_ids": ["s000"]}]}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    verdict = deepcopy(artifact.verdict)
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    _, view = project(artifact)
    assert view["display_selection"]["joint_selection_status"] == "applied"
    assert len(view["evidence_extracts"]) == 1
    assert view["display_selection"]["support_assessed"] is False
    assert artifact.verdict == verdict


@pytest.mark.parametrize("kind", ["quotation_check", "locator_check"])
def test_empty_advice_cannot_suppress_protected_evidence(monkeypatch, kind):
    artifact = prepared()
    getattr(artifact, kind).evidence_passage_ids = [artifact.passages[0].passage_id]
    monkeypatch.setattr(joint, "chat_completion_json", lambda *a, **k: {"selected": []})
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    assert artifact.joint_evidence_selection.status == "not_assessed"
    _, view = project(artifact)
    assert view["display_selection"]["joint_selection_status"] == "fallback"
    assert view["best_evidence"]["passage_id"] == artifact.passages[0].passage_id


def test_all_rejected_is_distinct_from_provider_call_returning_empty(monkeypatch):
    artifact = prepared()
    for assessment in artifact.passage_relevance.assessments:
        assessment.relevance = "not_relevant"
    def forbidden(*a, **k):
        pytest.fail("No eligible input must not spend a provider call")
    monkeypatch.setattr(joint, "chat_completion_json", forbidden)
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    receipt = artifact.joint_evidence_selection
    assert receipt.status == "complete" and receipt.call_count == 0
    assert receipt.candidates == receipt.selected == []
    _, view = project(artifact)
    assert view["best_evidence"] is None


def test_out_of_range_claim_endpoint_has_specific_private_diagnostic(monkeypatch):
    import re
    artifact = prepared()
    count = len(re.findall(r"\S+", joint._source_attributed_relevance_text(artifact.claim)))
    def response(system, prompt, **kwargs):
        pid = json.loads(prompt)["candidates"][0]["passage_id"]
        return {"selected": [{"passage_id": pid, "reason": "primary",
            "claim_token_ranges": [[0, count]], "source_sentence_ids": ["s000"]}]}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    artifact.joint_evidence_selection = joint.select_joint_evidence(artifact, source_title="Study")
    result = artifact.joint_evidence_selection
    assert result.status == "not_assessed"
    assert result.limitation_codes == ["claim_token_range_invalid"]
    assert result.selected == []  # No clipping or partial salvage.
    _, view = project(artifact)
    assert view["display_selection"]["joint_selection_status"] == "fallback"
