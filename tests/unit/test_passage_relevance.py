"""Regressions for the bounded passage-relevance gate."""

from datetime import datetime, timezone
import hashlib
import json

import pytest

from app.services.evidence_conditioned_judgment import (
    apply_evidence_conditioned_unit_judgment,
)
from app.services.passage_relevance import apply_passage_relevance_gate
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    VerificationVerdict,
    build_passage_evidence,
)


def _artifact(*, top_k=3):
    source_text = "\f".join(
        [
            "Licensing supports competition by preserving diverse providers.",
            "Licensing can reduce barriers to entry for new market participants.",
            "Licensing supports competition through filing procedures.",
            "Licensing reduces market barriers in historical settings.",
        ]
    )
    content = source_text.encode()
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="representation-1",
        canonical_work_id="work-1",
        content_object_id="object-1",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        representation_kind="plain_text",
        media_type="text/plain",
        provenance="authorized_upload",
        scope_type="personal_owner",
        scope_id="owner-1",
        identity_verdict="verified",
        identity_confidence=0.99,
        completeness_verdict="complete",
        text_quality="digital",
        edition_or_version="edition-1",
        created_at=now,
        admitted_at=now,
    )
    text = (
        "Licensing supports competition and reduces barriers "
        "(Smith, 2020). Contact reviewer@example.edu"
    )
    claim = ClaimEvidence(
        claim_id="citation-unit-1",
        paper_version_id="paper-v1",
        text=text,
        granularity="citation_unit",
        reference_ids=["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        extraction_confidence="high",
        passage_start=100,
        passage_end=100 + len(text),
    )
    return build_passage_evidence(source, claim=claim, top_k=top_k)


def test_relevant_and_partial_candidates_are_selected_without_changing_verdict(
    monkeypatch,
):
    artifact = _artifact()
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        captured["call_kwargs"] = kwargs
        ids = [passage["passage_id"] for passage in payload["passages"]]
        return {
            "assessments": [
                {
                    "passage_id": ids[0],
                    "relevance": "relevant",
                    "confidence": "high",
                    "rationale": "Addresses licensing and competition.",
                },
                {
                    "passage_id": ids[1],
                    "relevance": "partially_relevant",
                    "confidence": "medium",
                    "rationale": "Addresses barriers only.",
                },
                {
                    "passage_id": ids[2],
                    "relevance": "not_relevant",
                    "confidence": "high",
                    "rationale": "Procedural material.",
                },
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert assessed.passage_relevance.status == "complete"
    assert assessed.passage_relevance.outcome == "relevant_candidates_found"
    assert assessed.passage_relevance.relevant_passage_ids == [
        passage["passage_id"] for passage in captured["passages"][:2]
    ]
    assert len(captured["passages"]) == 3
    assert "reviewer@example.edu" not in captured["citation_unit"]
    assert assessed.passage_relevance.decision_applied is False
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert captured["call_kwargs"]["disable_thinking"] is True


def test_all_not_relevant_uses_all_three_in_shadow_judgment_fallback(
    monkeypatch,
):
    artifact = _artifact()

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "confidence": "high",
                    "rationale": "Does not address the citation assertion.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"

    captured = {}

    def fallback_judgment(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        return {
            "findings": [{
                "segments": [{
                    "start_id": payload["unit_tokens"][0][0],
                    "end_id": payload["unit_tokens"][4][0],
                }],
                "attribution": "cited_source",
                "relationship": "insufficient_evidence",
                "confidence": "low",
                "passage_ids": [payload["passages"][0]["passage_id"]],
                "rationale": "Recall-preserving fallback found no adequate support.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fallback_judgment,
    )
    judged = apply_evidence_conditioned_unit_judgment(assessed)
    assert judged.unit_judgment.status == "incomplete"
    assert judged.unit_judgment.outcome == "segmented_findings"
    assert len(captured["passages"]) == 3
    assert judged.unit_judgment.findings[0].relationship.value == (
        "insufficient_evidence"
    )
    assert any(
        "recall-preserving shadow fallback" in limitation
        for limitation in judged.unit_judgment.limitations
    )
    assert judged.verdict is VerificationVerdict.INCONCLUSIVE


def test_uncertain_relevance_stops_segmented_judgment(monkeypatch):
    artifact = _artifact()

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "uncertain",
                    "confidence": "low",
                    "rationale": "Insufficient context.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    assessed = apply_passage_relevance_gate(artifact)
    judged = apply_evidence_conditioned_unit_judgment(assessed)
    assert assessed.passage_relevance.outcome == "uncertain"
    assert judged.unit_judgment.status == "not_assessed"
    assert judged.unit_judgment.outcome == "uncertain_relevance"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "invented"])
def test_invalid_application_owned_passage_ids_fail_closed(monkeypatch, failure):
    artifact = _artifact()

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        ids = [passage["passage_id"] for passage in payload["passages"]]
        if failure == "missing":
            ids = ids[:-1]
        elif failure == "duplicate":
            ids[-1] = ids[0]
        else:
            ids[-1] = "invented-passage-id"
        return {
            "assessments": [
                {
                    "passage_id": passage_id,
                    "relevance": "relevant",
                    "confidence": "medium",
                    "rationale": "Test.",
                }
                for passage_id in ids
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.status == "not_assessed"
    assert assessed.passage_relevance.outcome == "not_assessed"
    assert assessed.passage_relevance.relevant_passage_ids == []


def test_mismatched_authorization_never_reaches_relevance_model(monkeypatch):
    artifact = _artifact()
    artifact = artifact.model_copy(
        update={
            "passages": [
                passage.model_copy(update={"authorization_scope_id": "other-owner"})
                for passage in artifact.passages
            ]
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("relevance model must not run")

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fail_if_called
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.status == "not_assessed"
    assert assessed.passage_relevance.method == "no_authorized_candidate_passage"


def test_retrieval_can_preserve_ten_candidates_for_calibration():
    artifact = _artifact(top_k=10)
    assert len(artifact.passages) == 4


def test_relevance_gate_uses_only_top_three_from_larger_calibration_pool(
    monkeypatch,
):
    artifact = _artifact(top_k=10)
    assert len(artifact.passages) == 4
    captured_ids = []

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured_ids.extend(
            passage["passage_id"] for passage in payload["passages"]
        )
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "confidence": "high",
                    "rationale": "Bounded cutoff regression.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    assessed = apply_passage_relevance_gate(artifact)

    expected_ids = [
        passage.passage_id
        for passage in sorted(
            artifact.passages,
            key=lambda passage: passage.retrieval_score,
            reverse=True,
        )[:3]
    ]
    assert captured_ids == expected_ids
    assert len(captured_ids) == 3
    assert artifact.passages[3].passage_id not in captured_ids
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"
