"""Bounded structured judgment security and decision-policy tests."""

from datetime import datetime, timezone
import hashlib
import json

from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimAntecedentDependency,
    ClaimEvidence,
    RelationshipStatus,
    VerificationVerdict,
    build_passage_evidence,
)
from app.services.verification_judgment import apply_bounded_structured_judgment


def _artifact(*, atomic=True, blocks=1, context_dependency=False):
    source_text = "\n\n".join(
        f"Passage {index} reports that cultural exports improve national image through repeated exposure."
        for index in range(blocks)
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
    claim_text = (
        "This pressure improves national image."
        if context_dependency
        else "Student: person@example.com argues cultural exports improve national image."
    )
    dependencies = (
        [
            ClaimAntecedentDependency(
                mention_text="This pressure",
                mention_local_start=0,
                mention_local_end=len("This pressure"),
                mention_paper_start=0,
                mention_paper_end=len("This pressure"),
                resolution_status="resolved",
                confidence="high",
                antecedent_context_index=0,
                antecedent_text=(
                    "competition among cultural exporters creates market pressure"
                ),
                antecedent_paper_start=100,
                antecedent_paper_end=160,
                method="test",
            )
        ]
        if context_dependency
        else []
    )
    claim = ClaimEvidence(
        claim_id="claim-1",
        paper_version_id="paper-v1",
        text=claim_text,
        granularity="atomic_claim" if atomic else "citation_unit",
        atomization_method="test" if atomic else "not_run",
        reference_ids=["reference-1"],
        passage_start=0,
        passage_end=len(claim_text),
        antecedent_dependencies=dependencies,
        context_dependency_status="resolved" if dependencies else "not_required",
    )
    return build_passage_evidence(source, claim=claim, top_k=5)


def test_valid_judgment_records_ids_and_stays_shadow_by_default(monkeypatch):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        captured["data"] = json.loads(user_prompt)
        return {
            "relationship": "supports",
            "confidence": "high",
            "passage_ids": [passage_id],
            "rationale": "The passage directly states the same relationship.",
            "limitations": [],
        }

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json", fake_call
    )
    judged = apply_bounded_structured_judgment(artifact)

    assert judged.judgment.status is RelationshipStatus.SUPPORTS
    assert judged.judgment.passage_ids == [passage_id]
    assert judged.judgment.decision_applied is False
    assert judged.verdict is VerificationVerdict.INCONCLUSIVE
    assert "person@example.com" not in captured["data"]["claim"]
    assert judged.judgment.direct_identifier_redactions == {"email": 1}


def test_adjudication_mode_can_apply_support_but_unrelated_stays_inconclusive(monkeypatch):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id

    def response(relationship):
        return {
            "relationship": relationship,
            "confidence": "high",
            "passage_ids": [passage_id],
            "rationale": "Bounded relationship assessment.",
            "limitations": [],
        }

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        lambda *args, **kwargs: response("supports"),
    )
    supported = apply_bounded_structured_judgment(
        artifact, decision_mode="adjudicate"
    )
    assert supported.verdict is VerificationVerdict.CONSISTENT
    assert supported.judgment.decision_applied is True

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        lambda *args, **kwargs: response("unrelated"),
    )
    unrelated = apply_bounded_structured_judgment(
        artifact, decision_mode="adjudicate"
    )
    assert unrelated.verdict is VerificationVerdict.INCONCLUSIVE
    assert unrelated.judgment.decision_applied is False
    assert "entire source" in unrelated.judgment.limitations[-1]


def test_fabricated_passage_id_and_extra_output_field_fail_closed(monkeypatch):
    artifact = _artifact()
    base = {
        "relationship": "contradicts",
        "confidence": "high",
        "passage_ids": ["model-invented-id"],
        "rationale": "Claim differs.",
        "limitations": [],
    }
    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        lambda *args, **kwargs: base,
    )
    fabricated = apply_bounded_structured_judgment(
        artifact, decision_mode="adjudicate"
    )
    assert fabricated.judgment.method == "judgment_fabricated_evidence_id"
    assert fabricated.verdict is VerificationVerdict.INCONCLUSIVE

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        lambda *args, **kwargs: {**base, "unexpected": "field"},
    )
    invalid = apply_bounded_structured_judgment(artifact)
    assert invalid.judgment.method == "judgment_invalid_or_unavailable"


def test_single_limitation_string_is_normalized_without_loosening_ids(monkeypatch):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id
    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        lambda *args, **kwargs: {
            "relationship": "supports",
            "confidence": "high",
            "passage_ids": [passage_id],
            "rationale": "The supplied passage supports the claim.",
            "limitations": "Only bounded evidence was supplied.",
        },
    )
    judged = apply_bounded_structured_judgment(artifact)
    assert judged.judgment.status is RelationshipStatus.SUPPORTS
    assert judged.judgment.limitations[0] == "Only bounded evidence was supplied."


def test_non_atomic_claim_never_calls_judge(monkeypatch):
    artifact = _artifact(atomic=False)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json", fail_if_called
    )
    judged = apply_bounded_structured_judgment(artifact)
    assert called is False
    assert judged.judgment.method == "atomic_claim_required"
    assert judged.verdict is VerificationVerdict.INCONCLUSIVE


def test_prompt_contains_at_most_three_application_passages(monkeypatch):
    artifact = _artifact(blocks=5)
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        captured.update(json.loads(user_prompt))
        return {
            "relationship": "insufficient_evidence",
            "confidence": "low",
            "passage_ids": [],
            "rationale": "The bounded evidence is insufficient.",
            "limitations": ["More context is needed."],
        }

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json", fake_call
    )
    apply_bounded_structured_judgment(artifact)
    assert 1 <= len(captured["passages"]) <= 3
    assert set(captured) == {
        "task",
        "claim",
        "resolved_context_dependencies",
        "coverage",
        "passages",
    }


def test_resolved_antecedent_is_context_not_source_evidence(monkeypatch):
    artifact = _artifact(context_dependency=True)
    passage_id = artifact.passages[0].passage_id
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["data"] = json.loads(user_prompt)
        return {
            "relationship": "supports",
            "confidence": "medium",
            "passage_ids": [passage_id],
            "rationale": "The source passage supports the resolved proposition.",
            "limitations": [],
        }

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_bounded_structured_judgment(artifact)

    assert judged.judgment.status is RelationshipStatus.SUPPORTS
    assert captured["data"]["claim"] == "This pressure improves national image."
    assert captured["data"]["resolved_context_dependencies"] == [
        {
            "mention": "This pressure",
            "antecedent": (
                "competition among cultural exporters creates market pressure"
            ),
            "status": "resolved",
            "confidence": "high",
        }
    ]
    assert "never count the context as evidence" in captured["system"]


def test_unresolved_antecedent_never_calls_judge(monkeypatch):
    artifact = _artifact(context_dependency=True)
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={"context_dependency_status": "ambiguous"}
            )
        }
    )
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        "app.services.verification_judgment.chat_completion_json",
        fail_if_called,
    )
    judged = apply_bounded_structured_judgment(artifact)

    assert called is False
    assert judged.judgment.method == "claim_context_dependency_unresolved"
