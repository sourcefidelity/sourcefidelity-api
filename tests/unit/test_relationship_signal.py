"""Local relationship-signal safety and aggregation regressions."""

from datetime import datetime, timezone
import hashlib
import os

import pytest

from app.services.relationship_signal import (
    NLIScore,
    RelationshipPolicy,
    RelationshipSignalError,
    TransformersNLIScorer,
    _resolve_label_indices,
    assess_local_relationship,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimAntecedentDependency,
    ClaimEvidence,
    RelationshipStatus,
    VerificationVerdict,
    build_passage_evidence,
)


class FakeScorer:
    model_id = "fixed-test-nli"
    model_revision = "revision-1"

    def __init__(self, scores: list[NLIScore]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[str], list[str]]] = []

    def score_pairs(self, premises, hypotheses) -> list[NLIScore]:
        self.calls.append((list(premises), list(hypotheses)))
        return self.scores


class FailingScorer(FakeScorer):
    def score_pairs(self, premises, hypotheses) -> list[NLIScore]:
        raise RuntimeError("model unavailable")


def _source(text: str, *, complete: bool = True) -> AuthorizedRepresentation:
    content = text.encode("utf-8")
    now = datetime.now(timezone.utc)
    return AuthorizedRepresentation(
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
        identity_confidence=0.98,
        completeness_verdict="complete" if complete else "uncertain",
        text_quality="digital",
        edition_or_version="version-1",
        created_at=now,
        admitted_at=now,
    )


def _claim(text: str, *, claim_type: str = "paraphrase", atomic: bool = True):
    return ClaimEvidence(
        claim_id="claim-1",
        paper_version_id="paper-v1",
        text=text,
        claim_type=claim_type,
        granularity="atomic_claim" if atomic else "citation_unit",
        atomization_method="test" if atomic else "not_run",
        reference_ids=["reference-1"],
        passage_start=10,
        passage_end=10 + len(text),
    )


def _artifact(
    claim_text: str,
    source_text: str,
    *,
    claim_type: str = "paraphrase",
    atomic: bool = True,
):
    return build_passage_evidence(
        _source(source_text),
        claim=_claim(claim_text, claim_type=claim_type, atomic=atomic),
    )


def test_exact_quotation_is_deterministic_support_without_nli() -> None:
    artifact = _artifact(
        'The source states, "careful verification improves accuracy."',
        "The evidence shows that careful verification improves accuracy. More context follows.",
        claim_type="quotation",
    )

    assessed = assess_local_relationship(artifact)

    assert assessed.relationship.status is RelationshipStatus.SUPPORTS
    assert assessed.relationship.method == "deterministic_exact_quotation"
    assert assessed.verdict is VerificationVerdict.CONSISTENT
    assert "exact_quotation_present_in_authorized_source" in assessed.reason_codes


def test_non_atomic_claim_never_reaches_scorer() -> None:
    artifact = _artifact(
        "Verification improves accuracy.",
        "The study reports that verification improves accuracy.",
        atomic=False,
    )
    scorer = FakeScorer([NLIScore(0.95, 0.03, 0.02)])

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert assessed.relationship.method == "atomic_claim_required"
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert scorer.calls == []


def test_context_dependent_claim_bypasses_unstructured_local_nli() -> None:
    artifact = _artifact(
        "This pressure improves national image.",
        "Competition among cultural exporters creates pressure and improves national image.",
    )
    dependency = ClaimAntecedentDependency(
        mention_text="This pressure",
        mention_local_start=0,
        mention_local_end=len("This pressure"),
        mention_paper_start=10,
        mention_paper_end=10 + len("This pressure"),
        resolution_status="resolved",
        confidence="high",
        antecedent_context_index=0,
        antecedent_text="competition among cultural exporters creates pressure",
        antecedent_paper_start=0,
        antecedent_paper_end=53,
        method="test",
    )
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={
                    "antecedent_dependencies": [dependency],
                    "context_dependency_status": "resolved",
                }
            )
        }
    )
    scorer = FakeScorer([NLIScore(0.95, 0.03, 0.02)])

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.method == (
        "context_dependent_claim_requires_structured_judgment"
    )
    assert scorer.calls == []


def test_no_candidate_passage_is_insufficient_not_contradiction() -> None:
    artifact = _artifact(
        "Marine temperatures determine coral survival.",
        "This monograph examines theatrical lighting and costume design.",
    )

    assessed = assess_local_relationship(
        artifact,
        scorer=FakeScorer([NLIScore(0.01, 0.98, 0.01)]),
    )

    assert assessed.relationship.status is RelationshipStatus.INSUFFICIENT_EVIDENCE
    assert assessed.relationship.method == "no_candidate_passage"
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE


def test_high_entailment_is_recorded_but_final_verdict_stays_inconclusive() -> None:
    artifact = _artifact(
        "Cultural exports improve national image.",
        "The study finds that cultural exports improve national image through repeated exposure.",
    )
    scorer = FakeScorer([NLIScore(0.94, 0.04, 0.02)])

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.status is RelationshipStatus.SUPPORTS
    assert assessed.relationship.confidence.value == "medium"
    assert assessed.relationship.model_id == "fixed-test-nli"
    assert assessed.relationship.model_revision == "revision-1"
    assert assessed.relationship.passage_scores[0].entailment == 0.94
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert "local_nli_support_candidate" in assessed.reason_codes


def test_high_contradiction_is_candidate_not_adverse_final_verdict() -> None:
    artifact = _artifact(
        "Cultural exports reduce national image.",
        "The evidence demonstrates that cultural exports increase national image rather than reduce it.",
    )
    scorer = FakeScorer([NLIScore(0.01, 0.02, 0.97)])

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.status is RelationshipStatus.CONTRADICTS
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert "local_nli_contradiction_candidate" in assessed.reason_codes
    assert "cannot alone produce an adverse" in assessed.relationship.limitations[0]


def test_neutral_and_weak_scores_remain_insufficient_not_unrelated() -> None:
    artifact = _artifact(
        "Cultural exports improve national image.",
        "The article discusses cultural exports and national image but reports no tested direction.",
    )
    scorer = FakeScorer([NLIScore(0.12, 0.84, 0.04)])

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.status is RelationshipStatus.INSUFFICIENT_EVIDENCE
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert "do not distinguish unrelated" in assessed.relationship.limitations[0]


def test_conflicting_passages_force_insufficient_evidence() -> None:
    artifact = _artifact(
        "Cultural exports improve national image.",
        "One analysis says cultural exports improve national image.\n\n"
        "Another analysis says cultural exports do not improve national image.",
    )
    assert len(artifact.passages) == 2
    scorer = FakeScorer(
        [NLIScore(0.95, 0.03, 0.02), NLIScore(0.01, 0.02, 0.97)]
    )

    assessed = assess_local_relationship(artifact, scorer=scorer)

    assert assessed.relationship.status is RelationshipStatus.INSUFFICIENT_EVIDENCE
    assert assessed.relationship.confidence.value == "low"
    assert len(assessed.relationship.passage_ids) == 2
    assert "conflicting" in assessed.relationship.limitations[0]


def test_model_failure_and_invalid_result_count_fail_safely() -> None:
    artifact = _artifact(
        "Cultural exports improve national image.",
        "The study examines how cultural exports improve national image.",
    )

    failed = assess_local_relationship(artifact, scorer=FailingScorer([]))
    invalid = assess_local_relationship(artifact, scorer=FakeScorer([]))

    assert failed.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert failed.relationship.method == "local_nli_failed"
    assert failed.verdict is VerificationVerdict.INCONCLUSIVE
    assert invalid.relationship.method == "local_nli_invalid_output"


def test_nli_probabilities_and_policy_are_validated() -> None:
    with pytest.raises(ValueError, match="sum"):
        NLIScore(0.9, 0.9, 0.1)
    with pytest.raises(ValueError, match="between"):
        RelationshipPolicy(support_threshold=1.1)


def test_label_mapping_is_explicit_and_unknown_models_fail_closed() -> None:
    assert _resolve_label_indices(
        {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
        "custom-model",
    ) == {"entailment": 0, "neutral": 1, "contradiction": 2}
    with pytest.raises(RelationshipSignalError, match="label mapping"):
        _resolve_label_indices({0: "LABEL_0", 1: "LABEL_1"}, "custom-model")


@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_NLI_TESTS") != "1",
    reason="set RUN_LOCAL_NLI_TESTS=1 after prefetching the pinned local model",
)
def test_pinned_local_nli_smoke_matrix() -> None:
    cases = [
        (
            "support",
            "The study found that cultural exports improved national image.",
            "Cultural exports improved national image.",
        ),
        (
            "support",
            "In the randomized trial, the intervention reduced symptoms by 18 percent.",
            "The intervention reduced symptoms.",
        ),
        (
            "support",
            "The survey included 428 undergraduate participants.",
            "The survey included more than 400 undergraduate participants.",
        ),
        (
            "support",
            "Researchers observed a positive association between sleep and memory performance.",
            "Sleep was positively associated with memory performance.",
        ),
        (
            "contradiction",
            "The study found that cultural exports did not improve national image.",
            "Cultural exports improved national image.",
        ),
        (
            "contradiction",
            "The intervention increased symptoms by 18 percent.",
            "The intervention reduced symptoms.",
        ),
        (
            "contradiction",
            "The survey included 24 undergraduate participants.",
            "The survey included more than 400 undergraduate participants.",
        ),
        (
            "contradiction",
            "Researchers found a negative association between sleep and memory performance.",
            "Sleep was positively associated with memory performance.",
        ),
        (
            "neutral",
            "The study examined cultural exports and national image but did not test their relationship.",
            "Cultural exports improved national image.",
        ),
        (
            "neutral",
            "The article reviews several interventions but reports no outcome data.",
            "The intervention reduced symptoms.",
        ),
        (
            "neutral",
            "The archive contains records of rainfall in coastal cities.",
            "Cultural exports improved national image.",
        ),
        (
            "neutral",
            "The study found a correlation between screen time and anxiety.",
            "Screen time causes anxiety.",
        ),
    ]
    scorer = TransformersNLIScorer(local_files_only=True, device="cpu")
    scores = scorer.score_pairs(
        [premise for _, premise, _ in cases],
        [hypothesis for _, _, hypothesis in cases],
    )
    predicted = [
        max(
            ("support", score.entailment),
            ("neutral", score.neutral),
            ("contradiction", score.contradiction),
            key=lambda item: item[1],
        )[0]
        for score in scores
    ]
    assert predicted == [expected for expected, _, _ in cases]
