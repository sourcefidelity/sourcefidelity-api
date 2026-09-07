"""Regressions for typed source-specific relevance obligations."""

from datetime import datetime, timezone
import hashlib

from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.evidence_obligations import attach_evidence_obligations
from app.services.evidence_package import build_evidence_package
from app.services.student_statement_interpretation import (
    StudentStatementInterpretation,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    CitationSourceBinding,
    ClaimEvidence,
    build_passage_evidence,
)


def _artifact(*, aggregate=False):
    content = b"A bounded source passage about translation practices."
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
    text = "Students use translation tools strategically (Smith, 2020)."
    claim = ClaimEvidence(
        claim_id="citation-unit-1",
        paper_version_id="paper-v1",
        text=text,
        granularity="citation_unit",
        reference_ids=["reference-1", "reference-2"] if aggregate else ["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        extraction_confidence="high",
        passage_start=10,
        passage_end=10 + len(text),
    )
    artifact = build_passage_evidence(
        source,
        claim=claim,
        top_k=1,
        active_reference_id="reference-1",
        cited_author_label="Smith",
    )
    return attach_citation_use_routes(attach_verification_candidates(artifact))


def test_aggregate_member_obligation_preserves_member_only_scope():
    artifact = attach_evidence_obligations(_artifact(aggregate=True))
    obligation = artifact.evidence_obligations.obligations[0]
    package = build_evidence_package(artifact)

    assert artifact.evidence_obligations.status == "complete"
    assert obligation.obligation_type == "aggregate_member_evidence"
    assert obligation.reference_id == "reference-1"
    assert obligation.aggregate_scope == "member_only"
    assert obligation.target_text_sha256 == hashlib.sha256(
        obligation.target_text.encode("utf-8")
    ).hexdigest()
    assert package.evidence_obligations == artifact.evidence_obligations


def test_source_blind_semantic_repair_creates_coverage_only_obligation():
    artifact = _artifact()
    candidate = next(
        item
        for item in artifact.verification_candidates.candidates
        if item.role == "relationship_candidate"
    )
    interpreted = "Students use translation tools as one part of their writing process."
    interpretation = StudentStatementInterpretation(
        interpretation_id="interpretation:repair-1",
        candidate_id=candidate.candidate_id,
        candidate_text_sha256=hashlib.sha256(candidate.text.encode()).hexdigest(),
        status="semantic_repair",
        interpreted_statement=interpreted,
        reason_code="single_plausible_semantic_repair",
        confidence="high",
        accuracy_judgment_allowed=False,
        coverage_judgment_allowed=True,
    )

    artifact = attach_evidence_obligations(
        artifact, interpretations=[interpretation]
    )
    coverage = next(
        item
        for item in artifact.evidence_obligations.obligations
        if item.obligation_type == "coverage_only_semantic_repair"
    )

    assert coverage.target_text == interpreted
    assert coverage.interpretation_id == interpretation.interpretation_id
    assert coverage.accuracy_judgment_allowed is False
    assert coverage.coverage_judgment_allowed is True
    package = build_evidence_package(artifact)
    assert package.student_statement_interpretations == [interpretation]


def test_repeated_marker_uses_exact_source_binding_coordinates():
    artifact = _artifact()
    marker = "(Smith, 2020)"
    text = f"{marker} states the factual proposition. {marker}"
    marker_start = text.rfind(marker)
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={
                    "text": text,
                    "citation_marker": marker,
                    "passage_end": artifact.claim.passage_start + len(text),
                }
            ),
            "source_binding": CitationSourceBinding(
                status="exact",
                reference_id="reference-1",
                cited_author_label="Smith",
                marker_text=marker,
                marker_local_start=marker_start,
                marker_local_end=marker_start + len(marker),
            ),
        }
    )

    assessed = attach_evidence_obligations(artifact)

    assert assessed.evidence_obligations.obligations[0].target_text == text
