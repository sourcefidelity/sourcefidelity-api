"""Typed, source-blind relevance obligations for one citation/source member."""

from __future__ import annotations

import hashlib

from app.services.student_statement_interpretation import (
    StudentStatementInterpretation,
)
from app.services.verification_evidence import (
    EvidenceObligation,
    EvidenceObligationSet,
    VerificationEvidenceArtifact,
)


EVIDENCE_OBLIGATION_VERSION = "typed-evidence-obligation-v2"


def attach_evidence_obligations(
    artifact: VerificationEvidenceArtifact,
    *,
    interpretations: list[StudentStatementInterpretation] | None = None,
) -> VerificationEvidenceArtifact:
    """Attach exact/member/Coverage targets without consulting source text."""

    binding = artifact.source_binding
    if binding is None or binding.status != "exact":
        return artifact.model_copy(
            update={
                "evidence_obligations": EvidenceObligationSet(
                    status="not_assessed",
                    method="exact_source_binding_required",
                    obligation_version=EVIDENCE_OBLIGATION_VERSION,
                    limitations=[
                        "A source-specific evidence obligation requires one exact citation/reference binding."
                    ],
                )
            }
        )
    original = exact_source_attributed_text(artifact)
    original_hash = _sha(original)
    aggregate = len(artifact.claim.reference_ids) > 1
    obligations = [
        EvidenceObligation(
            obligation_id=_obligation_id(
                artifact,
                "aggregate_member_evidence" if aggregate else "exact_factual_assertion",
                original,
            ),
            obligation_type=(
                "aggregate_member_evidence" if aggregate else "exact_factual_assertion"
            ),
            reference_id=binding.reference_id,
            target_text=original,
            target_text_sha256=original_hash,
            original_text_sha256=original_hash,
            derivation_method="exact_source_attributed_text",
            aggregate_scope="member_only" if aggregate else "not_aggregate",
            accuracy_judgment_allowed=True,
            coverage_judgment_allowed=False,
            limitations=(
                [
                    "This source may establish only its own membership or contribution; it cannot by itself prove an aggregate quantifier."
                ]
                if aggregate
                else []
            ),
        )
    ]
    candidate_ids = {
        candidate.candidate_id
        for candidate in artifact.verification_candidates.candidates
    }
    active_interpretations = (
        artifact.student_statement_interpretations
        if interpretations is None
        else interpretations
    )
    for interpretation in active_interpretations:
        if (
            interpretation.candidate_id not in candidate_ids
            or interpretation.status != "semantic_repair"
            or not interpretation.coverage_judgment_allowed
            or not interpretation.interpreted_statement
        ):
            continue
        repaired = interpretation.interpreted_statement.strip()
        obligation_id = _obligation_id(
            artifact, "coverage_only_semantic_repair", repaired
        )
        if any(item.obligation_id == obligation_id for item in obligations):
            continue
        obligations.append(
            EvidenceObligation(
                obligation_id=obligation_id,
                obligation_type="coverage_only_semantic_repair",
                reference_id=binding.reference_id,
                target_text=repaired,
                target_text_sha256=_sha(repaired),
                original_text_sha256=original_hash,
                derivation_method="source_blind_semantic_repair",
                interpretation_id=interpretation.interpretation_id,
                aggregate_scope="member_only" if aggregate else "not_aggregate",
                accuracy_judgment_allowed=False,
                coverage_judgment_allowed=True,
                limitations=[
                    "This obligation can show Coverage for an explicit source-blind repair but cannot establish accuracy of the original wording."
                ],
            )
        )
        if len(obligations) >= 4:
            break
    return artifact.model_copy(
        update={
            "student_statement_interpretations": list(active_interpretations)[:4],
            "evidence_obligations": EvidenceObligationSet(
                status="complete",
                method="source_blind_typed_obligations",
                obligation_version=EVIDENCE_OBLIGATION_VERSION,
                obligations=obligations,
                limitations=[
                    "Evidence obligations define relevance questions only; they do not decide support, contradiction, intent, or misconduct."
                ],
            )
        }
    )


def exact_source_attributed_text(artifact: VerificationEvidenceArtifact) -> str:
    """Return only the wording the visible source marker can attribute."""
    return claim_source_attributed_text(artifact.claim, artifact.source_binding)


def claim_source_attributed_text(claim, binding=None) -> str:
    """Shared source-blind scope projection; preserve the original claim."""
    if claim.source_segments:
        ordered = sorted(claim.source_segments, key=lambda item: item.local_start)
        joined = " ".join(item.text.strip() for item in ordered if item.text.strip())
        if joined:
            return joined
    marker = (claim.citation_marker or "").strip()
    if claim.citation_marker_type == "parenthetical" and marker:
        marker_start = -1
        if (
            binding is not None
            and binding.status == "exact"
            and binding.marker_text == marker
            and binding.marker_local_start >= 0
            and claim.text[
                binding.marker_local_start:binding.marker_local_end
            ] == marker
        ):
            marker_start = binding.marker_local_start
        if marker_start < 0:
            marker_start = claim.text.rfind(marker)
        if marker_start >= 0:
            marker_end = marker_start + len(marker)
            trailing = claim.text[marker_end:].strip()
            if trailing.strip(" ,;:.!?—–-"):
                attributed = claim.text[:marker_end].strip()
                if attributed:
                    return attributed
    return claim.text


def _obligation_id(artifact, kind: str, target: str) -> str:
    binding = artifact.source_binding
    return "obligation:" + _sha(
        f"{EVIDENCE_OBLIGATION_VERSION}:{artifact.claim.claim_id}:"
        f"{binding.reference_id}:{kind}:{target}"
    )[:32]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
