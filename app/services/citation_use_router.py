"""Deterministic citation-use routing over application-owned exact candidates.

The router chooses which evidence procedure is authorized to handle a fixed
candidate.  It does not judge whether a citation is correct, infer intent, or
score engagement.  High-confidence structural signals receive a typed route;
unsupported procedures and ambiguous scope fail closed.
"""

from __future__ import annotations

import hashlib
import re

from app.services.verification_evidence import (
    CitationUseRoute,
    CitationUseRoutingEvidence,
    VerificationCandidate,
    VerificationEvidenceArtifact,
)


CITATION_USE_ROUTER_VERSION = "deterministic-citation-use-router-v3"

_EXEMPLIFICATION = re.compile(
    r"\b(?:for\s+example|for\s+instance|as\s+an\s+example|e\.g\.)\b",
    re.IGNORECASE,
)
_LEADING_EXEMPLIFICATION = re.compile(
    r"^\s*(?:for\s+example|for\s+instance|as\s+an\s+example|e\.g\.)\b",
    re.IGNORECASE,
)
_APPLICATION = re.compile(
    # High precision by design: an application route requires explicit student
    # agency plus a named intellectual tool and a separate target. Generic
    # uses of "apply" (laws apply, policies applied) and source-reported
    # interpretations remain factual attribution.
    r"\b(?:i|we)\s+(?:will\s+)?(?:apply|use|employ)\s+"
    r"[^.;:]{0,100}\b(?:framework|theory|model|concept|typology|method|approach)\b"
    r"[^.;:]{0,80}\bto\b|"
    r"\bthis\s+(?:paper|essay|analysis|study)\s+"
    r"(?:will\s+)?(?:applies|apply|uses|use|employs|employ)\s+"
    r"[^.;:]{0,100}\b(?:framework|theory|model|concept|typology|method|approach)\b"
    r"[^.;:]{0,80}\b(?:to|on|for)\s+[A-Za-z0-9]|"
    r"^(?:by\s+)?(?:applying|using|employing)\s+"
    r"[^.;:]{0,100}\b(?:framework|theory|model|concept|typology|method|approach)\b"
    r"[^.;:]{0,80}\bto\b",
    re.IGNORECASE | re.DOTALL,
)
_APPLICATION_FRAME_WITHOUT_TARGET = re.compile(
    r"\b(?:i|we|this\s+(?:paper|essay|analysis|study))\s+"
    r"(?:will\s+)?(?:applies|apply|uses|use|employs|employ)\s+"
    r"[^.;:]{0,100}\b(?:framework|theory|model|concept|typology|method|approach)\b",
    re.IGNORECASE | re.DOTALL,
)
_FURTHER_REFERENCE_TEXT = re.compile(
    r"(?:\bfor\s+(?:further\s+)?(?:background|discussion|overview|review)\b"
    r".{0,80}\bsee(?:\s+also)?\b|\bsee\s+also\b|\bcf\.)",
    re.IGNORECASE | re.DOTALL,
)
_FURTHER_REFERENCE_MARKER = re.compile(
    r"^\s*\(\s*(?:see(?:\s+also)?|cf\.)\b",
    re.IGNORECASE,
)
_MULTI_SOURCE_COMPARISON = re.compile(
    r"\b(?:unlike|similarly|conversely|in\s+contrast\s+to|compared\s+with|"
    r"whereas|both\s+.+?\s+and|neither\s+.+?\s+nor|"
    r"agree(?:s|d)?\s+with|disagree(?:s|d)?\s+with)\b",
    re.IGNORECASE | re.DOTALL,
)
_EVALUATION_METHODS = {
    "explicit_source_evaluation_content",
    "explicit_agreement_source_content",
    "explicit_disagreement_source_content",
}
_NON_PROPOSITIONAL_CONTEXT_METHODS = {
    "narrative_leading_student_context",
    "narrative_reporting_frame",
}


def attach_citation_use_routes(
    artifact: VerificationEvidenceArtifact,
) -> VerificationEvidenceArtifact:
    """Attach one local route for every relationship-candidate exact span."""
    candidate_set = artifact.verification_candidates
    if candidate_set.status not in {"complete", "incomplete"}:
        return artifact.model_copy(
            update={
                "citation_use_routing": CitationUseRoutingEvidence(
                    status="not_assessed",
                    method="verification_candidates_required",
                    router_version=CITATION_USE_ROUTER_VERSION,
                    limitations=[
                        "Citation-use routing requires application-owned exact verification candidates."
                    ],
                )
            }
        )

    guard = next(
        (
            candidate.candidate_id
            for candidate in candidate_set.candidates
            if candidate.role == "whole_unit_guard"
        ),
        None,
    )
    routes = [
        _route_candidate(artifact, candidate)
        for candidate in candidate_set.candidates
        if candidate.role == "relationship_candidate"
        and candidate.generation_method not in _NON_PROPOSITIONAL_CONTEXT_METHODS
    ]
    unresolved = any(route.citation_use == "unresolved" for route in routes)
    status = (
        "incomplete"
        if unresolved or candidate_set.status == "incomplete"
        else "complete"
    )
    return artifact.model_copy(
        update={
            "citation_use_routing": CitationUseRoutingEvidence(
                status=status,
                method="deterministic_exact_candidate_scope_and_cue_routing",
                router_version=CITATION_USE_ROUTER_VERSION,
                complete_citation_unit_candidate_id=guard,
                routes=routes,
                limitations=[
                    "Routing selects an evidence procedure; it does not judge correctness, engagement, intent, grades, or misconduct.",
                    "Source-wide and multi-source procedures remain typed abstentions until their separate evidence contracts are validated.",
                    "Application routes require an exact isolated source proposition before source-relationship judgment; the student's application is never verified against the cited source as if the source made it.",
                    "The complete citation unit remains authoritative when a bounded route is uncertain or excluded.",
                ],
            )
        }
    )


def routed_relationship_candidate_ids(
    artifact: VerificationEvidenceArtifact,
) -> set[str]:
    """Return candidates authorized for one of the bounded relationship paths."""
    routing = artifact.citation_use_routing
    if routing.status == "not_run":
        # Backward-compatible inspection of historical artifacts. New runtime
        # paths attach routing before retrieval.
        return {
            candidate.candidate_id
            for candidate in artifact.verification_candidates.candidates
            if candidate.relationship_eligible
        }
    return {
        route.candidate_id
        for route in routing.routes
        if route.relationship_judgment_allowed
        and route.status == "ready"
        and route.evidence_procedure
        in {
            "bounded_passage_relationship",
            "bounded_source_proposition_relationship",
        }
    }


def _route_candidate(
    artifact: VerificationEvidenceArtifact,
    candidate: VerificationCandidate,
) -> CitationUseRoute:
    digest = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()

    def route(**values):
        return CitationUseRoute(
            candidate_id=candidate.candidate_id,
            candidate_text_sha256=digest,
            **values,
        )

    if "candidate_integrity:materially_redundant_candidate" in candidate.limitations:
        return route(
            citation_use="unresolved",
            evidence_procedure="unresolved",
            status="not_assessed",
            confidence="none",
            reason_code="materially_redundant_candidate_boundaries",
            relationship_judgment_allowed=False,
            limitations=[
                "Materially redundant candidate boundaries must be repaired before a relationship procedure is authorized."
            ],
        )

    if candidate.attribution == "ambiguous":
        return route(
            citation_use="unresolved",
            evidence_procedure="unresolved",
            status="not_assessed",
            confidence="none",
            reason_code="ambiguous_candidate_scope",
            relationship_judgment_allowed=False,
            limitations=[
                "Student/source voice is ambiguous, so no source-relationship procedure is authorized."
            ],
        )
    if (
        candidate.verification_scope == "not_source_verification"
        or candidate.attribution == "student"
    ):
        return route(
            citation_use="student_analysis",
            evidence_procedure="not_source_verification",
            status="excluded",
            confidence="high",
            reason_code="student_analysis_excluded",
            relationship_judgment_allowed=False,
            limitations=[
                "Student analysis remains inspectable but cannot acquire a relationship to the cited source."
            ],
        )
    if candidate.verification_scope == "source_wide_coverage":
        return route(
            citation_use="source_wide_coverage",
            evidence_procedure="source_wide_coverage_engine",
            status="deferred",
            confidence="high",
            reason_code="source_wide_engine_unavailable",
            relationship_judgment_allowed=False,
            limitations=[
                "Ordinary bounded passage retrieval cannot establish source-wide absence, omission, emphasis, or comparative coverage."
            ],
        )

    claim_text = artifact.claim.text
    if _is_further_reference(artifact):
        return route(
            citation_use="further_reference_background",
            evidence_procedure="no_relationship_proposition",
            status="excluded",
            confidence="high",
            reason_code="further_reference_no_relationship_proposition",
            relationship_judgment_allowed=False,
            limitations=[
                "The citation points to background/further reading but supplies no bounded source proposition to judge."
            ],
        )
    if (
        len(set(artifact.claim.reference_ids)) > 1
        and _MULTI_SOURCE_COMPARISON.search(claim_text)
    ):
        return route(
            citation_use="multi_source_connection_comparison",
            evidence_procedure="multi_source_synthesis",
            status="deferred",
            confidence="high",
            reason_code="multi_source_procedure_unavailable",
            relationship_judgment_allowed=False,
            limitations=[
                "A connection/comparison across multiple cited sources requires collective evidence and cannot be judged against one representation."
            ],
        )
    if _APPLICATION.search(claim_text):
        return route(
            citation_use="application",
            evidence_procedure="bounded_source_proposition_relationship",
            status="not_assessed",
            confidence="medium",
            reason_code="application_source_proposition_not_isolated",
            relationship_judgment_allowed=False,
            limitations=[
                "The source concept and the student's application are not yet isolated as separate exact spans."
            ],
        )
    if _APPLICATION_FRAME_WITHOUT_TARGET.search(claim_text):
        return route(
            citation_use="unresolved",
            evidence_procedure="unresolved",
            status="not_assessed",
            confidence="none",
            reason_code="application_target_unresolved",
            relationship_judgment_allowed=False,
            limitations=[
                "Student application language names an intellectual tool but no separate application target is explicit."
            ],
        )
    if candidate.generation_method in _EVALUATION_METHODS:
        return route(
            citation_use="evaluation",
            evidence_procedure="bounded_source_proposition_relationship",
            status="ready",
            confidence="high",
            reason_code="explicit_source_evaluation_proposition",
            relationship_judgment_allowed=candidate.relationship_eligible,
            limitations=[
                "Only the exact source proposition is relationship-eligible; the student's agreement, disagreement, or evaluation is not judged against the source."
            ],
        )
    if _is_exemplification(claim_text, candidate.text):
        return route(
            citation_use="exemplification",
            evidence_procedure="bounded_passage_relationship",
            status="ready",
            confidence="high",
            reason_code="explicit_exemplification",
            relationship_judgment_allowed=candidate.relationship_eligible,
            limitations=[],
        )
    return route(
        citation_use="factual_attribution",
        evidence_procedure="bounded_passage_relationship",
        status="ready" if candidate.relationship_eligible else "not_assessed",
        confidence="high" if candidate.relationship_eligible else "none",
        reason_code=(
            "ordinary_factual_relationship"
            if candidate.relationship_eligible
            else "ambiguous_candidate_scope"
        ),
        relationship_judgment_allowed=candidate.relationship_eligible,
        limitations=(
            []
            if candidate.relationship_eligible
            else ["Candidate scope did not authorize ordinary relationship judgment."]
        ),
    )


def _is_further_reference(artifact: VerificationEvidenceArtifact) -> bool:
    return bool(
        _FURTHER_REFERENCE_MARKER.search(artifact.claim.citation_marker)
        or _FURTHER_REFERENCE_TEXT.search(artifact.claim.text)
    )


def _is_exemplification(claim_text: str, candidate_text: str) -> bool:
    """Recognize student-level example cues without inheriting quoted wording."""
    if _LEADING_EXEMPLIFICATION.search(claim_text):
        return True
    return any(
        not _inside_quotation(candidate_text, match.start())
        for match in _EXEMPLIFICATION.finditer(candidate_text)
    )


def _inside_quotation(text: str, position: int) -> bool:
    prefix = text[:position]
    if prefix.count('"') % 2:
        return True
    return prefix.rfind("“") > prefix.rfind("”")
