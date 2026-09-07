"""Authoritative, judgment-free Evidence Package v1 construction.

One package binds one exact student citation/reference member to one admitted
source representation and the bounded evidence candidates retrieved from it.
Relationship labels remain experimental views outside this contract.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from app.services.verification_evidence import (
    CandidatePassageSelection,
    CitationSourceBinding,
    CoverageEvidence,
    EvidenceObligationSet,
    SourceIdentityEvidence,
    StudentStatementInterpretationEvidence,
    VerificationEvidenceArtifact,
)
from app.services.reference_discovery import ReferenceDiscoveryRecord


EVIDENCE_PACKAGE_VERSION = "evidence-package-v1"
MAX_PACKAGE_PASSAGE_CHARACTERS = 1_800
MAX_PACKAGE_STUDENT_CHARACTERS = 5_000


class EvidencePackageError(ValueError):
    """The artifact cannot produce an authoritative Evidence Package."""


class EvidencePackageWorkflowResult(BaseModel):
    """Explicitly separate an unrun workflow from a negative outcome."""

    status: Literal["not_run", "complete", "incomplete", "not_assessable"]
    outcome: str | None = Field(default=None, max_length=100)
    evidence_passage_ids: list[str] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list, max_length=8)


class EvidencePackagePassage(BaseModel):
    passage_id: str = Field(min_length=1, max_length=128)
    representation_id: str = Field(min_length=1, max_length=255)
    content_sha256: str = Field(min_length=64, max_length=64)
    page_index: int | None = None
    page_label: str | None = None
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    excerpt: str = Field(min_length=1, max_length=MAX_PACKAGE_PASSAGE_CHARACTERS)
    excerpt_truncated: bool = False
    passage_text_sha256: str = Field(min_length=64, max_length=64)
    retrieval_method: str = Field(min_length=1, max_length=200)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    passage_role: Literal[
        "body_prose",
        "abstract",
        "document_metadata",
        "citation_notes",
        "unknown",
    ]
    boundary_status: Literal[
        "sentence_complete",
        "bounded_fragment_or_nonprose",
        "unknown",
    ]
    retrieval_rule_version: str = Field(min_length=1, max_length=100)


class EvidencePackageRetrieval(BaseModel):
    """Inspectable protected union and its exact candidate-query bindings."""

    status: Literal["not_run", "complete", "incomplete", "not_assessable"]
    method: str = Field(min_length=1, max_length=200)
    retrieval_version: str | None = Field(default=None, max_length=100)
    whole_citation_passage_ids: list[str] = Field(default_factory=list)
    displayed_passage_ids: list[str] = Field(default_factory=list)
    display_consolidations: dict[str, list[str]] = Field(default_factory=dict)
    candidate_selections: list[CandidatePassageSelection] = Field(
        default_factory=list, max_length=16
    )
    excluded_block_counts: dict[str, int] = Field(default_factory=dict)
    semantic_rescue_status: Literal[
        "not_run", "complete", "incomplete", "not_assessed"
    ] = "not_run"
    semantic_rescue_version: str | None = Field(default=None, max_length=100)
    semantic_model_id: str | None = Field(default=None, max_length=300)
    semantic_model_revision: str | None = Field(default=None, max_length=100)
    semantic_prefilter_count: int = Field(default=0, ge=0, le=512)
    semantic_addition_count: int = Field(default=0, ge=0, le=64)
    candidate_availability: Literal[
        "candidates_retrieved",
        "no_candidates_retrieved",
        "retrieval_not_run",
        "not_assessable",
    ]
    source_absence_claim_permitted: Literal[False] = False
    limitations: list[str] = Field(default_factory=list, max_length=24)


class EvidencePackageV1(BaseModel):
    package_version: Literal["evidence-package-v1"] = EVIDENCE_PACKAGE_VERSION
    package_id: str = Field(min_length=64, max_length=64)
    package_sha256: str = Field(min_length=64, max_length=64)
    created_at: datetime
    artifact_version: str = Field(min_length=1, max_length=100)
    verification_id: str = Field(min_length=1, max_length=128)
    paper_version_id: str = Field(min_length=1, max_length=255)
    claim_id: str = Field(min_length=1, max_length=128)
    student_text: str = Field(min_length=1, max_length=MAX_PACKAGE_STUDENT_CHARACTERS)
    student_text_sha256: str = Field(min_length=64, max_length=64)
    paper_character_start: int
    paper_character_end: int
    source_binding: CitationSourceBinding
    source_identity: SourceIdentityEvidence
    coverage: CoverageEvidence
    evidence_obligations: EvidenceObligationSet = Field(
        default_factory=EvidenceObligationSet
    )
    student_statement_interpretations: list[
        StudentStatementInterpretationEvidence
    ] = Field(default_factory=list, max_length=4)
    extraction_version: str = Field(min_length=1, max_length=100)
    extracted_text_sha256: str = Field(min_length=64, max_length=64)
    passages: list[EvidencePackagePassage] = Field(default_factory=list)
    retrieval: EvidencePackageRetrieval
    reference_discovery: ReferenceDiscoveryRecord | EvidencePackageWorkflowResult
    quotation_check: EvidencePackageWorkflowResult
    locator_check: EvidencePackageWorkflowResult
    limitations: list[str] = Field(default_factory=list, max_length=24)


def build_evidence_package(
    artifact: VerificationEvidenceArtifact,
    *,
    reference_discovery: ReferenceDiscoveryRecord | None = None,
) -> EvidencePackageV1:
    """Build one immutable package without importing experimental judgments."""
    binding = artifact.source_binding
    if binding is None or binding.status != "exact":
        raise EvidencePackageError(
            "Evidence Package v1 requires one exact citation/reference binding"
        )
    identity = artifact.source_identity
    coverage = artifact.coverage
    if len(artifact.claim.text) > MAX_PACKAGE_STUDENT_CHARACTERS:
        raise EvidencePackageError(
            "Evidence Package citation-dependent student text exceeds its bound"
        )
    if len(coverage.extracted_text_sha256) != 64:
        raise EvidencePackageError(
            "Evidence Package v1 requires a current extracted-text hash"
        )
    _validate_interpretation_obligation_bindings(artifact)

    passage_ids: list[str] = []
    passages: list[EvidencePackagePassage] = []
    prohibited_roles = {"reference_list", "publication_metadata"}
    for passage in artifact.passages:
        if passage.passage_id in passage_ids:
            raise EvidencePackageError("Evidence Package passages must be unique")
        passage_ids.append(passage.passage_id)
        if passage.passage_role in prohibited_roles:
            raise EvidencePackageError(
                "Evidence Package cannot display a definitely excluded source role"
            )
        if (
            passage.representation_id != identity.representation_id
            or passage.content_sha256 != identity.content_sha256
            or passage.authorization_scope_type
            != identity.authorization_scope_type
            or passage.authorization_scope_id != identity.authorization_scope_id
            or passage.verification_run_id != identity.verification_run_id
        ):
            raise EvidencePackageError(
                "Evidence Package passage does not match its admitted representation"
            )
        excerpt = passage.text[:MAX_PACKAGE_PASSAGE_CHARACTERS]
        passages.append(
            EvidencePackagePassage(
                passage_id=passage.passage_id,
                representation_id=passage.representation_id,
                content_sha256=passage.content_sha256,
                page_index=passage.page_index,
                page_label=passage.page_label,
                character_start=passage.character_start,
                character_end=passage.character_end,
                excerpt=excerpt,
                excerpt_truncated=len(excerpt) != len(passage.text),
                passage_text_sha256=hashlib.sha256(
                    passage.text.encode("utf-8")
                ).hexdigest(),
                retrieval_method=passage.retrieval_method,
                retrieval_score=passage.retrieval_score,
                passage_role=passage.passage_role,
                boundary_status=passage.boundary_status,
                retrieval_rule_version=passage.retrieval_rule_version,
            )
        )

    candidate_retrieval = artifact.candidate_passage_retrieval
    selected_ids = [
        item.passage_id
        for selection in candidate_retrieval.selections
        for item in selection.passages
    ]
    unknown_selected = set(selected_ids) - set(passage_ids)
    if unknown_selected:
        raise EvidencePackageError(
            "Evidence Package retrieval refers to a passage outside its union"
        )
    whole_citation_ids = list(artifact.relationship.passage_ids)
    if set(whole_citation_ids) - set(passage_ids):
        raise EvidencePackageError(
            "Evidence Package whole-citation retrieval refers to unknown evidence"
        )
    candidate_order = []
    for selection in candidate_retrieval.selections:
        for item in sorted(selection.passages, key=lambda value: value.rank):
            if item.passage_id not in candidate_order:
                candidate_order.append(item.passage_id)
    ordered_raw_ids = _ordered_retrieval_ids(
        whole_citation_ids,
        candidate_order,
        passage_ids,
    )
    displayed_ids, display_consolidations = _display_consolidations(
        ordered_raw_ids, artifact
    )

    retrieval_status = _retrieval_status(candidate_retrieval.status, bool(passages))
    if passages:
        candidate_availability = "candidates_retrieved"
    elif retrieval_status == "not_run":
        candidate_availability = "retrieval_not_run"
    elif retrieval_status == "not_assessable":
        candidate_availability = "not_assessable"
    else:
        candidate_availability = "no_candidates_retrieved"

    retrieval_limitations = list(candidate_retrieval.limitations)
    neutral_absence_limit = (
        "An unmatched query or qualifier describes only the recorded retrieval "
        "channels and never establishes absence from the complete source."
    )
    if neutral_absence_limit not in retrieval_limitations:
        retrieval_limitations.append(neutral_absence_limit)
    retrieval = EvidencePackageRetrieval(
        status=retrieval_status,
        method=(
            candidate_retrieval.method
            if candidate_retrieval.method != "not_run"
            else "whole_citation_protected_retrieval_only"
        ),
        retrieval_version=candidate_retrieval.retrieval_version,
        whole_citation_passage_ids=whole_citation_ids,
        displayed_passage_ids=displayed_ids,
        display_consolidations=display_consolidations,
        candidate_selections=candidate_retrieval.selections,
        excluded_block_counts=dict(candidate_retrieval.excluded_block_counts),
        semantic_rescue_status=candidate_retrieval.semantic_rescue_status,
        semantic_rescue_version=candidate_retrieval.semantic_rescue_version,
        semantic_model_id=candidate_retrieval.semantic_model_id,
        semantic_model_revision=candidate_retrieval.semantic_model_revision,
        semantic_prefilter_count=candidate_retrieval.semantic_prefilter_count,
        semantic_addition_count=candidate_retrieval.semantic_addition_count,
        candidate_availability=candidate_availability,
        limitations=retrieval_limitations,
    )

    workflow_not_run = EvidencePackageWorkflowResult(
        status="not_run",
        limitations=[
            "This workflow has not yet been attached to Evidence Package v1."
        ],
    )
    if (
        reference_discovery is not None
        and reference_discovery.reference_id != binding.reference_id
    ):
        raise EvidencePackageError(
            "Evidence Package reference discovery does not match its source binding"
        )
    quotation_check = EvidencePackageWorkflowResult.model_validate(
        artifact.quotation_check.model_dump()
    )
    locator_check = EvidencePackageWorkflowResult.model_validate(
        artifact.locator_check.model_dump()
    )
    for check in (quotation_check, locator_check):
        if set(check.evidence_passage_ids) - set(passage_ids):
            raise EvidencePackageError(
                "Evidence Package workflow check refers to unknown evidence"
            )
    package_id = _stable_package_id(
        artifact.claim.paper_version_id,
        artifact.claim.claim_id,
        binding.reference_id,
        identity.representation_id,
        identity.content_sha256,
        coverage.extracted_text_sha256,
    )
    payload = {
        "package_version": EVIDENCE_PACKAGE_VERSION,
        "package_id": package_id,
        "created_at": artifact.created_at,
        "artifact_version": artifact.artifact_version,
        "verification_id": artifact.verification_id,
        "paper_version_id": artifact.claim.paper_version_id,
        "claim_id": artifact.claim.claim_id,
        "student_text": artifact.claim.text,
        "student_text_sha256": hashlib.sha256(
            artifact.claim.text.encode("utf-8")
        ).hexdigest(),
        "paper_character_start": artifact.claim.passage_start,
        "paper_character_end": artifact.claim.passage_end,
        "source_binding": binding,
        "source_identity": identity,
        "coverage": coverage,
        "evidence_obligations": artifact.evidence_obligations,
        "student_statement_interpretations": (
            artifact.student_statement_interpretations
        ),
        "extraction_version": coverage.extraction_version,
        "extracted_text_sha256": coverage.extracted_text_sha256,
        "passages": passages,
        "retrieval": retrieval,
        "reference_discovery": reference_discovery or workflow_not_run,
        "quotation_check": quotation_check,
        "locator_check": locator_check,
        "limitations": [
            "This package presents source-bound evidence and does not determine source-use accuracy, intent, misconduct, grades, or sanctions.",
            "Experimental relationship judgments are not part of the authoritative Evidence Package.",
        ],
    }
    package_sha256 = _package_payload_sha256(payload)
    return EvidencePackageV1(package_sha256=package_sha256, **payload)


def validate_evidence_package(package: EvidencePackageV1) -> None:
    """Reject any change to the content-addressed package payload."""
    payload = package.model_dump(mode="json", exclude={"package_sha256"})
    if _package_payload_sha256(payload) != package.package_sha256:
        raise EvidencePackageError("Evidence Package content hash does not match")


def _validate_interpretation_obligation_bindings(
    artifact: VerificationEvidenceArtifact,
) -> None:
    candidates = {
        item.candidate_id: item.text
        for item in artifact.verification_candidates.candidates
    }
    interpretations = {
        item.interpretation_id: item
        for item in artifact.student_statement_interpretations
    }
    for interpretation in interpretations.values():
        candidate_text = candidates.get(interpretation.candidate_id)
        if candidate_text is None or hashlib.sha256(
            candidate_text.encode("utf-8")
        ).hexdigest() != interpretation.candidate_text_sha256:
            raise EvidencePackageError(
                "Student interpretation is not bound to an exact candidate"
            )
        if interpretation.source_evidence_received is not False:
            raise EvidencePackageError(
                "Student interpretation exceeded the source-blind boundary"
            )
    for obligation in artifact.evidence_obligations.obligations:
        if obligation.obligation_type != "coverage_only_semantic_repair":
            continue
        interpretation = interpretations.get(obligation.interpretation_id)
        if (
            interpretation is None
            or interpretation.status != "semantic_repair"
            or not interpretation.coverage_judgment_allowed
            or interpretation.interpreted_statement != obligation.target_text
        ):
            raise EvidencePackageError(
                "Coverage-only obligation is not bound to its source-blind repair"
            )


def _retrieval_status(
    status: str, has_passages: bool
) -> Literal["not_run", "complete", "incomplete", "not_assessable"]:
    if status == "complete":
        return "complete"
    if status == "incomplete":
        return "incomplete"
    if status == "not_assessed":
        return "not_assessable"
    return "incomplete" if has_passages else "not_run"


def _display_consolidations(
    ordered_ids: list[str], artifact: VerificationEvidenceArtifact
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep one visible representative for substantially overlapping windows."""
    passages = {passage.passage_id: passage for passage in artifact.passages}
    displayed: list[str] = []
    groups: dict[str, list[str]] = {}
    for passage_id in ordered_ids:
        passage = passages[passage_id]
        representative = None
        for displayed_id in displayed:
            existing = passages[displayed_id]
            if passage.page_index != existing.page_index:
                continue
            overlap = max(
                0,
                min(passage.character_end, existing.character_end)
                - max(passage.character_start, existing.character_start),
            )
            shorter = min(
                passage.character_end - passage.character_start,
                existing.character_end - existing.character_start,
            )
            if shorter > 0 and overlap / shorter >= 0.50:
                representative = displayed_id
                break
        if representative is None:
            displayed.append(passage_id)
            groups[passage_id] = [passage_id]
        elif passage_id not in groups[representative]:
            groups[representative].append(passage_id)
    return displayed, groups


def _ordered_retrieval_ids(
    whole_citation_ids: list[str],
    candidate_ids: list[str],
    all_passage_ids: list[str],
) -> list[str]:
    """Lead with query-specific evidence while retaining the protected baseline."""
    order: list[str] = []
    for passage_id in (
        *candidate_ids[:2],
        *whole_citation_ids[:2],
        *candidate_ids[2:],
        *whole_citation_ids[2:],
        *all_passage_ids,
    ):
        if passage_id not in order:
            order.append(passage_id)
    return order


def _stable_package_id(*parts: str) -> str:
    value = "\x1f".join((EVIDENCE_PACKAGE_VERSION, *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_payload_sha256(payload: dict) -> str:
    normalized = to_jsonable_python(payload)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
