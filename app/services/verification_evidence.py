"""Authorization-bound evidence artifacts for the Phase 3.8 vertical slice.

This module deliberately stops before relationship judgment.  It proves which
representation was authorized, what content was inspected, and which bounded
passages deterministic retrieval located.  The safe result until a validated
relationship signal runs is ``inconclusive`` or ``not_assessed``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
import re
import unicodedata
import uuid
from typing import Literal

import fitz
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.models.source_repository import SourceRepresentationRecord
from app.services.schemas import CitationMarkerMember, InTextCitation
from app.services.source_repository import representation_is_expired
from app.services.storage.backend import StorageBackend


ARTIFACT_VERSION = "phase3.8-evidence-v16"
RETRIEVAL_RULE_VERSION = "deterministic-passage-v1"
CANDIDATE_RETRIEVAL_VERSION = "candidate-specific-union-v5"
MAX_SOURCE_PAGES = 2_000
MAX_PAGE_CHARACTERS = 250_000
MAX_SOURCE_CHARACTERS = 10_000_000
MAX_PASSAGE_CHARACTERS = 1_800
MAX_CANDIDATES = 10
MAX_CANDIDATE_PASSAGES = 3


class EvidenceAuthorizationError(ValueError):
    """A stored representation is not authorized and usable for this request."""


class IdentityStatus(str, Enum):
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    MISMATCH = "mismatch"
    NOT_ASSESSED = "not_assessed"


class CoverageLevel(str, Enum):
    FULL_TEXT = "full_text"
    PARTIAL_TEXT = "partial_text"
    ABSTRACT = "abstract"
    METADATA_ONLY = "metadata_only"
    UNAVAILABLE = "unavailable"


class RelationshipStatus(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNRELATED = "unrelated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_ASSESSED = "not_assessed"


class VerificationVerdict(str, Enum):
    CONSISTENT = "consistent"
    MISREPRESENTATION = "misrepresentation"
    TOPICAL_MISMATCH = "topical_mismatch"
    INCONCLUSIVE = "inconclusive"
    NOT_ASSESSED = "not_assessed"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ClaimSourceSegment(BaseModel):
    """Exact parent-text material used to construct an atomic claim."""

    role: str = Field(min_length=1, max_length=40)
    local_start: int = Field(ge=0)
    local_end: int = Field(gt=0)
    paper_start: int = Field(ge=0)
    paper_end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=50_000)


class ClaimContextSegment(BaseModel):
    """Exact bounded paper context supplied only to resolve claim dependencies."""

    context_index: int = Field(ge=0, le=1)
    distance_before: int = Field(ge=1, le=2)
    text: str = Field(min_length=1, max_length=4_000)
    paper_start: int = Field(ge=0)
    paper_end: int = Field(gt=0)


class ClaimAntecedentCandidateEvidence(BaseModel):
    """One exact local-paper phrase considered during antecedent rescue."""

    candidate_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=1_000)
    paper_start: int = Field(ge=0)
    paper_end: int = Field(gt=0)
    search_tier: Literal[
        "immediate_context",
        "full_paragraph",
        "document_anchor",
        "nearby_section_document",
    ]
    match_kind: Literal[
        "named_type",
        "coordinated_named_group",
        "exact_head_phrase",
        "compatible_group_phrase",
    ]


class ClaimAntecedentDependency(BaseModel):
    """Inspectable mapping from exact claim wording to exact paper context."""

    mention_text: str = Field(min_length=1, max_length=500)
    mention_local_start: int = Field(ge=0)
    mention_local_end: int = Field(gt=0)
    mention_paper_start: int = Field(ge=0)
    mention_paper_end: int = Field(gt=0)
    resolution_status: Literal["resolved", "ambiguous", "unresolved"]
    confidence: Literal["high", "medium", "low", "none"]
    antecedent_context_index: int | None = Field(default=None, ge=0, le=1)
    antecedent_text: str | None = Field(default=None, max_length=4_000)
    antecedent_paper_start: int | None = Field(default=None, ge=0)
    antecedent_paper_end: int | None = Field(default=None, gt=0)
    search_tier: Literal[
        "immediate_context",
        "full_paragraph",
        "document_anchor",
        "nearby_section_document",
    ] | None = None
    candidates: list[ClaimAntecedentCandidateEvidence] = Field(
        default_factory=list, max_length=8
    )
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=4)
    method: str = Field(min_length=1, max_length=100)


class ClaimDiscourseDependency(BaseModel):
    """Exact student-paper context that materially scopes a cited continuation."""

    relation: Literal["answers_preceding_question"]
    context_index: int = Field(ge=0, le=1)
    context_text: str = Field(min_length=1, max_length=4_000)
    context_paper_start: int = Field(ge=0)
    context_paper_end: int = Field(gt=0)
    confidence: Literal["high", "medium"]
    method: str = Field(min_length=1, max_length=100)


class ClaimEvidence(BaseModel):
    claim_id: str = Field(min_length=1, max_length=128)
    paper_version_id: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=50_000)
    claim_type: Literal["quotation", "paraphrase"] = "paraphrase"
    granularity: Literal["citation_unit", "atomic_claim"] = "citation_unit"
    atomization_method: str = "not_run"
    parent_claim_id: str | None = Field(default=None, max_length=128)
    proposition_voice: str = "unclassified"
    student_stance: str = "not_applicable"
    verification_task: str = "not_assigned"
    source_segments: list[ClaimSourceSegment] = Field(default_factory=list)
    antecedent_context: list[ClaimContextSegment] = Field(default_factory=list)
    antecedent_dependencies: list[ClaimAntecedentDependency] = Field(
        default_factory=list
    )
    discourse_dependencies: list[ClaimDiscourseDependency] = Field(
        default_factory=list, max_length=2
    )
    context_dependency_status: Literal[
        "not_required", "resolved", "ambiguous", "unresolved"
    ] = "not_required"
    reference_ids: list[str] = Field(default_factory=list)
    citation_marker: str = ""
    citation_markers: list[CitationMarkerMember] = Field(default_factory=list)
    citation_marker_type: str = "unknown"
    extraction_confidence: Literal["high", "medium", "low"] = "low"
    page_locator: str = Field(default="", max_length=100)
    passage_start: int = -1
    passage_end: int = -1


class CitationSourceBinding(BaseModel):
    """Exact source-specific binding for one fanned-out verification run."""

    status: Literal["exact", "unresolved"] = "exact"
    reference_id: str = Field(min_length=1, max_length=255)
    cited_author_label: str = Field(min_length=1, max_length=200)
    marker_text: str = Field(default="", max_length=1_000)
    marker_local_start: int = Field(default=-1, ge=-1)
    marker_local_end: int = Field(default=-1, ge=-1)

    @model_validator(mode="after")
    def _marker_coordinates_are_consistent(self):
        if self.status == "unresolved":
            if self.marker_text or (self.marker_local_start, self.marker_local_end) != (-1, -1):
                raise ValueError("Unresolved source binding cannot claim an exact marker")
            return self
        if (
            not self.marker_text
            or self.marker_local_start < 0
            or self.marker_local_end - self.marker_local_start != len(self.marker_text)
        ):
            raise ValueError("Source-binding marker coordinates do not match its text")
        return self


class SourceIdentityEvidence(BaseModel):
    status: IdentityStatus
    confidence: ConfidenceLevel
    method: str
    canonical_work_id: str
    representation_id: str
    content_sha256: str
    authorization_scope_type: str = ""
    authorization_scope_id: str = ""
    verification_run_id: str | None = None
    edition_or_version: str | None = None
    representation_created_at: datetime
    admitted_at: datetime | None = None
    limitations: list[str] = Field(default_factory=list)


class CoverageEvidence(BaseModel):
    level: CoverageLevel
    confidence: ConfidenceLevel
    method: str
    representation_kind: str
    media_type: str
    completeness_verdict: str
    text_quality: str
    pages_total: int | None = None
    pages_inspected: list[int] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SourcePassageEvidence(BaseModel):
    passage_id: str
    representation_id: str
    content_sha256: str
    authorization_scope_type: str
    authorization_scope_id: str
    verification_run_id: str | None = None
    page_index: int | None = None
    page_label: str | None = None
    character_start: int
    character_end: int
    text: str
    retrieval_method: str
    retrieval_score: float
    passage_role: Literal[
        "body_prose",
        "reference_list",
        "citation_notes",
        "publication_metadata",
        "unknown",
    ] = "unknown"
    retrieval_rule_version: str = RETRIEVAL_RULE_VERSION


class PassageRelationshipScore(BaseModel):
    passage_id: str
    entailment: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)


class ClaimRelationshipEvidence(BaseModel):
    status: RelationshipStatus = RelationshipStatus.NOT_ASSESSED
    confidence: ConfidenceLevel = ConfidenceLevel.NONE
    method: str = "not_run"
    model_id: str | None = None
    model_revision: str | None = None
    signal_version: str | None = None
    passage_ids: list[str] = Field(default_factory=list)
    passage_scores: list[PassageRelationshipScore] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "No validated claim-relationship signal has run; retrieved passages "
            "are candidates only."
        ]
    )


class StructuredJudgmentEvidence(BaseModel):
    """Validated output from the bounded high-stakes judgment pass."""

    status: RelationshipStatus = RelationshipStatus.NOT_ASSESSED
    confidence: ConfidenceLevel = ConfidenceLevel.NONE
    method: str = "not_run"
    model_id: str | None = None
    judgment_version: str | None = None
    passage_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=1_500)
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class CandidatePassageRelevanceEvidence(BaseModel):
    """One bounded relevance assessment over an application-owned passage."""

    passage_id: str = Field(min_length=1, max_length=128)
    relevance: Literal[
        "relevant", "partially_relevant", "not_relevant", "uncertain"
    ]
    confidence: ConfidenceLevel
    rationale: str = Field(default="", max_length=1_000)


class PassageRelevanceGateEvidence(BaseModel):
    """Shadow-only gate between lexical retrieval and relationship judgment."""

    status: Literal["not_run", "complete", "not_assessed"] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    gate_version: str | None = None
    outcome: Literal[
        "not_run",
        "relevant_candidates_found",
        "no_relevant_candidate_passage",
        "uncertain",
        "not_assessed",
    ] = "not_run"
    assessments: list[CandidatePassageRelevanceEvidence] = Field(
        default_factory=list, max_length=3
    )
    relevant_passage_ids: list[str] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class UnitClaimFinding(BaseModel):
    """One evidence-conditioned finding over exact citation-unit segments."""

    finding_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=50_000)
    segments: list[ClaimSourceSegment] = Field(min_length=1, max_length=8)
    attribution: Literal["cited_source", "student", "ambiguous"]
    relationship: RelationshipStatus
    confidence: ConfidenceLevel
    passage_ids: list[str] = Field(default_factory=list, max_length=3)
    rationale: str = Field(default="", max_length=1_500)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class EvidenceConditionedUnitJudgment(BaseModel):
    """Shadow-only segmented judgment that preserves the complete citation unit."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    outcome: Literal[
        "not_run",
        "segmented_findings",
        "no_relevant_candidate_passage",
        "uncertain_relevance",
        "not_assessed",
    ] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    judgment_version: str | None = None
    findings: list[UnitClaimFinding] = Field(default_factory=list, max_length=12)
    unresolved_segments: list[ClaimSourceSegment] = Field(
        default_factory=list, max_length=12
    )
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class VerificationCandidate(BaseModel):
    """Application-owned exact segments proposed for one bounded judgment."""

    candidate_id: str = Field(min_length=1, max_length=128)
    role: Literal["whole_unit_guard", "relationship_candidate"]
    kind: Literal[
        "whole_unit",
        "clause",
        "coordinated_predicate",
        "relative_clause",
        "participial_clause",
        "shared_predicate",
        "shared_complement",
        "stance_context",
        "source_emphasis",
        "source_coverage",
        "student_analysis",
    ]
    text: str = Field(min_length=1, max_length=50_000)
    segments: list[ClaimSourceSegment] = Field(min_length=1, max_length=8)
    generation_method: str = Field(min_length=1, max_length=100)
    attribution: Literal["cited_source", "student", "ambiguous"] = "cited_source"
    verification_scope: Literal[
        "bounded_passage_relationship",
        "source_wide_coverage",
        "not_source_verification",
    ] = "bounded_passage_relationship"
    requires_parent_context: bool = False
    requires_antecedent_context: bool = False
    relationship_eligible: bool = False
    parent_candidate_id: str | None = Field(default=None, max_length=128)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class VerificationCandidateSet(BaseModel):
    """Inspectable deterministic candidates plus any uncovered substantive text."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    candidate_version: str | None = None
    candidates: list[VerificationCandidate] = Field(default_factory=list, max_length=16)
    uncovered_segments: list[ClaimSourceSegment] = Field(
        default_factory=list, max_length=12
    )
    limitations: list[str] = Field(default_factory=list)


class CitationUseRoute(BaseModel):
    """Application-owned procedure route for one exact citation candidate."""

    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    citation_use: Literal[
        "factual_attribution",
        "exemplification",
        "application",
        "evaluation",
        "further_reference_background",
        "multi_source_connection_comparison",
        "source_wide_coverage",
        "student_analysis",
        "unresolved",
    ]
    evidence_procedure: Literal[
        "bounded_passage_relationship",
        "bounded_source_proposition_relationship",
        "source_wide_coverage_engine",
        "multi_source_synthesis",
        "not_source_verification",
        "no_relationship_proposition",
        "unresolved",
    ]
    status: Literal["ready", "deferred", "excluded", "not_assessed"]
    confidence: Literal["high", "medium", "none"]
    reason_code: Literal[
        "ordinary_factual_relationship",
        "explicit_exemplification",
        "explicit_source_evaluation_proposition",
        "application_source_proposition_not_isolated",
        "application_target_unresolved",
        "further_reference_no_relationship_proposition",
        "multi_source_procedure_unavailable",
        "source_wide_engine_unavailable",
        "student_analysis_excluded",
        "ambiguous_candidate_scope",
        "materially_redundant_candidate_boundaries",
    ]
    relationship_judgment_allowed: bool = False
    limitations: list[str] = Field(default_factory=list, max_length=5)


class CitationUseRoutingEvidence(BaseModel):
    """Local fail-closed routing before any source-relationship procedure."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = (
        "not_run"
    )
    method: str = "not_run"
    router_version: str | None = None
    complete_citation_unit_candidate_id: str | None = Field(
        default=None, max_length=128
    )
    routes: list[CitationUseRoute] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    decision_applied: Literal[False] = False


class CandidatePassageSelectionItem(BaseModel):
    """One authorized passage selected for one fixed verification candidate."""

    passage_id: str = Field(min_length=1, max_length=128)
    rank: int = Field(ge=1, le=MAX_CANDIDATE_PASSAGES)
    score: float = Field(ge=0.0, le=1.0)
    channels: list[str] = Field(min_length=1, max_length=12)


class CandidatePassageSelection(BaseModel):
    """Inspectable, deterministic source search for one candidate."""

    candidate_id: str = Field(min_length=1, max_length=128)
    query_sha256: str = Field(min_length=64, max_length=64)
    query_facet_sha256s: list[str] = Field(default_factory=list, max_length=4)
    passages: list[CandidatePassageSelectionItem] = Field(
        default_factory=list, max_length=MAX_CANDIDATE_PASSAGES
    )
    rescue_applied: bool = False
    diversity_applied: bool = False
    limitations: list[str] = Field(default_factory=list, max_length=5)


class CandidatePassageRetrievalEvidence(BaseModel):
    """Candidate-specific retrieval that supplements whole-citation retrieval."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    retrieval_version: str | None = None
    selections: list[CandidatePassageSelection] = Field(
        default_factory=list, max_length=16
    )
    excluded_block_counts: dict[
        Literal["reference_list", "citation_notes", "publication_metadata"], int
    ] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class CandidateRelationshipFinding(BaseModel):
    """One model classification over one fixed application-owned candidate."""

    candidate_id: str = Field(min_length=1, max_length=128)
    status: Literal["assessed", "not_proposition", "uncertain", "not_assessed"]
    attribution: Literal["cited_source", "student", "ambiguous"]
    evidence_coverage: Literal[
        "complete", "partial", "absent", "uncertain", "not_assessed"
    ] = "not_assessed"
    context_resolution: Literal[
        "not_required", "resolved", "ambiguous", "unresolved"
    ] = "not_required"
    relationship: RelationshipStatus
    confidence: ConfidenceLevel
    passage_ids: list[str] = Field(default_factory=list, max_length=3)
    rationale: str = Field(default="", max_length=1_500)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class CandidateRelationshipEvaluation(BaseModel):
    """Shadow-only results; model output cannot create or rewrite candidate text."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    judgment_version: str | None = None
    findings: list[CandidateRelationshipFinding] = Field(
        default_factory=list, max_length=16
    )
    unresolved_candidate_ids: list[str] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class CandidateFacet(BaseModel):
    """Application-owned exact candidate material; never model-written text."""

    facet_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "candidate_as_written",
        "exact_component",
        "quantity",
        "time",
        "scope_or_condition",
        "modality_or_frequency",
        "causality",
        "comparison",
        "negation",
        "coordinated_content",
        "compound_component",
        "interpretive_inference",
        "inherited_discourse_scope",
        "source_attribution",
    ]
    segments: list[ClaimSourceSegment] = Field(default_factory=list, max_length=8)
    context_segments: list[ClaimContextSegment] = Field(
        default_factory=list, max_length=2
    )
    text: str = Field(min_length=1, max_length=50_000)
    material_to_aggregate: bool = True
    generation_method: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _must_have_exact_input_segment(self):
        if not self.segments and not self.context_segments:
            raise ValueError("A candidate facet requires exact claim or context segments")
        return self


class SourceAttributionRelationEvidence(BaseModel):
    """Exact application-derived source–cue–content relation inside a sentence."""

    actor_text: str = Field(min_length=1, max_length=200)
    cue_text: str = Field(min_length=1, max_length=500)
    content_text: str = Field(default="", max_length=2_000)
    family: str = Field(min_length=1, max_length=100)
    resolution_status: Literal["resolved", "ambiguous", "unresolved"]
    reason_code: Literal[
        "explicit_actor_cue_content",
        "mixed_document_and_external_voice",
        "multiple_attribution_frames",
        "attributed_content_span_unresolved",
    ]
    actor_start: int = Field(ge=0)
    actor_end: int = Field(gt=0)
    cue_start: int = Field(ge=0)
    cue_end: int = Field(gt=0)
    content_start: int | None = Field(default=None, ge=0)
    content_end: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _content_state_is_consistent(self):
        if self.actor_end <= self.actor_start or self.cue_end <= self.cue_start:
            raise ValueError("Attribution actor and cue coordinates must be nonempty")
        coordinates = (self.content_start, self.content_end)
        if self.resolution_status in {"resolved", "ambiguous"}:
            if None in coordinates or not self.content_text:
                raise ValueError("A bounded attribution relation requires exact content")
            if self.content_end <= self.content_start:
                raise ValueError("Attribution content coordinates must be nonempty")
        elif any(value is not None for value in coordinates) or self.content_text:
            raise ValueError("An unresolved attribution relation cannot claim content")
        return self


class EpistemicCommitmentCueEvidence(BaseModel):
    """Exact local reporting expression; families are not ordinal verdicts."""

    cue_text: str = Field(min_length=1, max_length=300)
    family: Literal[
        "tentative_inference",
        "neutral_report",
        "position_assertion",
        "evidence_claim",
        "conclusive_evidence",
        "neutral_attribution",
        "attributed_position",
    ]
    holder_role: Literal["external_actor", "document_author"]
    actor_text: str = Field(min_length=1, max_length=200)
    resolution_status: Literal["resolved", "ambiguous", "unresolved"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    content_start: int | None = Field(default=None, ge=0)
    content_end: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _coordinates_are_consistent(self):
        if self.end <= self.start or self.end - self.start != len(self.cue_text):
            raise ValueError("Epistemic cue coordinates do not match its exact text")
        if (self.content_start is None) != (self.content_end is None):
            raise ValueError("Epistemic cue content coordinates must be paired")
        if (
            self.content_start is not None
            and self.content_end <= self.content_start
        ):
            raise ValueError("Epistemic cue content coordinates are empty")
        return self


class SourceEvidenceSentence(BaseModel):
    """One exact sentence-like source span inside an authorized passage."""

    sentence_id: str = Field(min_length=1, max_length=128)
    passage_id: str = Field(min_length=1, max_length=128)
    passage_start: int = Field(ge=0)
    passage_end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=2_000)
    voice_role: Literal[
        "unmarked_document_voice",
        "explicit_external_attribution",
        "mixed_or_uncertain",
    ] = "unmarked_document_voice"
    attributed_actor_texts: list[str] = Field(default_factory=list, max_length=8)
    voice_cues: list[str] = Field(default_factory=list, max_length=8)
    attribution_relations: list[SourceAttributionRelationEvidence] = Field(
        default_factory=list, max_length=8
    )
    epistemic_commitment_cues: list[EpistemicCommitmentCueEvidence] = Field(
        default_factory=list, max_length=8
    )
    discourse_role: Literal[
        "none",
        "document_reported_actor_scope",
    ] = "none"
    discourse_actor_texts: list[str] = Field(default_factory=list, max_length=4)
    discourse_evidence_sentence_ids: list[str] = Field(
        default_factory=list, max_length=4
    )


class CandidateFacetBundle(BaseModel):
    """Fixed facets and evidence sentence IDs supplied for one candidate."""

    candidate_id: str = Field(min_length=1, max_length=128)
    facets: list[CandidateFacet] = Field(min_length=1, max_length=24)
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=48)
    source_discourse_sentence_ids: list[str] = Field(
        default_factory=list, max_length=4
    )
    student_epistemic_commitment_cues: list[
        EpistemicCommitmentCueEvidence
    ] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class FacetEvidenceFoundation(BaseModel):
    """Deterministic inputs for bounded facet-to-evidence classification."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    foundation_version: str | None = None
    source_sentences: list[SourceEvidenceSentence] = Field(
        default_factory=list, max_length=256
    )
    candidate_bundles: list[CandidateFacetBundle] = Field(
        default_factory=list, max_length=16
    )
    limitations: list[str] = Field(default_factory=list)


class FacetEvidenceMapping(BaseModel):
    """Bounded model direction over one fixed facet and fixed sentence IDs."""

    facet_id: str = Field(min_length=1, max_length=128)
    direction: Literal[
        "supports", "contradicts", "qualifies", "mixed", "none", "uncertain"
    ]
    confidence: ConfidenceLevel
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(default="", max_length=1_000)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class CandidateFacetFinding(BaseModel):
    """Model mappings plus an application-derived candidate outcome."""

    candidate_id: str = Field(min_length=1, max_length=128)
    status: Literal["assessed", "uncertain", "not_assessed"]
    context_resolution: Literal[
        "not_required", "resolved", "ambiguous", "unresolved"
    ] = "not_required"
    mappings: list[FacetEvidenceMapping] = Field(default_factory=list, max_length=24)
    derived_outcome: Literal[
        "supports",
        "contradicts",
        "mixed_or_qualified",
        "insufficient_evidence",
        "not_assessed",
    ]
    evidence_coverage: Literal[
        "complete", "partial", "absent", "uncertain", "not_assessed"
    ]
    locator_status: Literal[
        "not_provided",
        "evidence_at_locator",
        "evidence_only_elsewhere",
        "no_evidence",
        "unresolved",
    ] = "unresolved"
    limitations: list[str] = Field(default_factory=list, max_length=5)


class FacetEvidenceLedger(BaseModel):
    """Shadow-only fixed-ID mappings and deterministic aggregate outcomes."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    judgment_version: str | None = None
    findings: list[CandidateFacetFinding] = Field(default_factory=list, max_length=16)
    derived_citation_outcome: Literal[
        "not_run",
        "supports",
        "contradicts",
        "mixed_or_qualified",
        "insufficient_evidence",
        "not_assessed",
    ] = "not_run"
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class PairwiseFacetEvidenceSelection(BaseModel):
    """Evidence IDs selected for exactly one fixed material facet."""

    facet_id: str = Field(min_length=1, max_length=128)
    status: Literal[
        "evidence_selected", "no_evidence", "uncertain", "not_assessed"
    ]
    confidence: ConfidenceLevel
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=6)
    rationale: str = Field(default="", max_length=1_000)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    failure_code: Literal[
        "none",
        "context_unresolved",
        "prompt_budget_exceeded",
        "response_schema_invalid",
        "facet_id_mismatch",
        "evidence_not_authorized",
        "selection_contract_invalid",
        "candidate_or_facet_id_mismatch",
        "holder_contract_invalid",
        "no_relevant_evidence",
        "selection_unavailable",
        "provider_or_runtime_failure",
    ] = "none"


class PairwiseFacetDecision(BaseModel):
    """Direction for one fixed facet against only its selected evidence."""

    facet_id: str = Field(min_length=1, max_length=128)
    status: Literal["assessed", "uncertain", "not_assessed"]
    direction: Literal[
        "supports", "contradicts", "qualifies", "mixed", "none", "uncertain"
    ]
    confidence: ConfidenceLevel
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=6)
    rationale: str = Field(default="", max_length=1_000)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    failure_code: Literal[
        "none",
        "selection_unavailable",
        "prompt_budget_exceeded",
        "response_schema_invalid",
        "facet_id_mismatch",
        "evidence_not_selected",
        "evidence_not_authorized",
        "direction_contract_invalid",
        "candidate_or_facet_id_mismatch",
        "holder_contract_invalid",
        "no_relevant_evidence",
        "provider_or_runtime_failure",
    ] = "none"


class PropositionHolderAssessment(BaseModel):
    """Dedicated source-voice result, separate from semantic direction."""

    candidate_id: str = Field(min_length=1, max_length=128)
    facet_id: str = Field(min_length=1, max_length=128)
    status: Literal["assessed", "uncertain", "not_assessed"]
    holder_relation: Literal[
        "document_author",
        "different_actor",
        "mixed",
        "uncertain",
        "not_assessed",
    ]
    mapped_direction: Literal[
        "supports", "contradicts", "qualifies", "uncertain"
    ]
    confidence: ConfidenceLevel
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=8)
    reported_actor_texts: list[str] = Field(default_factory=list, max_length=4)
    rationale: str = Field(default="", max_length=1_000)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    failure_code: Literal[
        "none",
        "context_unresolved",
        "prompt_budget_exceeded",
        "response_schema_invalid",
        "candidate_or_facet_id_mismatch",
        "evidence_not_authorized",
        "holder_contract_invalid",
        "no_relevant_evidence",
        "selection_unavailable",
        "provider_or_runtime_failure",
    ] = "none"


class PairwiseCandidateFinding(BaseModel):
    """Decomposed shadow trace plus its application-derived candidate finding."""

    candidate_id: str = Field(min_length=1, max_length=128)
    evidence_selections: list[PairwiseFacetEvidenceSelection] = Field(
        default_factory=list, max_length=24
    )
    facet_decisions: list[PairwiseFacetDecision] = Field(
        default_factory=list, max_length=24
    )
    proposition_holder: PropositionHolderAssessment | None = None
    derived_finding: CandidateFacetFinding


class PairwiseFacetEvaluation(BaseModel):
    """Shadow architecture: select per facet, judge pairwise, aggregate locally."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    selection_version: str | None = None
    direction_version: str | None = None
    proposition_holder_version: str | None = None
    findings: list[PairwiseCandidateFinding] = Field(default_factory=list, max_length=16)
    derived_citation_outcome: Literal[
        "not_run",
        "supports",
        "contradicts",
        "mixed_or_qualified",
        "insufficient_evidence",
        "not_assessed",
    ] = "not_run"
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class DecisiveCriticCheck(BaseModel):
    """One neutral, fixed-type check over authorized facets and evidence."""

    check_type: Literal[
        "missing_material_detail",
        "scope_mismatch",
        "agency_or_attribution_mismatch",
        "incompatible_counterevidence",
        "locator_conflict",
    ]
    result: Literal["no_defect", "defect", "uncertain"]
    facet_ids: list[str] = Field(default_factory=list, max_length=24)
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=24)
    mapping_reconciliation: Literal[
        "not_required",
        "mapping_supports_proposed_label",
        "mapping_does_not_establish_facet",
        "uncertain",
    ] = "not_required"
    rationale: str = Field(default="", max_length=600)


class DecisiveCriticFinding(BaseModel):
    """Application-derived outcome from the complete neutral checklist."""

    candidate_id: str = Field(min_length=1, max_length=128)
    proposed_outcome: Literal["supports", "contradicts"]
    status: Literal["upheld", "challenged", "uncertain", "not_assessed"]
    challenge_types: list[
        Literal[
            "missing_material_detail",
            "scope_mismatch",
            "agency_or_attribution_mismatch",
            "incompatible_counterevidence",
            "locator_conflict",
        ]
    ] = Field(default_factory=list, max_length=5)
    checks: list[DecisiveCriticCheck] = Field(default_factory=list, max_length=5)
    reviewed_facet_ids: list[str] = Field(default_factory=list, max_length=24)
    challenged_facet_ids: list[str] = Field(default_factory=list, max_length=24)
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=24)
    rationale: str = Field(default="", max_length=1_000)
    limitations: list[str] = Field(default_factory=list, max_length=5)
    failure_code: Literal[
        "none",
        "evidence_bundle_missing",
        "prompt_budget_exceeded",
        "response_schema_invalid",
        "response_type_invalid",
        "candidate_id_mismatch",
        "material_facet_review_incomplete",
        "checklist_type_coverage_mismatch",
        "checklist_facet_not_authorized",
        "checklist_evidence_not_authorized",
        "defect_missing_facet",
        "defect_missing_evidence",
        "missing_detail_mapping_not_reconciled",
        "no_defect_carries_decision_data",
        "checklist_result_contract_invalid",
        "provider_or_runtime_failure",
    ] = "none"
    effective_outcome: Literal["supports", "contradicts", "not_assessed"]


class DecisiveCriticEvaluation(BaseModel):
    """Shadow-only neutral checklist; it can remove but never create a label."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    model_id: str | None = None
    critic_version: str | None = None
    findings: list[DecisiveCriticFinding] = Field(default_factory=list, max_length=16)
    challenged_candidate_ids: list[str] = Field(default_factory=list, max_length=16)
    derived_citation_outcome: Literal[
        "not_run",
        "supports",
        "contradicts",
        "mixed_or_qualified",
        "insufficient_evidence",
        "not_assessed",
    ] = "not_run"
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class VerificationEvidenceArtifact(BaseModel):
    artifact_version: str = ARTIFACT_VERSION
    verification_id: str
    created_at: datetime
    claim: ClaimEvidence
    # Historical private calibration artifacts predate source-specific fan-out.
    # New builders always populate this field; downstream holder logic must
    # abstain when it is absent rather than deriving one author from a combined
    # citation marker.
    source_binding: CitationSourceBinding | None = None
    source_identity: SourceIdentityEvidence
    coverage: CoverageEvidence
    passages: list[SourcePassageEvidence] = Field(default_factory=list)
    relationship: ClaimRelationshipEvidence
    passage_relevance: PassageRelevanceGateEvidence = Field(
        default_factory=PassageRelevanceGateEvidence
    )
    judgment: StructuredJudgmentEvidence = Field(
        default_factory=StructuredJudgmentEvidence
    )
    unit_judgment: EvidenceConditionedUnitJudgment = Field(
        default_factory=EvidenceConditionedUnitJudgment
    )
    verification_candidates: VerificationCandidateSet = Field(
        default_factory=VerificationCandidateSet
    )
    citation_use_routing: CitationUseRoutingEvidence = Field(
        default_factory=CitationUseRoutingEvidence
    )
    candidate_passage_retrieval: CandidatePassageRetrievalEvidence = Field(
        default_factory=CandidatePassageRetrievalEvidence
    )
    candidate_relationships: CandidateRelationshipEvaluation = Field(
        default_factory=CandidateRelationshipEvaluation
    )
    facet_evidence_foundation: FacetEvidenceFoundation = Field(
        default_factory=FacetEvidenceFoundation
    )
    facet_evidence_ledger: FacetEvidenceLedger = Field(
        default_factory=FacetEvidenceLedger
    )
    pairwise_facet_evaluation: PairwiseFacetEvaluation = Field(
        default_factory=PairwiseFacetEvaluation
    )
    decisive_critic: DecisiveCriticEvaluation = Field(
        default_factory=DecisiveCriticEvaluation
    )
    verdict: VerificationVerdict
    reason_codes: list[str] = Field(default_factory=list)
    retrieval_route: str

    def report_payload(self) -> dict:
        """Return the inspectable JSON-safe payload without source binaries."""
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class AuthorizedRepresentation:
    representation_id: str
    canonical_work_id: str
    content_object_id: str
    content_sha256: str
    content: bytes
    representation_kind: str
    media_type: str
    provenance: str
    scope_type: str
    scope_id: str
    identity_verdict: str
    identity_confidence: float | None
    completeness_verdict: str
    text_quality: str
    edition_or_version: str | None
    created_at: datetime
    admitted_at: datetime | None
    verification_run_id: str | None = None


@dataclass(frozen=True)
class _SourcePage:
    index: int | None
    label: str | None
    text: str


@dataclass(frozen=True)
class _PassageCandidate:
    page_index: int | None
    page_label: str | None
    start: int
    end: int
    text: str
    method: str
    score: float


def authorize_representation(
    session: Session,
    backend: StorageBackend,
    *,
    representation_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
) -> AuthorizedRepresentation:
    """Load bytes only after exact-scope, state, expiry, and hash validation."""
    try:
        record_id = (
            representation_id
            if isinstance(representation_id, uuid.UUID)
            else uuid.UUID(str(representation_id))
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceAuthorizationError("Invalid representation identifier") from exc

    record = session.get(SourceRepresentationRecord, record_id)
    if record is None:
        raise EvidenceAuthorizationError("Representation does not exist")
    requested_scope_type = scope_type.strip().casefold()
    requested_scope_id = scope_id.strip()
    if not requested_scope_type or not requested_scope_id:
        raise EvidenceAuthorizationError("Requesting authorization scope is required")
    if (
        record.scope_type != requested_scope_type
        or record.scope_id != requested_scope_id
    ):
        raise EvidenceAuthorizationError(
            "Representation is not authorized in the requesting scope"
        )
    if record.admission_state != "accepted":
        raise EvidenceAuthorizationError("Representation is not accepted for use")
    if representation_is_expired(record, now=now):
        raise EvidenceAuthorizationError("Representation authorization has expired")
    if record.identity_verdict not in {"match", "verified"}:
        raise EvidenceAuthorizationError("Representation identity is not verified")

    content_object = record.content_object
    if content_object.deletion_pending:
        raise EvidenceAuthorizationError("Representation content is pending deletion")
    if not backend.exists(content_object.storage_key):
        raise EvidenceAuthorizationError("Authorized representation content is missing")
    content = backend.download(content_object.storage_key)
    digest = hashlib.sha256(content).hexdigest()
    if digest != content_object.content_sha256 or len(content) != content_object.byte_size:
        raise EvidenceAuthorizationError(
            "Authorized representation failed immutable-object verification"
        )

    audit_scope = (record.validation_evidence or {}).get("authorization_scope")
    if audit_scope and audit_scope != {
        "type": record.scope_type,
        "id": record.scope_id,
    }:
        raise EvidenceAuthorizationError("Authorization audit evidence is inconsistent")

    return AuthorizedRepresentation(
        representation_id=str(record.id),
        canonical_work_id=str(record.canonical_work_id),
        content_object_id=str(record.content_object_id),
        content_sha256=digest,
        content=content,
        representation_kind=record.representation_kind,
        media_type=content_object.media_type,
        provenance=record.provenance,
        scope_type=record.scope_type,
        scope_id=record.scope_id,
        identity_verdict=record.identity_verdict,
        identity_confidence=record.identity_confidence,
        completeness_verdict=record.completeness_verdict,
        text_quality=record.text_quality or "unknown",
        edition_or_version=record.edition_or_version,
        created_at=record.created_at,
        admitted_at=record.admitted_at,
    )


def claim_evidence_from_citation(
    citation: InTextCitation,
    *,
    paper_version_id: str,
    antecedent_context: list[ClaimContextSegment] | None = None,
) -> ClaimEvidence:
    """Convert an accepted, exactly linked paper citation into a claim."""
    version_id = paper_version_id.strip()
    if not version_id:
        raise ValueError("paper_version_id is required")
    if citation.drop_reason is not None:
        raise ValueError("Rejected citation spans cannot enter verification")
    if citation.link_status != "linked" or not citation.reference_ids:
        raise ValueError("Verification requires uniquely linked reference membership")
    if (
        citation.passage_start < 0
        or citation.passage_end <= citation.passage_start
        or not citation.text.strip()
    ):
        raise ValueError("Citation passage coordinates are invalid")
    marker_members = list(citation.citation_markers)
    if marker_members:
        assigned_reference_ids: set[str] = set()
        previous_end = -1
        for marker in marker_members:
            if (
                marker.local_end > len(citation.text)
                or citation.text[marker.local_start:marker.local_end] != marker.text
                or not marker.reference_ids
                or not set(marker.reference_ids).issubset(citation.reference_ids)
                or marker.local_start < previous_end
            ):
                raise ValueError("Citation marker membership is not exact")
            assigned_reference_ids.update(marker.reference_ids)
            previous_end = marker.local_end
        if assigned_reference_ids != set(citation.reference_ids):
            raise ValueError("Citation marker membership does not cover every reference")
    else:
        marker_start = citation.text.find(citation.citation_marker)
        if marker_start >= 0 and citation.citation_marker:
            marker_members = [
                CitationMarkerMember(
                    text=citation.citation_marker,
                    local_start=marker_start,
                    local_end=marker_start + len(citation.citation_marker),
                    reference_ids=list(citation.reference_ids),
                    marker_type=citation.marker_type,
                )
            ]

    claim_id = _stable_id(
        "paper-claim-v1",
        version_id,
        "|".join(sorted(citation.reference_ids)),
        str(citation.passage_start),
        str(citation.passage_end),
        citation.text,
    )
    return ClaimEvidence(
        claim_id=claim_id,
        paper_version_id=version_id,
        text=citation.text,
        claim_type=citation.claim_type,
        granularity="citation_unit",
        atomization_method="not_run",
        reference_ids=list(citation.reference_ids),
        citation_marker=citation.citation_marker,
        citation_markers=marker_members,
        citation_marker_type=citation.marker_type,
        extraction_confidence=citation.confidence,
        page_locator=citation.page_number,
        passage_start=citation.passage_start,
        passage_end=citation.passage_end,
        antecedent_context=list(antecedent_context or []),
    )


def build_passage_evidence(
    source: AuthorizedRepresentation,
    *,
    claim: ClaimEvidence,
    top_k: int = 3,
    active_reference_id: str | None = None,
    cited_author_label: str | None = None,
) -> VerificationEvidenceArtifact:
    """Retrieve bounded inspectable passages and safely abstain from judgment."""
    source_binding = _source_binding(
        claim,
        active_reference_id=active_reference_id,
        cited_author_label=cited_author_label,
    )
    bounded_top_k = max(1, min(top_k, MAX_CANDIDATES))
    pages, extraction_limitations = _extract_pages(source)
    candidates = _retrieve_candidates(
        pages,
        claim_text=_retrieval_claim_text(claim),
        claim_type=claim.claim_type,
        page_locator=claim.page_locator,
        top_k=bounded_top_k,
    )
    passages = [
        _passage_evidence(source, candidate) for candidate in candidates
    ]

    identity = SourceIdentityEvidence(
        status=IdentityStatus.VERIFIED,
        confidence=_identity_confidence(source.identity_confidence),
        method=(
            "transient_verification_run_record"
            if source.verification_run_id
            else "durable_admission_record"
        ),
        canonical_work_id=source.canonical_work_id,
        representation_id=source.representation_id,
        content_sha256=source.content_sha256,
        authorization_scope_type=source.scope_type,
        authorization_scope_id=source.scope_id,
        verification_run_id=source.verification_run_id,
        edition_or_version=source.edition_or_version,
        representation_created_at=source.created_at,
        admitted_at=source.admitted_at,
        limitations=(
            []
            if source.edition_or_version
            else ["Edition/version was not explicitly recorded."]
        ),
    )
    coverage = _coverage_evidence(source, pages, extraction_limitations)
    reason_codes: list[str] = []
    usable_text = any(page.text.strip() for page in pages)
    if not usable_text:
        verdict = VerificationVerdict.NOT_ASSESSED
        reason_codes.append("source_text_unavailable")
    else:
        verdict = VerificationVerdict.INCONCLUSIVE
        reason_codes.append(
            "candidate_passages_located_relation_not_assessed"
            if passages
            else "no_passage_located_in_available_evidence"
        )
    if source.completeness_verdict not in {"complete", "not_applicable"}:
        reason_codes.append("source_completeness_limited")
    if source.text_quality not in {"digital", "born_digital"}:
        reason_codes.append("source_text_quality_limited")
    if claim.granularity != "atomic_claim":
        reason_codes.append("claim_not_atomized")

    relationship = ClaimRelationshipEvidence(
        passage_ids=[passage.passage_id for passage in passages]
    )
    verification_id = _stable_id(
        ARTIFACT_VERSION,
        claim.paper_version_id,
        claim.claim_id,
        source_binding.reference_id,
        source.representation_id,
        source.content_sha256,
    )
    return VerificationEvidenceArtifact(
        verification_id=verification_id,
        created_at=datetime.now(timezone.utc),
        claim=claim,
        source_binding=source_binding,
        source_identity=identity,
        coverage=coverage,
        passages=passages,
        relationship=relationship,
        verdict=verdict,
        reason_codes=reason_codes,
        retrieval_route=source.provenance,
    )


def _source_binding(
    claim: ClaimEvidence,
    *,
    active_reference_id: str | None,
    cited_author_label: str | None,
) -> CitationSourceBinding:
    reference_id = (active_reference_id or "").strip()
    if not reference_id:
        if len(claim.reference_ids) != 1:
            raise ValueError(
                "Multi-source verification requires an active reference binding"
            )
        reference_id = claim.reference_ids[0]
    if reference_id not in claim.reference_ids:
        raise ValueError("Active reference binding is not a member of the claim")

    members = [
        marker
        for marker in claim.citation_markers
        if reference_id in marker.reference_ids
    ]
    if not members and claim.citation_marker:
        marker_start = claim.text.find(claim.citation_marker)
        if marker_start >= 0:
            members = [
                CitationMarkerMember(
                    text=claim.citation_marker,
                    local_start=marker_start,
                    local_end=marker_start + len(claim.citation_marker),
                    reference_ids=[reference_id],
                    marker_type=claim.citation_marker_type,
                )
            ]
    if len(members) != 1:
        if len(claim.reference_ids) != 1:
            raise ValueError(
                "Active reference binding requires one exact citation marker member"
            )
        return CitationSourceBinding(
            status="unresolved",
            reference_id=reference_id,
            cited_author_label=(cited_author_label or "unknown").strip() or "unknown",
        )
    marker = members[0]
    if (
        marker.local_end > len(claim.text)
        or claim.text[marker.local_start:marker.local_end] != marker.text
    ):
        raise ValueError("Active citation marker member is not exact")

    author = (cited_author_label or "").strip()
    if not author:
        if len(claim.reference_ids) != 1:
            raise ValueError(
                "Multi-source verification requires a source-specific author label"
            )
        author = _marker_author_label(marker.text)
    if not author:
        raise ValueError("A source-specific cited-author label is required")
    return CitationSourceBinding(
        reference_id=reference_id,
        cited_author_label=author,
        marker_text=marker.text,
        marker_local_start=marker.local_start,
        marker_local_end=marker.local_end,
    )


def _marker_author_label(marker_text: str) -> str:
    value = re.sub(r"[()]", " ", marker_text)
    value = re.sub(
        r"\b(?:19|20)\d{2}[a-z]?\b", " ", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"\b(?:p{1,2}\.?\s*)?\d+(?:\s*[-–—]\s*\d+)?\b", " ", value
    )
    value = re.sub(r"\bet\s+al\.?\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,;:")
    return value[:200]


def attach_candidate_passage_retrieval(
    source: AuthorizedRepresentation,
    artifact: VerificationEvidenceArtifact,
    *,
    top_k: int = MAX_CANDIDATE_PASSAGES,
) -> VerificationEvidenceArtifact:
    """Search the complete authorized source separately for each fixed candidate.

    Whole-citation passages remain in ``artifact.passages`` and continue to feed
    the broad relevance assessment. This pass adds an explicit per-candidate
    selection whose IDs are later used by the one-candidate relationship judge.
    No search result changes a verdict.
    """
    from app.services.citation_use_router import (
        attach_citation_use_routes,
        routed_relationship_candidate_ids,
    )

    if artifact.citation_use_routing.status == "not_run":
        artifact = attach_citation_use_routes(artifact)
    candidate_set = artifact.verification_candidates
    if candidate_set.status not in {"complete", "incomplete"}:
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": CandidatePassageRetrievalEvidence(
                    status="not_assessed",
                    method="verification_candidates_required",
                    retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
                    limitations=[
                        "Candidate-specific retrieval requires application-owned verification candidates."
                    ],
                )
            }
        )
    if not _source_matches_artifact(source, artifact):
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": CandidatePassageRetrievalEvidence(
                    status="not_assessed",
                    method="authorized_source_mismatch",
                    retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
                    limitations=[
                        "The supplied source did not match the artifact representation, hash, scope, and verification run."
                    ],
                )
            }
        )

    routed_ids = routed_relationship_candidate_ids(artifact)
    eligible = [
        candidate
        for candidate in candidate_set.candidates
        if candidate.relationship_eligible and candidate.candidate_id in routed_ids
    ]
    if not eligible:
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": CandidatePassageRetrievalEvidence(
                    status="not_assessed",
                    method="no_routed_bounded_relationship_candidate",
                    retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
                    limitations=[
                        "No fixed candidate was routed to an available bounded source-relationship procedure."
                    ],
                )
            }
        )

    bounded_top_k = max(1, min(top_k, MAX_CANDIDATE_PASSAGES))
    pages, extraction_limitations = _extract_pages(source)
    if not any(page.text.strip() for page in pages):
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": CandidatePassageRetrievalEvidence(
                    status="not_assessed",
                    method="source_text_unavailable",
                    retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
                    limitations=[
                        "The complete authorized source could not be searched for fixed candidates.",
                        *extraction_limitations[:4],
                    ],
                )
            }
        )

    excluded_block_counts: Counter[str] = Counter()
    for _page, _start, _end, _block_text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            excluded_block_counts[role] += 1

    broad_by_id = {
        passage.passage_id: passage
        for passage in artifact.passages
        if _passage_matches_source(source, passage)
    }
    evidence_by_id = dict(broad_by_id)
    selections: list[CandidatePassageSelection] = []
    incomplete = False
    for candidate in eligible:
        query_text = _candidate_retrieval_text(artifact.claim, candidate)
        ranked, rescue_applied, facet_queries = _candidate_union_candidates(
            pages,
            query_text=query_text,
            page_locator=artifact.claim.page_locator,
            broad_passages=list(broad_by_id.values()),
            top_k=bounded_top_k,
        )
        items: list[CandidatePassageSelectionItem] = []
        for rank, (passage_candidate, channels) in enumerate(ranked, start=1):
            passage = _passage_evidence(source, passage_candidate)
            evidence_by_id.setdefault(passage.passage_id, passage)
            items.append(
                CandidatePassageSelectionItem(
                    passage_id=passage.passage_id,
                    rank=rank,
                    score=round(passage_candidate.score, 6),
                    channels=channels,
                )
            )
        if not items:
            incomplete = True
        limitations = []
        if rescue_applied:
            limitations.append(
                "A relaxed bounded lexical/concept rescue supplemented the normal candidate search."
            )
        if facet_queries:
            limitations.append(
                "Material quantity facets were searched separately and passage slots were diversified across those exact query fragments."
            )
        if not items:
            limitations.append(
                "No candidate-specific passage was located; this is not a source-wide absence finding."
            )
        selections.append(
            CandidatePassageSelection(
                candidate_id=candidate.candidate_id,
                query_sha256=hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                query_facet_sha256s=[
                    hashlib.sha256(query.encode("utf-8")).hexdigest()
                    for query in facet_queries
                ],
                passages=items,
                rescue_applied=rescue_applied,
                diversity_applied=bool(facet_queries),
                limitations=limitations,
            )
        )

    retrieval = CandidatePassageRetrievalEvidence(
        status="incomplete" if incomplete else "complete",
        method="candidate_specific_exact_lexical_concept_facet_union",
        retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
        selections=selections,
        excluded_block_counts=dict(excluded_block_counts),
        limitations=[
            "Candidate-specific retrieval is recall-oriented and does not establish support, contradiction, or source-wide absence.",
            "Definite publication-metadata, reference-list, and citation-only note blocks are excluded before ranking; ordinary cited body prose and ambiguous blocks remain eligible.",
            "Material quantity facets may reserve distinct passage slots, but the later evidence ledger still determines whether the complete qualifier is established.",
            "Dense-semantic retrieval remains an unvalidated optional channel and is not represented as active in this version.",
            *extraction_limitations,
        ],
    )
    return artifact.model_copy(
        update={
            "passages": list(evidence_by_id.values()),
            "candidate_passage_retrieval": retrieval,
        }
    )


def _retrieval_claim_text(claim: ClaimEvidence) -> str:
    """Expand paraphrase retrieval with validated antecedents without rewriting."""
    if claim.claim_type == "quotation":
        return claim.text
    resolved = [
        dependency.antecedent_text
        for dependency in claim.antecedent_dependencies
        if dependency.resolution_status == "resolved"
        and dependency.confidence == "high"
        and dependency.antecedent_text
    ]
    return "\n".join(dict.fromkeys([claim.text, *resolved]))


def _candidate_retrieval_text(
    claim: ClaimEvidence, candidate: VerificationCandidate
) -> str:
    """Use exact candidate wording plus only already-validated antecedents."""
    resolved = [
        dependency.antecedent_text
        for dependency in claim.antecedent_dependencies
        if candidate.requires_antecedent_context
        and dependency.resolution_status == "resolved"
        and dependency.confidence == "high"
        and dependency.antecedent_text
    ]
    return "\n".join(dict.fromkeys([candidate.text, *resolved]))


_DISTRIBUTED_QUANTITY_CUE = re.compile(
    r"\b(?:a\s+large\s+number\s+of|large\s+numbers\s+of|numerous|many|"
    r"several|multiple)\b",
    re.IGNORECASE,
)


def _candidate_retrieval_facets(query_text: str) -> list[str]:
    """Return exact retrieval-only fragments for distributed quantity claims.

    These fragments never replace or rewrite the fixed candidate. The rule is
    deliberately narrow: it activates only for an explicit plurality cue and
    retains only conjunction pieces with enough lexical material to search.
    """
    if not _DISTRIBUTED_QUANTITY_CUE.search(query_text):
        return []
    quantity = _DISTRIBUTED_QUANTITY_CUE.search(query_text)
    coordinated_heads = re.search(
        r"\b(?P<left>[A-Za-z][A-Za-z'’\-]*)\s+(?:and|or)\s+"
        r"(?P<right>[A-Za-z][A-Za-z'’\-]*)\b",
        query_text[quantity.end() :],
        re.IGNORECASE,
    )
    if coordinated_heads is not None:
        facets = [
            coordinated_heads.group("left"),
            coordinated_heads.group("right"),
        ]
        if all(_meaningful_tokens(facet) for facet in facets):
            return facets
    pieces = re.split(r"\s+(?:and|or)\s+", query_text, flags=re.IGNORECASE)
    facets = []
    for piece in pieces:
        value = re.sub(r"\s+", " ", piece).strip(" ,;:")
        if len(_meaningful_tokens(value)) < 3:
            continue
        if value.casefold() == re.sub(r"\s+", " ", query_text).strip().casefold():
            continue
        facets.append(value)
    return list(dict.fromkeys(facets))[:4] if len(facets) >= 2 else []


def _source_matches_artifact(
    source: AuthorizedRepresentation, artifact: VerificationEvidenceArtifact
) -> bool:
    identity = artifact.source_identity
    return (
        source.representation_id == identity.representation_id
        and source.content_sha256 == identity.content_sha256
        and source.scope_type == identity.authorization_scope_type
        and source.scope_id == identity.authorization_scope_id
        and source.verification_run_id == identity.verification_run_id
    )


def _passage_matches_source(
    source: AuthorizedRepresentation, passage: SourcePassageEvidence
) -> bool:
    return (
        passage.representation_id == source.representation_id
        and passage.content_sha256 == source.content_sha256
        and passage.authorization_scope_type == source.scope_type
        and passage.authorization_scope_id == source.scope_id
        and passage.verification_run_id == source.verification_run_id
    )


def _extract_pages(
    source: AuthorizedRepresentation,
) -> tuple[list[_SourcePage], list[str]]:
    limitations: list[str] = []
    if source.representation_kind == "pdf":
        try:
            document = fitz.open(stream=source.content, filetype="pdf")
            try:
                if document.page_count > MAX_SOURCE_PAGES:
                    return [], ["Source exceeds the verification page limit."]
                pages = []
                total_characters = 0
                for index, page in enumerate(document):
                    text = page.get_text("text")
                    if len(text) > MAX_PAGE_CHARACTERS:
                        text = text[:MAX_PAGE_CHARACTERS]
                        limitations.append(f"Page {index + 1} text was truncated by policy.")
                    remaining = MAX_SOURCE_CHARACTERS - total_characters
                    if remaining <= 0:
                        limitations.append(
                            "Source text inspection stopped at the total character limit."
                        )
                        break
                    source_limit_reached = len(text) > remaining
                    if source_limit_reached:
                        text = text[:remaining]
                        limitations.append(
                            "Source text inspection stopped at the total character limit."
                        )
                    pages.append(
                        _SourcePage(
                            index=index,
                            label=page.get_label() or None,
                            text=text,
                        )
                    )
                    total_characters += len(text)
                    if source_limit_reached:
                        break
                return pages, limitations
            finally:
                document.close()
        except (fitz.FileDataError, fitz.mupdf.FzErrorBase, RuntimeError, ValueError):
            return [], ["Accepted PDF could not be text-extracted for verification."]

    if source.representation_kind == "plain_text":
        text = source.content.decode("utf-8", "replace")
        replacement_ratio = text.count("\ufffd") / max(len(text), 1)
        if replacement_ratio > 0.01:
            limitations.append("Plain-text decoding produced replacement characters.")
        raw_pages = text.split("\f")
        return [
            _SourcePage(
                index=index if len(raw_pages) > 1 else None,
                label=str(index + 1) if len(raw_pages) > 1 else None,
                text=page_text[:MAX_PAGE_CHARACTERS],
            )
            for index, page_text in enumerate(raw_pages[:MAX_SOURCE_PAGES])
        ], limitations

    return [], [
        f"Representation kind {source.representation_kind!r} has no Phase 3.8 text extractor."
    ]


def _retrieve_candidates(
    pages: list[_SourcePage],
    *,
    claim_text: str,
    claim_type: str,
    page_locator: str,
    top_k: int,
) -> list[_PassageCandidate]:
    if not claim_text.strip():
        return []
    locator_pages = _page_locator_values(page_locator)
    exact_target = _quotation_target(claim_text) if claim_type == "quotation" else None
    exact: list[_PassageCandidate] = []
    if exact_target:
        reference_section_started = False
        for page in pages:
            reference_heading = _REFERENCE_HEADING.search(page.text)
            match = _normalized_find(page.text, exact_target)
            if match is None:
                reference_section_started = bool(
                    reference_section_started or reference_heading
                )
                continue
            start, end = _context_bounds(page.text, *match)
            exact_text = page.text[start:end].strip()
            role = (
                "reference_list"
                if reference_section_started
                or (reference_heading is not None and start >= reference_heading.start())
                else passage_role_from_text(exact_text)
            )
            reference_section_started = bool(
                reference_section_started or reference_heading
            )
            if role in _EXCLUDED_RETRIEVAL_ROLES:
                continue
            exact.append(
                _PassageCandidate(
                    page_index=page.index,
                    page_label=page.label,
                    start=start,
                    end=end,
                    text=exact_text,
                    method="exact_quotation",
                    score=min(1.0, 0.95 + _page_boost(page, locator_pages)),
                )
            )
    if exact:
        return _deduplicate_candidates(exact)[:top_k]

    claim_tokens = _meaningful_tokens(claim_text)
    if not claim_tokens:
        return []
    scored: list[_PassageCandidate] = []
    for page, start, end, text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        score = _lexical_score(claim_tokens, claim_text, text)
        if score <= 0:
            continue
        score = min(0.94, score + _page_boost(page, locator_pages))
        scored.append(
            _PassageCandidate(
                page_index=page.index,
                page_label=page.label,
                start=start,
                end=end,
                text=text.strip(),
                method="lexical_overlap",
                score=score,
            )
        )
    scored.sort(key=lambda candidate: (candidate.score, -candidate.start), reverse=True)
    return _deduplicate_candidates(scored)[:top_k]


def _candidate_union_candidates(
    pages: list[_SourcePage],
    *,
    query_text: str,
    page_locator: str,
    broad_passages: list[SourcePassageEvidence],
    top_k: int,
) -> tuple[list[tuple[_PassageCandidate, list[str]]], bool, list[str]]:
    """Union bounded channels and preserve distinct material-facet evidence."""
    entries: dict[
        tuple[int | None, int, int], tuple[_PassageCandidate, set[str]]
    ] = {}

    def add(candidate: _PassageCandidate, channel: str) -> None:
        key = (candidate.page_index, candidate.start, candidate.end)
        current = entries.get(key)
        if current is None:
            entries[key] = (candidate, {channel})
            return
        existing, channels = current
        channels.add(channel)
        if candidate.score > existing.score:
            entries[key] = (candidate, channels)

    for candidate in _exact_phrase_candidates(pages, query_text):
        add(candidate, "candidate_exact_phrase")
    for candidate in _equivalent_phrase_candidates(pages, query_text):
        add(candidate, "candidate_equivalent_phrase")
    normal = _retrieve_candidates(
        pages,
        claim_text=query_text,
        claim_type="paraphrase",
        page_locator=page_locator,
        top_k=max(top_k * 2, top_k),
    )
    for candidate in normal:
        add(candidate, "candidate_lexical")

    facet_queries = _candidate_retrieval_facets(query_text)
    for index, facet_query in enumerate(facet_queries, start=1):
        for candidate in _retrieve_candidates(
            pages,
            claim_text=facet_query,
            claim_type="paraphrase",
            page_locator=page_locator,
            top_k=max(top_k * 2, top_k),
        ):
            add(candidate, f"candidate_facet_{index}_lexical")
        for candidate in _bounded_concept_rescue_candidates(
            pages,
            query_text=facet_query,
            page_locator=page_locator,
            top_k=max(top_k * 3, top_k),
        ):
            add(candidate, f"candidate_facet_{index}_concept")

    query_tokens = _meaningful_tokens(query_text)
    for passage in broad_passages:
        score = _lexical_score(query_tokens, query_text, passage.text)
        if score <= 0:
            continue
        add(
            _PassageCandidate(
                page_index=passage.page_index,
                page_label=passage.page_label,
                start=passage.character_start,
                end=passage.character_end,
                text=passage.text,
                method="whole_citation_context",
                score=min(0.94, score),
            ),
            "whole_citation_context",
        )

    best_normal_score = max((candidate.score for candidate in normal), default=0.0)
    distinct_before_rescue = _consolidate_nested_passage_entries(
        list(entries.values())
    )
    rescue_applied = len(distinct_before_rescue) < top_k or best_normal_score < 0.45
    if rescue_applied:
        for candidate in _bounded_concept_rescue_candidates(
            pages,
            query_text=query_text,
            page_locator=page_locator,
            top_k=max(top_k * 3, top_k),
        ):
            add(candidate, "candidate_concept_rescue")

    consolidated = _consolidate_nested_passage_entries(list(entries.values()))
    ranked = _select_diverse_candidate_entries(
        consolidated,
        query_text=query_text,
        facet_queries=facet_queries,
        top_k=top_k,
    )
    return [
        (candidate, sorted(channels)) for candidate, channels in ranked
    ], rescue_applied, facet_queries


def _candidate_entry_rank_key(item: tuple[_PassageCandidate, set[str]]):
    candidate, channels = item
    return (
        "candidate_exact_phrase" in channels,
        "candidate_equivalent_phrase" in channels,
        candidate.score,
        "candidate_lexical" in channels,
        -candidate.start,
    )


def _select_diverse_candidate_entries(
    entries: list[tuple[_PassageCandidate, set[str]]],
    *,
    query_text: str,
    facet_queries: list[str],
    top_k: int,
) -> list[tuple[_PassageCandidate, set[str]]]:
    """Reserve evidence slots for distinct material retrieval facets."""
    ordered = sorted(entries, key=_candidate_entry_rank_key, reverse=True)
    if not facet_queries:
        return ordered[:top_k]

    selected: list[tuple[_PassageCandidate, set[str]]] = []
    selected_keys: set[tuple[int | None, int, int]] = set()

    def add(item):
        candidate = item[0]
        key = (candidate.page_index, candidate.start, candidate.end)
        if key in selected_keys or len(selected) >= top_k:
            return
        selected_keys.add(key)
        selected.append(item)

    exact = next(
        (
            item
            for item in ordered
            if "candidate_exact_phrase" in item[1]
            or "candidate_equivalent_phrase" in item[1]
        ),
        None,
    )
    if exact is not None:
        add(exact)

    scope_concepts = {
        _retrieval_stem(match.group(0).casefold())
        for match in re.finditer(r"(?<![.!?]\s)\b[A-Z][A-Za-z'’\-]+\b", query_text)
        if match.group(0).casefold() not in {"however", "therefore", "thus", "moreover"}
    }
    all_facet_concepts = [set(_concept_tokens(query)) for query in facet_queries]
    for index, facet_query in enumerate(facet_queries, start=1):
        prefix = f"candidate_facet_{index}_"
        facet_concepts = all_facet_concepts[index - 1]
        other_facet_concepts = set().union(
            *(concepts for offset, concepts in enumerate(all_facet_concepts) if offset != index - 1)
        )
        eligible = [
            item
            for item in ordered
            if any(channel.startswith(prefix) for channel in item[1])
            and (set(_concept_tokens(item[0].text)) & facet_concepts)
            and (
                item[0].page_index,
                item[0].start,
                item[0].end,
            )
            not in selected_keys
        ]
        if eligible:
            add(
                max(
                    eligible,
                    key=lambda item: (
                        len(set(_concept_tokens(item[0].text)) & facet_concepts)
                        / max(len(facet_concepts), 1),
                        len(set(_concept_tokens(item[0].text)) & scope_concepts),
                        -len(set(_concept_tokens(item[0].text)) & other_facet_concepts),
                        _candidate_entry_rank_key(item),
                    ),
                )
            )

    selected_concepts = set().union(
        *(set(_concept_tokens(item[0].text)) for item in selected)
    ) if selected else set()
    while len(selected) < top_k:
        remaining = [
            item
            for item in ordered
            if (item[0].page_index, item[0].start, item[0].end)
            not in selected_keys
        ]
        if not remaining:
            break
        facet_remaining = [
            item
            for item in remaining
            if any(channel.startswith("candidate_facet_") for channel in item[1])
        ]
        if facet_remaining:
            remaining = facet_remaining
        chosen = max(
            remaining,
            key=lambda item: (
                _candidate_entry_rank_key(item),
                item[0].page_index not in {row[0].page_index for row in selected},
                len(set(_concept_tokens(item[0].text)) - selected_concepts)
                / max(len(set(_concept_tokens(item[0].text))), 1),
            ),
        )
        add(chosen)
        selected_concepts.update(_concept_tokens(chosen[0].text))
    return selected


def _consolidate_nested_passage_entries(
    entries: list[tuple[_PassageCandidate, set[str]]],
) -> list[tuple[_PassageCandidate, set[str]]]:
    """Merge same-page containment before top-k consumes a redundant slot."""
    consolidated: list[tuple[_PassageCandidate, set[str]]] = []
    for candidate, channels in entries:
        merged = False
        for index, (existing, existing_channels) in enumerate(consolidated):
            if candidate.page_index != existing.page_index:
                continue
            candidate_contains = (
                candidate.start <= existing.start and candidate.end >= existing.end
            )
            existing_contains = (
                existing.start <= candidate.start and existing.end >= candidate.end
            )
            if not candidate_contains and not existing_contains:
                continue
            broader = candidate if candidate_contains else existing
            combined_channels = set(existing_channels) | set(channels)
            consolidated[index] = (
                _PassageCandidate(
                    page_index=broader.page_index,
                    page_label=broader.page_label,
                    start=broader.start,
                    end=broader.end,
                    text=broader.text,
                    method=broader.method,
                    score=max(existing.score, candidate.score),
                ),
                combined_channels,
            )
            merged = True
            break
        if not merged:
            consolidated.append((candidate, set(channels)))
    return consolidated


def _exact_phrase_candidates(
    pages: list[_SourcePage], query_text: str
) -> list[_PassageCandidate]:
    """Locate a sufficiently substantive fixed candidate verbatim."""
    if len(_meaningful_tokens(query_text)) < 3:
        return []
    candidates: list[_PassageCandidate] = []
    reference_section_started = False
    for page in pages:
        reference_heading = _REFERENCE_HEADING.search(page.text)
        match = _normalized_find(page.text, query_text)
        if match is None:
            reference_section_started = bool(
                reference_section_started or reference_heading
            )
            continue
        start, end = _context_bounds(page.text, *match)
        text = page.text[start:end].strip()
        role = (
            "reference_list"
            if reference_section_started
            or (reference_heading is not None and start >= reference_heading.start())
            else passage_role_from_text(text)
        )
        reference_section_started = bool(reference_section_started or reference_heading)
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        candidates.append(
            _PassageCandidate(
                page_index=page.index,
                page_label=page.label,
                start=start,
                end=end,
                text=text,
                method="candidate_exact_phrase",
                score=0.98,
            )
        )
    return _deduplicate_candidates(candidates)


_RETRIEVAL_PHRASE_EQUIVALENTS = (
    (
        re.compile(
            r"\b(?:provid(?:e|es|ed|ing)\s+)?information\s+to\s+the\s+public\b",
            re.IGNORECASE,
        ),
        "transparency",
    ),
)


def _equivalent_phrase_candidates(
    pages: list[_SourcePage], query_text: str
) -> list[_PassageCandidate]:
    """High-priority exact anchors for a small reviewed phrase mapping."""
    targets = [
        target
        for pattern, target in _RETRIEVAL_PHRASE_EQUIVALENTS
        if pattern.search(query_text)
    ]
    candidates = []
    for target in targets:
        for page, block_start, _block_end, block_text, role in _source_blocks(pages):
            if role in _EXCLUDED_RETRIEVAL_ROLES:
                continue
            for match in re.finditer(rf"\b{re.escape(target)}\b", block_text, re.IGNORECASE):
                local_start, local_end = _context_bounds(block_text, *match.span())
                start, end = block_start + local_start, block_start + local_end
                text = block_text[local_start:local_end].strip()
                if not text or _is_retrieval_noise_block(text):
                    continue
                candidates.append(
                    _PassageCandidate(
                        page_index=page.index,
                        page_label=page.label,
                        start=start,
                        end=end,
                        text=text,
                        method="candidate_equivalent_phrase",
                        score=0.96,
                    )
                )
    return _deduplicate_candidates(candidates)


def _bounded_concept_rescue_candidates(
    pages: list[_SourcePage],
    *,
    query_text: str,
    page_locator: str,
    top_k: int,
) -> list[_PassageCandidate]:
    """Run a bounded, polarity-tolerant concept search over every source block.

    This deliberately relaxes the normal 15% literal-token threshold. It is a
    recall fallback only: downstream judgment must still establish relevance and
    relationship from the selected passage text.
    """
    query_concepts = set(_concept_tokens(query_text))
    if not query_concepts:
        return []
    blocks: list[tuple[_SourcePage, int, int, str, set[str]]] = []
    document_frequency: Counter[str] = Counter()
    for page, start, end, text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        concepts = set(_concept_tokens(text))
        if not concepts:
            continue
        blocks.append((page, start, end, text, concepts))
        document_frequency.update(concepts)
    if not blocks:
        return []

    total = len(blocks)
    query_weights = {
        token: math.log((total + 1) / (document_frequency.get(token, 0) + 1)) + 1.0
        for token in query_concepts
    }
    total_weight = sum(query_weights.values()) or 1.0
    maximum_query_weight = max(query_weights.values(), default=1.0)
    locator_pages = _page_locator_values(page_locator)
    rescued: list[_PassageCandidate] = []
    for page, start, end, text, concepts in blocks:
        matched = query_concepts & concepts
        if not matched:
            continue
        weighted_coverage = sum(query_weights[token] for token in matched) / total_weight
        rare_anchor = max(query_weights[token] for token in matched) / maximum_query_weight
        # One rare concept can be enough to surface a possible counter-direction
        # passage, while two matches are accepted regardless of query length.
        rare_single = len(matched) == 1 and query_weights[next(iter(matched))] >= 2.0
        if len(matched) < 2 and weighted_coverage < 0.12 and not rare_single:
            continue
        density = len(matched) / max(math.sqrt(len(concepts)), 1.0)
        score = min(
            0.89,
            0.50 * weighted_coverage
            + 0.35 * rare_anchor
            + 0.15 * min(1.0, density)
            + 0.10 * _page_boost(page, locator_pages),
        )
        rescued.append(
            _PassageCandidate(
                page_index=page.index,
                page_label=page.label,
                start=start,
                end=end,
                text=text.strip(),
                method="bounded_concept_rescue",
                score=score,
            )
        )
    rescued.sort(key=lambda candidate: (candidate.score, -candidate.start), reverse=True)
    return _deduplicate_candidates(rescued)[:top_k]


def _passage_evidence(
    source: AuthorizedRepresentation,
    candidate: _PassageCandidate,
) -> SourcePassageEvidence:
    passage_id = _stable_id(
        source.content_sha256,
        str(candidate.page_index),
        str(candidate.start),
        str(candidate.end),
        candidate.text,
    )
    return SourcePassageEvidence(
        passage_id=passage_id,
        representation_id=source.representation_id,
        content_sha256=source.content_sha256,
        authorization_scope_type=source.scope_type,
        authorization_scope_id=source.scope_id,
        verification_run_id=source.verification_run_id,
        page_index=candidate.page_index,
        page_label=candidate.page_label,
        character_start=candidate.start,
        character_end=candidate.end,
        text=candidate.text,
        retrieval_method=candidate.method,
        retrieval_score=round(candidate.score, 6),
        passage_role=passage_role_from_text(candidate.text),
    )


def _coverage_evidence(
    source: AuthorizedRepresentation,
    pages: list[_SourcePage],
    extraction_limitations: list[str],
) -> CoverageEvidence:
    complete = source.completeness_verdict in {"complete", "not_applicable"}
    usable_text = any(page.text.strip() for page in pages)
    if not pages or not usable_text:
        level = CoverageLevel.UNAVAILABLE
        confidence = ConfidenceLevel.NONE
    elif complete and not extraction_limitations:
        level = CoverageLevel.FULL_TEXT
        confidence = (
            ConfidenceLevel.HIGH
            if source.text_quality in {"digital", "born_digital"}
            else ConfidenceLevel.MEDIUM
        )
    else:
        level = CoverageLevel.PARTIAL_TEXT
        confidence = ConfidenceLevel.LOW
    limitations = list(extraction_limitations)
    if not complete:
        limitations.append(
            f"Representation completeness is {source.completeness_verdict!r}."
        )
    if complete and extraction_limitations:
        limitations.append(
            "Verification inspected only a policy-bounded subset of the source text."
        )
    if source.text_quality not in {"digital", "born_digital"}:
        limitations.append(f"Text quality is {source.text_quality!r}.")
    return CoverageEvidence(
        level=level,
        confidence=confidence,
        method="validated_representation_text_extraction",
        representation_kind=source.representation_kind,
        media_type=source.media_type,
        completeness_verdict=source.completeness_verdict,
        text_quality=source.text_quality,
        pages_total=len(pages) if pages else None,
        pages_inspected=[page.index for page in pages if page.index is not None],
        limitations=limitations,
    )


def _identity_confidence(value: float | None) -> ConfidenceLevel:
    if value is None:
        return ConfidenceLevel.MEDIUM
    if value >= 0.9:
        return ConfidenceLevel.HIGH
    if value >= 0.7:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "in",
        "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
        "this", "to", "was", "were", "which", "with",
    }
)


def _meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w]+", unicodedata.normalize("NFKC", value).casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    ]


_CONCEPT_EQUIVALENTS = {
    "higher": "high",
    "highest": "high",
    "lower": "low",
    "lowest": "low",
    "incapable": "capable",
    "inability": "ability",
    "unable": "able",
    "impossible": "possible",
    "disabilities": "disability",
    "disabled": "disability",
    "individuals": "individual",
    "people": "individual",
    "persons": "individual",
    "prosperity": "wellbeing",
    "prosperous": "wellbeing",
    "welfare": "wellbeing",
    "transparent": "transparency",
    "openness": "transparency",
    "disclosure": "transparency",
    "gore": "blood",
    "gory": "blood",
    "sex": "sexual",
    "sexuality": "sexual",
    "sexualised": "sexual",
    "sexualized": "sexual",
    "erotic": "sexual",
    "erotically": "sexual",
    "restored": "reappear",
    "reemerged": "reappear",
    "emerged": "reappear",
    "reappeared": "reappear",
}


def _concept_tokens(value: str) -> list[str]:
    """Return conservative retrieval-only stems without changing claim meaning.

    Polarity pairs intentionally share a retrieval concept so passages stating
    the opposite direction remain eligible for later contradiction judgment.
    The original wording remains the only text supplied as the claim.
    """
    normalized = re.sub(
        r"\bwell[\s\-‐‑–—]+being\b",
        "wellbeing",
        unicodedata.normalize("NFKC", value).casefold(),
    )
    for pattern, replacement in _RETRIEVAL_PHRASE_EQUIVALENTS:
        normalized = pattern.sub(replacement, normalized)
    return [_retrieval_stem(token) for token in _meaningful_tokens(normalized)]


_METADATA_NOISE_MARKERS = (
    re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.IGNORECASE),
    re.compile(r"\b(?:issn|isbn)\s*[:\-]", re.IGNORECASE),
    re.compile(r"\b(?:received|accepted|published)\s*[:\-]?\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:volume|vol\.|issue|no\.)\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:corresponding author|author affiliations?|e-?mail)\b", re.IGNORECASE),
    re.compile(r"\b(?:downloaded from|journal homepage|copyright|all rights reserved)\b", re.IGNORECASE),
)

_REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(?:references|bibliography|works\s+cited)\s*$"
)
_NOTES_HEADING = re.compile(r"(?im)^\s*(?:notes|endnotes)\s*$")
_REFERENCE_ENTRY_LINE = re.compile(
    r"(?m)^\s*(?:\[?\d+\]?\.?\s+)?[A-Z][^\n]{0,140}"
    r"\((?:18|19|20)\d{2}[a-z]?\)"
)
_NUMBERED_NOTE_LINE = re.compile(r"(?m)^\s*\d{1,3}[.)]?\s+[A-Z]")
_PARENTHETICAL_YEAR = re.compile(
    r"\((?:[^()]*)\b(?:18|19|20)\d{2}[a-z]?(?:[^()]*)\)"
)
_LEGAL_CITATION_CUE = re.compile(
    r"\b\d+\s+(?:U\.S\.|F\.?\s*(?:2d|3d|4th)?|S\.\s*Ct\.|Stat\.|"
    r"L\.\s*(?:Ed\.|Rev\.)|WL\b)",
    re.IGNORECASE,
)
_EXCLUDED_RETRIEVAL_ROLES = {
    "reference_list",
    "citation_notes",
    "publication_metadata",
}


def _is_metadata_noise_block(text: str) -> bool:
    """Exclude compact publication boilerplate, never ordinary source prose.

    The filter deliberately requires multiple independent metadata cues, or one
    cue in a very short non-sentence block, so abstracts and scholarly prose
    containing an incidental DOI or year remain eligible for retrieval.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return True
    marker_count = sum(bool(pattern.search(normalized)) for pattern in _METADATA_NOISE_MARKERS)
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", normalized))
    word_count = len(re.findall(r"\b\w+\b", normalized))
    if marker_count >= 2 and word_count <= 180 and sentence_count <= 6:
        return True
    return marker_count >= 1 and word_count <= 35 and sentence_count <= 1


def passage_role_from_text(text: str) -> Literal[
    "body_prose",
    "reference_list",
    "citation_notes",
    "publication_metadata",
    "unknown",
]:
    """Classify only high-confidence retrieval roles from exact source text.

    Dense ordinary scholarly citation is not enough to exclude a block. A
    reference list needs a structural heading or repeated entry-shaped lines;
    citation notes need repeated numbered lines plus bibliographic/legal cues.
    Ambiguous blocks remain eligible as ``unknown``.
    """
    if _is_metadata_noise_block(text):
        return "publication_metadata"
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return "unknown"
    reference_entries = len(_REFERENCE_ENTRY_LINE.findall(text))
    if _REFERENCE_HEADING.search(text):
        return "reference_list"

    numbered_notes = len(_NUMBERED_NOTE_LINE.findall(text))
    citation_cues = len(_PARENTHETICAL_YEAR.findall(text)) + len(
        _LEGAL_CITATION_CUE.findall(text)
    )
    if (
        (_NOTES_HEADING.search(text) and numbered_notes >= 2)
        or (numbered_notes >= 4 and citation_cues >= 2)
    ):
        return "citation_notes"
    if reference_entries >= 3:
        return "reference_list"

    word_count = len(re.findall(r"\b\w+\b", normalized))
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", normalized))
    if word_count >= 8 and sentence_count >= 2:
        return "body_prose"
    return "unknown"


def _is_retrieval_noise_block(text: str) -> bool:
    return passage_role_from_text(text) in _EXCLUDED_RETRIEVAL_ROLES


def _retrieval_stem(token: str) -> str:
    token = _CONCEPT_EQUIVALENTS.get(token, token)
    if token.endswith("ism") and len(token) > 7:
        return token[:-3]
    if token.endswith("ist") and len(token) > 7:
        return token[:-3]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        stem = token[:-3]
        if len(stem) >= 3 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _quotation_target(claim_text: str) -> str:
    quoted = re.findall(r"[\"“”‘’]([^\"“”‘’]{6,})[\"“”‘’]", claim_text)
    return max(quoted, key=len).strip() if quoted else claim_text.strip()


def _normalized_with_positions(value: str) -> tuple[str, list[int]]:
    output: list[str] = []
    positions: list[int] = []
    index = 0
    while index < len(value):
        character = value[index]
        if (
            character in {"-", "‐", "­"}
            and index > 0
            and index + 1 < len(value)
            and value[index - 1].isalpha()
            and value[index + 1].isspace()
        ):
            next_index = index + 1
            while next_index < len(value) and value[next_index].isspace():
                next_index += 1
            if next_index < len(value) and value[next_index].isalpha():
                index = next_index
                continue
        normalized = unicodedata.normalize("NFKC", character).casefold()
        normalized = normalized.translate(
            str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
        )
        if normalized.isspace():
            if output and output[-1] != " ":
                output.append(" ")
                positions.append(index)
        else:
            for emitted in normalized:
                output.append(emitted)
                positions.append(index)
        index += 1
    while output and output[0] == " ":
        output.pop(0)
        positions.pop(0)
    while output and output[-1] == " ":
        output.pop()
        positions.pop()
    return "".join(output), positions


def _normalized_find(source: str, target: str) -> tuple[int, int] | None:
    normalized_source, positions = _normalized_with_positions(source)
    normalized_target, _ = _normalized_with_positions(target)
    if not normalized_target:
        return None
    start = normalized_source.find(normalized_target)
    if start < 0:
        return None
    end = start + len(normalized_target) - 1
    return positions[start], positions[end] + 1


def _context_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    paragraph_start = text.rfind("\n\n", 0, start)
    paragraph_end = text.find("\n\n", end)
    bounded_start = 0 if paragraph_start < 0 else paragraph_start + 2
    bounded_end = len(text) if paragraph_end < 0 else paragraph_end
    if bounded_end - bounded_start <= MAX_PASSAGE_CHARACTERS:
        return bounded_start, bounded_end
    margin = max(0, (MAX_PASSAGE_CHARACTERS - (end - start)) // 2)
    return max(0, start - margin), min(len(text), end + margin)


def _text_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.DOTALL):
        start, end = match.span()
        block = match.group(0)
        if len(block) <= MAX_PASSAGE_CHARACTERS:
            blocks.append((start, end, block))
            continue
        cursor = 0
        while cursor < len(block):
            chunk_end = min(len(block), cursor + MAX_PASSAGE_CHARACTERS)
            if chunk_end < len(block):
                boundary = max(
                    block.rfind(". ", cursor, chunk_end),
                    block.rfind("\n", cursor, chunk_end),
                )
                if boundary > cursor + MAX_PASSAGE_CHARACTERS // 2:
                    chunk_end = boundary + 1
            chunk = block[cursor:chunk_end]
            blocks.append((start + cursor, start + chunk_end, chunk))
            cursor = chunk_end
    return blocks


def _source_blocks(
    pages: list[_SourcePage],
) -> list[tuple[_SourcePage, int, int, str, str]]:
    """Return exact blocks with document-aware structural roles.

    Once an explicit reference heading is encountered, later blocks and pages
    remain reference-list material. This catches reference entries separated by
    blank lines without treating citation-dense body prose as bibliography.
    """
    output = []
    reference_section_started = False
    for page in pages:
        reference_heading = _REFERENCE_HEADING.search(page.text)
        for start, end, block_text in _text_blocks(page.text):
            if (
                not reference_section_started
                and reference_heading is not None
                and start < reference_heading.start() < end
            ):
                body_start = start
                body_end = reference_heading.start()
                while body_end > body_start and page.text[body_end - 1].isspace():
                    body_end -= 1
                if body_end > body_start:
                    body_text = page.text[body_start:body_end]
                    output.append(
                        (
                            page,
                            body_start,
                            body_end,
                            body_text,
                            passage_role_from_text(body_text),
                        )
                    )
                reference_start = reference_heading.start()
                reference_text = page.text[reference_start:end]
                output.append(
                    (page, reference_start, end, reference_text, "reference_list")
                )
                continue
            if reference_section_started or (
                reference_heading is not None and end > reference_heading.start()
            ):
                role = "reference_list"
            else:
                role = passage_role_from_text(block_text)
            output.append((page, start, end, block_text, role))
        if reference_heading is not None:
            reference_section_started = True
    return output


def _lexical_score(claim_tokens: list[str], claim_text: str, passage: str) -> float:
    passage_tokens = _meaningful_tokens(passage)
    if not passage_tokens:
        return 0.0
    claim_set = set(claim_tokens)
    passage_set = set(passage_tokens)
    coverage = len(claim_set & passage_set) / len(claim_set)
    if coverage < 0.15:
        return 0.0
    claim_bigrams = set(zip(claim_tokens, claim_tokens[1:]))
    passage_bigrams = set(zip(passage_tokens, passage_tokens[1:]))
    bigram = (
        len(claim_bigrams & passage_bigrams) / len(claim_bigrams)
        if claim_bigrams
        else 0.0
    )
    length_penalty = min(1.0, math.sqrt(len(claim_tokens) / max(len(passage_tokens), 1)))
    return (0.72 * coverage + 0.20 * bigram + 0.08 * length_penalty)


def _page_locator_values(locator: str) -> set[int]:
    numbers = [int(value) for value in re.findall(r"\d+", locator or "")]
    if not numbers:
        return set()
    if len(numbers) >= 2 and re.search(r"[-–—]", locator):
        start, end = numbers[0], numbers[1]
        if 0 < start <= end and end - start <= 50:
            return set(range(start, end + 1))
    return {numbers[0]}


def _page_boost(page: _SourcePage, locator_pages: set[int]) -> float:
    if not locator_pages:
        return 0.0
    label_numbers = {
        int(value) for value in re.findall(r"\d+", page.label or "")
    }
    if label_numbers & locator_pages:
        return 0.05
    if not page.label and page.index is not None and page.index + 1 in locator_pages:
        return 0.02
    return 0.0


def passage_matches_page_locator(
    passage: SourcePassageEvidence, locator: str
) -> bool | None:
    """Return whether persisted passage coordinates match a supplied locator."""
    locator_pages = _page_locator_values(locator)
    if not locator_pages:
        return None
    label_numbers = {
        int(value) for value in re.findall(r"\d+", passage.page_label or "")
    }
    if label_numbers:
        return bool(label_numbers & locator_pages)
    if passage.page_index is not None:
        return passage.page_index + 1 in locator_pages
    return False


def _deduplicate_candidates(
    candidates: list[_PassageCandidate],
) -> list[_PassageCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    seen: set[tuple[int | None, int, int]] = set()
    result: list[_PassageCandidate] = []
    for candidate in ordered:
        key = (candidate.page_index, candidate.start, candidate.end)
        if key in seen or not candidate.text:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
