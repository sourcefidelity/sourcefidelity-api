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
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from app.services.schemas import (
    BoundedFieldsMixin,
    bound_text_fields,
    declared_max_length,
    note_bounded,
)
from sqlalchemy.orm import Session

from app.models.source_repository import SourceRepresentationRecord
from app.services.sentence_splitter import split_sentences
from app.services.schemas import CitationMarkerMember, InTextCitation
from app.services.source_repository import representation_is_expired
from app.services.storage.backend import StorageBackend


ARTIFACT_VERSION = "phase3.8-evidence-v40"
EXTRACTION_VERSION = "source-text-extraction-v6-narrow-columns"
RETRIEVAL_RULE_VERSION = "all-channel-section-boundary-v14"
CANDIDATE_RETRIEVAL_VERSION = "candidate-specific-union-v19"
SEMANTIC_RETRIEVAL_RESCUE_VERSION = "bm25-prefilter-local-nli-v1"
MAX_SOURCE_PAGES = 2_000
MAX_PAGE_CHARACTERS = 250_000
MAX_SOURCE_CHARACTERS = 10_000_000
MAX_PASSAGE_CHARACTERS = 1_800
PASSAGE_WINDOW_STRIDE_CHARACTERS = MAX_PASSAGE_CHARACTERS // 3
SUBSTANTIAL_PASSAGE_OVERLAP_RATIO = 0.50
MAX_CANDIDATES = 10
MAX_CANDIDATE_PASSAGES = 10
_NONSEMANTIC_C0_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class EvidenceAuthorizationError(ValueError):
    """A stored representation is not authorized and usable for this request."""


class SourceTextUnusable(EvidenceAuthorizationError):
    """The source's damaged pages could not be repaired; its text is not used."""


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


class StudentInterpretationRepairOperationEvidence(BaseModel):
    """One exact student-text span changed by a source-blind repair."""

    kind: str = Field(min_length=1, max_length=80)
    problem_segments: list[ClaimSourceSegment] = Field(min_length=1, max_length=4)


class StudentStatementInterpretationEvidence(BaseModel):
    """Inspectable source-blind interpretation; never a source judgment."""

    contract_version: str = "source-blind-student-statement-interpretation-v1"
    interpretation_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "as_written",
        "mechanically_normalized",
        "semantic_repair",
        "not_assessed",
        "uncertain",
    ]
    interpreted_statement: str | None = Field(default=None, max_length=2_000)
    reason_code: str = Field(min_length=1, max_length=100)
    confidence: Literal["high", "medium", "low", "none"]
    repair_operations: list[StudentInterpretationRepairOperationEvidence] = Field(
        default_factory=list, max_length=8
    )
    accuracy_judgment_allowed: bool
    coverage_judgment_allowed: bool
    source_evidence_received: Literal[False] = False
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_code: Literal[
        "none",
        "candidate_not_eligible",
        "context_unresolved",
        "prompt_budget_exceeded",
        "provider_or_contract_failure",
    ] = "none"
    limitations: list[str] = Field(default_factory=list, max_length=6)
    decision_applied: Literal[False] = False


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
        "derived_verb_phrase",
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


class ClaimEvidence(BoundedFieldsMixin):
    @model_validator(mode="before")
    @classmethod
    def _bound_text(cls, data):
        """Student-supplied detail must not crash the run, and must say so.

        A locator is whatever the paper wrote between the citation's brackets,
        so a mis-parsed citation can put a sentence there. Owner decision,
        2026-09-23: an unexpectedly long field must not crash a paper run. The
        altered field names are recorded in `bounded_fields`.
        """
        return bound_text_fields(cls, data, ("page_locator", "citation_marker"))

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


CITED_AUTHOR_LABEL_LIMIT = 200


def bounded_author_label(value: str) -> str:
    """Keep an identifying prefix of an over-long author list.

    The label exists to match a cited surname against an actor named in the
    source, and the leading authors carry that. The cut falls on a comma, so a
    surname is never halved into a fragment that was never cited; the trailing
    entry may lose its initials, which surname matching does not use.
    """
    if len(value) <= CITED_AUTHOR_LABEL_LIMIT:
        return value
    clipped = value[:CITED_AUTHOR_LABEL_LIMIT]
    boundary = clipped.rfind(",")
    bounded = (clipped[:boundary] if boundary > 0 else clipped).rstrip(" ,;&.")
    # A single overlong first token leaves nothing to cut back to; the hard
    # clip is still better than failing the paper.
    return bounded or clipped


class CitationSourceBinding(BoundedFieldsMixin):
    """Exact source-specific binding for one fanned-out verification run."""

    status: Literal["exact", "unresolved"] = "exact"
    reference_id: str = Field(min_length=1, max_length=255)
    cited_author_label: str = Field(min_length=1, max_length=CITED_AUTHOR_LABEL_LIMIT)

    @model_validator(mode="before")
    @classmethod
    def _degrade_unrepresentable_marker(cls, data):
        """A marker too long to store becomes unresolved, never a failed run.

        `marker_text` is the in-text citation exactly as the student wrote it,
        paired with the character offsets where it sits. Truncating it would
        leave the offsets describing a span longer than the text they claim, so
        a report highlight would land on the wrong words. Dropping to
        `unresolved` instead says plainly that this citation could not be
        pinned to an exact span, which is what the evidence actually supports.

        Owner decision, 2026-09-23: an unexpectedly long field must not crash a
        paper run; losing one citation's exact span is acceptable where losing
        the whole submission is not.
        """
        if not isinstance(data, dict):
            return data
        out = None
        text = data.get("marker_text")
        limit = declared_max_length(cls, "marker_text")
        if isinstance(text, str) and limit and len(text) > limit:
            out = dict(data)
            out.update(
                status="unresolved", marker_text="",
                marker_local_start=-1, marker_local_end=-1,
            )
            note_bounded(out, "marker_text")
        # A reference with dozens of authors overruns the label: Wu et al.
        # (2016) supplies 223 characters, and the resulting ValidationError
        # once failed a 65-reference paper at persist time with no report.
        # Bounding on the model means no construction path can reintroduce it.
        label = data.get("cited_author_label")
        if isinstance(label, str) and len(label) > CITED_AUTHOR_LABEL_LIMIT:
            out = out or dict(data)
            out["cited_author_label"] = bounded_author_label(label)
            note_bounded(out, "cited_author_label")
        return out if out is not None else data
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
    parent_content_sha256: str | None = None
    derivation_method: str | None = None
    derivation_manifest_sha256: str | None = None
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
    extraction_version: str = "legacy-unrecorded"
    extracted_text_sha256: str = ""
    pages_total: int | None = None
    pages_inspected: list[int] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AcademicPracticeCheckEvidence(BaseModel):
    """Bounded deterministic check that never implies intent or misconduct."""

    status: Literal["not_run", "complete", "incomplete", "not_assessable"] = "not_run"
    outcome: str | None = Field(default=None, max_length=100)
    evidence_passage_ids: list[str] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list, max_length=8)


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
    consolidated_from_spans: list[tuple[int, int]] = Field(default_factory=list)
    passage_role: Literal[
        "body_prose",
        "abstract",
        "document_metadata",
        "reference_list",
        "citation_notes",
        "publication_metadata",
        "unknown",
    ] = "unknown"
    boundary_status: Literal[
        "sentence_complete",
        "bounded_fragment_or_nonprose",
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


class PassageDisplayObservation(BaseModel):
    """Bounded selection clues, not facets or source-support judgments."""

    model_config = {"extra": "forbid"}
    basis: Literal["direct_attribution", "general_framework", "illustrative_example", "necessary_context", "unclear"]
    claim_spans: list[str] = Field(default_factory=list, max_length=4)
    source_span: str = Field(min_length=1, max_length=1400)


class CandidatePassageRelevanceEvidence(BaseModel):
    """One bounded relevance assessment over an application-owned passage."""

    passage_id: str = Field(min_length=1, max_length=128)
    relevance: Literal[
        "relevant", "partially_relevant", "not_relevant", "uncertain"
    ]
    confidence: ConfidenceLevel
    evidence_role: Literal[
        "source_own_claim_or_finding",
        "source_synthesis_or_conclusion",
        "document_level_member_evidence",
        "representation_of_other_work",
        "methods_or_background",
        "unclear",
    ] = "unclear"
    assessed_text_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    model_input_text_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    assessed_text_offset_start: int = Field(default=0, ge=0)
    assessed_text_offset_end: int | None = Field(default=None, gt=0)
    assessment_input_truncated: bool = False
    rationale: str = Field(default="", max_length=1_000)
    display_observation: PassageDisplayObservation | None = None


class ObligationPassageRelevanceEvidence(BaseModel):
    """One relevance result for exactly one source-blind evidence obligation."""

    obligation_id: str = Field(min_length=1, max_length=128)
    obligation_type: Literal[
        "exact_factual_assertion",
        "aggregate_member_evidence",
        "coverage_only_semantic_repair",
    ]
    status: Literal["complete", "not_assessed"]
    method: str = Field(min_length=1, max_length=100)
    model_id: str | None = None
    gate_version: str
    outcome: Literal[
        "relevant_candidates_found",
        "no_relevant_candidate_passage",
        "uncertain",
        "not_assessed",
    ]
    assessments: list[CandidatePassageRelevanceEvidence] = Field(
        default_factory=list, max_length=18
    )
    relevant_passage_ids: list[str] = Field(default_factory=list, max_length=18)
    candidate_count_assessed: int = Field(default=0, ge=0, le=18)
    batch_count: int = Field(default=0, ge=0, le=6)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


class SourceScopeAssessmentEvidence(BaseModel):
    """Scope comparison for a retrieved document, bound to the text assessed.

    Until this existed the comparison ran only when the source could NOT be
    obtained, so retrieving a document removed the check that a work about
    somewhere or someone else is not evidence for the citation. The excerpt is
    carried with its hash because the displayed passage answers a different
    question -- what was cited, not what this work is about -- and a reader
    must be able to see which text the judgment was made against.
    """

    status: Literal["not_run", "complete", "not_assessed"] = "not_run"
    coverage: str = ""
    excerpt: str = ""
    excerpt_sha256: str = ""
    assessment: dict = Field(default_factory=dict)
    # How much of the citing sentence's vocabulary the COMPLETE document
    # contains, counted locally over text the excerpt does not include. A work
    # that discusses what was attributed to it is not a different subject,
    # whatever its opening happens to mention.
    claim_terms_present: int = 0
    claim_terms_total: int = 0


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
        default_factory=list, max_length=18
    )
    relevant_passage_ids: list[str] = Field(default_factory=list, max_length=18)
    candidate_count_assessed: int = Field(default=0, ge=0, le=18)
    # Up to 18 candidates are assessed in provider-sized batches. The current
    # conservative three-passage boundary therefore requires as many as six.
    batch_count: int = Field(default=0, ge=0, le=6)
    limitations: list[str] = Field(default_factory=list)
    decision_applied: bool = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)
    obligation_findings: list[ObligationPassageRelevanceEvidence] = Field(
        default_factory=list, max_length=4
    )


class JointEvidenceCandidate(BaseModel):
    model_config = {"extra": "forbid"}
    passage_id: str = Field(min_length=1, max_length=128)
    source_span: str = Field(min_length=1)
    source_span_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class JointEvidenceSelectedPassage(BaseModel):
    model_config = {"extra": "forbid"}
    passage_id: str = Field(min_length=1, max_length=128)
    reason: Literal["primary", "distinct_aspect", "necessary_context"]
    claim_spans: list[str] = Field(default_factory=list, max_length=4)
    source_sentence_ids: list[str] = Field(default_factory=list)
    source_span: str = Field(default="", max_length=1400)
    source_span_sha256: str = ""


class JointEvidenceSelection(BaseModel):
    """Optional joint display advice; never a support assessment."""

    model_config = {"extra": "forbid"}
    status: Literal["not_run", "complete", "not_assessed"] = "not_run"
    version: str = "joint-evidence-selection-v4"
    model_id: str | None = None
    source_title: str = ""
    input_fingerprint: str = ""
    model_input_sha256: str = ""
    target_text_sha256: str = ""
    candidates: list[JointEvidenceCandidate] = Field(default_factory=list)
    selected: list[JointEvidenceSelectedPassage] = Field(default_factory=list, max_length=3)
    limitation_codes: list[str] = Field(default_factory=list)
    call_count: Literal[0, 1] = 0
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)
    support_assessed: Literal[False] = False


class EvidenceObligation(BaseModel):
    """One exact relevance question for one citation/source member."""

    obligation_id: str = Field(min_length=1, max_length=128)
    obligation_type: Literal[
        "exact_factual_assertion",
        "aggregate_member_evidence",
        "coverage_only_semantic_repair",
    ]
    reference_id: str = Field(min_length=1, max_length=255)
    target_text: str = Field(min_length=1, max_length=50_000)
    target_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_method: Literal[
        "exact_source_attributed_text",
        "source_blind_semantic_repair",
    ]
    interpretation_id: str | None = Field(default=None, max_length=128)
    aggregate_scope: Literal["not_aggregate", "member_only"]
    accuracy_judgment_allowed: bool
    coverage_judgment_allowed: bool
    limitations: list[str] = Field(default_factory=list, max_length=6)


class EvidenceObligationSet(BaseModel):
    """Typed relevance targets kept separate from retrieved source text."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = (
        "not_run"
    )
    method: str = "not_run"
    obligation_version: str | None = None
    obligations: list[EvidenceObligation] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=8)


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


class StudentClaimClarityFinding(BaseModel):
    """Shadow assessment of whether one exact candidate defines a safe question."""

    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["clear", "not_assessed", "uncertain"]
    reason_code: Literal[
        "interpretable_relationship",
        "unresolved_local_reference",
        "internally_underspecified_relationship",
        "semantically_uninterpretable_wording",
        "conflicting_internal_scope",
        "clarity_uncertain",
        "clarity_assessment_unavailable",
    ]
    confidence: Literal["high", "medium", "low", "none"]
    problem_segments: list[ClaimSourceSegment] = Field(
        default_factory=list, max_length=4
    )
    explanation: str = Field(min_length=1, max_length=500)


class StudentClaimClarityEvidence(BaseModel):
    """Bounded, exact-text clarity gate kept shadow-only during calibration."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = (
        "not_run"
    )
    method: str = "not_run"
    gate_version: str | None = None
    findings: list[StudentClaimClarityFinding] = Field(
        default_factory=list, max_length=16
    )
    blocked_candidate_ids: list[str] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    decision_applied: Literal[False] = False
    processing_boundary: Literal["local", "configured_remote", "unknown"] = "unknown"
    direct_identifier_redactions: dict[str, int] = Field(default_factory=dict)


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


class ExcludedSourceBlockEvidence(BaseModel):
    """Text-free provenance for a source span excluded from ordinary ranking."""

    block_id: str = Field(min_length=64, max_length=64)
    page_index: int | None = None
    page_label: str | None = None
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    text_sha256: str = Field(min_length=64, max_length=64)
    role: Literal[
        "reference_list",
        "citation_notes",
        "publication_metadata",
        "page_furniture",
        "author_biography",
    ]


class CandidatePassageRetrievalEvidence(BaseModel):
    """Candidate-specific retrieval that supplements whole-citation retrieval."""

    status: Literal["not_run", "complete", "incomplete", "not_assessed"] = "not_run"
    method: str = "not_run"
    retrieval_version: str | None = None
    evidence_only_passage_ids: list[str] = Field(default_factory=list, max_length=10)
    evidence_only_query_sha256: str | None = None
    selections: list[CandidatePassageSelection] = Field(
        default_factory=list, max_length=16
    )
    excluded_block_counts: dict[
        Literal[
            "reference_list",
            "citation_notes",
            "publication_metadata",
            "page_furniture",
            "author_biography",
        ],
        int,
    ] = Field(default_factory=dict)
    excluded_blocks: list[ExcludedSourceBlockEvidence] = Field(
        default_factory=list, max_length=10_000
    )
    semantic_rescue_status: Literal[
        "not_run", "complete", "incomplete", "not_assessed"
    ] = "not_run"
    semantic_rescue_version: str | None = Field(default=None, max_length=100)
    semantic_model_id: str | None = Field(default=None, max_length=300)
    semantic_model_revision: str | None = Field(default=None, max_length=100)
    semantic_prefilter_count: int = Field(default=0, ge=0, le=512)
    semantic_addition_count: int = Field(default=0, ge=0, le=64)
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
    # Up to 128 per candidate plus document-scope sentences (foundation v11).
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=136)
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
        default_factory=list, max_length=1024
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


class SentenceEvidenceItem(BaseModel):
    """One source sentence the evidence selector chose, bound to its exact span."""

    passage_id: str = Field(min_length=1, max_length=128)
    passage_start: int = Field(ge=0)
    passage_end: int = Field(ge=0)
    page_index: int | None = None
    text: str = Field(min_length=1, max_length=4_000)
    reason: Literal["bears_on_statement", "qualifies_or_contradicts", "necessary_context"]


class SentenceEvidenceSelection(BaseModel):
    """The citation's displayed evidence: numbered source sentences chosen by GLM.

    Owner decision 2026-09-28 (one evidence source per citation). `selected`
    and `empty` are displayed; any other status falls back to the relevance
    gate's passage display. Presentation only: the Evidence Package and its
    passages are unchanged.
    """

    status: Literal["not_run", "selected", "empty", "unavailable", "over_budget", "invalid"] = "not_run"
    version: str | None = None
    model: str | None = None
    endpoint_host: str | None = None
    request_fingerprint: str | None = None
    items: list[SentenceEvidenceItem] = Field(default_factory=list, max_length=8)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    limitations: list[str] = Field(default_factory=list, max_length=5)


class VerificationEvidenceArtifact(BaseModel):
    # Judgment layout sentence reserve (judgment_reserve.py). Private: never
    # serialized, never part of the Evidence Package or its digest; persisted
    # beside the verification report only when Judgment is enabled.
    _judgment_reserve: dict | None = PrivateAttr(default=None)

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
    quotation_check: AcademicPracticeCheckEvidence = Field(
        default_factory=AcademicPracticeCheckEvidence
    )
    locator_check: AcademicPracticeCheckEvidence = Field(
        default_factory=AcademicPracticeCheckEvidence
    )
    passages: list[SourcePassageEvidence] = Field(default_factory=list)
    relationship: ClaimRelationshipEvidence
    passage_relevance: PassageRelevanceGateEvidence = Field(
        default_factory=PassageRelevanceGateEvidence
    )
    source_scope_assessment: SourceScopeAssessmentEvidence = Field(
        default_factory=SourceScopeAssessmentEvidence
    )
    joint_evidence_selection: JointEvidenceSelection = Field(
        default_factory=JointEvidenceSelection
    )
    sentence_evidence: SentenceEvidenceSelection = Field(
        default_factory=SentenceEvidenceSelection
    )
    evidence_obligations: EvidenceObligationSet = Field(
        default_factory=EvidenceObligationSet
    )
    student_statement_interpretations: list[
        StudentStatementInterpretationEvidence
    ] = Field(default_factory=list, max_length=4)
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
    student_claim_clarity: StudentClaimClarityEvidence = Field(
        default_factory=StudentClaimClarityEvidence
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
    parent_content_sha256: str | None = None
    derivation_method: str | None = None
    derivation_manifest_sha256: str | None = None
    page_labels: tuple[str | None, ...] | None = None
    # Pages whose damaged text was re-read by local OCR (page_ocr_repair receipt).
    page_repairs: tuple[tuple[int, str], ...] = ()
    page_repair_manifest_sha256: str | None = None


@dataclass(frozen=True)
class _SourceStructuralSpan:
    start: int
    end: int
    role: Literal[
        "abstract",
        "document_metadata",
        "citation_notes",
        "publication_metadata",
        "page_furniture",
        "author_biography",
    ]


@dataclass(frozen=True)
class _SourcePage:
    index: int | None
    label: str | None
    text: str
    structural_spans: tuple[_SourceStructuralSpan, ...] = ()


@dataclass(frozen=True)
class _QuotationLocation:
    fragments: tuple[tuple[int | None, int, int], ...]
    method: str


@dataclass(frozen=True)
class _PdfLayoutSpan:
    page_index: int
    start: int
    end: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float


@dataclass(frozen=True)
class _PassageCandidate:
    page_index: int | None
    page_label: str | None
    start: int
    end: int
    text: str
    method: str
    score: float
    passage_role: Literal[
        "body_prose",
        "abstract",
        "document_metadata",
        "reference_list",
        "citation_notes",
        "publication_metadata",
        "unknown",
    ] = "unknown"
    consolidated_from_spans: tuple[tuple[int, int], ...] = ()


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

    validation_evidence = record.validation_evidence or {}
    derivative_evidence = validation_evidence.get("ocr_derivative") or {}
    if derivative_evidence:
        try:
            parent_id = uuid.UUID(derivative_evidence["parent_representation_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceAuthorizationError(
                "OCR derivative parent representation binding is invalid"
            ) from exc
        parent_record = session.get(SourceRepresentationRecord, parent_id)
        expected_parent_hash = derivative_evidence.get("parent_content_sha256")
        if (
            parent_record is None
            or parent_record.canonical_work_id != record.canonical_work_id
            or parent_record.scope_type != record.scope_type
            or parent_record.scope_id != record.scope_id
            or representation_is_expired(parent_record, now=now)
            or parent_record.content_object.deletion_pending
            or parent_record.content_object.content_sha256 != expected_parent_hash
            or not backend.exists(parent_record.content_object.storage_key)
        ):
            raise EvidenceAuthorizationError(
                "OCR derivative immutable parent is unavailable or inconsistent"
            )
    page_labels = derivative_evidence.get("page_labels")
    repairs = None
    from app.config import settings as _settings
    if record.representation_kind == "pdf" and _settings.SOURCE_TEXT_QUALITY_CHECK_ENABLED:
        # Owner decision 2026-09-28: damaged pages are re-read by OCR; a source
        # whose damaged pages cannot be repaired is not used at all.
        from app.services.page_ocr_repair import ensure_receipt, validated_repairs
        from app.services.text_quality import WordListUnavailable
        try:
            receipt = ensure_receipt(session, record.id, content, digest)
        except WordListUnavailable as exc:
            raise EvidenceAuthorizationError("Source text quality cannot be checked") from exc
        if receipt.get("status") == "unusable":
            raise SourceTextUnusable(
                "Representation text is damaged and could not be repaired")
        repairs = validated_repairs(receipt, digest)
        if receipt.get("status") == "repaired" and repairs is None:
            raise EvidenceAuthorizationError("Page repair receipt does not validate")
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
        parent_content_sha256=derivative_evidence.get("parent_content_sha256"),
        derivation_method=derivative_evidence.get("derivation_method"),
        derivation_manifest_sha256=derivative_evidence.get(
            "derivation_manifest_sha256"
        ),
        page_labels=(
            tuple(label if isinstance(label, str) else None for label in page_labels)
            if isinstance(page_labels, list)
            else None
        ),
        page_repairs=tuple(sorted(repairs.texts.items())) if repairs else (),
        page_repair_manifest_sha256=repairs.manifest_sha256 if repairs else None,
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
    if source.identity_verdict == 'possible_match' and not source.verification_run_id:
        raise EvidenceAuthorizationError('Possible matches require a submission-scoped verification run')
    _validate_derivative_provenance(source)
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
        allow_ocr_token_sequence=bool(
            source.text_quality == "scan_ocr" and source.derivation_method
        ),
    )
    passages = [
        _passage_evidence(source, candidate) for candidate in candidates
    ]

    identity = SourceIdentityEvidence(
        status=IdentityStatus.UNCERTAIN if source.identity_verdict == 'possible_match' else IdentityStatus.VERIFIED,
        confidence=ConfidenceLevel.MEDIUM if source.identity_verdict == 'possible_match' else _identity_confidence(source.identity_confidence),
        method=(
            "submission-possible-match-v1" if source.identity_verdict == 'possible_match' else
            "transient_verification_run_record"
            if source.verification_run_id
            else "durable_admission_record"
        ),
        canonical_work_id=source.canonical_work_id,
        representation_id=source.representation_id,
        content_sha256=source.content_sha256,
        parent_content_sha256=source.parent_content_sha256,
        derivation_method=source.derivation_method,
        derivation_manifest_sha256=source.derivation_manifest_sha256,
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
    quotation_check, locator_check = _academic_practice_checks(
        claim=claim,
        source=source,
        pages=pages,
        passages=passages,
    )
    if source.identity_verdict == 'possible_match':
        identity.limitations.insert(0, 'Possible source match—identity not confirmed')
        quotation_check = AcademicPracticeCheckEvidence(status='not_assessable',
            outcome='source_identity_unconfirmed')
        locator_check = AcademicPracticeCheckEvidence(status='not_assessable',
            outcome='source_identity_unconfirmed')
        verdict = VerificationVerdict.NOT_ASSESSED
        reason_codes.append('source_identity_unconfirmed')
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
        quotation_check=quotation_check,
        locator_check=locator_check,
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
    if len(members) > 1:
        # Repeated attribution to this same reference is not a second source.
        # Keep every marker on the claim; bind this package to one occurrence.
        for member in members:
            if (member.local_end > len(claim.text)
                    or claim.text[member.local_start:member.local_end] != member.text):
                raise ValueError("Active citation marker member is not exact")
        controlling = [m for m in members if m.text == claim.citation_marker]
        members = [min(controlling or members, key=lambda m: (m.local_start, m.local_end))]
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


def _attach_evidence_only_fallback(source, artifact, *, top_k):
    """Retrieve exact citation context without authorizing candidate judgment."""
    if not _source_matches_artifact(source, artifact):
        return artifact
    pages, limitations = _extract_pages(source)
    query = artifact.claim.text
    broad = [p for p in artifact.passages if _passage_matches_source(source, p)]
    ranked, _, _ = _candidate_union_candidates(
        pages, query_text=query, page_locator=artifact.claim.page_locator,
        broad_passages=broad, top_k=max(1, min(top_k, MAX_CANDIDATE_PASSAGES)),
    )
    passages = {p.passage_id: p for p in broad}
    ids = []
    for candidate, _ in ranked:
        if candidate.passage_role != "body_prose":
            continue
        passage = _passage_evidence(source, candidate)
        passages.setdefault(passage.passage_id, passage)
        ids.append(passage.passage_id)
    return artifact.model_copy(update={
        "passages": list(passages.values()),
        "candidate_passage_retrieval": CandidatePassageRetrievalEvidence(
            status="not_assessed", method="exact_citation_evidence_only_fallback_v1",
            retrieval_version=CANDIDATE_RETRIEVAL_VERSION,
            evidence_only_passage_ids=ids,
            evidence_only_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            limitations=[
                "Candidate judgment remains unavailable. Additional passages were retrieved only for manual comparison with the unchanged citation.",
                *limitations,
            ],
        ),
    })


def attach_candidate_passage_retrieval(
    source: AuthorizedRepresentation,
    artifact: VerificationEvidenceArtifact,
    *,
    top_k: int = MAX_CANDIDATE_PASSAGES,
    accepted_facet_queries_by_candidate: dict[str, list[str]] | None = None,
    excluded_passage_ids_by_candidate: dict[str, set[str]] | None = None,
) -> VerificationEvidenceArtifact:
    """Search the complete authorized source separately for each fixed candidate.

    ``excluded_passage_ids_by_candidate`` (Judgment wider-search reserve only)
    skips passages already selected, so the same ranking yields the next ones.

    Whole-citation passages remain in ``artifact.passages`` and continue to feed
    the broad relevance assessment. This pass adds an explicit per-candidate
    selection whose IDs are later used by the one-candidate relationship judge.
    No search result changes a verdict.
    """
    _validate_derivative_provenance(source)
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
        # Failed decomposition is not a prohibition on retrieving evidence for
        # the unchanged single-source citation. Never promote rejected proposals
        # or create candidate selections that could authorize a judgment.
        redundant_only = any(
            "candidate_integrity:materially_redundant_candidate" in c.limitations
            for c in candidate_set.candidates
        )
        binding = artifact.source_binding
        if redundant_only and binding and binding.status == "exact" and len(artifact.claim.reference_ids) == 1:
            return _attach_evidence_only_fallback(source, artifact, top_k=top_k)
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
    excluded_blocks: list[ExcludedSourceBlockEvidence] = []
    for excluded_page, excluded_start, excluded_end, excluded_text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            excluded_block_counts[role] += 1
            text_sha256 = hashlib.sha256(excluded_text.encode("utf-8")).hexdigest()
            excluded_blocks.append(
                ExcludedSourceBlockEvidence(
                    block_id=_stable_id(
                        source.content_sha256,
                        str(excluded_page.index),
                        str(excluded_start),
                        str(excluded_end),
                        role,
                        text_sha256,
                    ),
                    page_index=excluded_page.index,
                    page_label=excluded_page.label,
                    character_start=excluded_start,
                    character_end=excluded_end,
                    text_sha256=text_sha256,
                    role=role,
                )
            )

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
        skip = (excluded_passage_ids_by_candidate or {}).get(candidate.candidate_id, set())
        ranked, rescue_applied, facet_queries = _candidate_union_candidates(
            pages,
            query_text=query_text,
            page_locator=artifact.claim.page_locator,
            broad_passages=list(broad_by_id.values()),
            top_k=bounded_top_k + len(skip),
            include_document_metadata=len(set(artifact.claim.reference_ids)) > 1,
            accepted_facet_queries=(accepted_facet_queries_by_candidate or {}).get(
                candidate.candidate_id, []
            ),
        )
        items: list[CandidatePassageSelectionItem] = []
        rank = 0
        for passage_candidate, channels in ranked:
            passage = _passage_evidence(source, passage_candidate)
            if passage.passage_id in skip or len(items) >= bounded_top_k:
                continue
            rank += 1
            evidence_by_id.setdefault(passage.passage_id, passage)
            items.append(
                CandidatePassageSelectionItem(
                    passage_id=passage.passage_id,
                    rank=rank,
                    score=round(passage_candidate.score, 6),
                    # Consolidation can merge two retrieval channels per facet
                    # into one overlapping source window. The selection schema
                    # is intentionally bounded; complete query provenance is
                    # retained separately in ``query_facet_sha256s``.
                    channels=channels[:12],
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
                "Accepted material facets were searched separately and passage slots were diversified across those complete propositions or exact constraint fragments."
            )
        if any(
            "candidate_document_metadata" in channels
            for _passage_candidate, channels in ranked
        ):
            limitations.append(
                "The source title was retained as document-level member evidence for a multi-source aggregate citation; it does not establish the aggregate claim."
            )
        if any(
            "candidate_explicit_note" in channels
            or "candidate_note_fallback" in channels
            for _passage_candidate, channels in ranked
        ):
            limitations.append(
                "Labelled citation-note evidence was searched separately by explicit locator or after ordinary body-prose retrieval failed."
            )
        if any(
            evidence_by_id[item.passage_id].boundary_status
            == "bounded_fragment_or_nonprose"
            for item in items
        ):
            limitations.append(
                "At least one selected passage has a bounded fragment or non-prose boundary; inspect its source-page coordinates before treating omitted context as absent."
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
        excluded_blocks=excluded_blocks,
        limitations=[
            "Candidate-specific retrieval is recall-oriented and does not establish support, contradiction, or source-wide absence.",
            "Definite page furniture, publication metadata, author biography, reference-list, and citation-note blocks are excluded from ordinary body ranking; ambiguous blocks remain eligible.",
            "Labelled citation notes are retained with exact coordinates and may be searched only by an explicit note locator or after ordinary body-prose retrieval fails.",
            "Accepted material facets may reserve distinct passage slots, but retrieval does not establish that any facet is supported by the source.",
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


def attach_local_semantic_retrieval_rescue(
    source: AuthorizedRepresentation,
    artifact: VerificationEvidenceArtifact,
    *,
    scorer=None,
    prefilter_count: int | None = None,
    max_additions: int | None = None,
) -> VerificationEvidenceArtifact:
    """Add a bounded local semantic channel after a true candidate miss.

    The protected lexical/locator passages remain in ``artifact.passages``.
    Local NLI scores only rank a deterministic BM25-prefiltered subset; they do
    not establish relevance, support, contradiction, or source-wide absence.
    A second relevance-gate pass must assess any newly surfaced passages.
    """
    if artifact.passage_relevance.outcome != "no_relevant_candidate_passage":
        return artifact
    retrieval = artifact.candidate_passage_retrieval
    if retrieval.status not in {"complete", "incomplete"} or not retrieval.selections:
        return artifact
    if not _source_matches_artifact(source, artifact):
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": retrieval.model_copy(
                    update={
                        "semantic_rescue_status": "not_assessed",
                        "semantic_rescue_version": SEMANTIC_RETRIEVAL_RESCUE_VERSION,
                        "limitations": [
                            *retrieval.limitations,
                            "Local semantic rescue did not run because the supplied source did not match the authorized artifact.",
                        ],
                    }
                )
            }
        )

    from app.config import settings
    from app.services.relationship_signal import TransformersNLIScorer

    bounded_prefilter = max(
        1,
        min(
            prefilter_count
            if prefilter_count is not None
            else settings.EVIDENCE_RETRIEVAL_SEMANTIC_PREFILTER,
            32,
        ),
    )
    bounded_additions = max(
        1,
        min(
            max_additions
            if max_additions is not None
            else settings.EVIDENCE_RETRIEVAL_SEMANTIC_MAX_ADDITIONS,
            4,
        ),
    )
    local_scorer = scorer or TransformersNLIScorer(
        model_id=settings.RELATIONSHIP_MODEL_NAME,
        model_revision=settings.RELATIONSHIP_MODEL_REVISION,
        local_files_only=settings.RELATIONSHIP_MODEL_LOCAL_FILES_ONLY,
        device=settings.RELATIONSHIP_MODEL_DEVICE,
        batch_size=settings.EVIDENCE_RETRIEVAL_SEMANTIC_BATCH_SIZE,
    )
    pages, extraction_limitations = _extract_pages(source)
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in artifact.verification_candidates.candidates
    }
    evidence_by_id = {passage.passage_id: passage for passage in artifact.passages}
    updated_selections: list[CandidatePassageSelection] = []
    total_prefiltered = 0
    total_added = 0
    try:
        for selection in retrieval.selections:
            candidate = candidates_by_id.get(selection.candidate_id)
            if candidate is None:
                updated_selections.append(selection)
                continue
            query_text = _candidate_retrieval_text(artifact.claim, candidate)
            prefiltered = _bm25_concept_candidates(
                pages,
                query_text=query_text,
                page_locator=artifact.claim.page_locator,
                top_k=bounded_prefilter,
            )
            total_prefiltered += len(prefiltered)
            scores = local_scorer.score_pairs(
                [item.text for item in prefiltered],
                [query_text] * len(prefiltered),
            )
            if len(scores) != len(prefiltered):
                raise ValueError("Local semantic scorer returned the wrong row count")
            ranked = sorted(
                zip(prefiltered, scores, strict=True),
                key=lambda item: (
                    max(item[1].entailment, item[1].contradiction),
                    item[1].entailment,
                    item[0].score,
                    -item[0].start,
                ),
                reverse=True,
            )[:bounded_additions]

            previous_items = {item.passage_id: item for item in selection.passages}
            semantic_items: list[CandidatePassageSelectionItem] = []
            for passage_candidate, nli_score in ranked:
                semantic_candidate = _PassageCandidate(
                    page_index=passage_candidate.page_index,
                    page_label=passage_candidate.page_label,
                    start=passage_candidate.start,
                    end=passage_candidate.end,
                    text=passage_candidate.text,
                    method="local_nli_bm25_prefilter",
                    score=max(nli_score.entailment, nli_score.contradiction),
                    passage_role=passage_candidate.passage_role,
                )
                passage = _passage_evidence(source, semantic_candidate)
                evidence_by_id.setdefault(passage.passage_id, passage)
                previous = previous_items.get(passage.passage_id)
                channels = list(previous.channels) if previous is not None else []
                channels = [
                    channel
                    for channel in channels
                    if channel != "candidate_local_nli_rescue"
                ][:11]
                channels.append("candidate_local_nli_rescue")
                semantic_items.append(
                    CandidatePassageSelectionItem(
                        passage_id=passage.passage_id,
                        rank=1,
                        score=round(semantic_candidate.score, 6),
                        channels=channels,
                    )
                )
                if previous is None:
                    total_added += 1

            ordered: list[CandidatePassageSelectionItem] = []
            seen: set[str] = set()
            for item in [*semantic_items, *selection.passages]:
                if item.passage_id in seen:
                    continue
                seen.add(item.passage_id)
                ordered.append(item)
                if len(ordered) >= MAX_CANDIDATE_PASSAGES:
                    break
            reranked = [
                item.model_copy(update={"rank": rank})
                for rank, item in enumerate(ordered, start=1)
            ]
            updated_selections.append(
                selection.model_copy(
                    update={
                        "passages": reranked,
                        "rescue_applied": True,
                        "limitations": list(
                            dict.fromkeys(
                                [
                                    *selection.limitations,
                                    "A pinned local NLI model ranked a 32-block-or-smaller BM25 prefilter after the bounded relevance gate found no connected candidate.",
                                ]
                            )
                        ),
                    }
                )
            )
    except Exception:
        return artifact.model_copy(
            update={
                "candidate_passage_retrieval": retrieval.model_copy(
                    update={
                        "semantic_rescue_status": "incomplete",
                        "semantic_rescue_version": SEMANTIC_RETRIEVAL_RESCUE_VERSION,
                        "semantic_model_id": getattr(local_scorer, "model_id", None),
                        "semantic_model_revision": getattr(
                            local_scorer, "model_revision", None
                        ),
                        "semantic_prefilter_count": total_prefiltered,
                        "limitations": list(
                            dict.fromkeys(
                                [
                                    *retrieval.limitations,
                                    "Local semantic retrieval rescue was unavailable or returned invalid output; the protected retrieval union was preserved.",
                                    *extraction_limitations[:4],
                                ]
                            )
                        ),
                    }
                )
            }
        )

    limitations = [
        limitation
        for limitation in retrieval.limitations
        if not limitation.startswith("Dense-semantic retrieval remains")
    ]
    limitations.extend(
        [
            "A pinned local NLI model ranked only a deterministic BM25-prefiltered subset after the bounded relevance gate found no connected candidate.",
            "Semantic rescue is retrieval-only: its scores do not establish relevance, support, contradiction, or source-wide absence.",
            "The protected lexical/locator union remains retained in the artifact even when semantic candidates are promoted into bounded selection slots.",
            *extraction_limitations,
        ]
    )
    updated_retrieval = retrieval.model_copy(
        update={
            "method": f"{retrieval.method}+local_nli_bm25_prefilter",
            "retrieval_version": CANDIDATE_RETRIEVAL_VERSION,
            "selections": updated_selections,
            "semantic_rescue_status": "complete",
            "semantic_rescue_version": SEMANTIC_RETRIEVAL_RESCUE_VERSION,
            "semantic_model_id": getattr(local_scorer, "model_id", None),
            "semantic_model_revision": getattr(local_scorer, "model_revision", None),
            "semantic_prefilter_count": total_prefiltered,
            "semantic_addition_count": total_added,
            "limitations": list(dict.fromkeys(limitations)),
        }
    )
    return artifact.model_copy(
        update={
            "passages": list(evidence_by_id.values()),
            "candidate_passage_retrieval": updated_retrieval,
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
        and source.parent_content_sha256 == identity.parent_content_sha256
        and source.derivation_method == identity.derivation_method
        and source.derivation_manifest_sha256 == identity.derivation_manifest_sha256
    )


def _validate_derivative_provenance(source: AuthorizedRepresentation) -> None:
    values = (
        source.parent_content_sha256,
        source.derivation_method,
        source.derivation_manifest_sha256,
        source.page_labels,
    )
    if not any(value is not None for value in values):
        return
    if any(value is None for value in values):
        raise EvidenceAuthorizationError(
            "Derivative source provenance must be complete and page-bound"
        )
    assert source.parent_content_sha256 is not None
    assert source.derivation_manifest_sha256 is not None
    if not re.fullmatch(r"[0-9a-f]{64}", source.parent_content_sha256):
        raise EvidenceAuthorizationError("Derivative parent hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", source.derivation_manifest_sha256):
        raise EvidenceAuthorizationError("Derivative manifest hash is invalid")
    if source.parent_content_sha256 == source.content_sha256:
        raise EvidenceAuthorizationError(
            "Derivative content must remain distinct from its parent bytes"
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


def _structural_role_for_range(
    page: _SourcePage, start: int, end: int
) -> str | None:
    for span in page.structural_spans:
        if start < span.end and end > span.start:
            return span.role
    return None


def _layout_signature(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _probable_article_header_indices(
    page_index: int,
    spans: list[_PdfLayoutSpan],
) -> set[int]:
    """Identify a high-confidence opening-page title/byline/affiliation group.

    A large multiword heading alone is not enough: a nearby short name-shaped
    byline is required so an ordinary section heading stays body. Publisher
    cover sheets can precede the article, so only the first three physical PDF
    pages are eligible rather than assuming the article begins on page zero.
    """
    if page_index < 0 or page_index > 2:
        return set()
    if any(
        re.search(r"\b(?:award|prize)\s+winners?\b", span.text, re.IGNORECASE)
        and span.y0 <= span.page_height * 0.35
        for span in spans
    ):
        return set()
    non_header_labels = {
        "award",
        "awards",
        "winner",
        "winners",
        "prize",
        "prizes",
    }
    for title_index, title in enumerate(spans):
        title_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+", title.text)
        title_lines = max(len(title.text.splitlines()), 1)
        average_line_height = (title.y1 - title.y0) / title_lines
        if not (
            3 <= len(title_words) <= 60
            and not any(word.casefold() in non_header_labels for word in title_words)
            and average_line_height >= 12.0
            and title.page_height * 0.08 <= title.y0 <= title.page_height * 0.55
            and title.x0
            >= title.page_width * (0.10 if page_index == 0 else 0.05)
        ):
            continue
        for byline_index in range(title_index + 1, min(title_index + 4, len(spans))):
            byline = spans[byline_index]
            byline_name_text = byline.text
            byline_lines = [line.strip() for line in byline.text.splitlines() if line.strip()]
            if len(byline_lines) >= 2 and re.search(
                r"\b(?:department|faculty|school|college|university|institute|"
                r"institution|centre|center|academy)\b",
                " ".join(byline_lines[1:]),
                re.IGNORECASE,
            ):
                byline_name_text = byline_lines[0]
            byline_words = re.findall(
                r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+", byline_name_text
            )
            semantic_byline_words = [
                word
                for word in byline_words
                if not (len(word) == 1 and word.islower())
                and word.casefold() not in {"id", "orcid"}
            ]
            capitalized = sum(
                word[0].isupper() for word in semantic_byline_words if word
            )
            connectors = {"and", "et", "al"}
            name_shaped = (
                2 <= len(semantic_byline_words) <= 16
                and capitalized >= 2
                and all(
                    word[0].isupper() or word.casefold() in connectors
                    for word in semantic_byline_words
                )
                and not any(
                    word.casefold() in non_header_labels
                    for word in semantic_byline_words
                )
                and not re.search(r"\d", byline.text)
            )
            nearby = (
                byline.y0 >= title.y1 - 2.0
                and byline.y0 - title.y1 <= max(24.0, title.page_height * 0.06)
                and abs(byline.x0 - title.x0) <= title.page_width * 0.12
            )
            if name_shaped and nearby:
                if page_index > 0:
                    nearby_opening_spans = spans[
                        max(0, title_index - 3) : min(len(spans), byline_index + 7)
                    ]
                    has_abstract = any(
                        re.match(r"^\s*ABSTRACT\b", item.text, re.IGNORECASE)
                        for item in nearby_opening_spans
                    )
                    has_scholarly_header = any(
                        re.search(
                            r"(?:https?://doi\.org/10\.|\bdoi\s*:\s*10\.|"
                            r"\b(?:journal|volume|vol\.|issue|no\.)\b[^\n]{0,100}"
                            r"\b(?:18|19|20)\d{2}\b)",
                            item.text,
                            re.IGNORECASE,
                        )
                        for item in spans[:title_index]
                    )
                    if not (has_abstract or has_scholarly_header):
                        continue
                header_indices = {title_index, byline_index}
                prior = byline
                for affiliation_index in range(
                    byline_index + 1, min(byline_index + 4, len(spans))
                ):
                    affiliation = spans[affiliation_index]
                    normalized_affiliation = re.sub(
                        r"\s+", " ", affiliation.text
                    ).strip()
                    affiliation_cue = re.search(
                        r"\b(?:department|faculty|school|college|university|"
                        r"institute|institution|centre|center|academy)\b",
                        normalized_affiliation,
                        re.IGNORECASE,
                    )
                    gap = affiliation.y0 - prior.y1
                    affiliation_word_count = len(
                        re.findall(r"\b\w+\b", normalized_affiliation)
                    )
                    affiliation_sentence_count = len(
                        re.findall(r"[.!?](?:\s|$)", normalized_affiliation)
                    )
                    if (
                        not affiliation_cue
                        or affiliation_word_count > 80
                        or affiliation_sentence_count > 2
                        or affiliation.y1 - affiliation.y0
                        > max(50.0, title.page_height * 0.09)
                        or gap > max(24.0, title.page_height * 0.06)
                        or affiliation.y0 > title.page_height * 0.60
                    ):
                        break
                    header_indices.add(affiliation_index)
                    prior = affiliation
                return header_indices
    return set()


def _probable_article_title_indices(
    spans: list[_PdfLayoutSpan], header_indices: set[int]
) -> set[int]:
    """Extend a detected article title over adjacent same-style layout blocks.

    Some publisher PDFs split a multi-line title so its final line is one block
    while the preceding title line shares neither a block nor a reliable text
    boundary with it. Extension is deliberately backward-only and requires
    close vertical, typographic, and horizontal alignment. It stops before
    journal mastheads and other bibliographic furniture.
    """
    if not header_indices:
        return set()
    first = min(header_indices)
    title_indices = {first}
    current = spans[first]
    for prior_index in range(first - 1, max(-1, first - 3), -1):
        prior = spans[prior_index]
        prior_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+", prior.text)
        prior_lines = max(len(prior.text.splitlines()), 1)
        current_lines = max(len(current.text.splitlines()), 1)
        prior_line_height = (prior.y1 - prior.y0) / prior_lines
        current_line_height = (current.y1 - current.y0) / current_lines
        vertical_gap = current.y0 - prior.y1
        center_distance = abs(
            ((prior.x0 + prior.x1) / 2) - ((current.x0 + current.x1) / 2)
        )
        metadata_cue = bool(
            re.search(
                r"(?:https?://|\bdoi\b|\b(?:vol(?:ume)?|issue|issn)\b|"
                r"\b(?:18|19|20)\d{2}\b)",
                prior.text,
                re.IGNORECASE,
            )
        )
        aligned = (
            3 <= len(prior_words) <= 30
            and 0 <= vertical_gap <= max(24.0, current.page_height * 0.04)
            and 0.72 <= prior_line_height / max(current_line_height, 1.0) <= 1.38
            and center_distance <= current.page_width * 0.12
            and prior.y0 >= current.page_height * 0.08
            and not metadata_cue
        )
        if not aligned:
            break
        title_indices.add(prior_index)
        current = prior
    return title_indices


def _pdf_layout_spans(
    page: fitz.Page, page_text: str, page_index: int
) -> list[_PdfLayoutSpan]:
    """Map PyMuPDF layout blocks back to the exact extracted text offsets."""
    spans: list[_PdfLayoutSpan] = []
    cursor = 0
    occupied: list[tuple[int, int]] = []
    # Classification needs visual order for footer and marginal-note rules.
    # The occupied-coordinate selection below prevents a visually sorted
    # repeated header or note number from mapping onto an earlier occurrence.
    for block in page.get_text("blocks", sort=True):
        if int(block[6]) != 0:
            continue
        block_text = str(block[4])
        if not block_text.strip():
            continue
        occurrences: list[int] = []
        search_from = 0
        while True:
            occurrence = page_text.find(block_text, search_from)
            if occurrence < 0:
                break
            occurrences.append(occurrence)
            search_from = occurrence + 1
        non_overlapping = [
            occurrence
            for occurrence in occurrences
            if not any(
                occurrence < occupied_end
                and occurrence + len(block_text) > occupied_start
                for occupied_start, occupied_end in occupied
            )
        ]
        after_cursor = [occurrence for occurrence in non_overlapping if occurrence >= cursor]
        start = after_cursor[0] if after_cursor else (
            non_overlapping[0] if non_overlapping else -1
        )
        if start < 0:
            continue
        end = start + len(block_text)
        spans.append(
            _PdfLayoutSpan(
                page_index=page_index,
                start=start,
                end=end,
                text=block_text,
                x0=float(block[0]),
                y0=float(block[1]),
                x1=float(block[2]),
                y1=float(block[3]),
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
            )
        )
        occupied.append((start, end))
        cursor = max(cursor, end)
    return spans


def _pdf_reading_order_text_and_spans(
    page: fitz.Page, page_index: int
) -> tuple[str, list[_PdfLayoutSpan], bool]:
    """Rebuild clear two-column pages in visual column-major reading order.

    PyMuPDF's content-stream order can place a complete right column before a
    complete left column. Sentence windows over that stream then splice
    unrelated column edges. Only pages with strong, balanced two-column layout
    evidence are reordered; ambiguous layouts retain the ordinary extractor.
    """
    raw_text = page.get_text("text")
    raw_blocks = [
        block
        for block in page.get_text("blocks")
        if int(block[6]) == 0 and str(block[4]).strip()
    ]
    if len(raw_blocks) < 4:
        return raw_text, _pdf_layout_spans(page, raw_text, page_index), False

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    # Newspaper/pressbook pages can have four narrow columns. Treating each
    # half as one column interleaves unrelated articles at equal y positions.
    seeds = [b for b in raw_blocks if len(str(b[4]).strip()) >= 80
             and page_width * .10 <= float(b[2])-float(b[0]) <= page_width * .23]
    clusters = []
    for block in sorted(seeds, key=lambda b: float(b[0])):
        if not clusters or float(block[0])-float(clusters[-1][0][0]) > page_width*.04:
            clusters.append([])
        clusters[-1].append(block)
    if 3 <= len(clusters) <= 5 and all(len(c) >= 3 for c in clusters):
        bounds = [(min(float(b[0]) for b in c), max(float(b[2]) for b in c)) for c in clusters]
        if all(a[1] < b[0] for a,b in zip(bounds,bounds[1:])):
            groups = [[] for _ in bounds]; furniture = []
            for block in raw_blocks:
                owners = [i for i,(left_edge,right_edge) in enumerate(bounds)
                          if left_edge-page_width*.025 <= float(block[0]) <= right_edge]
                if owners and float(block[1]) < page_height*.94:
                    groups[owners[0]].append(block)
                else:
                    furniture.append(block)
            ordered = [b for group in groups for b in sorted(group,key=lambda b:(float(b[1]),float(b[0])))] + furniture
            text = ''; spans = []
            for block in ordered:
                block_text = str(block[4]); start = len(text); text += block_text + '\n'
                spans.append(_PdfLayoutSpan(page_index=page_index,start=start,end=start+len(block_text),text=block_text,
                    x0=float(block[0]),y0=float(block[1]),x1=float(block[2]),y1=float(block[3]),
                    page_width=page_width,page_height=page_height))
            return text, spans, True
    # A short body fragment at the top of each column is not a running header.
    # Require the block to start in the actual upper margin as well as end high.
    headers = [block for block in raw_blocks
               if float(block[1]) <= page_height * 0.065
               and float(block[3]) <= page_height * 0.12]
    footers = [block for block in raw_blocks if float(block[1]) >= page_height * 0.88]
    body = [block for block in raw_blocks if block not in headers and block not in footers]
    left = [
        block
        for block in body
        if float(block[2]) <= page_width * 0.58
        and (float(block[0]) + float(block[2])) / 2 < page_width * 0.48
    ]
    right = [
        block
        for block in body
        if float(block[0]) >= page_width * 0.42
        and (float(block[0]) + float(block[2])) / 2 > page_width * 0.52
    ]
    assigned_ids = {id(block) for block in [*left, *right]}
    unassigned = [block for block in body if id(block) not in assigned_ids]
    left_characters = sum(len(str(block[4]).strip()) for block in left)
    right_characters = sum(len(str(block[4]).strip()) for block in right)
    unassigned_characters = sum(len(str(block[4]).strip()) for block in unassigned)
    assigned_characters = left_characters + right_characters
    clear_two_column = (
        len(left) >= 2
        and len(right) >= 2
        and left_characters >= 200
        and right_characters >= 200
        and unassigned_characters <= max(80, assigned_characters * 0.08)
    )
    if not clear_two_column:
        return raw_text, _pdf_layout_spans(page, raw_text, page_index), False

    ordered = [
        *sorted(headers, key=lambda block: (float(block[1]), float(block[0]))),
        *sorted(left, key=lambda block: (float(block[1]), float(block[0]))),
        *sorted(right, key=lambda block: (float(block[1]), float(block[0]))),
        *sorted(unassigned, key=lambda block: (float(block[1]), float(block[0]))),
        *sorted(footers, key=lambda block: (float(block[1]), float(block[0]))),
    ]
    text_parts: list[str] = []
    spans: list[_PdfLayoutSpan] = []
    cursor = 0
    for block in ordered:
        block_text = str(block[4])
        text_parts.append(block_text)
        end = cursor + len(block_text)
        spans.append(
            _PdfLayoutSpan(
                page_index=page_index,
                start=cursor,
                end=end,
                text=block_text,
                x0=float(block[0]),
                y0=float(block[1]),
                x1=float(block[2]),
                y1=float(block[3]),
                page_width=page_width,
                page_height=page_height,
            )
        )
        cursor = end
    return "".join(text_parts), spans, True


def _normalize_pdf_extracted_text(text: str) -> tuple[str, int]:
    """Replace only nonsemantic C0 debris without shifting coordinates."""
    return _NONSEMANTIC_C0_CONTROLS.subn(" ", text)


def _normalize_pdf_page_label(label: str | None) -> str | None:
    """Decode PyMuPDF's literal UTF-16 page-label prefix when present."""
    if not label:
        return None
    match = re.fullmatch(r"<FEFF([0-9A-Fa-f]+)>(.*)", label)
    if match:
        try:
            prefix = bytes.fromhex(match.group(1)).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            return label
        return f"{prefix}{match.group(2)}"
    return label


def _visible_pdf_page_label(
    text: str,
    structural_spans: tuple[_SourceStructuralSpan, ...],
) -> str | None:
    """Recover one printed page number from high-confidence margin furniture."""
    candidates: set[str] = set()
    bracketed_candidates: set[str] = set()
    for span in structural_spans:
        if span.role != "page_furniture":
            continue
        for line in text[span.start : span.end].splitlines():
            bracketed = re.fullmatch(r'\s*\[\s*(\d{1,4})\s*\]\s*',line)
            if bracketed: bracketed_candidates.add(bracketed[1])
            match = re.fullmatch(r"\s*(?:[-–—]\s*)?(?:\[\s*)?(\d{1,4})(?:\s*\])?(?:\s*[-–—])?\s*", line)
            if match:
                candidates.add(match.group(1))
    if len(bracketed_candidates)==1:return next(iter(bracketed_candidates))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _resolved_pdf_page_labels(
    pages: list[_SourcePage],
    structural_by_page: dict[int, tuple[_SourceStructuralSpan, ...]],
) -> dict[int, str | None]:
    """Prefer corroborated printed pagination over conflicting numeric labels.

    Never infer a missing printed number from an offset. Non-numeric embedded
    labels (e.g. appendix prefixes) retain their explicit document meaning.
    """
    visible = {
        page.index: _visible_pdf_page_label(
            page.text, structural_by_page.get(page.index, ())
        )
        for page in pages if page.index is not None
    }
    resolved = {}
    for page in pages:
        if page.index is None:
            continue
        printed = visible.get(page.index)
        embedded = page.label
        corroborated = bool(printed) and any(
            visible.get(page.index + delta) is not None
            and int(visible[page.index + delta]) == int(printed) + delta
            for delta in (-1, 1)
        )
        resolved[page.index] = (
            printed
            if printed and (
                not embedded or embedded == printed
                or (embedded.isdecimal() and corroborated)
            )
            else embedded
        )
    return resolved


def _byline_first_opening_roles(page_index, spans):
    """Bounded author/affiliation → large title → abstract opening layout.

    Require an independent DOI masthead and geometric alignment. A large
    section heading or a person's name alone must never exclude body text.
    """
    if page_index not in range(3) or not any(
        s.y1 < s.page_height * .15 and
        re.search(r"\bdoi\s*:\s*10\.\d{4,9}/", s.text, re.I)
        for s in spans
    ):
        return {}
    for bi, byline in enumerate(spans):
        lines = [line.strip() for line in byline.text.splitlines() if line.strip()]
        if len(lines) < 2 or len(lines) > 4:
            continue
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+", lines[0])
        if not (2 <= len(words) <= 8 and all(w[0].isupper() for w in words)
                and re.search(r"\b(?:university|institute|college|department)\b", ' '.join(lines[1:]), re.I)
                and byline.y1 - byline.y0 <= 55):
            continue
        for ti, title in enumerate(spans):
            height = (title.y1-title.y0)/max(1, len(title.text.splitlines()))
            if not (3 <= len(title.text.split()) <= 60 and height >= 18
                    and 0 <= title.y0-byline.y1 <= title.page_height*.08
                    and abs(title.x0-byline.x0) <= title.page_width*.04
                    and title.y1 < title.page_height*.65):
                continue
            abstracts = [(i,s) for i,s in enumerate(spans)
                         if re.fullmatch(r"\s*abstract\s*", s.text, re.I)
                         and 0 <= s.y0-title.y1 <= title.page_height*.08
                         and abs(s.x0-title.x0) <= title.page_width*.04]
            if len(abstracts) != 1:
                continue
            ai, abstract = abstracts[0]
            result = {bi: 'publication_metadata', ti: 'document_metadata', ai: 'abstract'}
            # The abstract and optional keyword column are independent blocks;
            # never absorb another column by text-stream adjacency.
            for hi, heading in [(ai, abstract), *[(i,s) for i,s in enumerate(spans)
                    if re.fullmatch(r"\s*keywords\s*", s.text, re.I)
                    and abs(s.y0-abstract.y0) <= 8 and s.x0 > abstract.x0]]:
                role = 'abstract' if hi == ai else 'publication_metadata'
                result[hi] = role
                followers = [(i,s) for i,s in enumerate(spans)
                             if 0 <= s.y0-heading.y1 <= 15
                             and abs(s.x0-heading.x0) <= heading.page_width*.03]
                if len(followers) == 1:
                    result[followers[0][0]] = role
            return result
    return {}


def _pdf_structural_spans(
    layout_by_page: dict[int, list[_PdfLayoutSpan]],
) -> dict[int, tuple[_SourceStructuralSpan, ...]]:
    """Classify only high-confidence layout-derived non-body source spans."""
    signature_pages: dict[str, set[int]] = {}
    for page_index, spans in layout_by_page.items():
        for span in spans:
            near_margin = (
                span.y1 <= span.page_height * 0.14
                or span.y0 >= span.page_height * 0.84
                or span.x1 <= span.page_width * 0.12
                or span.x0 >= span.page_width * 0.88
            )
            signature = _layout_signature(span.text)
            word_count = len(re.findall(r"\b\w+\b", signature))
            if near_margin and signature and word_count <= 35:
                signature_pages.setdefault(signature, set()).add(page_index)

    roles: dict[tuple[int, int], str] = {}
    for page_index, spans in layout_by_page.items():
        opening_roles = _byline_first_opening_roles(page_index, spans)
        article_header_indices = _probable_article_header_indices(page_index, spans)
        article_title_indices = _probable_article_title_indices(
            spans, article_header_indices
        )
        for index, span in enumerate(spans):
            normalized = re.sub(r"\s+", " ", span.text).strip()
            near_margin = (
                span.y1 <= span.page_height * 0.14
                or span.y0 >= span.page_height * 0.84
                or span.x1 <= span.page_width * 0.12
                or span.x0 >= span.page_width * 0.88
            )
            standalone_page_number = bool(re.fullmatch(r"(?:[-–—]\s*)?(?:\[\s*)?\d{1,4}(?:\s*\])?(?:\s*[-–—])?", normalized))
            repeated_margin = (
                near_margin
                and len(signature_pages.get(_layout_signature(span.text), set())) >= 2
            )
            marked_furniture = any(
                pattern.search(normalized)
                for pattern in _STRONG_PAGE_FURNITURE_MARKERS
            ) or (
                any(pattern.search(normalized) for pattern in _PAGE_FURNITURE_MARKERS)
                and (near_margin or span.y0 >= span.page_height * 0.65)
            )
            if index in opening_roles:
                roles[(page_index, index)] = opening_roles[index]
            elif index in article_title_indices:
                roles[(page_index, index)] = "document_metadata"
            elif index in article_header_indices:
                roles[(page_index, index)] = "publication_metadata"
            elif article_header_indices and re.match(
                r"^\s*ABSTRACT\b", span.text, re.IGNORECASE
            ):
                roles[(page_index, index)] = "abstract"
            elif (standalone_page_number and near_margin) or repeated_margin or marked_furniture:
                roles[(page_index, index)] = "page_furniture"
            elif _is_metadata_noise_block(span.text):
                roles[(page_index, index)] = "publication_metadata"

        # A definite footer cue makes every later layout line furniture. This
        # retains odd IP/control lines between repeated download notices.
        footer_indices = [
            index
            for index, span in enumerate(spans)
            if roles.get((page_index, index)) == "page_furniture"
            and span.y0 >= span.page_height * 0.72
        ]
        if footer_indices:
            for index in range(min(footer_indices), len(spans)):
                roles[(page_index, index)] = "page_furniture"

        for index, span in enumerate(spans):
            if _AUTHOR_BIOGRAPHY_START.search(span.text) or re.fullmatch(
                r"\s*Notes? on (?:the )?contributors?\s*", span.text, re.IGNORECASE
            ):
                roles[(page_index, index)] = "author_biography"
                prior = span
                for continuation_index in range(index + 1, len(spans)):
                    continuation = spans[continuation_index]
                    if _REFERENCE_HEADING.search(continuation.text):
                        break
                    gap = continuation.y0 - prior.y1
                    if gap > max(7.0, (prior.y1 - prior.y0) * 0.8):
                        break
                    if roles.get((page_index, continuation_index)) == "page_furniture":
                        break
                    if _NUMBERED_NOTE_LINE.search(continuation.text):
                        break
                    roles[(page_index, continuation_index)] = "author_biography"
                    prior = continuation

        note_starts = []
        for index, span in enumerate(spans):
            numbered = len(_NUMBERED_NOTE_LINE.findall(span.text))
            citation_cues = len(_PARENTHETICAL_YEAR.findall(span.text)) + len(
                re.findall(r"\b(?:18|19|20)\d{2}\b", span.text)
            )
            span_width = max(1.0, span.x1 - span.x0)
            adjacent_main_column = any(
                other_index != index
                and max(0.0, min(span.y1, other.y1) - max(span.y0, other.y0)) > 0
                and other.x1 - other.x0 >= span_width * 1.5
                and (
                    (
                        span.x1 <= span.page_width * 0.35
                        and 0 <= other.x0 - span.x1 <= span.page_width * 0.12
                    )
                    or (
                        span.x0 >= span.page_width * 0.65
                        and 0 <= span.x0 - other.x1 <= span.page_width * 0.12
                    )
                )
                for other_index, other in enumerate(spans)
            )
            narrow_margin_note = (
                numbered >= 1
                and (
                    span.y0 >= span.page_height * 0.10
                    # Some journal side notes begin beside the first body
                    # paragraph. Keep the column geometry gate and require
                    # substantive note text, not a running page/issue number.
                    or (span.y0 >= span.page_height * 0.02
                        and len(span.text.split()) >= 8)
                )
                and adjacent_main_column
            )
            if (
                narrow_margin_note
                or (
                    (_NOTES_HEADING.search(span.text) or numbered >= 2)
                    and citation_cues >= 1
                    # A numbered main-column method/results block in the lower
                    # half of a two-column article is not a footnote section.
                    # Ordinary bottom notes begin substantially closer to the
                    # footer; marginal notes use the separate column rule.
                    and span.y0 >= span.page_height * 0.62
                )
            ):
                note_starts.append(index)
        for start_index in note_starts:
            seed = spans[start_index]
            roles[(page_index, start_index)] = "citation_notes"
            prior = seed
            continuations = sorted(
                (
                    (index, span)
                    for index, span in enumerate(spans)
                    if index != start_index and span.y0 >= seed.y0
                ),
                key=lambda item: (item[1].y0, item[1].x0),
            )
            for index, span in continuations:
                if span.y0 < prior.y1 - 1.0:
                    continue
                horizontal_overlap = max(
                    0.0, min(seed.x1, span.x1) - max(seed.x0, span.x0)
                )
                minimum_width = max(1.0, min(seed.x1 - seed.x0, span.x1 - span.x0))
                if horizontal_overlap / minimum_width < 0.55:
                    continue
                gap = span.y0 - prior.y1
                if gap > max(8.0, (prior.y1 - prior.y0) * 0.85):
                    break
                if roles.get((page_index, index)) == "page_furniture":
                    break
                roles[(page_index, index)] = "citation_notes"
                prior = span

    output: dict[int, tuple[_SourceStructuralSpan, ...]] = {}
    for page_index, spans in layout_by_page.items():
        classified: list[_SourceStructuralSpan] = []
        last_classified_index: int | None = None
        for index, span in enumerate(spans):
            role = roles.get((page_index, index))
            if role is None:
                continue
            item = _SourceStructuralSpan(start=span.start, end=span.end, role=role)
            if (
                classified
                and last_classified_index is not None
                and index == last_classified_index + 1
                and classified[-1].role == item.role
                and item.start == classified[-1].end
            ):
                classified[-1] = _SourceStructuralSpan(
                    start=classified[-1].start,
                    end=item.end,
                    role=item.role,
                )
            else:
                classified.append(item)
            last_classified_index = index
        ordered = sorted(classified, key=lambda item: (item.start, item.end))
        consolidated: list[_SourceStructuralSpan] = []
        for item in ordered:
            if (
                consolidated
                and consolidated[-1].role == item.role
                and consolidated[-1].end == item.start
            ):
                consolidated[-1] = _SourceStructuralSpan(
                    start=consolidated[-1].start,
                    end=item.end,
                    role=item.role,
                )
            else:
                consolidated.append(item)
        output[page_index] = tuple(consolidated)
    return output


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
                layout_by_page: dict[int, list[_PdfLayoutSpan]] = {}
                total_characters = 0
                normalized_control_characters = False
                reordered_two_column_pages = 0
                repaired_text = dict(getattr(source, "page_repairs", ()) or ())
                for index, page in enumerate(document):
                    if index in repaired_text:
                        # Receipt-bound OCR text replaces a damaged text layer; the
                        # layer's layout does not describe it, so none is kept.
                        raw_text, layout_spans, reading_order_rebuilt = repaired_text[index], [], False
                    else:
                        raw_text, layout_spans, reading_order_rebuilt = (
                            _pdf_reading_order_text_and_spans(page, index)
                        )
                    reordered_two_column_pages += int(reading_order_rebuilt)
                    text, substitutions = _normalize_pdf_extracted_text(raw_text)
                    normalized_control_characters = (
                        normalized_control_characters or substitutions > 0
                    )
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
                            label=_normalize_pdf_page_label(page.get_label()),
                            text=text,
                        )
                    )
                    layout_by_page[index] = [
                        span for span in layout_spans if span.end <= len(text)
                    ]
                    total_characters += len(text)
                    if source_limit_reached:
                        break
                structural_by_page = _pdf_structural_spans(layout_by_page)
                resolved_labels = _resolved_pdf_page_labels(pages, structural_by_page)
                if any(
                    page.label and resolved_labels.get(page.index) != page.label
                    for page in pages
                ):
                    limitations.append(
                        "Conflicting numeric PDF labels were replaced by "
                        "printed page numbers corroborated on adjacent pages."
                    )
                if normalized_control_characters:
                    limitations.append(
                        "PDF text contained nonsemantic control characters that "
                        "were normalized to same-length spaces."
                    )
                if reordered_two_column_pages:
                    limitations.append(
                        "PDF visual column order was reconstructed on "
                        f"{reordered_two_column_pages} page(s)."
                    )
                if repaired_text:
                    limitations.append(
                        f"Damaged PDF text on {len(repaired_text)} page(s) was re-read by local OCR "
                        f"(receipt {getattr(source, 'page_repair_manifest_sha256', None)})."
                    )
                return [
                    _SourcePage(
                        index=page.index,
                        label=resolved_labels.get(page.index),
                        text=page.text,
                        structural_spans=structural_by_page.get(
                            page.index if page.index is not None else 0, ()
                        ),
                    )
                    for page in pages
                ], limitations
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
        supplied_labels = source.page_labels
        if supplied_labels is not None and len(supplied_labels) != len(raw_pages):
            return [], ["OCR derivative page labels do not match its page count."]
        pages: list[_SourcePage] = []
        for index, page_text in enumerate(raw_pages[:MAX_SOURCE_PAGES]):
            bounded_text = page_text[:MAX_PAGE_CHARACTERS]
            label = (
                supplied_labels[index]
                if supplied_labels is not None
                else (str(index + 1) if len(raw_pages) > 1 else None)
            )
            structural_spans: tuple[_SourceStructuralSpan, ...] = ()
            if source.derivation_method and label:
                matches = list(
                    re.finditer(
                        rf"(?m)^\s*{re.escape(label)}\s*(?:\n|$)", bounded_text
                    )
                )
                if len(matches) == 1:
                    structural_spans = (
                        _SourceStructuralSpan(
                            start=matches[0].start(),
                            end=matches[0].end(),
                            role="page_furniture",
                        ),
                    )
            pages.append(
                _SourcePage(
                    index=(
                        index
                        if len(raw_pages) > 1 or source.derivation_method
                        else None
                    ),
                    label=label,
                    text=bounded_text,
                    structural_spans=structural_spans,
                )
            )
        return pages, limitations

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
    allow_ocr_token_sequence: bool = False,
) -> list[_PassageCandidate]:
    if not claim_text.strip():
        return []
    locator_pages = _page_locator_values(page_locator)
    exact_targets = _quotation_targets(claim_text) if claim_type == "quotation" else []
    exact: list[_PassageCandidate] = []
    if exact_targets:
        reference_section_started = False
        for page in pages:
            reference_heading = _REFERENCE_HEADING.search(page.text)
            for exact_target in exact_targets:
                quotation_match = _quotation_match(
                    page.text,
                    exact_target,
                    allow_ocr_token_sequence=allow_ocr_token_sequence,
                )
                if quotation_match is None:
                    continue
                match = quotation_match[:2]
                start, end = _context_bounds_on_page(page, *match)
                exact_text = page.text[start:end].strip()
                structural_role = _structural_role_for_range(page, *match)
                role = structural_role or (
                    "reference_list"
                    if reference_section_started
                    or (reference_heading is not None and start >= reference_heading.start())
                    else passage_role_from_text(exact_text)
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
                        method=(
                            "ocr_token_sequence"
                            if quotation_match[2] == "ocr_token_sequence"
                            else "exact_quotation"
                        ),
                        score=min(1.0, 0.95 + _page_boost(page, locator_pages)),
                        passage_role=role,
                    )
                )
            reference_section_started = bool(
                reference_section_started or reference_heading
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
                passage_role=role,
            )
        )
    scored.sort(key=lambda candidate: (candidate.score, -candidate.start), reverse=True)
    consolidated = _consolidate_nested_passage_entries(
        [(candidate, {"whole_citation_lexical"}) for candidate in _deduplicate_candidates(scored)],
        page_text_by_index={page.index: page.text for page in pages},
    )
    return [candidate for candidate, _channels in consolidated[:top_k]]


def _candidate_union_candidates(
    pages: list[_SourcePage],
    *,
    query_text: str,
    page_locator: str,
    broad_passages: list[SourcePassageEvidence],
    top_k: int,
    include_document_metadata: bool = False,
    accepted_facet_queries: list[str] | None = None,
) -> tuple[list[tuple[_PassageCandidate, list[str]]], bool, list[str]]:
    """Union bounded channels and preserve distinct material-facet evidence."""
    entries: dict[
        tuple[int | None, int, int], tuple[_PassageCandidate, set[str]]
    ] = {}

    def add(candidate: _PassageCandidate, channel: str) -> None:
        # A title/byline block is admissible only through its own
        # document-metadata channel, which runs for multi-reference claims.
        # Ordinary lexical and BM25 channels rank it highly because it repeats
        # the cited title, but its bare words cannot reproduce that layout, so
        # its role is not text-derivable and it is not usable candidate
        # evidence. Admitting it elsewhere also overwrote the dedicated
        # channel's retrieval method whenever a lexical score was higher.
        if (
            candidate.passage_role == "document_metadata"
            and channel != "candidate_document_metadata"
        ):
            return
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
    if include_document_metadata:
        for page, start, end, text, role in _source_blocks(pages):
            if role != "document_metadata":
                continue
            if not document_metadata_member_admissible(
                page_index=page.index, text=text.strip()
            ):
                continue
            add(
                _PassageCandidate(
                    page_index=page.index,
                    page_label=page.label,
                    start=start,
                    end=end,
                    text=text.strip(),
                    method="document_level_member_evidence",
                    score=0.95,
                    passage_role="document_metadata",
                ),
                "candidate_document_metadata",
            )
    normal = _retrieve_candidates(
        pages,
        claim_text=query_text,
        claim_type="paraphrase",
        page_locator=page_locator,
        top_k=max(top_k * 2, top_k),
    )
    for candidate in normal:
        add(candidate, "candidate_lexical")
    for candidate in _bm25_concept_candidates(
        pages,
        query_text=query_text,
        page_locator=page_locator,
        top_k=top_k,
    ):
        add(candidate, "candidate_bm25_concept")

    facet_queries = list(
        dict.fromkeys(
            query.strip()
            for query in [
                *_candidate_retrieval_facets(query_text),
                *(accepted_facet_queries or []),
            ]
            if query and query.strip() and query.strip() != query_text.strip()
        )
    )[:4]
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
        for candidate in _bm25_concept_candidates(
            pages,
            query_text=facet_query,
            page_locator=page_locator,
            top_k=top_k,
        ):
            add(candidate, f"candidate_facet_{index}_bm25")

    query_tokens = _meaningful_tokens(query_text)
    page_by_index = {page.index: page for page in pages}
    for passage in broad_passages:
        current_page = page_by_index.get(passage.page_index)
        if current_page is None:
            continue
        if (
            passage.character_start < 0
            or passage.character_end > len(current_page.text)
            or current_page.text[
                passage.character_start : passage.character_end
            ]
            != passage.text
        ):
            continue
        current_structural_role = _structural_role_for_range(
            current_page, passage.character_start, passage.character_end
        )
        current_role = current_structural_role or passage_role_from_text(
            passage.text
        )
        if current_role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
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
                passage_role=current_role,
            ),
            "whole_citation_context",
        )

    best_normal_score = max((candidate.score for candidate in normal), default=0.0)
    distinct_before_rescue = _consolidate_nested_passage_entries(
        list(entries.values()),
        page_text_by_index={page.index: page.text for page in pages},
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

    note_search_required = _has_explicit_note_locator(page_locator) or not entries
    if note_search_required:
        for candidate in _citation_note_candidates(
            pages,
            query_text=query_text,
            page_locator=page_locator,
            top_k=max(top_k * 2, top_k),
        ):
            add(
                candidate,
                "candidate_explicit_note" if _has_explicit_note_locator(page_locator)
                else "candidate_note_fallback",
            )
    else:
        # A body hit does not imply that explanatory notes lack useful context.
        # Keep notes labelled and bounded; never promote bibliography-only notes.
        linked_notes = _linked_explanatory_notes(
            pages, [candidate for candidate, _channels in entries.values()]
        )
        for candidate in linked_notes[:1]:
            add(candidate, "candidate_explanatory_note_context")
            add(candidate, "candidate_same_page_note_link")
        for candidate in ([] if linked_notes else _citation_note_candidates(
            pages, query_text=query_text, page_locator=page_locator, top_k=1,
            substantive_only=True,
        )):
            add(candidate, "candidate_explanatory_note_context")

    consolidated = _consolidate_nested_passage_entries(
        list(entries.values()),
        page_text_by_index={page.index: page.text for page in pages},
    )
    ranked = _select_diverse_candidate_entries(
        consolidated,
        query_text=query_text,
        facet_queries=facet_queries,
        top_k=top_k,
    )
    note = next((item for item in consolidated
                 if "candidate_explanatory_note_context" in item[1]), None)
    if note is not None and note not in ranked:
        if len(ranked) < top_k:
            ranked.append(note)
        elif top_k > 1:
            # Preserve the leading candidate and every exact/locator channel.
            for index in range(len(ranked) - 1, 0, -1):
                if not any("exact" in channel or "equivalent" in channel
                           or "locator" in channel for channel in ranked[index][1]):
                    ranked[index] = note
                    break
    continuation = _following_body_context(pages, ranked, query_text=query_text)
    if continuation is not None:
        candidate, anchor_key = continuation
        if not any(c.page_index == candidate.page_index and c.start <= candidate.start
                   and c.end >= candidate.end for c, _channels in ranked):
            item = (candidate, {"candidate_following_body_context"})
            if len(ranked) < top_k:
                ranked.append(item)
            else:
                for index in range(len(ranked) - 1, 1, -1):
                    current, channels = ranked[index]
                    if (current.page_index, current.start, current.end) == anchor_key:
                        continue
                    if any(any(token in channel for token in ("exact", "equivalent", "locator", "note"))
                           for channel in channels):
                        continue
                    ranked[index] = item
                    break
    return [
        (candidate, sorted(channels)) for candidate, channels in ranked
    ], rescue_applied, facet_queries


def _following_body_context(pages, ranked, *, query_text):
    """One non-recursive, same-page continuation within existing source windows.

    Context must have a positive lexical connection and immediately follow a
    selected body window. No section crossing, arbitrary neighbours, or increased
    excerpt budget. Keep the originating window in the selection.
    """
    tokens = _meaningful_tokens(query_text)
    if not tokens:
        return None
    candidates = []
    for page, start, end, text, role in _source_blocks(pages):
        if role != "body_prose" or len(text) > MAX_PASSAGE_CHARACTERS:
            continue
        if _passage_boundary_status(text) != "sentence_complete":
            continue
        if _NUMBERED_SECTION_HEADING_LINE.search(text):
            continue
        score = _lexical_score(tokens, query_text, text)
        if score <= 0:
            continue
        for anchor, _channels in ranked:
            if anchor.passage_role != "body_prose" or page.index != anchor.page_index:
                continue
            if not 0 <= start - anchor.end <= 2:
                continue
            if page.text[anchor.end:start].strip():
                continue
            candidate = _PassageCandidate(
                page_index=page.index, page_label=page.label, start=start, end=end,
                text=text.strip(), method="following_body_context", score=score,
                passage_role="body_prose",
            )
            candidates.append((candidate, (anchor.page_index, anchor.start, anchor.end)))
    return max(candidates, key=lambda item: (item[0].score, -item[0].start), default=None)


def _has_explicit_note_locator(locator: str) -> bool:
    return bool(
        re.search(r"\b(?:n\.?|note|endnote)\s*\d{1,3}\b", locator or "", re.IGNORECASE)
    )


def _linked_explanatory_notes(
    pages: list[_SourcePage], body_candidates: list[_PassageCandidate],
) -> list[_PassageCandidate]:
    """Follow only unique same-page printed markers from retrieved body context.

    The anchor, not semantic similarity or source-wide proximity, authorizes the
    context link. Ambiguous numbering and bibliography-only notes abstain.
    """
    notes: dict[tuple[int | None, str], list[tuple]] = {}
    for page, start, end, text, role in _source_blocks(pages):
        if role != "citation_notes":
            continue
        markers = list(re.finditer(r"(?m)^\s*(\d{1,3})[.)]?\s+(?=[A-Za-z])", text))
        if len(markers) != 1 or markers[0].start() != 0:
            continue
        notes.setdefault((page.index, markers[0].group(1)), []).append(
            (page, start, end, text)
        )
    found: dict[tuple[int | None, int, int], _PassageCandidate] = {}
    for body in body_candidates:
        if body.passage_role not in {"body_prose", "unknown"}:
            continue
        # Require attached sentence-end markers, not years, decimals or list items.
        for marker in re.finditer(r"[A-Za-z][.!?](\d{1,3})(?=\s|$)", body.text):
            matches = notes.get((body.page_index, marker.group(1)), [])
            if len(matches) != 1:
                continue
            page, start, end, text = matches[0]
            if not _is_explanatory_note(text):
                continue
            key = (page.index, start, end)
            candidate = _PassageCandidate(
                page_index=page.index, page_label=page.label, start=start, end=end,
                text=text.strip(), method="explanatory_note_context",
                score=min(0.90, body.score), passage_role="citation_notes",
            )
            if key not in found or candidate.score > found[key].score:
                found[key] = candidate
    return sorted(found.values(), key=lambda item: (item.score, -item.start), reverse=True)


def _citation_note_candidates(
    pages: list[_SourcePage],
    *,
    query_text: str,
    page_locator: str,
    top_k: int,
    substantive_only: bool = False,
) -> list[_PassageCandidate]:
    """Search labelled notes; optional conservative prose gate for body coexistence."""
    query_tokens = _meaningful_tokens(query_text)
    if not query_tokens:
        return []
    locator_pages = _page_locator_values(page_locator)
    candidates: list[_PassageCandidate] = []
    for page, start, end, text, role in _source_blocks(pages):
        if role != "citation_notes":
            continue
        if substantive_only and not _is_explanatory_note(text):
            continue
        score = _lexical_score(query_tokens, query_text, text)
        if score <= 0:
            continue
        candidates.append(
            _PassageCandidate(
                page_index=page.index,
                page_label=page.label,
                start=start,
                end=end,
                text=text.strip(),
                method="explanatory_note_context" if substantive_only else "citation_note_fallback",
                score=min(0.90, score + _page_boost(page, locator_pages)),
                passage_role="citation_notes",
            )
        )
    candidates.sort(key=lambda item: (item.score, -item.start), reverse=True)
    return _deduplicate_candidates(candidates)[:top_k]


def _is_explanatory_note(text: str) -> bool:
    """Conservative deterministic eligibility, not a relevance/accuracy judgment.

    Require a prose predicate outside parenthetical citations and URLs. Bibliographic
    titles alone and bare cross-references remain ineligible; ambiguity abstains.
    """
    prose = re.sub(r"https?://\S+|\([^)]*\)", " ", text)
    if len(list(re.finditer(r"(?m)^\s*\d{1,3}[.)]?\s+(?=[A-Za-z])", text))) > 1:
        return False
    if re.search(r"\b(?:cf\.|see\s+also|op\.\s*cit)", prose, re.IGNORECASE):
        return False
    if re.match(r"\s*(?:\d+[.)]?\s*)?(?:see|cf\.?|ibid\.?|op\.\s*cit)\b", text, re.IGNORECASE):
        return False
    if re.match(r"\s*(?:\d+[.)]?\s*)?[A-Z][\w’-]+,\s*[A-Z]\.", text):
        return False
    if len(re.findall(r"\b[A-Za-z]+\b", prose)) < 10:
        return False
    return bool(re.search(
        r"\b(?:was|were|is|are|had|has|have|documents|shows|describes|"
        r"reports|demonstrates|broadcast|aired|released)\b", prose, re.IGNORECASE
    ))


def _candidate_entry_rank_key(item: tuple[_PassageCandidate, set[str]]):
    candidate, channels = item
    return (
        "candidate_exact_phrase" in channels,
        "candidate_equivalent_phrase" in channels,
        "candidate_document_metadata" in channels,
        "candidate_bm25_concept" in channels,
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
    *,
    page_text_by_index: dict[int | None, str] | None = None,
) -> list[tuple[_PassageCandidate, set[str]]]:
    """Consolidate contained and substantially overlapping same-page windows.

    Sentence-complete source windows deliberately overlap to protect paragraph
    recall. Retrieval must not then present those windows as independent
    evidence. Containment and partial overlap retain the broader exact context
    while recording every source span that was consolidated into it. A narrow
    higher-scoring window must not erase material context already retrieved by
    a broader candidate.
    """

    def source_spans(candidate: _PassageCandidate) -> set[tuple[int, int]]:
        return set(candidate.consolidated_from_spans) or {
            (candidate.start, candidate.end)
        }

    def with_sources(
        candidate: _PassageCandidate, spans: set[tuple[int, int]]
    ) -> _PassageCandidate:
        return _PassageCandidate(
            page_index=candidate.page_index,
            page_label=candidate.page_label,
            start=candidate.start,
            end=candidate.end,
            text=candidate.text,
            method=candidate.method,
            score=candidate.score,
            passage_role=candidate.passage_role,
            consolidated_from_spans=tuple(sorted(spans)),
        )

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
                with_sources(
                    _PassageCandidate(
                        page_index=broader.page_index,
                        page_label=broader.page_label,
                        start=broader.start,
                        end=broader.end,
                        text=broader.text,
                        method=broader.method,
                        score=max(existing.score, candidate.score),
                        passage_role=broader.passage_role,
                    ),
                    source_spans(existing) | source_spans(candidate),
                ),
                combined_channels,
            )
            merged = True
            break
        if not merged:
            consolidated.append((candidate, set(channels)))

    ranked = sorted(
        consolidated,
        key=lambda item: (_candidate_entry_rank_key(item), len(item[0].text)),
        reverse=True,
    )
    distinct: list[tuple[_PassageCandidate, set[str]]] = []
    for candidate, channels in ranked:
        matches = [
            (index, _passage_overlap_ratio(candidate, existing))
            for index, (existing, _existing_channels) in enumerate(distinct)
            if candidate.page_index == existing.page_index
            and _passage_overlap_ratio(candidate, existing)
            >= SUBSTANTIAL_PASSAGE_OVERLAP_RATIO
        ]
        if not matches:
            distinct.append((candidate, set(channels)))
            continue
        match_index, _ratio = max(matches, key=lambda item: item[1])
        existing, existing_channels = distinct[match_index]
        broader = max(
            (existing, candidate),
            key=lambda item: (item.end - item.start, item.score),
        )
        union_start = min(existing.start, candidate.start)
        union_end = max(existing.end, candidate.end)
        page_text = (page_text_by_index or {}).get(broader.page_index)
        if page_text is not None and 0 <= union_start < union_end <= len(page_text):
            while union_start < union_end and page_text[union_start].isspace():
                union_start += 1
            while union_end > union_start and page_text[union_end - 1].isspace():
                union_end -= 1
            union_text = page_text[union_start:union_end]
            # The union covers page text neither candidate was scored on. When
            # the combined span reads as a different structural role it has
            # crossed a boundary - body prose running into a reference list -
            # and the inherited role would no longer describe the passage.
            # Keep the attested candidate instead of inventing a wider one.
            # A geometry-derived document_metadata role is exempt: its words
            # cannot reproduce it, so a text comparison says nothing.
            if broader.passage_role != "document_metadata" and (
                passage_role_from_text(union_text) != broader.passage_role
            ):
                union_start, union_end = broader.start, broader.end
                union_text = broader.text
        else:
            union_start = broader.start
            union_end = broader.end
            union_text = broader.text
        combined_channels = set(existing_channels) | set(channels)
        combined_channels.add("substantial_overlap_consolidated")
        distinct[match_index] = (
            with_sources(
                _PassageCandidate(
                    page_index=broader.page_index,
                    page_label=broader.page_label,
                    start=union_start,
                    end=union_end,
                    text=union_text,
                    method=broader.method,
                    score=max(existing.score, candidate.score),
                    passage_role=broader.passage_role,
                ),
                source_spans(existing) | source_spans(candidate),
            ),
            combined_channels,
        )
    return distinct


def _passage_overlap_ratio(
    left: _PassageCandidate, right: _PassageCandidate
) -> float:
    """Return same-page character overlap as a share of the shorter span."""
    if left.page_index != right.page_index:
        return 0.0
    overlap = max(0, min(left.end, right.end) - max(left.start, right.start))
    shorter = min(left.end - left.start, right.end - right.start)
    return overlap / shorter if shorter > 0 else 0.0


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
        start, end = _context_bounds_on_page(page, *match)
        text = page.text[start:end].strip()
        structural_role = _structural_role_for_range(page, *match)
        role = structural_role or (
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
                passage_role=role,
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
                        passage_role=role,
                    )
                )
    return _deduplicate_candidates(candidates)


def _bm25_concept_candidates(
    pages: list[_SourcePage],
    *,
    query_text: str,
    page_locator: str,
    top_k: int,
) -> list[_PassageCandidate]:
    """Rank eligible source blocks with deterministic document-local BM25.

    Concept normalization improves morphological recall but never changes the
    student proposition or establishes relevance. The channel is additive and
    the ordinary protected lexical/locator candidates remain retained.
    """
    query_tokens = _concept_tokens(query_text)
    if not query_tokens:
        return []
    blocks: list[tuple[_SourcePage, int, int, str, str, list[str]]] = []
    document_frequency: Counter[str] = Counter()
    for page, start, end, text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        tokens = _concept_tokens(text)
        if not tokens:
            continue
        blocks.append((page, start, end, text, role, tokens))
        document_frequency.update(set(tokens))
    if not blocks:
        return []

    average_length = sum(len(item[5]) for item in blocks) / len(blocks)
    query_terms = set(query_tokens)
    locator_pages = _page_locator_values(page_locator)
    scored: list[tuple[float, _SourcePage, int, int, str, str]] = []
    for page, start, end, text, role, tokens in blocks:
        counts = Counter(tokens)
        score = 0.0
        for token in query_terms:
            frequency = counts[token]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (len(blocks) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(tokens) / max(average_length, 1.0)
            )
            score += inverse_document_frequency * (frequency * 2.2 / denominator)
        if score <= 0:
            continue
        score += 0.1 * _page_boost(page, locator_pages)
        scored.append((score, page, start, end, text, role))
    scored.sort(key=lambda item: (item[0], -item[2]), reverse=True)
    output = []
    for rank, (_score, page, start, end, text, role) in enumerate(
        scored[:top_k], start=1
    ):
        output.append(
            _PassageCandidate(
                page_index=page.index,
                page_label=page.label,
                start=start,
                end=end,
                text=text.strip(),
                method="bm25_concept",
                score=max(0.50, 0.94 - (rank - 1) * 0.01),
                passage_role=role,
            )
        )
    return _deduplicate_candidates(output)


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
    blocks: list[tuple[_SourcePage, int, int, str, str, set[str]]] = []
    document_frequency: Counter[str] = Counter()
    for page, start, end, text, role in _source_blocks(pages):
        if role in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        concepts = set(_concept_tokens(text))
        if not concepts:
            continue
        blocks.append((page, start, end, text, role, concepts))
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
    for page, start, end, text, role, concepts in blocks:
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
                passage_role=role,
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
        consolidated_from_spans=list(candidate.consolidated_from_spans),
        passage_role=(
            candidate.passage_role
            if candidate.passage_role != "unknown"
            else passage_role_from_text(candidate.text)
        ),
        boundary_status=_passage_boundary_status(candidate.text),
    )


def _passage_boundary_status(
    text: str,
) -> Literal["sentence_complete", "bounded_fragment_or_nonprose", "unknown"]:
    """Expose when a bounded passage is not visibly sentence-complete."""
    normalized = text.strip()
    if not normalized:
        return "unknown"
    visible_start = re.sub(r'^["“‘(\[]+', "", normalized).lstrip()
    if visible_start and visible_start[0].islower():
        return "bounded_fragment_or_nonprose"
    if re.search(r"[.!?][\"')\]]*\s*$", normalized):
        return "sentence_complete"
    return "bounded_fragment_or_nonprose"


def _coverage_evidence(
    source: AuthorizedRepresentation,
    pages: list[_SourcePage],
    extraction_limitations: list[str],
) -> CoverageEvidence:
    complete = source.completeness_verdict in {"complete", "not_applicable"}
    subset_limitations = [
        item for item in extraction_limitations if _limits_source_coverage(item)
    ]
    usable_text = any(page.text.strip() for page in pages)
    if not pages or not usable_text:
        level = CoverageLevel.UNAVAILABLE
        confidence = ConfidenceLevel.NONE
    elif complete and not subset_limitations:
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
    if complete and subset_limitations:
        limitations.append(
            "Verification inspected only a policy-bounded subset of the source text."
        )
    if source.text_quality not in {"digital", "born_digital"}:
        limitations.append(f"Text quality is {source.text_quality!r}.")
    if source.derivation_method:
        limitations.append(
            "Verification used a separately hashed OCR derivative; OCR evidence remains confidence-limited."
        )
    return CoverageEvidence(
        level=level,
        confidence=confidence,
        method=(
            "validated_ocr_derivative_text_extraction"
            if source.derivation_method
            else "validated_representation_text_extraction"
        ),
        representation_kind=source.representation_kind,
        media_type=source.media_type,
        completeness_verdict=source.completeness_verdict,
        text_quality=source.text_quality,
        extraction_version=EXTRACTION_VERSION,
        extracted_text_sha256=_extracted_pages_sha256(pages),
        pages_total=len(pages) if pages else None,
        pages_inspected=[page.index for page in pages if page.index is not None],
        limitations=limitations,
    )


def _limits_source_coverage(limitation: str) -> bool:
    """Separate true omission/truncation from lossless normalization notes."""
    normalized = limitation.casefold()
    return any(
        marker in normalized
        for marker in (
            "truncated",
            "page limit",
            "policy limit",
            "excluded page",
            "missing page",
            "unavailable page",
        )
    )


def _extracted_pages_sha256(pages: list[_SourcePage]) -> str:
    """Bind page order, labels and normalized extracted text deterministically."""
    digest = hashlib.sha256()
    for page in pages:
        for value in (str(page.index), page.label or "", page.text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


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
    "incapable": "adequacy",
    "inability": "adequacy",
    "unable": "adequacy",
    "adequate": "adequacy",
    "inadequate": "adequacy",
    "sufficient": "adequacy",
    "insufficient": "adequacy",
    "sufficiency": "adequacy",
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
    re.compile(
        r"(?:\bdoi\s*:\s*|https?://doi\.org/)10\.\d{4,9}/",
        re.IGNORECASE,
    ),
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
_NUMBERED_SECTION_HEADING_LINE = re.compile(
    r"(?m)^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^\n.!?]{2,120}\s*$"
)
_REFERENCE_ENTRY_LINE = re.compile(
    r"(?m)^(?:\s*(?:\[?\d+\]?\.?\s+)?[A-Z][^\n]{0,140}"
    r"\((?:18|19|20)\d{2}[a-z]?\)"
    r"|[ \t]*[A-Z][\w’'\-]+,[ \t]+(?:[A-Z]\.[ \t]*){1,4}"
    r"(?:18|19|20)\d{2}[a-z]?\.[ \t]+\S)"
)
_NUMBERED_NOTE_LINE = re.compile(r"(?m)^\s*\d{1,3}[.)]?\s+(?=\S)")
_PARENTHETICAL_YEAR = re.compile(
    r"\((?:[^()]*)\b(?:18|19|20)\d{2}[a-z]?(?:[^()]*)\)"
)
_LEGAL_CITATION_CUE = re.compile(
    r"(?:\b\d+\s+(?:U\.S\.|F\.?\s*(?:2d|3d|4th)?|S\.\s*Ct\.|Stat\.|"
    r"L\.\s*(?:Ed\.|Rev\.))\s+\d+\b|\b(?:18|19|20)\d{2}\s+WL\s+\d+\b)",
    re.IGNORECASE,
)
_EXCLUDED_RETRIEVAL_ROLES = {
    "reference_list",
    "citation_notes",
    "publication_metadata",
    "page_furniture",
    "author_biography",
}

_PAGE_FURNITURE_MARKERS = (
    re.compile(r"\bjournal homepage\b", re.IGNORECASE),
    re.compile(r"(?:\bcopyright\b|©|\ball rights reserved\b)", re.IGNORECASE),
)
_STRONG_PAGE_FURNITURE_MARKERS = (
    re.compile(r"\bthis content downloaded from\b", re.IGNORECASE),
    re.compile(r"\ball use subject to\b", re.IGNORECASE),
    re.compile(r"^\s*CONTACT\b", re.IGNORECASE),
    re.compile(r"\bthis article has been republished\b", re.IGNORECASE),
)
_AUTHOR_BIOGRAPHY_START = re.compile(
    r"^\s*[A-Z][\w .,'’\-]{1,100}\s+is\s+(?:an?\s+|the\s+)?"
    r"(?:Lecturer|Professor|Reader|Researcher|Fellow|Instructor|Dean|Chair)\b",
    re.IGNORECASE,
)


def _is_metadata_noise_block(text: str) -> bool:
    """Exclude compact publication boilerplate, never ordinary source prose.

    The filter deliberately requires multiple independent metadata cues, or one
    cue in a very short non-sentence block, so abstracts and scholarly prose
    containing an incidental DOI or year remain eligible for retrieval.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return True
    # Publisher platform accessibility/conformance statements are not book
    # contents. Require both a product-specific declaration and formal audit
    # language, not merely discussion of accessibility in scholarly prose.
    platform_statement = re.search(
        r"\b(?:ebook|platform|smartbook|reader\s+app)\b.{0,100}\b(?:was built with accessibility|provides a student experience|conforms to|complies with)\b",
        normalized, re.I,
    )
    conformance = re.search(r"\b(?:VPAT|Accessibility Conformance Report)\b", normalized, re.I)
    standard = re.search(r"\bWCAG\s+2\.[012]\b", normalized, re.I)
    if platform_statement and conformance and standard:
        return True
    # A copyright/cataloging leaf may contain substantial permissions prose.
    # Require the cataloging heading, an actual ISBN-shaped identifier and
    # copyright evidence together; discussion of ISBNs/cataloging alone is not
    # metadata. Do not apply the compact-boilerplate sentence ceiling here.
    if (re.search(r"\b(?:Library of Congress )?Cataloging[- ]in[- ]Publication Data\b", normalized, re.I)
            and re.search(r"\bISBN(?:-1[03])?\s*:?[\s-]*[0-9][0-9Xx\s-]{8,20}", normalized)
            and re.search(r"(?:©\s*(?:18|19|20)\d{2}|\bcopyright\b|\ball rights reserved\b)", normalized, re.I)):
        return True
    marker_count = sum(bool(pattern.search(normalized)) for pattern in _METADATA_NOISE_MARKERS)
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", normalized))
    word_count = len(re.findall(r"\b\w+\b", normalized))
    if marker_count >= 2 and word_count <= 180 and sentence_count <= 6:
        return True
    return marker_count >= 1 and word_count <= 35 and sentence_count == 0


def passage_role_from_text(text: str) -> Literal[
    "body_prose",
    "abstract",
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
    if re.match(r"^ABSTRACT\b", normalized, re.IGNORECASE):
        return "abstract"
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


DOCUMENT_METADATA_MEMBER_PAGE_LIMIT = 3

# A title/byline block carries a geometry-derived role that its bare words
# cannot reproduce. Reading as publication metadata is the *expected* plain-text
# result for such a block, not evidence that the role was supplied from outside
# the application, so it belongs here beside the two weaker readings. Provenance
# is established by the producing method and retrieval channel, which only this
# module sets.
_DOCUMENT_METADATA_MEMBER_TEXT_ROLES = frozenset(
    {"unknown", "body_prose", "publication_metadata"}
)


def document_metadata_member_admissible(*, page_index: int | None, text: str) -> bool:
    """Whether an opening-page title block may stand as document-level evidence.

    Producer and validator share this rule. When only the validator held it, a
    block the producer had already offered could be refused at persist time,
    failing the whole paper instead of simply not being offered.
    """
    return bool(
        page_index is not None
        and 0 <= page_index < DOCUMENT_METADATA_MEMBER_PAGE_LIMIT
        and passage_role_from_text(text) in _DOCUMENT_METADATA_MEMBER_TEXT_ROLES
    )


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
    targets = _quotation_targets(claim_text)
    return max(targets, key=len) if targets else claim_text.strip()


def _quotation_targets(claim_text: str) -> list[str]:
    """Return every complete double-quoted span in paper order."""
    matches = []
    for pattern in (r'"([^"]{2,})"', r"“([^”]{2,})”"):
        matches.extend(
            (match.start(), match.group(1).strip())
            for match in re.finditer(pattern, claim_text)
            if match.group(1).strip()
        )
    return [value for _start, value in sorted(matches)]


def _ocr_token_sequence_match(
    source_text: str, target: str
) -> tuple[int, int, str] | None:
    """Match a long exact word sequence while ignoring OCR punctuation noise."""

    def unicode_casefold_with_positions(value: str) -> tuple[str, list[int]]:
        output: list[str] = []
        positions: list[int] = []
        for index, character in enumerate(value):
            for emitted in unicodedata.normalize("NFKC", character).casefold():
                output.append(emitted)
                positions.append(index)
        return "".join(output), positions

    normalized_target, _target_positions = unicode_casefold_with_positions(target)
    target_words = [
        match.group(0) for match in re.finditer(r"[^\W_]+", normalized_target)
    ]
    if len(target_words) < 8 or sum(len(word) for word in target_words) < 40:
        return None
    normalized_source, source_positions = unicode_casefold_with_positions(source_text)
    source_matches = list(re.finditer(r"[^\W_]+", normalized_source))
    source_words = [match.group(0) for match in source_matches]
    width = len(target_words)
    for index in range(0, len(source_words) - width + 1):
        if source_words[index : index + width] != target_words:
            continue
        normalized_start = source_matches[index].start()
        normalized_end = source_matches[index + width - 1].end()
        if not source_positions or normalized_end <= normalized_start:
            return None
        return (
            source_positions[normalized_start],
            source_positions[normalized_end - 1] + 1,
            "ocr_token_sequence",
        )
    return None


def _quotation_match(
    source_text: str,
    target: str,
    *,
    allow_ocr_token_sequence: bool = False,
) -> tuple[int, int, str] | None:
    """Locate one complete quote with bounded editorial normalization.

    Ellipses must have substantive text on both sides, preventing a short
    prefix from being treated as a complete-span match.
    """
    # First retain a lexical hyphen across a PDF wrap. The historical
    # dehyphenated alternative remains available for split ordinary words.
    direct = _normalized_find(source_text, target, preserve_line_hyphens=True)
    if direct is None:
        direct = _normalized_find(source_text, target)
    if direct is not None:
        raw = source_text[direct[0] : direct[1]]
        method = "literal" if raw == target else "normalized"
        return direct[0], direct[1], method
    if re.search(r"(?<=\w)-\s*\n\s*(?=\w)", target):
        # Each proven line-wrap hyphen may be lexical or discretionary.
        # Resolve them independently rather than dropping every hyphen at once.
        for preserve in (True, False):
            normalized, positions = _normalized_with_positions(source_text, preserve_line_hyphens=preserve)
            found = re.search(_literal_quote_pattern(target), normalized)
            if found and found.end() > found.start():
                return positions[found.start()], positions[found.end()-1]+1, "linewrap_normalized"

    editorial_target = re.sub(
        r"\[\s*(?:sic|emphasis\s+added)\s*\]",
        "",
        target,
        flags=re.IGNORECASE,
    )
    editorial_target = re.sub(
        r"\[([^\]]+)\]",
        lambda match: match.group(1),
        editorial_target,
    )
    if editorial_target != target:
        bracket_match = _normalized_find(source_text, editorial_target)
        if bracket_match is not None:
            return bracket_match[0], bracket_match[1], "bracket_normalized"

    editorial = _marked_editorial_match(source_text, target)
    if editorial is None and re.search(r"(?<=\w)-\s*\n\s*(?=\w)", target):
        editorial = _marked_editorial_match(source_text, re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", target))
    if editorial is not None:
        return editorial

    segments = [
        segment.strip()
        for segment in re.split(r"(?:\.(?:\s*\.){2,}|…)", editorial_target)
        if segment.strip()
    ]
    segment_word_counts = [len(_meaningful_tokens(segment)) for segment in segments]
    if (
        len(segments) < 2
        or any(count < 2 for count in segment_word_counts)
        or sum(segment_word_counts) < 8
    ):
        return (
            _ocr_token_sequence_match(source_text, target)
            if allow_ocr_token_sequence
            else None
        )
    matched_segments: list[tuple[int, int]] = []
    cursor = 0
    for segment in segments:
        match = _normalized_find(source_text[cursor:], segment)
        if match is None:
            return (
                _ocr_token_sequence_match(source_text, target)
                if allow_ocr_token_sequence
                else None
            )
        absolute = (cursor + match[0], cursor + match[1])
        if matched_segments and absolute[0] - matched_segments[-1][1] > 3_000:
            return None
        matched_segments.append(absolute)
        cursor = absolute[1]
    return matched_segments[0][0], matched_segments[-1][1], "ellipsis_normalized"


def _literal_quote_pattern(value: str) -> str:
    normalized, positions = _normalized_with_positions(value, preserve_line_hyphens=True)
    return ''.join('-?' if character == '-' and re.match(r'-\s*\n\s*\w', value[positions[index]:]) else re.escape(character)
                   for index, character in enumerate(normalized))


def _marked_editorial_match(source_text: str, target: str) -> tuple[int, int, str] | None:
    """Match unchanged wording around explicitly marked edits, not their meaning.

    Brackets may insert or replace text; ellipses may omit it. Keep gaps bounded,
    require substantive unchanged wording, and never tolerate an unmarked edit.
    The distinct method requires a visible context-review limitation.
    """
    marker = re.compile(r"\[[^\[\]]*\]|\.(?:\s*\.){2,}|…")
    marks = list(marker.finditer(target))
    if not marks or len(marks) > 12:
        return None
    pieces, cursor, literal_words = [], 0, 0
    has_bracket = False
    for item in marks:
        literal = target[cursor:item.start()]
        literal_words += len(_meaningful_tokens(literal))
        pieces.append(_literal_quote_pattern(literal))
        content = item.group()
        ellipsis = bool(re.fullmatch(r"\[?\s*(?:\.(?:\s*\.){2,}|…)\s*\]?", content))
        has_bracket |= not ellipsis
        pieces.append(r".{0,3000}?" if ellipsis else r".{0,160}?")
        cursor = item.end()
    tail = target[cursor:]
    literal_words += len(_meaningful_tokens(tail))
    pieces.append(_literal_quote_pattern(tail))
    if literal_words < 8:
        return None
    # Leading/trailing editorial marks cannot establish a replacement span;
    # locate the unchanged wording and show its enclosing source context.
    if not pieces[0]:
        pieces = pieces[2:]
    if pieces and not pieces[-1]:
        pieces = pieces[:-2]
    if not pieces or not pieces[0] or not pieces[-1]:
        return None
    pattern = re.compile("".join(pieces), re.DOTALL)
    for preserve_hyphens in (True, False):
        normalized_source, positions = _normalized_with_positions(source_text, preserve_line_hyphens=preserve_hyphens)
        match = pattern.search(normalized_source)
        if match is not None and match.end() > match.start():
            return positions[match.start()], positions[match.end()-1]+1, (
                "marked_editorial_match" if has_bracket else "ellipsis_normalized")
    return None


def _quotation_location(
    pages: list[_SourcePage],
    target: str,
    *,
    allow_ocr_token_sequence: bool = False,
) -> _QuotationLocation | None:
    """Locate a quote on one page or across adjacent eligible page blocks."""
    for page in pages:
        quotation_match = _quotation_match(
            page.text,
            target,
            allow_ocr_token_sequence=allow_ocr_token_sequence,
        )
        if quotation_match is None:
            continue
        start, end, method = quotation_match
        if _structural_role_for_range(page, start, end) in _EXCLUDED_RETRIEVAL_ROLES:
            continue
        return _QuotationLocation(
            fragments=((page.index, start, end),),
            method=method,
        )

    eligible_blocks = [
        (page, start, end, text)
        for page, start, end, text, role in _source_blocks(pages)
        if role not in _EXCLUDED_RETRIEVAL_ROLES
    ]
    if not eligible_blocks:
        return None

    virtual_parts: list[tuple[_SourcePage, int, int, int, int]] = []
    virtual_text_parts: list[str] = []
    cursor = 0
    for page, start, end, text in eligible_blocks:
        if virtual_text_parts:
            virtual_text_parts.append("\n")
            cursor += 1
        virtual_start = cursor
        virtual_text_parts.append(text)
        cursor += len(text)
        virtual_parts.append((page, start, end, virtual_start, cursor))
    virtual_text = "".join(virtual_text_parts)
    quotation_match = _quotation_match(
        virtual_text,
        target,
        allow_ocr_token_sequence=allow_ocr_token_sequence,
    )
    if quotation_match is None:
        return None
    virtual_start, virtual_end, method = quotation_match
    fragments: list[tuple[int | None, int, int]] = []
    for page, source_start, _source_end, part_start, part_end in virtual_parts:
        overlap_start = max(virtual_start, part_start)
        overlap_end = min(virtual_end, part_end)
        if overlap_start >= overlap_end:
            continue
        fragments.append(
            (
                page.index,
                source_start + overlap_start - part_start,
                source_start + overlap_end - part_start,
            )
        )
    if len(fragments) < 2:
        return None
    page_indices = [page_index for page_index, _start, _end in fragments]
    concrete_pages = sorted({value for value in page_indices if value is not None})
    if len(concrete_pages) < 2 or any(
        following != previous + 1
        for previous, following in zip(concrete_pages, concrete_pages[1:])
    ):
        return None
    return _QuotationLocation(fragments=tuple(fragments), method=method)


def _academic_practice_checks(
    *,
    claim: ClaimEvidence,
    source: AuthorizedRepresentation,
    pages: list[_SourcePage],
    passages: list[SourcePassageEvidence],
) -> tuple[AcademicPracticeCheckEvidence, AcademicPracticeCheckEvidence]:
    """Check complete quotation spans and only evidence-backed locators."""
    if claim.claim_type != "quotation":
        quotation = AcademicPracticeCheckEvidence(
            status="complete", outcome="not_applicable"
        )
        locator = (
            AcademicPracticeCheckEvidence(
                status="not_assessable",
                outcome="paraphrase_locator_requires_relevant_evidence",
                limitations=[
                    "A locator on a paraphrase cannot be validated from retrieval rank alone."
                ],
            )
            if claim.page_locator
            else AcademicPracticeCheckEvidence(
                status="complete", outcome="not_applicable"
            )
        )
        return quotation, locator

    targets = _quotation_targets(claim.text)
    if not targets:
        unresolved = AcademicPracticeCheckEvidence(
            status="incomplete",
            outcome="quotation_boundaries_unavailable",
            limitations=[
                "The citation was classified as a quotation but no complete double-quoted span was available."
            ],
        )
        locator = (
            AcademicPracticeCheckEvidence(
                status="not_assessable",
                outcome="quotation_not_located",
            )
            if claim.page_locator
            else AcademicPracticeCheckEvidence(status="complete", outcome="not_applicable")
        )
        return unresolved, locator

    target_matches: list[_QuotationLocation] = []
    missing = 0
    missing_editorial = 0
    allow_ocr_token_sequence = bool(
        source.text_quality == "scan_ocr" and source.derivation_method
    )
    for target in targets:
        found = _quotation_location(
            pages,
            target,
            allow_ocr_token_sequence=allow_ocr_token_sequence,
        )
        if found is None:
            missing += 1
            if re.search(r"\[[^\]]*\]|\.(?:\s*\.){2,}|…", target):
                missing_editorial += 1
        else:
            target_matches.append(found)

    evidence_ids = [
        passage.passage_id
        for passage in passages
        if any(
            passage.page_index == page_index
            and passage.character_start <= start
            and passage.character_end >= end
            for match in target_matches
            for page_index, start, end in match.fragments
        )
    ]
    reliable_negative = (
        source.completeness_verdict in {"complete", "not_applicable"}
        and source.text_quality in {"digital", "born_digital"}
    )
    if missing == 0:
        if all(match.method == "literal" for match in target_matches):
            outcome = "all_spans_literal_match"
        elif any(match.method == "ocr_token_sequence" for match in target_matches):
            outcome = "all_spans_ocr_token_sequence_match"
        else:
            outcome = "all_spans_normalized_match"
        quotation = AcademicPracticeCheckEvidence(
            status="complete",
            outcome=outcome,
            evidence_passage_ids=evidence_ids,
            limitations=([
                "Unchanged quoted wording matches around marked brackets or omissions. The meaning of editorial changes and omitted context requires human review."
            ] if any(match.method in {"marked_editorial_match", "ellipsis_normalized"} for match in target_matches) else []),
        )
    elif missing_editorial == missing:
        quotation = AcademicPracticeCheckEvidence(
            status="not_assessable", outcome="marked_editorial_changes_require_review",
            evidence_passage_ids=evidence_ids,
            limitations=["Marked quotation edits could not be matched reliably. Compare the unchanged wording and editorial changes with the source; this is not an automatic quotation-fidelity failure."],
        )
    elif reliable_negative:
        quotation = AcademicPracticeCheckEvidence(
            status="complete",
            outcome="some_spans_not_located" if target_matches else "no_span_located",
            evidence_passage_ids=evidence_ids,
            limitations=[
                "A non-match establishes only that the complete quoted span was not located in this admitted representation."
            ],
        )
    else:
        quotation = AcademicPracticeCheckEvidence(
            status="not_assessable",
            outcome="inconclusive_source_text_quality_or_coverage",
            evidence_passage_ids=evidence_ids,
        )

    version = (
        (source.edition_or_version or "")
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    pagination_may_differ = version in {
        "acceptedversion",
        "accepted_version",
        "accepted_manuscript",
        "author_accepted_manuscript",
        "submittedversion",
        "submitted_version",
        "submitted_manuscript",
        "preprint",
    }
    if not claim.page_locator:
        locator = AcademicPracticeCheckEvidence(status="complete", outcome="not_applicable")
    elif pagination_may_differ:
        locator = AcademicPracticeCheckEvidence(
            status="not_assessable",
            outcome="source_version_pagination_may_differ",
            evidence_passage_ids=evidence_ids,
            limitations=[
                "The available manuscript/preprint may not preserve the published version's pagination."
            ],
        )
    elif not _page_locator_values(claim.page_locator):
        locator = AcademicPracticeCheckEvidence(
            status="incomplete", outcome="locator_not_parseable"
        )
    elif missing:
        locator = AcademicPracticeCheckEvidence(
            status="not_assessable",
            outcome="quotation_not_fully_located",
            evidence_passage_ids=evidence_ids,
        )
    else:
        locator_pages = _page_locator_values(claim.page_locator)
        matched_pages = {
            page_index
            for match in target_matches
            for page_index, _start, _end in match.fragments
            if page_index is not None
        }
        page_labels = {
            page.index: {
                int(value) for value in re.findall(r"\d+", page.label or "")
            }
            for page in pages
        }
        derivative_label_missing = bool(source.derivation_method) and any(
            not page_labels.get(page_index)
            for page_index in matched_pages
            if page_index is not None
        )
        if derivative_label_missing:
            locator = AcademicPracticeCheckEvidence(
                status="not_assessable",
                outcome="ocr_page_label_unavailable",
                evidence_passage_ids=evidence_ids,
                limitations=[
                    "The OCR derivative did not preserve a reliable printed-page label for every matched span."
                ],
            )
        else:
            consistent = bool(matched_pages) and all(
                (
                    page_labels.get(page_index)
                    or ({page_index + 1} if not source.derivation_method else set())
                )
                & locator_pages
                for page_index in matched_pages
                if page_index is not None
            )
            locator = AcademicPracticeCheckEvidence(
                status="complete",
                outcome=(
                    "located_span_matches_supplied_locator"
                    if consistent
                    else "located_span_outside_supplied_locator"
                ),
                evidence_passage_ids=evidence_ids,
            )
    return quotation, locator


def _normalized_with_positions(value: str, *, preserve_line_hyphens: bool = False) -> tuple[str, list[int]]:
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
                if preserve_line_hyphens:
                    output.append("-")
                    positions.append(index)
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


def _normalized_find(source: str, target: str, *, preserve_line_hyphens: bool = False) -> tuple[int, int] | None:
    normalized_source, positions = _normalized_with_positions(source, preserve_line_hyphens=preserve_line_hyphens)
    normalized_target, _ = _normalized_with_positions(target, preserve_line_hyphens=preserve_line_hyphens)
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
    local_start = start - bounded_start
    local_end = end - bounded_start
    containing = [
        (window_start, window_end)
        for window_start, window_end, _window_text in _sentence_complete_windows(
            text[bounded_start:bounded_end]
        )
        if window_start <= local_start and local_end <= window_end
    ]
    if containing:
        window_start, window_end = max(
            containing,
            key=lambda bounds: (
                min(local_start - bounds[0], bounds[1] - local_end),
                bounds[1] - bounds[0],
                -bounds[0],
            ),
        )
        return bounded_start + window_start, bounded_start + window_end
    return start, end


def _context_bounds_on_page(
    page: _SourcePage, start: int, end: int
) -> tuple[int, int]:
    """Keep exact-match context inside its current structural/body interval.

    Numbered section headings are hard semantic boundaries for every retrieval
    channel. Ordinary lexical windows inherit those boundaries from
    ``_text_blocks``; page-level exact matching must apply the same boundary
    explicitly.
    """
    bounded_start, bounded_end = _context_bounds(page.text, start, end)
    containing = [
        span
        for span in page.structural_spans
        if span.start <= start and end <= span.end
    ]
    if containing:
        span = min(containing, key=lambda item: item.end - item.start)
        return max(bounded_start, span.start), min(bounded_end, span.end)
    left_boundaries = [
        span.end for span in page.structural_spans if span.end <= start
    ]
    right_boundaries = [
        span.start for span in page.structural_spans if span.start >= end
    ]
    if left_boundaries:
        bounded_start = max(bounded_start, max(left_boundaries))
    if right_boundaries:
        bounded_end = min(bounded_end, min(right_boundaries))

    numbered_headings = list(_NUMBERED_SECTION_HEADING_LINE.finditer(page.text))
    prior_headings = [
        heading.start()
        for heading in numbered_headings
        if heading.start() <= start
    ]
    following_headings = [
        heading.start()
        for heading in numbered_headings
        if heading.start() >= end
    ]
    if prior_headings:
        bounded_start = max(bounded_start, max(prior_headings))
    if following_headings:
        bounded_end = min(bounded_end, min(following_headings))
    return bounded_start, bounded_end


def _exact_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Map the shared sentence splitter's output back to exact source offsets."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            # Binary/control debris in a PDF footer can be transformed by the
            # shared splitter's abbreviation placeholders. Keep the exact
            # body sentences already recovered rather than discarding the
            # entire page or manufacturing coordinates for the damaged tail.
            break
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def _sentence_complete_windows(text: str) -> list[tuple[int, int, str]]:
    """Build overlapping bounded windows that begin and end on sentences.

    PDF extraction often represents visual lines rather than paragraphs. A
    disjoint character window can therefore split the most relevant paragraph
    and leave retrieval with only its tail. Two-thirds sentence-aligned overlap
    ensures that an ordinary paragraph up to roughly 1,200 characters can
    appear intact in at least one 1,800-character retrieval window.
    """
    spans = _exact_sentence_spans(text)
    if not spans:
        return []
    bounded_spans: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start <= MAX_PASSAGE_CHARACTERS:
            bounded_spans.append((start, end))
            continue
        # A PDF/OCR text layer can expose several thousand characters as one
        # sentence. Preserve exact coordinates and the hard passage contract by
        # splitting only this pathological span at the latest available clause
        # or word boundary. Ordinary sentences remain whole.
        fragment_start = start
        while fragment_start < end:
            hard_end = min(fragment_start + MAX_PASSAGE_CHARACTERS, end)
            fragment_end = hard_end
            if hard_end < end:
                bounded = text[fragment_start:hard_end]
                clause_breaks = [match.end() for match in re.finditer(r"[;:]\s+", bounded)]
                word_breaks = [match.start() for match in re.finditer(r"\s+", bounded)]
                candidates = [
                    offset
                    for offset in clause_breaks + word_breaks
                    if offset >= MAX_PASSAGE_CHARACTERS // 2
                ]
                if candidates:
                    fragment_end = fragment_start + max(candidates)
            if fragment_end <= fragment_start:
                fragment_end = hard_end
            bounded_spans.append((fragment_start, fragment_end))
            fragment_start = fragment_end
    spans = bounded_spans
    windows: list[tuple[int, int, str]] = []
    start_index = 0
    while start_index < len(spans):
        start = spans[start_index][0]
        end_index = start_index
        while (
            end_index + 1 < len(spans)
            and spans[end_index + 1][1] - start <= MAX_PASSAGE_CHARACTERS
        ):
            end_index += 1
        end = spans[end_index][1]
        windows.append((start, end, text[start:end]))
        if end_index == len(spans) - 1:
            break
        threshold = start + PASSAGE_WINDOW_STRIDE_CHARACTERS
        next_index = start_index + 1
        while next_index < len(spans) and spans[next_index][0] < threshold:
            next_index += 1
        start_index = min(next_index, end_index + 1)
    return windows


def _text_blocks(text: str) -> list[tuple[int, int, str]]:
    raw_blocks: list[tuple[int, int]] = []
    for match in re.finditer(
        r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.DOTALL
    ):
        start, end = match.start(), match.end()
        section_starts = [
            start + heading.start()
            for heading in _NUMBERED_SECTION_HEADING_LINE.finditer(text[start:end])
            if heading.start() > 0
        ]
        cursor = start
        for section_start in section_starts:
            boundary = section_start
            while boundary > cursor and text[boundary - 1].isspace():
                boundary -= 1
            if boundary > cursor:
                raw_blocks.append((cursor, boundary))
            cursor = section_start
        if cursor < end:
            raw_blocks.append((cursor, end))
    merged_blocks: list[tuple[int, int]] = []
    for start, end in raw_blocks:
        if merged_blocks:
            prior_start, prior_end = merged_blocks[-1]
            prior_text = text[prior_start:prior_end].strip()
            current_text = text[start:end].strip()
            if (
                prior_text
                and current_text
                and not re.search(r"[.!?][\"')\]]*\s*$", prior_text)
                and re.match(r"^[\"'(\[]*[a-z]", current_text)
                and not text[prior_end:start].strip()
            ):
                # PDF block/column boundaries sometimes bisect a sentence. The
                # exact whitespace gap remains part of the merged coordinate
                # range; completed sentences and uppercase paragraph starts do
                # not merge.
                merged_blocks[-1] = (prior_start, end)
                continue
        merged_blocks.append((start, end))

    blocks: list[tuple[int, int, str]] = []
    for start, end in merged_blocks:
        block = text[start:end]
        if len(block) <= MAX_PASSAGE_CHARACTERS:
            blocks.append((start, end, block))
            continue
        windows = _sentence_complete_windows(block)
        if windows:
            blocks.extend(
                (start + window_start, start + window_end, window_text)
                for window_start, window_end, window_text in windows
            )
    return blocks


def _page_text_blocks(
    page: _SourcePage,
) -> list[tuple[int, int, str, str | None]]:
    """Split body intervals while retaining classified exact structural spans."""
    output: list[tuple[int, int, str, str | None]] = []
    cursor = 0
    for span in sorted(page.structural_spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor or span.end > len(page.text):
            continue
        if cursor < span.start:
            for start, end, text in _text_blocks(page.text[cursor:span.start]):
                output.append((cursor + start, cursor + end, text, None))
        structural_text = page.text[span.start:span.end]
        if structural_text.strip():
            output.append((span.start, span.end, structural_text, span.role))
        cursor = span.end
    if cursor < len(page.text):
        for start, end, text in _text_blocks(page.text[cursor:]):
            output.append((cursor + start, cursor + end, text, None))
    return output


def _source_blocks(
    pages: list[_SourcePage],
) -> list[tuple[_SourcePage, int, int, str, str]]:
    """Return exact blocks with document-aware structural roles.

    A reference heading starts a reference section only when at least one
    entry-shaped line follows it on the same page. This prevents a table of
    contents entry from excluding the rest of a book or report. Continuation is
    also page-bounded: later pages remain reference material only while they
    contain entry-shaped lines, so a chapter bibliography cannot swallow the
    chapters that follow it.
    """
    output = []
    reference_section_started = False
    for page in pages:
        reference_heading = _REFERENCE_HEADING.search(page.text)
        reference_entries = len(_REFERENCE_ENTRY_LINE.findall(page.text))
        heading_starts_section = bool(
            reference_heading is not None
            and _REFERENCE_ENTRY_LINE.search(page.text[reference_heading.end() :])
        )
        reference_continuation = bool(
            reference_section_started and reference_entries >= 1
        )
        page_is_reference_section = heading_starts_section or reference_continuation
        for start, end, block_text, structural_role in _page_text_blocks(page):
            if structural_role is not None:
                output.append((page, start, end, block_text, structural_role))
                continue
            if (
                heading_starts_section
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
            if page_is_reference_section and (
                reference_heading is None or end > reference_heading.start()
            ):
                role = "reference_list"
            else:
                role = passage_role_from_text(block_text)
            output.append((page, start, end, block_text, role))
        reference_section_started = page_is_reference_section
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
    value = re.sub(r"^\s*(?:pages?|pp?\.?)\s*(?=\d)", "", locator or "", flags=re.IGNORECASE)
    if len(value) > 512 or not re.fullmatch(
        r"\s*\d{1,5}(?:\s*[-–—]\s*\d{1,5})?(?:\s*,\s*\d{1,5}(?:\s*[-–—]\s*\d{1,5})?)*\s*",
        value,
    ):
        return set()
    pages: set[int] = set()
    for item in value.split(","):
        numbers = [int(number) for number in re.findall(r"\d+", item)]
        start, end = numbers[0], numbers[-1]
        if not 0 < start <= end or end - start > 50:
            return set()
        pages.update(range(start, end + 1))
        if len(pages) > 100:
            return set()
    return pages


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


# Matched to the report's bound on stored source text. The judgment must be
# made on exactly the text the report can keep and a reader can see: assessing
# more than is stored would leave the comparison unverifiable, and storing more
# would widen how much of a source the report retains.
MAX_SCOPE_EXCERPT_CHARACTERS = 1_200


def leading_source_excerpt(
    source: AuthorizedRepresentation,
    *,
    limit: int = MAX_SCOPE_EXCERPT_CHARACTERS,
) -> str:
    """The opening of a retrieved document, where a work states what it covers.

    A later passage says what was cited; only the opening states the work's own
    subject and the limits it sets for itself, which is what a scope comparison
    needs. Extraction failures return an empty string, which is abstention.
    """
    try:
        pages, _ = _extract_pages(source)
    except Exception:  # extraction failure is abstention, never a judgment
        return ""
    collected: list[str] = []
    total = 0
    for page in pages:
        text = re.sub(r"\s+", " ", str(getattr(page, "text", "") or "")).strip()
        if not text:
            continue
        collected.append(text)
        total += len(text) + 1
        if total >= limit:
            break
    return " ".join(collected)[:limit].strip()


# Stopwords plus the reporting verbs a citing sentence uses about its source;
# neither tells us what the source is about.
_CLAIM_TERM_STOPWORDS = frozenset("""
the a an and or of in on for to with by from as is are was were be been being this that these
those it its their his her they he she we you at not no but if then than so such which who whom
whose what when where how why can could may might will would shall should must have has had do
does did done also more most other another some any each every both few many much own same very
just only about into over under between during through above below states stated state according
meanwhile however therefore thus because while although though new used using use make makes made
known argues argued stated said claim claims claimed show shows showed suggest suggests suggested
note notes noted write writes wrote point points pointed follow follows followed
found finds finding report reports reported describe describes described
""".split())
MAX_CLAIM_TERMS = 12
CLAIM_TERM_MIN_LENGTH = 5


def claim_topic_terms(claim_text: str) -> list[str]:
    """The distinctive vocabulary a citing sentence attributes to its source."""
    words = re.findall(r"[a-z][a-z\-']*", (claim_text or "").lower())
    terms = []
    for word in words:
        # A possessive is the same topic: "hero's journey" is about heroes.
        word = re.sub(r"'s?$", "", word)
        if len(word) < CLAIM_TERM_MIN_LENGTH or word in _CLAIM_TERM_STOPWORDS:
            continue
        if word not in terms:
            terms.append(word)
    return terms[:MAX_CLAIM_TERMS]


def claim_terms_present_in_source(
    source: AuthorizedRepresentation, claim_text: str,
) -> tuple[int, int]:
    """How much of the claim's vocabulary the WHOLE document actually contains.

    The scope judgment sees a bounded opening -- 1,200 characters of a work
    that may run to 126 pages -- so "the source does not discuss this" cannot
    be concluded from it. Measured 2026-09-22, both full-text different-subject
    marks were exactly that mistake: Pallant's opening omitted narrative and
    Khan's omitted the Telecommunications Act, and both works discuss them at
    length. This reads the complete extracted text locally, sends nothing to a
    model and retains no document text, and returns counts only.
    """
    if source is None or not isinstance(claim_text, str):
        return 0, 0
    terms = claim_topic_terms(claim_text)
    if not terms:
        return 0, 0
    try:
        pages, _ = _extract_pages(source)
    except Exception:
        # A total of zero reads as not measured, and the gate abstains.
        # Returning "none of them present" would turn an extraction
        # failure into evidence that the source discusses nothing.
        return 0, 0
    document = " ".join(
        str(getattr(page, "text", "") or "") for page in pages).lower()
    if not document:
        return 0, 0
    present = 0
    for term in terms:
        # A prefix match absorbs plurals and simple inflection; a work that
        # discusses "narratives" discusses "narrative".
        stem = re.escape(term[:max(CLAIM_TERM_MIN_LENGTH, len(term) - 2)])
        if re.search(r"\b" + stem, document):
            present += 1
    return present, len(terms)


# The opening states what a work covers; the passages show what it discusses.
# Measured 2026-09-23 over 409 retained documents, a clear geographic scope was
# evident in 63% of document text and in only 3% of openings, which is why a
# judgment made from the opening alone missed that Khan's article is United
# States law throughout while the citing sentence concerned Canada.
MAX_SCOPE_EVIDENCE_CHARACTERS = 1_800
_SCOPE_EVIDENCE_HEADING = "\n\nFurther passages from the same document:\n"


def scope_evidence_block(passages, *, limit: int = MAX_SCOPE_EVIDENCE_CHARACTERS) -> str:
    """Bounded further text from the same document, in document order.

    These passages were retrieved because they match the citation, so their
    agreement with it proves little. Their DISAGREEMENT is what counts: text
    selected to match a claim that still speaks about somewhere else is
    evidence the retrieval could not find what was attributed to the source.
    Ordering is by position, never by score, so the sample does not shift with
    the claim beyond the selection already made.
    """
    if not isinstance(passages, (list, tuple)):
        return ""
    ordered = sorted(
        (p for p in passages if str(getattr(p, "text", "") or "").strip()),
        key=lambda p: (getattr(p, "page_index", None) is None,
                       getattr(p, "page_index", 0) or 0,
                       getattr(p, "character_start", 0) or 0),
    )
    collected: list[str] = []
    used = 0
    for passage in ordered:
        text = re.sub(r"\s+", " ", str(passage.text)).strip()
        if not text:
            continue
        room = limit - used
        if room <= 0:
            break
        collected.append(text[:room])
        used += min(len(text), room) + 1
    return " ".join(collected).strip()
