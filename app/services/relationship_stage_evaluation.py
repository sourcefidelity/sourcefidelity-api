"""Capability-separated relationship calibration.

This module does not adjudicate a citation.  It builds and validates human
stage labels for three independent questions:

1. which authorized source sentences bear on one fixed facet;
2. what semantic direction those fixed sentences have toward that facet; and
3. whose proposition the selected source material expresses.

The same labels can evaluate local NLI or deterministic source-voice signals
without letting either signal change a verification verdict.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import re
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.schemas import BoundedFieldsMixin, note_bounded

from app.services.relationship_signal import NLIScore
from app.services.source_attribution import source_voice_fields
from app.services.verification_evidence import (
    ConfidenceLevel,
    SourceAttributionRelationEvidence,
    VerificationEvidenceArtifact,
)


STAGE_REVIEW_VERSION = "relationship-stage-review-v1"
STAGE_NLI_VERSION = "fixed-human-evidence-local-nli-v1"


class StageReviewError(ValueError):
    """A stage-review manifest or export violated its fixed-ID contract."""


class StageSourceSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_id: str = Field(min_length=1, max_length=128)
    passage_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2_000)
    voice_role: Literal[
        "unmarked_document_voice",
        "explicit_external_attribution",
        "mixed_or_uncertain",
    ]
    attributed_actor_texts: list[str] = Field(default_factory=list, max_length=8)
    attribution_relations: list[SourceAttributionRelationEvidence] = Field(
        default_factory=list, max_length=8
    )
    discourse_role: Literal["none", "document_reported_actor_scope"]
    discourse_actor_texts: list[str] = Field(default_factory=list, max_length=4)
    discourse_evidence_sentence_ids: list[str] = Field(
        default_factory=list, max_length=4
    )


class StageFacetCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facet_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=50_000)
    allowed_sentence_ids: list[str] = Field(default_factory=list, max_length=52)
    semantic_direction_eligible: bool
    proposition_holder_eligible: bool


CITED_AUTHOR_LABEL_LIMIT = 200


class StageCandidateCase(BoundedFieldsMixin):
    model_config = ConfigDict(extra="forbid")

    unit_alias: str = Field(min_length=1, max_length=200)
    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_text: str = Field(min_length=1, max_length=50_000)
    complete_citation_unit: str = Field(min_length=1, max_length=50_000)
    cited_author_label: str = Field(default="", max_length=CITED_AUTHOR_LABEL_LIMIT)

    @model_validator(mode="before")
    @classmethod
    def _bound_author_label(cls, data):
        """`_citation_author_label` already truncates, so this is not reachable
        through the current producer. It is here because that is precisely the
        fragile arrangement: the bound lived at one call site and the limit was
        written twice. A marker of 328 characters exists in the corpus, so a
        second construction path would fail the run rather than the field. An
        alteration is recorded in `bounded_fields`."""
        if not isinstance(data, dict):
            return data
        value = data.get("cited_author_label")
        if isinstance(value, str) and len(value) > CITED_AUTHOR_LABEL_LIMIT:
            data = dict(data)
            data["cited_author_label"] = value[:CITED_AUTHOR_LABEL_LIMIT]
            note_bounded(data, "cited_author_label")
        return data
    accepted_candidate_relationship: Literal[
        "supports",
        "contradicts",
        "mixed_or_qualified",
        "insufficient_evidence",
        "not_assessed",
    ]
    source_sentences: list[StageSourceSentence] = Field(
        min_length=1, max_length=256
    )
    facets: list[StageFacetCase] = Field(min_length=1, max_length=24)


class RelationshipStageReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["relationship-stage-review-manifest-v1"] = (
        "relationship-stage-review-manifest-v1"
    )
    source_artifact: str = Field(min_length=1, max_length=500)
    source_sha256: str = Field(min_length=64, max_length=64)
    review_scope: Literal["bounded_authorized_evidence"] = (
        "bounded_authorized_evidence"
    )
    holdout_included: Literal[False] = False
    candidates: list[StageCandidateCase] = Field(min_length=1)


class FacetStageReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facet_id: str = Field(min_length=1, max_length=128)
    selection_status: Literal[
        "pending", "evidence_selected", "no_evidence", "uncertain"
    ] = "pending"
    selected_sentence_ids: list[str] = Field(default_factory=list, max_length=52)
    semantic_direction: Literal[
        "pending",
        "supports",
        "contradicts",
        "qualifies",
        "mixed",
        "none",
        "uncertain",
        "not_applicable",
    ] = "pending"
    proposition_holder: Literal[
        "pending",
        "document_author",
        "different_actor",
        "mixed",
        "ambiguous",
        "not_assessed_no_relevant_evidence",
        "not_applicable",
    ] = "pending"
    confidence: ConfidenceLevel = ConfidenceLevel.NONE
    notes: str = Field(default="", max_length=2_000)


class CandidateStageReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_alias: str = Field(min_length=1, max_length=200)
    candidate_id: str = Field(min_length=1, max_length=128)
    facets: list[FacetStageReview] = Field(min_length=1, max_length=24)


class RelationshipStageReviewExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["relationship-stage-review-export-v1"] = (
        "relationship-stage-review-export-v1"
    )
    source_sha256: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: datetime | None = None
    review_scope: Literal["bounded_authorized_evidence"] = (
        "bounded_authorized_evidence"
    )
    candidates: list[CandidateStageReview] = Field(min_length=1)


class LocalHolderProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "human_review_required", "not_applicable"]
    relation: Literal[
        "document_author",
        "different_actor",
        "mixed",
        "ambiguous",
        "not_applicable",
    ]
    confidence: ConfidenceLevel
    evidence_sentence_ids: list[str] = Field(default_factory=list, max_length=56)
    actor_texts: list[str] = Field(default_factory=list, max_length=8)
    reason_code: Literal[
        "unmarked_document_voice",
        "explicit_cited_author_voice",
        "explicit_different_actor",
        "document_reported_actor_scope",
        "mixed_source_voice",
        "source_voice_uncertain",
        "evidence_selection_required",
        "not_source_attribution_facet",
    ]
    limitations: list[str] = Field(default_factory=list, max_length=5)


class FixedPairNLISignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_alias: str
    candidate_id: str
    facet_id: str
    reference_direction: Literal[
        "supports", "contradicts", "qualifies", "mixed", "none", "uncertain"
    ]
    predicted_class: Literal["supports", "contradicts", "neutral"]
    entailment: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)


class FixedPairNLIResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["fixed-pair-nli-ablation-v1"] = (
        "fixed-pair-nli-ablation-v1"
    )
    signal_version: Literal["fixed-human-evidence-local-nli-v1"] = (
        "fixed-human-evidence-local-nli-v1"
    )
    model_id: str
    model_revision: str
    reference_kind: Literal["human_stage_gold", "non_gold_control_proposal"]
    processing_boundary: Literal["local"] = "local"
    decision_applied: Literal[False] = False
    signals: list[FixedPairNLISignal] = Field(default_factory=list)
    metrics: dict
    limitations: list[str] = Field(default_factory=list)


class FixedPairScorer(Protocol):
    model_id: str
    model_revision: str

    def score_pairs(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[NLIScore]: ...


def build_stage_review_manifest(
    rows: list[dict], *, source_artifact: str, source_bytes: bytes
) -> RelationshipStageReviewManifest:
    """Build a fixed private review manifest from reviewed development rows."""
    cases: list[StageCandidateCase] = []
    seen_candidates: set[str] = set()
    for row in rows:
        artifact = VerificationEvidenceArtifact.model_validate(row["artifact"])
        human = row["independent_review"]["candidate_reviews"]
        candidates = {
            candidate.candidate_id: candidate
            for candidate in artifact.verification_candidates.candidates
            if candidate.relationship_eligible
        }
        sentences = {
            sentence.sentence_id: sentence
            for sentence in artifact.facet_evidence_foundation.source_sentences
        }
        for bundle in artifact.facet_evidence_foundation.candidate_bundles:
            candidate = candidates.get(bundle.candidate_id)
            review = human.get(bundle.candidate_id)
            if candidate is None or review is None:
                raise StageReviewError("Candidate and accepted review IDs must match")
            if bundle.candidate_id in seen_candidates:
                raise StageReviewError("Candidate IDs must be unique across the manifest")
            seen_candidates.add(bundle.candidate_id)
            allowed_ids = list(
                dict.fromkeys(
                    [
                        *bundle.evidence_sentence_ids,
                        *bundle.source_discourse_sentence_ids,
                    ]
                )
            )
            missing = [item for item in allowed_ids if item not in sentences]
            if missing:
                raise StageReviewError("A facet bundle references a missing sentence")
            facets = [
                StageFacetCase(
                    facet_id=facet.facet_id,
                    kind=facet.kind,
                    text=facet.text,
                    allowed_sentence_ids=allowed_ids,
                    semantic_direction_eligible=(
                        facet.material_to_aggregate
                        and facet.kind != "source_attribution"
                    ),
                    proposition_holder_eligible=(
                        facet.material_to_aggregate
                        and facet.kind == "source_attribution"
                    ),
                )
                for facet in bundle.facets
                if facet.material_to_aggregate
            ]
            if not facets:
                raise StageReviewError("Each candidate requires a material review facet")
            cases.append(
                StageCandidateCase(
                    unit_alias=row["unit_alias"],
                    candidate_id=bundle.candidate_id,
                    candidate_text=candidate.text,
                    complete_citation_unit=artifact.claim.text,
                    cited_author_label=_citation_author_label(
                        artifact.claim.citation_marker
                    ),
                    accepted_candidate_relationship=review["final_relationship"],
                    source_sentences=[
                        _stage_source_sentence(sentences[sentence_id])
                        for sentence_id in allowed_ids
                    ],
                    facets=facets,
                )
            )
    return RelationshipStageReviewManifest(
        source_artifact=source_artifact,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        candidates=cases,
    )


def _stage_source_sentence(sentence) -> StageSourceSentence:
    voice = source_voice_fields(sentence.text)
    return StageSourceSentence(
        sentence_id=sentence.sentence_id,
        passage_id=sentence.passage_id,
        text=sentence.text,
        voice_role=voice["voice_role"],
        attributed_actor_texts=voice["attributed_actor_texts"],
        attribution_relations=voice["attribution_relations"],
        discourse_role=sentence.discourse_role,
        discourse_actor_texts=sentence.discourse_actor_texts,
        discourse_evidence_sentence_ids=sentence.discourse_evidence_sentence_ids,
    )


def blank_stage_review_export(
    manifest: RelationshipStageReviewManifest, *, reviewer: str
) -> RelationshipStageReviewExport:
    return RelationshipStageReviewExport(
        source_sha256=manifest.source_sha256,
        reviewer=reviewer,
        candidates=[
            CandidateStageReview(
                unit_alias=case.unit_alias,
                candidate_id=case.candidate_id,
                facets=[
                    FacetStageReview(
                        facet_id=facet.facet_id,
                        semantic_direction=(
                            "pending"
                            if facet.semantic_direction_eligible
                            else "not_applicable"
                        ),
                        proposition_holder=(
                            "pending"
                            if facet.proposition_holder_eligible
                            else "not_applicable"
                        ),
                    )
                    for facet in case.facets
                ],
            )
            for case in manifest.candidates
        ],
    )


def validate_stage_review_export(
    manifest: RelationshipStageReviewManifest,
    review: RelationshipStageReviewExport,
    *,
    require_complete: bool = False,
) -> RelationshipStageReviewExport:
    """Validate exact candidates, facets, authorized evidence and task boundaries."""
    if review.source_sha256 != manifest.source_sha256:
        raise StageReviewError("The review targets a different source snapshot")
    cases = {(item.unit_alias, item.candidate_id): item for item in manifest.candidates}
    reviewed = {(item.unit_alias, item.candidate_id): item for item in review.candidates}
    if len(reviewed) != len(review.candidates) or set(reviewed) != set(cases):
        raise StageReviewError("Review candidate IDs must exactly match the manifest")

    for key, case in cases.items():
        decision = reviewed[key]
        facets = {item.facet_id: item for item in case.facets}
        decisions = {item.facet_id: item for item in decision.facets}
        if len(decisions) != len(decision.facets) or set(decisions) != set(facets):
            raise StageReviewError("Review facet IDs must exactly match the manifest")
        for facet_id, facet in facets.items():
            item = decisions[facet_id]
            selected = set(item.selected_sentence_ids)
            if len(selected) != len(item.selected_sentence_ids):
                raise StageReviewError("Selected sentence IDs must be unique")
            if not selected.issubset(set(facet.allowed_sentence_ids)):
                raise StageReviewError("A review selected unauthorized evidence")
            if item.selection_status == "evidence_selected" and not selected:
                raise StageReviewError("Evidence-selected requires a sentence ID")
            if item.selection_status == "no_evidence" and selected:
                raise StageReviewError("No-evidence cannot carry sentence IDs")
            if facet.semantic_direction_eligible:
                if item.proposition_holder != "not_applicable":
                    raise StageReviewError("Semantic facets cannot receive holder labels")
                if item.semantic_direction in {
                    "supports", "contradicts", "qualifies", "mixed"
                } and not selected:
                    raise StageReviewError("Evidentiary direction requires selected evidence")
                if item.semantic_direction == "none" and selected:
                    raise StageReviewError("None direction cannot retain evidence IDs")
                if item.selection_status == "no_evidence" and item.semantic_direction not in {
                    "none", "pending"
                }:
                    raise StageReviewError("No-evidence selection requires none direction")
                if item.selection_status == "uncertain" and item.semantic_direction not in {
                    "uncertain", "pending"
                }:
                    raise StageReviewError("Uncertain selection requires uncertain direction")
                if item.selection_status == "evidence_selected" and item.semantic_direction == "none":
                    raise StageReviewError("Selected material evidence cannot have none direction")
                if require_complete and (
                    item.selection_status == "pending"
                    or item.semantic_direction == "pending"
                ):
                    raise StageReviewError("A required semantic facet is incomplete")
            else:
                if item.semantic_direction != "not_applicable":
                    raise StageReviewError("Holder facets cannot receive semantic direction")
                if require_complete and (
                    item.selection_status == "pending"
                    or item.proposition_holder == "pending"
                ):
                    raise StageReviewError("A required holder facet is incomplete")
                if item.proposition_holder in {
                    "document_author", "different_actor", "mixed"
                } and not selected:
                    raise StageReviewError("A resolved holder requires selected evidence")
                if item.selection_status == "no_evidence" and (
                    selected
                    or item.proposition_holder
                    != "not_assessed_no_relevant_evidence"
                ):
                    raise StageReviewError(
                        "Holder no-evidence requires the explicit not-assessed state"
                    )
                if (
                    item.proposition_holder
                    == "not_assessed_no_relevant_evidence"
                    and item.selection_status != "no_evidence"
                ):
                    raise StageReviewError(
                        "No-relevant-evidence holder state requires no-evidence selection"
                    )
    return review


def propose_local_proposition_holder(
    case: StageCandidateCase,
    facet: StageFacetCase,
    selected_sentence_ids: Sequence[str],
) -> LocalHolderProposal:
    """Return only high-precision local source-voice proposals or human routing."""
    if not facet.proposition_holder_eligible:
        return LocalHolderProposal(
            status="not_applicable",
            relation="not_applicable",
            confidence=ConfidenceLevel.NONE,
            reason_code="not_source_attribution_facet",
        )
    if not selected_sentence_ids:
        return _holder_human("evidence_selection_required")
    sentences = {item.sentence_id: item for item in case.source_sentences}
    selected = [sentences[item] for item in selected_sentence_ids if item in sentences]
    if len(selected) != len(set(selected_sentence_ids)):
        return _holder_human("source_voice_uncertain")

    cited_actors: set[str] = set()
    external_actors: set[str] = set()
    evidence_ids = list(dict.fromkeys(selected_sentence_ids))
    has_uncertain_voice = False
    has_unmarked = False
    has_external_scope = False
    for sentence in selected:
        if sentence.voice_role == "mixed_or_uncertain":
            has_uncertain_voice = True
        if sentence.voice_role == "unmarked_document_voice":
            has_unmarked = True
        actors = [
            *sentence.attributed_actor_texts,
            *sentence.discourse_actor_texts,
        ]
        for actor in actors:
            if _actor_matches_cited_author(actor, case.cited_author_label):
                cited_actors.add(actor)
            elif _informative_actor(actor):
                external_actors.add(actor)
        if sentence.discourse_role == "document_reported_actor_scope":
            has_external_scope = bool(external_actors)
            evidence_ids.extend(sentence.discourse_evidence_sentence_ids)

    if has_uncertain_voice or (cited_actors and external_actors):
        return LocalHolderProposal(
            status="human_review_required",
            relation="mixed" if cited_actors and external_actors else "ambiguous",
            confidence=ConfidenceLevel.LOW,
            evidence_sentence_ids=list(dict.fromkeys(evidence_ids)),
            actor_texts=sorted(cited_actors | external_actors)[:8],
            reason_code="mixed_source_voice" if cited_actors and external_actors else "source_voice_uncertain",
            limitations=["Source voice requires human review before relationship aggregation."],
        )
    if external_actors:
        return LocalHolderProposal(
            status="resolved",
            relation="different_actor",
            confidence=ConfidenceLevel.HIGH,
            evidence_sentence_ids=list(dict.fromkeys(evidence_ids)),
            actor_texts=sorted(external_actors)[:8],
            reason_code=(
                "document_reported_actor_scope"
                if has_external_scope
                else "explicit_different_actor"
            ),
            limitations=["This identifies proposition holder, not student intent or misconduct."],
        )
    if cited_actors:
        return LocalHolderProposal(
            status="resolved",
            relation="document_author",
            confidence=ConfidenceLevel.HIGH,
            evidence_sentence_ids=evidence_ids,
            actor_texts=sorted(cited_actors)[:8],
            reason_code="explicit_cited_author_voice",
        )
    if has_unmarked and all(
        sentence.voice_role == "unmarked_document_voice"
        and sentence.discourse_role == "none"
        for sentence in selected
    ):
        return LocalHolderProposal(
            status="resolved",
            relation="document_author",
            confidence=ConfidenceLevel.MEDIUM,
            evidence_sentence_ids=evidence_ids,
            reason_code="unmarked_document_voice",
            limitations=[
                "Unmarked scholarly prose is treated as document voice only within the selected bounded evidence."
            ],
        )
    return _holder_human("source_voice_uncertain", evidence_ids=evidence_ids)


def run_fixed_pair_nli_ablation(
    manifest: RelationshipStageReviewManifest,
    review: RelationshipStageReviewExport,
    scorer: FixedPairScorer,
    *,
    reference_kind: Literal[
        "human_stage_gold", "non_gold_control_proposal"
    ] = "human_stage_gold",
) -> FixedPairNLIResult:
    """Score only human-selected semantic pairs; never source-attribution facets."""
    validate_stage_review_export(manifest, review, require_complete=True)
    cases = {(item.unit_alias, item.candidate_id): item for item in manifest.candidates}
    premises: list[str] = []
    hypotheses: list[str] = []
    metadata = []
    for candidate_review in review.candidates:
        case = cases[(candidate_review.unit_alias, candidate_review.candidate_id)]
        facets = {item.facet_id: item for item in case.facets}
        sentences = {item.sentence_id: item for item in case.source_sentences}
        for decision in candidate_review.facets:
            facet = facets[decision.facet_id]
            if not facet.semantic_direction_eligible:
                continue
            if decision.selection_status != "evidence_selected":
                continue
            if decision.semantic_direction not in {
                "supports", "contradicts", "qualifies", "mixed", "none", "uncertain"
            }:
                continue
            premises.append(
                " ".join(
                    sentences[sentence_id].text
                    for sentence_id in decision.selected_sentence_ids
                )
            )
            hypotheses.append(facet.text)
            metadata.append((candidate_review, decision))
    scores = scorer.score_pairs(premises, hypotheses)
    if len(scores) != len(metadata):
        raise StageReviewError("The local NLI scorer returned an invalid result count")

    signals = []
    pairs = []
    for (candidate_review, decision), score in zip(metadata, scores):
        predicted = max(
            ("supports", score.entailment),
            ("neutral", score.neutral),
            ("contradicts", score.contradiction),
            key=lambda item: item[1],
        )[0]
        gold = decision.semantic_direction
        signals.append(
            FixedPairNLISignal(
                unit_alias=candidate_review.unit_alias,
                candidate_id=candidate_review.candidate_id,
                facet_id=decision.facet_id,
                reference_direction=gold,
                predicted_class=predicted,
                entailment=score.entailment,
                neutral=score.neutral,
                contradiction=score.contradiction,
            )
        )
        gold_class = {
            "supports": "supports",
            "contradicts": "contradicts",
            "qualifies": "neutral",
            "mixed": "neutral",
            "none": "neutral",
            "uncertain": "neutral",
        }[gold]
        pairs.append((gold_class, predicted))
    return FixedPairNLIResult(
        model_id=scorer.model_id,
        model_revision=scorer.model_revision,
        reference_kind=reference_kind,
        signals=signals,
        metrics=_classification_metrics(pairs),
        limitations=[
            "Development-stage fixed pairs, not holdout accuracy.",
            "Qualifies, mixed, none and uncertain are grouped as NLI-neutral only for this diagnostic.",
            "Source-attribution facets are excluded because NLI cannot establish proposition holder.",
            *(
                ["Agreement with a non-gold control proposal is not an accuracy measure."]
                if reference_kind == "non_gold_control_proposal"
                else []
            ),
        ],
    )


def _classification_metrics(pairs: list[tuple[str, str]]) -> dict:
    labels = ("supports", "contradicts", "neutral")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for gold, predicted in pairs:
        confusion[gold][predicted] += 1
    return {
        "n": len(pairs),
        "exact_agreement": sum(gold == predicted for gold, predicted in pairs),
        "exact_agreement_rate": (
            round(sum(gold == predicted for gold, predicted in pairs) / len(pairs), 4)
            if pairs
            else None
        ),
        "confusion": {
            label: dict(confusion[label]) for label in labels if confusion[label]
        },
    }


def _holder_human(reason_code, *, evidence_ids=()):
    return LocalHolderProposal(
        status="human_review_required",
        relation="ambiguous",
        confidence=ConfidenceLevel.NONE,
        evidence_sentence_ids=list(evidence_ids),
        reason_code=reason_code,
        limitations=["Source voice requires human review before relationship aggregation."],
    )


def _citation_author_label(marker):
    value = (marker or "").strip().strip("()[]")
    value = value.split(";", 1)[0]
    value = re.split(r",?\s+(?:19|20)\d{2}[a-z]?\b", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", value.strip(" ,"))[:CITED_AUTHOR_LABEL_LIMIT]


def _actor_matches_cited_author(actor, cited_author_label):
    ignored = {"et", "al", "and"}
    actor_tokens = {
        item for item in re.findall(r"[a-z]+", (actor or "").casefold()) if item not in ignored
    }
    cited_tokens = {
        item
        for item in re.findall(r"[a-z]+", (cited_author_label or "").casefold())
        if item not in ignored
    }
    return bool(actor_tokens & cited_tokens)


def _informative_actor(actor):
    ignored = {
        "he", "her", "hers", "his", "i", "it", "its", "she", "their",
        "theirs", "them", "they", "to", "we", "our", "ours", "you", "your",
    }
    return (actor or "").casefold().strip(" .") not in ignored


def dump_manifest_json(manifest: RelationshipStageReviewManifest) -> str:
    """Stable serialization helper for ignored private calibration artifacts."""
    return json.dumps(manifest.model_dump(mode="json"), indent=2)
