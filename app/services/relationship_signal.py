"""Bounded local claim/passage relationship signal.

The specialist NLI model is evidence, not the product's final judge. Neutral
does not establish topical mismatch, and model disagreement or weak margins
remain insufficient evidence. Application code owns thresholds, aggregation,
coverage ceilings, and final-verdict gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Protocol, Sequence

from app.config import settings
from app.services.verification_evidence import (
    ClaimRelationshipEvidence,
    ConfidenceLevel,
    CoverageLevel,
    PassageRelationshipScore,
    RelationshipStatus,
    VerificationEvidenceArtifact,
    VerificationVerdict,
)


logger = logging.getLogger(__name__)

RELATIONSHIP_SIGNAL_VERSION = "local-nli-v1"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
MAX_RELATIONSHIP_PASSAGES = 5
MAX_RELATIONSHIP_TEXT_CHARACTERS = 2_000


class RelationshipSignalError(RuntimeError):
    """The configured local relationship signal could not run safely."""


@dataclass(frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float

    def __post_init__(self) -> None:
        values = (self.entailment, self.neutral, self.contradiction)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("NLI probabilities must be between zero and one")
        if abs(sum(values) - 1.0) > 0.02:
            raise ValueError("NLI probabilities must sum to approximately one")


class LocalNLIScorer(Protocol):
    model_id: str
    model_revision: str

    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> list[NLIScore]: ...


@dataclass(frozen=True)
class RelationshipPolicy:
    support_threshold: float = 0.82
    support_margin: float = 0.20
    contradiction_threshold: float = 0.90
    contradiction_margin: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.support_threshold,
            self.support_margin,
            self.contradiction_threshold,
            self.contradiction_margin,
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Relationship policy values must be between zero and one")


class TransformersNLIScorer:
    """Lazy, local-only Hugging Face NLI adapter with no remote fallback."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_NLI_MODEL,
        model_revision: str = DEFAULT_NLI_REVISION,
        local_files_only: bool = True,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Unsupported local relationship model device")
        if batch_size < 1 or batch_size > 32:
            raise ValueError("Relationship model batch size must be between 1 and 32")
        self.model_id = model_id
        self.model_revision = model_revision
        self.local_files_only = local_files_only
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._label_indices: dict[str, int] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RelationshipSignalError(
                "Local NLI dependencies are not installed; install requirements-nli.txt"
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
            )
            model.to(self.device)
            model.eval()
        except Exception as exc:
            raise RelationshipSignalError(
                "The pinned local NLI model could not be loaded"
            ) from exc
        self._label_indices = _resolve_label_indices(model.config.id2label, self.model_id)
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> list[NLIScore]:
        if len(premises) != len(hypotheses):
            raise ValueError("Every premise must have one hypothesis")
        if not premises:
            return []
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._label_indices is not None

        output: list[NLIScore] = []
        for start in range(0, len(premises), self.batch_size):
            premise_batch = [
                value[:MAX_RELATIONSHIP_TEXT_CHARACTERS]
                for value in premises[start:start + self.batch_size]
            ]
            hypothesis_batch = [
                value[:MAX_RELATIONSHIP_TEXT_CHARACTERS]
                for value in hypotheses[start:start + self.batch_size]
            ]
            encoded = self._tokenizer(
                premise_batch,
                hypothesis_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probabilities:
                output.append(
                    NLIScore(
                        entailment=float(row[self._label_indices["entailment"]]),
                        neutral=float(row[self._label_indices["neutral"]]),
                        contradiction=float(row[self._label_indices["contradiction"]]),
                    )
                )
        return output


def assess_local_relationship(
    artifact: VerificationEvidenceArtifact,
    *,
    scorer: LocalNLIScorer | None = None,
    policy: RelationshipPolicy | None = None,
) -> VerificationEvidenceArtifact:
    """Attach a bounded relationship signal without overstating final verdicts."""
    if artifact.claim.granularity != "atomic_claim":
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="atomic_claim_required",
            reason_code="relationship_requires_atomic_claim",
            limitation="Relationship assessment requires an eligible atomic claim.",
        )
    if artifact.claim.antecedent_dependencies:
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="context_dependent_claim_requires_structured_judgment",
            reason_code="relationship_context_dependency_requires_structured_judgment",
            limitation=(
                "The local NLI pair interface is not calibrated for structured "
                "antecedent context; use the bounded structured judgment path."
            ),
        )
    if artifact.coverage.level is CoverageLevel.UNAVAILABLE:
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="source_text_unavailable",
            reason_code="relationship_source_text_unavailable",
            limitation="No usable authorized source text was available.",
        )
    if not artifact.passages:
        return _abstain(
            artifact,
            status=RelationshipStatus.INSUFFICIENT_EVIDENCE,
            method="no_candidate_passage",
            reason_code="relationship_insufficient_no_passage",
            limitation=(
                "No candidate passage was located; absence is not contradiction or "
                "topical mismatch."
            ),
        )

    exact = [
        passage
        for passage in artifact.passages
        if artifact.claim.claim_type == "quotation"
        and passage.retrieval_method == "exact_quotation"
    ]
    if exact:
        relationship = ClaimRelationshipEvidence(
            status=RelationshipStatus.SUPPORTS,
            confidence=ConfidenceLevel.HIGH,
            method="deterministic_exact_quotation",
            signal_version=RELATIONSHIP_SIGNAL_VERSION,
            passage_ids=[passage.passage_id for passage in exact],
            limitations=[
                "Exact text presence verifies the quoted wording, not every surrounding interpretation."
            ],
        )
        return _updated_artifact(
            artifact,
            relationship=relationship,
            verdict=VerificationVerdict.CONSISTENT,
            reason_code="exact_quotation_present_in_authorized_source",
        )

    active_scorer = scorer or get_configured_local_nli_scorer()
    if active_scorer is None:
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="local_nli_disabled",
            reason_code="relationship_local_signal_unavailable",
            limitation="The optional local relationship signal is not enabled.",
        )

    passages = artifact.passages[:MAX_RELATIONSHIP_PASSAGES]
    try:
        scores = active_scorer.score_pairs(
            [passage.text for passage in passages],
            [artifact.claim.text] * len(passages),
        )
    except Exception as exc:
        logger.warning("Local relationship signal failed: %s", exc)
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="local_nli_failed",
            reason_code="relationship_local_signal_failed",
            limitation="The local relationship signal failed safely.",
        )
    if len(scores) != len(passages):
        return _abstain(
            artifact,
            status=RelationshipStatus.NOT_ASSESSED,
            method="local_nli_invalid_output",
            reason_code="relationship_local_signal_invalid",
            limitation="The local relationship signal returned an invalid result count.",
        )

    relationship = _aggregate_nli(
        passages=passages,
        scores=scores,
        scorer=active_scorer,
        policy=policy or RelationshipPolicy(),
    )
    reason_code = {
        RelationshipStatus.SUPPORTS: "local_nli_support_candidate",
        RelationshipStatus.CONTRADICTS: "local_nli_contradiction_candidate",
        RelationshipStatus.INSUFFICIENT_EVIDENCE: "local_nli_insufficient_evidence",
        RelationshipStatus.UNRELATED: "local_nli_unrelated_candidate",
        RelationshipStatus.NOT_ASSESSED: "relationship_not_assessed",
    }[relationship.status]
    # Until task-specific calibration and bounded adjudication are complete,
    # NLI cannot independently produce an adverse or affirmative final verdict.
    return _updated_artifact(
        artifact,
        relationship=relationship,
        verdict=VerificationVerdict.INCONCLUSIVE,
        reason_code=reason_code,
    )


def _aggregate_nli(
    *,
    passages,
    scores: list[NLIScore],
    scorer: LocalNLIScorer,
    policy: RelationshipPolicy,
) -> ClaimRelationshipEvidence:
    passage_scores = [
        PassageRelationshipScore(
            passage_id=passage.passage_id,
            entailment=round(score.entailment, 6),
            neutral=round(score.neutral, 6),
            contradiction=round(score.contradiction, 6),
        )
        for passage, score in zip(passages, scores)
    ]
    supports = [
        index
        for index, score in enumerate(scores)
        if score.entailment >= policy.support_threshold
        and score.entailment - max(score.neutral, score.contradiction)
        >= policy.support_margin
    ]
    contradictions = [
        index
        for index, score in enumerate(scores)
        if score.contradiction >= policy.contradiction_threshold
        and score.contradiction - max(score.entailment, score.neutral)
        >= policy.contradiction_margin
    ]

    if supports and contradictions:
        status = RelationshipStatus.INSUFFICIENT_EVIDENCE
        confidence = ConfidenceLevel.LOW
        selected = sorted(set(supports + contradictions))
        limitations = [
            "Candidate passages produced conflicting local signals; bounded adjudication is required."
        ]
    elif contradictions:
        status = RelationshipStatus.CONTRADICTS
        confidence = ConfidenceLevel.MEDIUM
        selected = contradictions
        limitations = [
            "Local contradiction is a candidate signal and cannot alone produce an adverse final verdict."
        ]
    elif supports:
        status = RelationshipStatus.SUPPORTS
        confidence = ConfidenceLevel.MEDIUM
        selected = supports
        limitations = [
            "Local entailment is a candidate signal pending task-specific calibration and adjudication."
        ]
    else:
        status = RelationshipStatus.INSUFFICIENT_EVIDENCE
        confidence = ConfidenceLevel.LOW
        selected = list(range(len(passages)))
        limitations = [
            "NLI-neutral or weak scores do not distinguish unrelated content from insufficient passage evidence."
        ]

    return ClaimRelationshipEvidence(
        status=status,
        confidence=confidence,
        method="local_deberta_nli_candidate",
        model_id=scorer.model_id,
        model_revision=scorer.model_revision,
        signal_version=RELATIONSHIP_SIGNAL_VERSION,
        passage_ids=[passages[index].passage_id for index in selected],
        passage_scores=passage_scores,
        limitations=limitations,
    )


def _abstain(
    artifact: VerificationEvidenceArtifact,
    *,
    status: RelationshipStatus,
    method: str,
    reason_code: str,
    limitation: str,
) -> VerificationEvidenceArtifact:
    relationship = ClaimRelationshipEvidence(
        status=status,
        confidence=ConfidenceLevel.NONE,
        method=method,
        signal_version=RELATIONSHIP_SIGNAL_VERSION,
        passage_ids=[passage.passage_id for passage in artifact.passages],
        limitations=[limitation],
    )
    verdict = (
        VerificationVerdict.NOT_ASSESSED
        if status is RelationshipStatus.NOT_ASSESSED
        and artifact.coverage.level is CoverageLevel.UNAVAILABLE
        else VerificationVerdict.INCONCLUSIVE
    )
    return _updated_artifact(
        artifact,
        relationship=relationship,
        verdict=verdict,
        reason_code=reason_code,
    )


def _updated_artifact(
    artifact: VerificationEvidenceArtifact,
    *,
    relationship: ClaimRelationshipEvidence,
    verdict: VerificationVerdict,
    reason_code: str,
) -> VerificationEvidenceArtifact:
    replaced = {
        "candidate_passages_located_relation_not_assessed",
        "no_passage_located_in_available_evidence",
        "relationship_not_assessed",
    }
    reason_codes = [code for code in artifact.reason_codes if code not in replaced]
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)
    return artifact.model_copy(
        update={
            "relationship": relationship,
            "verdict": verdict,
            "reason_codes": reason_codes,
        }
    )


def _resolve_label_indices(id2label, model_id: str) -> dict[str, int]:
    aliases = {
        "entailment": "entailment",
        "entails": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "contradicts": "contradiction",
    }
    resolved: dict[str, int] = {}
    for raw_index, raw_label in dict(id2label or {}).items():
        label = aliases.get(str(raw_label).strip().casefold())
        if label:
            resolved[label] = int(raw_index)
    if set(resolved) == {"entailment", "neutral", "contradiction"}:
        return resolved
    if model_id == DEFAULT_NLI_MODEL:
        # The official model card specifies this exact logit order.
        return {"entailment": 0, "neutral": 1, "contradiction": 2}
    raise RelationshipSignalError(
        "Configured NLI model does not expose a recognized three-way label mapping"
    )


@lru_cache(maxsize=1)
def get_configured_local_nli_scorer() -> TransformersNLIScorer | None:
    if settings.RELATIONSHIP_SIGNAL_BACKEND == "disabled":
        return None
    return TransformersNLIScorer(
        model_id=settings.RELATIONSHIP_MODEL_NAME,
        model_revision=settings.RELATIONSHIP_MODEL_REVISION,
        local_files_only=settings.RELATIONSHIP_MODEL_LOCAL_FILES_ONLY,
        device=settings.RELATIONSHIP_MODEL_DEVICE,
        batch_size=settings.RELATIONSHIP_MODEL_BATCH_SIZE,
    )
