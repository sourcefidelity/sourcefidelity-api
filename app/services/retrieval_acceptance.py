"""Source-separated complete-source retrieval acceptance measurement."""

from __future__ import annotations

from datetime import datetime
import hashlib
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.services.evidence_package import EvidencePackageV1


LEGACY_ACCEPTANCE_CONTRACT_VERSION = "evidence-package-retrieval-acceptance-v1"
ACCEPTANCE_CONTRACT_VERSION = "evidence-package-retrieval-acceptance-v2"
EXCLUDED_DISPLAY_ROLES = frozenset(
    {"reference_list", "publication_metadata", "page_furniture", "author_biography"}
)


class RetrievalAcceptanceError(ValueError):
    """A result cannot be compared with its frozen acceptance input."""


class MaterialEvidenceFacet(BaseModel):
    """One complete material proposition or explicit constraint used for retrieval."""

    facet_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=3, max_length=2_000)
    text_sha256: str = Field(min_length=64, max_length=64)
    kind: Literal[
        "proposition",
        "scope_constraint",
        "quantity_constraint",
        "comparison_constraint",
        "causal_constraint",
        "temporal_constraint",
        "modality_constraint",
        "negation_constraint",
    ]
    required_for_complete_coverage: bool = True
    derivation: Literal[
        "deterministic_quote",
        "source_blind_model",
        "human_review",
    ]

    @model_validator(mode="after")
    def _text_hash_matches(self):
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("Material facet text hash does not match its text")
        return self


class GoldEvidenceSpan(BaseModel):
    gold_span_id: str = Field(min_length=1, max_length=128)
    page_index: int = Field(ge=0)
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    text: str | None = Field(default=None, min_length=1, max_length=20_000)
    text_sha256: str = Field(min_length=64, max_length=64)
    page_label: str | None = Field(default=None, max_length=50)
    material_facet_ids: list[str] = Field(min_length=1, max_length=16)
    passage_role: Literal["body_prose", "abstract", "citation_notes", "unknown"]
    evidence_role: Literal[
        "direct",
        "distributed",
        "qualification",
        "contradiction",
        "context",
    ] = "direct"

    @model_validator(mode="after")
    def _coordinates_are_ordered(self):
        if self.character_end <= self.character_start:
            raise ValueError("Gold evidence span coordinates are not ordered")
        if len(self.material_facet_ids) != len(set(self.material_facet_ids)):
            raise ValueError("Gold evidence span contains duplicate facet IDs")
        if self.text is not None:
            digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if digest != self.text_sha256:
                raise ValueError("Gold evidence span text hash does not match its text")
        return self


class RetrievalAcceptanceCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    source_group_id: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(min_length=64, max_length=64)
    extracted_text_sha256: str = Field(min_length=64, max_length=64)
    query_sha256: str = Field(min_length=64, max_length=64)
    paper_version_id: str | None = Field(default=None, max_length=255)
    citation_id: str | None = Field(default=None, max_length=128)
    claim_text: str | None = Field(default=None, min_length=1, max_length=10_000)
    claim_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    source_extraction_version: str | None = Field(default=None, max_length=128)
    review_scope: Literal["complete_source"]
    reviewer_provenance: str = Field(min_length=1, max_length=200)
    reviewed_at: datetime | None = None
    acceptance_basis: Literal[
        "deterministic_exact_quote",
        "complete_source_human_review",
        "complete_source_multi_model_consensus",
    ] | None = None
    task_types: list[
        Literal["paraphrase", "quotation", "supplied_locator", "note", "qualifier"]
    ] = Field(min_length=1, max_length=5)
    material_facet_ids: list[str] = Field(min_length=1, max_length=24)
    material_facets: list[MaterialEvidenceFacet] = Field(
        default_factory=list, max_length=24
    )
    gold_spans: list[GoldEvidenceSpan] = Field(min_length=1, max_length=64)
    protected_baseline_passage_ids: list[str] = Field(default_factory=list, max_length=10)
    supplied_locator_page_indices: list[int] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _gold_covers_declared_facets(self):
        declared = set(self.material_facet_ids)
        covered = {
            facet_id for span in self.gold_spans for facet_id in span.material_facet_ids
        }
        if not declared.issubset(covered):
            raise ValueError("Every material facet requires complete-source evidence gold")
        if "supplied_locator" in self.task_types and not self.supplied_locator_page_indices:
            raise ValueError("A supplied-locator case requires expected page indices")
        if self.claim_text is not None:
            digest = hashlib.sha256(self.claim_text.encode("utf-8")).hexdigest()
            if digest != self.claim_sha256:
                raise ValueError("Acceptance claim hash does not match its text")
        if self.material_facets:
            facet_ids = [facet.facet_id for facet in self.material_facets]
            if len(facet_ids) != len(set(facet_ids)):
                raise ValueError("Acceptance case contains duplicate material facets")
            if set(facet_ids) != declared:
                raise ValueError(
                    "Material facet records must match declared material facet IDs"
                )
        return self


class RetrievalAcceptanceCorpus(BaseModel):
    contract_version: Literal[
        "evidence-package-retrieval-acceptance-v1",
        "evidence-package-retrieval-acceptance-v2",
    ] = ACCEPTANCE_CONTRACT_VERSION
    corpus_id: str = Field(min_length=1, max_length=128)
    partition: Literal["development", "acceptance"]
    holdout_included: Literal[False] = False
    source_separated: Literal[True] = True
    cases: list[RetrievalAcceptanceCase] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _case_ids_are_unique(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Acceptance case IDs must be unique")
        if self.contract_version == ACCEPTANCE_CONTRACT_VERSION:
            for case in self.cases:
                if not all(
                    (
                        case.paper_version_id,
                        case.citation_id,
                        case.claim_text,
                        case.claim_sha256,
                        case.source_extraction_version,
                        case.reviewed_at,
                        case.acceptance_basis,
                        case.material_facets,
                    )
                ):
                    raise ValueError(
                        "Retrieval acceptance v2 requires exact claim, facet, "
                        "review, and extraction provenance"
                    )
                if any(span.text is None for span in case.gold_spans):
                    raise ValueError(
                        "Retrieval acceptance v2 requires inspectable gold-span text"
                    )
        return self


class RetrievedAcceptancePassage(BaseModel):
    passage_id: str = Field(min_length=1, max_length=128)
    rank: int = Field(ge=1, le=10_000)
    page_index: int | None = None
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    passage_role: str = Field(min_length=1, max_length=100)
    channels: list[str] = Field(min_length=1, max_length=32)
    protected_baseline: bool = False
    consolidated_source_passage_ids: list[str] = Field(
        default_factory=list, max_length=64
    )


class RetrievalAcceptanceCaseResult(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    source_group_id: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(min_length=64, max_length=64)
    extracted_text_sha256: str = Field(min_length=64, max_length=64)
    query_sha256: str = Field(min_length=64, max_length=64)
    passages: list[RetrievedAcceptancePassage] = Field(default_factory=list)


class RetrievalAcceptanceMetrics(BaseModel):
    contract_version: str = ACCEPTANCE_CONTRACT_VERSION
    case_count: int = Field(ge=1)
    source_count: int = Field(ge=1)
    source_binding_rate: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    material_facet_coverage_at_10: float = Field(ge=0.0, le=1.0)
    exact_quote_retrieval_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    supplied_locator_retrieval_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    excluded_role_slot_count: int = Field(ge=0)
    redundant_slot_pair_count: int = Field(ge=0)
    protected_baseline_retention: float = Field(ge=0.0, le=1.0)
    gates: dict[str, bool]


def retrieval_result_from_package(
    package: EvidencePackageV1,
    *,
    case_id: str,
    source_group_id: str,
    query_sha256: str,
) -> RetrievalAcceptanceCaseResult:
    """Project one package into a stable ranked acceptance result."""
    passages_by_id = {passage.passage_id: passage for passage in package.passages}
    representative_by_raw_id = {
        raw_id: displayed_id
        for displayed_id, raw_ids in package.retrieval.display_consolidations.items()
        for raw_id in raw_ids
    }
    protected = list(package.retrieval.whole_citation_passage_ids)
    allowed: set[str] = set()
    channels: dict[str, set[str]] = {}
    protected_representatives: set[str] = set()
    for passage_id in protected:
        representative = representative_by_raw_id.get(passage_id, passage_id)
        allowed.add(representative)
        channels.setdefault(representative, set()).add("whole_citation_protected")
        protected_representatives.add(representative)
    for selection in package.retrieval.candidate_selections:
        if selection.query_sha256 == query_sha256:
            matching = True
        else:
            matching = query_sha256 in selection.query_facet_sha256s
        if not matching:
            continue
        for item in sorted(selection.passages, key=lambda value: value.rank):
            representative = representative_by_raw_id.get(
                item.passage_id, item.passage_id
            )
            allowed.add(representative)
            channels.setdefault(representative, set()).update(item.channels)
    order = [
        passage_id
        for passage_id in package.retrieval.displayed_passage_ids
        if passage_id in allowed
    ]
    results = []
    for rank, passage_id in enumerate(order, start=1):
        passage = passages_by_id.get(passage_id)
        if passage is None:
            raise RetrievalAcceptanceError(
                "Evidence Package ranking refers to an unknown displayed passage"
            )
        results.append(
            RetrievedAcceptancePassage(
                passage_id=passage_id,
                rank=rank,
                page_index=passage.page_index,
                character_start=passage.character_start,
                character_end=passage.character_end,
                passage_role=passage.passage_role,
                channels=sorted(channels.get(passage_id) or {passage.retrieval_method}),
                protected_baseline=passage_id in protected_representatives,
                consolidated_source_passage_ids=list(
                    package.retrieval.display_consolidations.get(
                        passage_id, [passage_id]
                    )
                ),
            )
        )
    return RetrievalAcceptanceCaseResult(
        case_id=case_id,
        source_group_id=source_group_id,
        content_sha256=package.source_identity.content_sha256,
        extracted_text_sha256=package.extracted_text_sha256,
        query_sha256=query_sha256,
        passages=results,
    )


def evaluate_retrieval_acceptance(
    corpus: RetrievalAcceptanceCorpus,
    results: list[RetrievalAcceptanceCaseResult],
) -> RetrievalAcceptanceMetrics:
    """Measure only hash-compatible complete-source cases."""
    by_id = {result.case_id: result for result in results}
    if len(by_id) != len(results) or set(by_id) != {case.case_id for case in corpus.cases}:
        raise RetrievalAcceptanceError(
            "Acceptance results must cover every frozen case exactly once"
        )

    binding_passes = 0
    gold_total = gold_hit_5 = gold_hit_10 = 0
    facet_total = facet_hit_10 = 0
    quote_total = quote_hits = 0
    locator_total = locator_hits = 0
    excluded_roles = redundant_pairs = 0
    protected_total = protected_retained = 0

    for case in corpus.cases:
        result = by_id[case.case_id]
        binding_ok = (
            result.source_group_id == case.source_group_id
            and result.content_sha256 == case.content_sha256
            and result.extracted_text_sha256 == case.extracted_text_sha256
            and result.query_sha256 == case.query_sha256
        )
        binding_passes += int(binding_ok)
        if not binding_ok:
            raise RetrievalAcceptanceError(
                f"Acceptance result binding mismatch for case {case.case_id}"
            )
        ranked = sorted(result.passages, key=lambda passage: passage.rank)
        if [passage.rank for passage in ranked] != list(range(1, len(ranked) + 1)):
            raise RetrievalAcceptanceError("Acceptance passage ranks must be contiguous")
        top_5, top_10 = ranked[:5], ranked[:10]
        excluded_roles += sum(
            passage.passage_role in EXCLUDED_DISPLAY_ROLES for passage in top_10
        )
        redundant_pairs += _redundant_pair_count(top_10)

        hits_5 = {
            gold.gold_span_id
            for gold in case.gold_spans
            if _span_is_retrieved(gold, top_5)
        }
        hits_10 = {
            gold.gold_span_id
            for gold in case.gold_spans
            if _span_is_retrieved(gold, top_10)
        }
        gold_total += len(case.gold_spans)
        gold_hit_5 += len(hits_5)
        gold_hit_10 += len(hits_10)
        covered_facets = {
            facet_id
            for gold in case.gold_spans
            if gold.gold_span_id in hits_10
            for facet_id in gold.material_facet_ids
        }
        facet_total += len(case.material_facet_ids)
        facet_hit_10 += len(set(case.material_facet_ids) & covered_facets)

        if "quotation" in case.task_types:
            quote_total += 1
            quote_hits += int(len(hits_10) == len(case.gold_spans))
        if "supplied_locator" in case.task_types:
            locator_total += 1
            locator_hits += int(
                any(
                    passage.page_index in case.supplied_locator_page_indices
                    and any(
                        _passage_contains_gold(passage, gold)
                        for gold in case.gold_spans
                    )
                    for passage in top_10
                )
            )

        protected_total += len(case.protected_baseline_passage_ids)
        retained_ids = {
            raw_id
            for passage in top_10
            for raw_id in (
                passage.consolidated_source_passage_ids or [passage.passage_id]
            )
        }
        protected_retained += len(
            set(case.protected_baseline_passage_ids) & retained_ids
        )

    source_binding_rate = binding_passes / len(corpus.cases)
    recall_5 = gold_hit_5 / gold_total
    recall_10 = gold_hit_10 / gold_total
    facet_coverage = facet_hit_10 / facet_total
    protected_retention = (
        protected_retained / protected_total if protected_total else 1.0
    )
    exact_quote_rate = quote_hits / quote_total if quote_total else None
    locator_rate = locator_hits / locator_total if locator_total else None
    gates = {
        "source_binding_100_percent": source_binding_rate == 1.0,
        "zero_excluded_role_slots": excluded_roles == 0,
        "zero_redundant_slots": redundant_pairs == 0,
        "exact_quote_100_percent": exact_quote_rate in {None, 1.0},
        "supplied_locator_100_percent": locator_rate in {None, 1.0},
        "recall_at_5_at_least_90_percent": recall_5 >= 0.90,
        "recall_at_10_at_least_95_percent": recall_10 >= 0.95,
        "facet_coverage_at_10_at_least_90_percent": facet_coverage >= 0.90,
        "protected_baseline_no_regression": protected_retention == 1.0,
    }
    return RetrievalAcceptanceMetrics(
        contract_version=corpus.contract_version,
        case_count=len(corpus.cases),
        source_count=len({case.source_group_id for case in corpus.cases}),
        source_binding_rate=source_binding_rate,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        material_facet_coverage_at_10=facet_coverage,
        exact_quote_retrieval_rate=exact_quote_rate,
        supplied_locator_retrieval_rate=locator_rate,
        excluded_role_slot_count=excluded_roles,
        redundant_slot_pair_count=redundant_pairs,
        protected_baseline_retention=protected_retention,
        gates=gates,
    )


def _span_is_retrieved(
    gold: GoldEvidenceSpan, passages: list[RetrievedAcceptancePassage]
) -> bool:
    return any(_passage_contains_gold(passage, gold) for passage in passages)


def _passage_contains_gold(
    passage: RetrievedAcceptancePassage, gold: GoldEvidenceSpan
) -> bool:
    return (
        passage.page_index == gold.page_index
        and passage.character_start <= gold.character_start
        and passage.character_end >= gold.character_end
    )


def _redundant_pair_count(passages: list[RetrievedAcceptancePassage]) -> int:
    count = 0
    for left, right in combinations(passages, 2):
        if left.page_index != right.page_index:
            continue
        overlap = max(
            0,
            min(left.character_end, right.character_end)
            - max(left.character_start, right.character_start),
        )
        shorter = min(
            left.character_end - left.character_start,
            right.character_end - right.character_start,
        )
        if shorter > 0 and overlap / shorter >= 0.50:
            count += 1
    return count
