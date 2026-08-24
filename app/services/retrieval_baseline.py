"""Build a protected retrieval baseline from reviewed relationship artifacts.

The builder intentionally consumes only the development relationship packet.
It converts exact human-selected sentence IDs to their containing authorized
passage IDs, preserving source-file hashes as source-group boundaries.  It
does not infer negative labels from passages the reviewer did not select.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from app.services.relationship_stage_evaluation import (
    RelationshipStageReviewExport,
    RelationshipStageReviewManifest,
    validate_stage_review_export,
)
from app.services.retrieval_ablation import (
    RankedRetrievalPassage,
    RetrievalAblationCase,
    RetrievalAblationError,
    RetrievalAblationManifest,
    RetrievalArm,
    RetrievalFacetGold,
    RetrievalInputSnapshot,
)


def build_reviewed_retrieval_baseline(
    stage_manifest: RelationshipStageReviewManifest,
    stage_review: RelationshipStageReviewExport,
    development_artifact: dict[str, Any],
    *,
    input_snapshots: Iterable[RetrievalInputSnapshot] = (),
) -> RetrievalAblationManifest:
    """Create an evidence-positive, source-grouped baseline manifest."""
    validate_stage_review_export(stage_manifest, stage_review, require_complete=True)

    results = development_artifact.get("results")
    if not isinstance(results, list):
        raise RetrievalAblationError("Development artifact must contain a results list")
    by_alias = _unique_by_key(results, "unit_alias", "development result")
    review_by_candidate = {
        (item.unit_alias, item.candidate_id): item
        for item in stage_review.candidates
    }

    cases: list[RetrievalAblationCase] = []
    for candidate in stage_manifest.candidates:
        review = review_by_candidate[(candidate.unit_alias, candidate.candidate_id)]
        reviewed_facets = {item.facet_id: item for item in review.facets}
        facet_gold: list[RetrievalFacetGold] = []
        sentence_to_passage = {
            item.sentence_id: item.passage_id for item in candidate.source_sentences
        }
        for facet in candidate.facets:
            decision = reviewed_facets[facet.facet_id]
            if not facet.semantic_direction_eligible:
                continue
            if decision.selection_status != "evidence_selected":
                continue
            passage_ids = _ordered_unique(
                sentence_to_passage[sentence_id]
                for sentence_id in decision.selected_sentence_ids
            )
            facet_gold.append(
                RetrievalFacetGold(
                    facet_id=facet.facet_id,
                    useful_passage_ids=passage_ids,
                )
            )
        if not facet_gold:
            continue

        result = by_alias.get(candidate.unit_alias)
        if result is None:
            raise RetrievalAblationError(
                f"Missing development result for {candidate.unit_alias}"
            )
        artifact = result.get("artifact")
        if not isinstance(artifact, dict):
            raise RetrievalAblationError("Development result is missing its artifact")
        source_identity = artifact.get("source_identity") or {}
        source_group_id = source_identity.get("content_sha256")
        if not isinstance(source_group_id, str) or len(source_group_id) != 64:
            raise RetrievalAblationError(
                f"Missing source content hash for {candidate.unit_alias}"
            )

        retrieval = artifact.get("candidate_passage_retrieval") or {}
        selections = retrieval.get("selections") or []
        selection_by_candidate = _unique_by_key(
            selections, "candidate_id", "candidate passage selection"
        )
        selection = selection_by_candidate.get(candidate.candidate_id)
        if selection is None:
            raise RetrievalAblationError(
                f"Missing passage selection for {candidate.candidate_id}"
            )
        passage_records = _unique_by_key(
            artifact.get("passages") or [], "passage_id", "authorized passage"
        )
        ranked_records = []
        for ranked in selection.get("passages") or []:
            passage_id = ranked.get("passage_id")
            passage = passage_records.get(passage_id)
            if passage is None:
                raise RetrievalAblationError(
                    f"Ranked passage is not in the artifact: {passage_id}"
                )
            ranked_records.append((ranked, passage))

        useful_passage_ids = _ordered_unique(
            passage_id
            for facet in facet_gold
            for passage_id in facet.useful_passage_ids
        )
        ranked_ids = {item[0].get("passage_id") for item in ranked_records}
        missing_gold = set(useful_passage_ids) - ranked_ids
        if missing_gold:
            raise RetrievalAblationError(
                "Human-selected passages are absent from the preserved baseline: "
                f"{sorted(missing_gold)}"
            )

        overlap_groups = _overlap_groups(
            [passage for _, passage in ranked_records]
        )
        baseline_passages = []
        for expected_rank, (ranked, passage) in enumerate(ranked_records, start=1):
            if ranked.get("rank") != expected_rank:
                raise RetrievalAblationError("Baseline ranks are not contiguous")
            channels = ranked.get("channels") or ["legacy_hybrid"]
            passage_role = passage.get("passage_role") or "unknown"
            baseline_passages.append(
                RankedRetrievalPassage(
                    passage_id=ranked["passage_id"],
                    rank=expected_rank,
                    channels=list(channels),
                    passage_role=passage_role,
                    redundancy_group_id=overlap_groups[ranked["passage_id"]],
                    source_context_sufficient=None,
                )
            )

        cases.append(
            RetrievalAblationCase(
                case_id=f"development:{candidate.candidate_id}",
                candidate_id=candidate.candidate_id,
                source_group_id=source_group_id,
                gold_review_scope="bounded_candidates",
                gold_reviewer=stage_review.reviewer,
                evidence_labels="positive_only",
                useful_passage_ids=useful_passage_ids,
                facet_gold=facet_gold,
                baseline=RetrievalArm(passages=baseline_passages),
            )
        )

    if not cases:
        raise RetrievalAblationError("No evidence-positive semantic facets were found")
    return RetrievalAblationManifest(
        input_snapshots=list(input_snapshots),
        limitations=[
            "This baseline contains only evidence-positive semantic facets from the reviewed development packet.",
            "No-evidence and uncertain facets remain a separate retrieval-abstention evaluation and are not counted here.",
            "Passage redundancy groups use deterministic substantial character-span overlap; source-context sufficiency and operational cost were not retrospectively labelled.",
        ],
        cases=cases,
    )


def _unique_by_key(
    items: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise RetrievalAblationError(f"A {label} lacks {key}")
        if value in result:
            raise RetrievalAblationError(f"Duplicate {label} {key}: {value}")
        result[value] = item
    return result


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _overlap_groups(passages: list[dict[str, Any]]) -> dict[str, str]:
    """Group exact/substantially overlapping ranked spans without source text."""
    parent = {item["passage_id"]: item["passage_id"] for item in passages}

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(passages):
        for right in passages[index + 1 :]:
            if _substantially_overlaps(left, right):
                union(left["passage_id"], right["passage_id"])
    return {
        item["passage_id"]: f"span-overlap:{find(item['passage_id'])}"
        for item in passages
    }


def _substantially_overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Passage character offsets are page-local for PDFs.  The shared
    # content_sha256 identifies the complete source representation, not the
    # passage text, so neither equal offsets on different pages nor equal
    # source hashes establish passage overlap.
    if left.get("page_index") != right.get("page_index"):
        return False
    try:
        left_start, left_end = int(left["character_start"]), int(left["character_end"])
        right_start, right_end = int(right["character_start"]), int(right["character_end"])
    except (KeyError, TypeError, ValueError):
        return False
    left_length = max(0, left_end - left_start)
    right_length = max(0, right_end - right_start)
    shorter = min(left_length, right_length)
    if not shorter:
        return False
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    return intersection / shorter >= 0.75
