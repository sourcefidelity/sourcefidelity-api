"""Reviewed relationship packet to protected retrieval baseline regressions."""

from copy import deepcopy

import pytest

from app.services.relationship_stage_evaluation import (
    RelationshipStageReviewExport,
    RelationshipStageReviewManifest,
)
from app.services.retrieval_ablation import (
    RetrievalAblationError,
    RetrievalInputSnapshot,
    evaluate_retrieval_ablation,
)
from app.services.retrieval_baseline import build_reviewed_retrieval_baseline


SOURCE_SHA = "a" * 64
SNAPSHOT_SHA = "b" * 64


def _stage_manifest():
    return RelationshipStageReviewManifest.model_validate(
        {
            "source_artifact": "private-development.json",
            "source_sha256": SNAPSHOT_SHA,
            "candidates": [
                {
                    "unit_alias": "unit-1",
                    "candidate_id": "candidate-1",
                    "candidate_text": "fixed candidate",
                    "complete_citation_unit": "fixed citation unit",
                    "cited_author_label": "Author",
                    "accepted_candidate_relationship": "supports",
                    "source_sentences": [
                        {
                            "sentence_id": "sentence-1",
                            "passage_id": "passage-2",
                            "text": "authorized source sentence",
                            "voice_role": "unmarked_document_voice",
                            "discourse_role": "none",
                        }
                    ],
                    "facets": [
                        {
                            "facet_id": "facet-positive",
                            "kind": "candidate_as_written",
                            "text": "material facet",
                            "allowed_sentence_ids": ["sentence-1"],
                            "semantic_direction_eligible": True,
                            "proposition_holder_eligible": False,
                        },
                        {
                            "facet_id": "facet-no-evidence",
                            "kind": "quantity",
                            "text": "unestablished qualifier",
                            "allowed_sentence_ids": ["sentence-1"],
                            "semantic_direction_eligible": True,
                            "proposition_holder_eligible": False,
                        },
                    ],
                }
            ],
        }
    )


def _stage_review():
    return RelationshipStageReviewExport.model_validate(
        {
            "source_sha256": SNAPSHOT_SHA,
            "reviewer": "owner",
            "candidates": [
                {
                    "unit_alias": "unit-1",
                    "candidate_id": "candidate-1",
                    "facets": [
                        {
                            "facet_id": "facet-positive",
                            "selection_status": "evidence_selected",
                            "selected_sentence_ids": ["sentence-1"],
                            "semantic_direction": "supports",
                            "proposition_holder": "not_applicable",
                            "confidence": "high",
                        },
                        {
                            "facet_id": "facet-no-evidence",
                            "selection_status": "no_evidence",
                            "selected_sentence_ids": [],
                            "semantic_direction": "none",
                            "proposition_holder": "not_applicable",
                            "confidence": "high",
                        },
                    ],
                }
            ],
        }
    )


def _development_artifact():
    passages = []
    for rank in range(1, 4):
        passages.append(
            {
                "passage_id": f"passage-{rank}",
                "content_sha256": SOURCE_SHA,
                "page_index": rank,
                "character_start": 0,
                "character_end": 100,
                "passage_role": "body_prose",
            }
        )
    return {
        "results": [
            {
                "unit_alias": "unit-1",
                "artifact": {
                    "source_identity": {"content_sha256": SOURCE_SHA},
                    "passages": passages,
                    "candidate_passage_retrieval": {
                        "selections": [
                            {
                                "candidate_id": "candidate-1",
                                "passages": [
                                    {
                                        "passage_id": f"passage-{rank}",
                                        "rank": rank,
                                        "channels": ["deterministic"],
                                    }
                                    for rank in range(1, 4)
                                ],
                            }
                        ]
                    },
                },
            }
        ]
    }


def test_builds_positive_only_source_grouped_baseline_and_metrics():
    manifest = build_reviewed_retrieval_baseline(
        _stage_manifest(),
        _stage_review(),
        _development_artifact(),
        input_snapshots=[
            RetrievalInputSnapshot(name="private-development.json", sha256=SNAPSHOT_SHA)
        ],
    )
    result = evaluate_retrieval_ablation(manifest)

    assert len(manifest.cases) == 1
    assert manifest.cases[0].source_group_id == SOURCE_SHA
    assert manifest.cases[0].gold_review_scope == "bounded_candidates"
    assert manifest.cases[0].gold_reviewer == "owner"
    assert manifest.cases[0].useful_passage_ids == ["passage-2"]
    assert [item.facet_id for item in manifest.cases[0].facet_gold] == [
        "facet-positive"
    ]
    assert result.case_count == 1
    assert result.source_group_count == 1
    assert result.baseline.hit_rate_at_3 == 1.0
    assert result.baseline.mean_reciprocal_rank == 0.5
    assert result.baseline.useful_precision_at_3 is None
    assert result.baseline.redundant_slot_rate_at_5 == 0.0


def test_page_local_offsets_do_not_create_cross_page_redundancy():
    manifest = build_reviewed_retrieval_baseline(
        _stage_manifest(), _stage_review(), _development_artifact()
    )

    groups = [
        item.redundancy_group_id for item in manifest.cases[0].baseline.passages
    ]
    assert len(groups) == len(set(groups))


def test_rejects_reviewed_gold_missing_from_preserved_baseline():
    artifact = deepcopy(_development_artifact())
    artifact["results"][0]["artifact"]["candidate_passage_retrieval"]["selections"][0][
        "passages"
    ] = artifact["results"][0]["artifact"]["candidate_passage_retrieval"][
        "selections"
    ][0]["passages"][:1]

    with pytest.raises(RetrievalAblationError, match="absent from the preserved baseline"):
        build_reviewed_retrieval_baseline(
            _stage_manifest(), _stage_review(), artifact
        )
