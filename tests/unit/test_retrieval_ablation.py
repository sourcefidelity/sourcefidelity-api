"""Protected retrieval-ablation contract and metrics regressions."""

import pytest

from app.services.retrieval_ablation import (
    RankedRetrievalPassage,
    RetrievalAblationError,
    RetrievalAblationManifest,
    RetrievalArm,
    RetrievalArmCost,
    RetrievalAblationCase,
    RetrievalFacetGold,
    build_overlap_cleanup_experiment,
    build_protected_discovery_experiment,
    evaluate_retrieval_ablation,
)


def _passage(
    passage_id,
    rank,
    *,
    channels=("deterministic",),
    role="body_prose",
    group=None,
    context=True,
):
    return RankedRetrievalPassage(
        passage_id=passage_id,
        rank=rank,
        channels=list(channels),
        passage_role=role,
        redundancy_group_id=group,
        source_context_sufficient=context,
    )


def _case(*, experiment=None, exhaustive=False):
    return RetrievalAblationCase(
        case_id="case-1",
        candidate_id="candidate-1",
        source_group_id="source-hash-1",
        evidence_labels="exhaustive" if exhaustive else "positive_only",
        useful_passage_ids=["useful-a", "useful-b"],
        facet_gold=[
            RetrievalFacetGold(facet_id="facet-a", useful_passage_ids=["useful-a"]),
            RetrievalFacetGold(facet_id="facet-b", useful_passage_ids=["useful-b"]),
        ],
        baseline=RetrievalArm(
            passages=[
                _passage("safety-1", 1, group="overlap"),
                _passage("useful-a", 2, group="overlap"),
                _passage("noise", 3, role="publication_metadata"),
                _passage("other", 4),
                _passage("useful-b", 5),
            ]
        ),
        experiment=experiment,
    )


def _improved_experiment():
    return RetrievalArm(
        passages=[
            _passage(
                "safety-1", 1, channels=("protected_baseline", "deterministic")
            ),
            _passage(
                "useful-a", 2, channels=("protected_baseline", "deterministic")
            ),
            _passage("useful-b", 3, channels=("dense_shadow",)),
            _passage("other", 4),
        ]
    )


def test_baseline_only_reports_recall_but_not_positive_only_precision():
    result = evaluate_retrieval_ablation(
        RetrievalAblationManifest(cases=[_case()])
    )

    assert result.decision == "baseline_only"
    assert result.baseline.hit_rate_at_3 == 1.0
    assert result.baseline.labelled_passage_recall_at_3 == 0.5
    assert result.baseline.labelled_passage_recall_at_5 == 1.0
    assert result.baseline.facet_coverage_at_3 == 0.5
    assert result.baseline.facet_coverage_at_5 == 1.0
    assert result.baseline.useful_precision_at_3 is None
    assert result.baseline.redundant_slot_rate_at_5 == 0.2
    assert result.baseline.excluded_role_rate_at_5 == 0.2


def test_exhaustive_labels_enable_useful_precision():
    result = evaluate_retrieval_ablation(
        RetrievalAblationManifest(cases=[_case(exhaustive=True)])
    )

    assert result.baseline.useful_precision_at_3 == pytest.approx(1 / 3)
    assert result.baseline.useful_precision_at_5 == pytest.approx(2 / 5)


def test_additive_experiment_must_preserve_and_mark_baseline_safety_passages():
    missing = RetrievalArm(
        passages=[
            _passage(
                "safety-1", 1, channels=("protected_baseline", "deterministic")
            ),
            _passage("useful-b", 2, channels=("dense_shadow",)),
        ]
    )
    with pytest.raises(ValueError, match="retain every protected"):
        RetrievalAblationManifest(cases=[_case(experiment=missing)])

    unmarked = RetrievalArm(
        passages=[
            _passage("safety-1", 1),
            _passage("useful-a", 2),
            _passage("useful-b", 3, channels=("dense_shadow",)),
        ]
    )
    with pytest.raises(ValueError, match="protected_baseline channel"):
        RetrievalAblationManifest(cases=[_case(experiment=unmarked)])


def test_improved_additive_result_continues_without_precision_claim():
    result = evaluate_retrieval_ablation(
        RetrievalAblationManifest(
            cases=[_case(experiment=_improved_experiment())]
        )
    )

    assert result.decision == "continue"
    assert result.protected_passage_retention == 1.0
    assert "material_facet_coverage_at_3_improved" in result.reason_codes
    assert "excluded_role_noise_reduced" in result.reason_codes
    assert result.experiment.labelled_passage_recall_at_5 == 1.0
    assert result.experiment.useful_precision_at_5 is None


def test_recall_regression_stops_even_when_protected_passages_remain():
    regressed = RetrievalArm(
        passages=[
            _passage(
                "safety-1", 1, channels=("protected_baseline", "deterministic")
            ),
            _passage(
                "useful-a", 2, channels=("protected_baseline", "deterministic")
            ),
            _passage("other", 3, channels=("dense_shadow",)),
        ]
    )
    result = evaluate_retrieval_ablation(
        RetrievalAblationManifest(cases=[_case(experiment=regressed)])
    )

    assert result.decision == "stop"
    assert "labelled_passage_recall_regression" in result.reason_codes
    assert "material_facet_coverage_regression" in result.reason_codes
    assert result.decision_applied is False


def test_manifest_rejects_mixed_baseline_and_experiment_cases():
    first = _case(experiment=_improved_experiment())
    second = _case()
    second.case_id = "case-2"
    second.candidate_id = "candidate-2"
    with pytest.raises(ValueError, match="cannot mix"):
        RetrievalAblationManifest(cases=[first, second])


def test_overlap_cleanup_preserves_top_two_and_removes_later_group_duplicate():
    case = _case()
    case.baseline.passages[2].redundancy_group_id = "overlap"
    manifest = build_overlap_cleanup_experiment(
        RetrievalAblationManifest(cases=[case])
    )
    experiment = manifest.cases[0].experiment
    assert experiment is not None

    assert [item.passage_id for item in experiment.passages] == [
        "safety-1",
        "useful-a",
        "other",
        "useful-b",
    ]
    assert all(
        "protected_baseline" in item.channels
        for item in experiment.passages[:2]
    )
    assert all(
        "deterministic_overlap_cleanup" in item.channels
        for item in experiment.passages
    )
    result = evaluate_retrieval_ablation(manifest)
    assert result.decision == "continue"
    assert result.protected_passage_retention == 1.0
    assert "redundant_slots_reduced" in result.reason_codes


def test_overlap_cleanup_rejects_a_manifest_that_already_has_an_experiment():
    manifest = RetrievalAblationManifest(
        cases=[_case(experiment=_improved_experiment())]
    )

    with pytest.raises(RetrievalAblationError, match="baseline-only"):
        build_overlap_cleanup_experiment(manifest)


def test_protected_discovery_guarantees_top_two_not_old_rank_three():
    case = _case()
    discovery = [
        _passage("dense-a", 1, channels=("bge_m3_hybrid",)),
        _passage("dense-b", 2, channels=("bge_m3_hybrid",)),
    ]
    manifest = build_protected_discovery_experiment(
        RetrievalAblationManifest(cases=[case]),
        discoveries={"case-1": discovery},
        costs={"case-1": RetrievalArmCost(query_latency_ms=12, index_bytes=100)},
    )
    experiment = manifest.cases[0].experiment
    assert experiment is not None

    assert [item.passage_id for item in experiment.passages] == [
        "safety-1",
        "useful-a",
        "dense-a",
        "dense-b",
        "noise",
    ]
    assert all(
        "protected_baseline" in item.channels
        for item in experiment.passages[:2]
    )
    assert all(
        "bge_m3_hybrid_discovery" in item.channels
        for item in experiment.passages[2:4]
    )
    assert experiment.cost.query_latency_ms == 12


def test_protected_discovery_requires_every_case_ranking():
    with pytest.raises(RetrievalAblationError, match="every and only"):
        build_protected_discovery_experiment(
            RetrievalAblationManifest(cases=[_case()]),
            discoveries={},
        )
