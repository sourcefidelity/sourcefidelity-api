"""Judicial-proximity calibration contract regressions."""

import pytest

from app.services.judicial_proximity import (
    JUDICIAL_PROXIMITY_ORDER,
    assess_judicial_panel,
    compare_judgments,
    judicial_distance,
)


def test_fixed_calibration_order_and_distances():
    assert JUDICIAL_PROXIMITY_ORDER == (
        "contradicts",
        "mixed",
        "supports",
        "qualifies",
        "uncertain",
        "none",
    )
    assert judicial_distance("supports", "qualifies") == 1
    assert judicial_distance("qualifies", "uncertain") == 1
    assert judicial_distance("uncertain", "none") == 1
    assert judicial_distance("contradicts", "none") == 5


@pytest.mark.parametrize(
    ("left", "right", "classification"),
    [
        ("supports", "supports", "exact"),
        ("contradicts", "mixed", "adjacent"),
        ("mixed", "supports", "adjacent"),
        ("supports", "qualifies", "adjacent"),
        ("supports", "uncertain", "moderate"),
        ("qualifies", "none", "moderate"),
        ("mixed", "none", "material"),
        ("contradicts", "supports", "directional"),
        ("contradicts", "none", "directional"),
    ],
)
def test_pair_classification_preserves_directional_override(left, right, classification):
    result = compare_judgments("left", left, "right", right)
    assert result.classification == classification
    assert result.directional_disagreement is (classification == "directional")


def test_exact_formal_convergence_retains_provisional_exact_label():
    result = assess_judicial_panel(
        {"a": "qualifies", "b": "qualifies", "c": "qualifies"}
    )
    assert result.formal_status == "exact_convergence"
    assert result.formal_convergence_accepted is True
    assert result.provisional_exact_label == "qualifies"


def test_adjacent_formal_labels_are_strong_proximity_without_canonical_label():
    result = assess_judicial_panel(
        {"a": "supports", "b": "qualifies", "c": "qualifies"}
    )
    assert result.formal_status == "strong_proximity"
    assert result.formal_convergence_accepted is True
    assert result.formal_neighborhood == ["supports", "qualifies"]
    assert result.provisional_exact_label is None


def test_two_step_dispersion_is_review_not_accepted_convergence():
    result = assess_judicial_panel(
        {"a": "supports", "b": "qualifies", "c": "uncertain"}
    )
    assert result.formal_status == "moderate_review"
    assert result.formal_convergence_accepted is False
    assert result.formal_max_distance == 2


def test_directional_disagreement_overrides_small_numeric_distance():
    result = assess_judicial_panel(
        {"a": "contradicts", "b": "mixed", "c": "supports"}
    )
    assert result.formal_status == "directional_disagreement"
    assert result.formal_convergence_accepted is False


def test_missing_formal_arm_is_operationally_incomplete():
    result = assess_judicial_panel({"a": "qualifies", "b": "uncertain"})
    assert result.formal_status == "operationally_incomplete"
    assert result.formal_convergence_accepted is False


def test_supplemental_failure_is_not_a_label_and_does_not_change_formal_panel():
    result = assess_judicial_panel(
        {"a": "qualifies", "b": "qualifies", "c": "qualifies"},
        supplemental_labels={},
        anchor_labels={"owner": "uncertain"},
    )
    assert result.formal_status == "exact_convergence"
    assert result.formal_convergence_accepted is True
    assert result.external_material_or_directional_disagreement is False
    assert {item.distance for item in result.external_comparisons} == {1}


def test_supplemental_directional_reading_is_preserved_as_external_flag():
    result = assess_judicial_panel(
        {"a": "supports", "b": "qualifies", "c": "supports"},
        supplemental_labels={"shadow": "contradicts"},
    )
    assert result.formal_status == "strong_proximity"
    assert result.formal_convergence_accepted is True
    assert result.external_material_or_directional_disagreement is True


def test_duplicate_judge_ids_across_roles_fail_closed():
    with pytest.raises(ValueError, match="unique"):
        assess_judicial_panel(
            {"same": "supports", "b": "supports", "c": "supports"},
            anchor_labels={"same": "qualifies"},
        )


def test_unknown_label_fails_closed():
    with pytest.raises(ValueError, match="unsupported"):
        judicial_distance("supports", "invalid")  # type: ignore[arg-type]
