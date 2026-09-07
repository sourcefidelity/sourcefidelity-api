"""Deterministic proximity analysis for factual-direction calibration.

The ordering in this module is an evaluation aid, not a severity scale and not
an instruction to rewrite one judge's label.  Runtime factual meanings and
evidence requirements remain owned by the judgment contracts.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


JudicialLabel = Literal[
    "contradicts",
    "mixed",
    "supports",
    "qualifies",
    "uncertain",
    "none",
]
PairClassification = Literal[
    "exact",
    "adjacent",
    "moderate",
    "material",
    "directional",
]
PanelStatus = Literal[
    "operationally_incomplete",
    "exact_convergence",
    "strong_proximity",
    "moderate_review",
    "material_disagreement",
    "directional_disagreement",
]

JUDICIAL_PROXIMITY_ORDER: tuple[JudicialLabel, ...] = (
    "contradicts",
    "mixed",
    "supports",
    "qualifies",
    "uncertain",
    "none",
)
_POSITION = {label: index for index, label in enumerate(JUDICIAL_PROXIMITY_ORDER)}


class JudicialPairComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_judge_id: str = Field(min_length=1, max_length=200)
    right_judge_id: str = Field(min_length=1, max_length=200)
    left_label: JudicialLabel
    right_label: JudicialLabel
    distance: int = Field(ge=0, le=5)
    classification: PairClassification
    directional_disagreement: bool


class JudicialPanelAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judicial-proximity-assessment-v1"] = (
        "judicial-proximity-assessment-v1"
    )
    ordering: list[JudicialLabel]
    required_formal_count: int = Field(ge=2, le=20)
    formal_valid_count: int = Field(ge=0, le=20)
    formal_status: PanelStatus
    formal_convergence_accepted: bool
    formal_labels: dict[str, JudicialLabel]
    formal_neighborhood: list[JudicialLabel]
    formal_max_distance: int | None = Field(default=None, ge=0, le=5)
    provisional_exact_label: JudicialLabel | None = None
    formal_comparisons: list[JudicialPairComparison]
    supplemental_labels: dict[str, JudicialLabel]
    anchor_labels: dict[str, JudicialLabel]
    external_comparisons: list[JudicialPairComparison]
    external_material_or_directional_disagreement: bool
    limitations: list[str]


def judicial_distance(left: JudicialLabel, right: JudicialLabel) -> int:
    """Return ordinal calibration distance; this is not semantic severity."""
    try:
        return abs(_POSITION[left] - _POSITION[right])
    except KeyError as error:
        raise ValueError(f"unsupported judicial label: {error.args[0]!r}") from error


def compare_judgments(
    left_judge_id: str,
    left_label: JudicialLabel,
    right_judge_id: str,
    right_label: JudicialLabel,
) -> JudicialPairComparison:
    distance = judicial_distance(left_label, right_label)
    labels = {left_label, right_label}
    directional = "contradicts" in labels and any(
        label not in {"contradicts", "mixed"} for label in labels
    )
    if directional:
        classification: PairClassification = "directional"
    elif distance == 0:
        classification = "exact"
    elif distance == 1:
        classification = "adjacent"
    elif distance == 2:
        classification = "moderate"
    else:
        classification = "material"
    return JudicialPairComparison(
        left_judge_id=left_judge_id,
        right_judge_id=right_judge_id,
        left_label=left_label,
        right_label=right_label,
        distance=distance,
        classification=classification,
        directional_disagreement=directional,
    )


def assess_judicial_panel(
    formal_labels: Mapping[str, JudicialLabel],
    *,
    supplemental_labels: Mapping[str, JudicialLabel] | None = None,
    anchor_labels: Mapping[str, JudicialLabel] | None = None,
    required_formal_count: int = 3,
) -> JudicialPanelAssessment:
    """Assess formal convergence while retaining supplemental/anchor opinions.

    Supplemental judges and expert anchors cannot repair an incomplete formal
    denominator or manufacture a single canonical label.  They expose whether
    another competent reading falls outside the formal judicial neighborhood.
    """
    supplemental = dict(supplemental_labels or {})
    anchors = dict(anchor_labels or {})
    formal = dict(formal_labels)
    all_ids = [*formal, *supplemental, *anchors]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("judge IDs must be unique across panel roles")
    if required_formal_count < 2:
        raise ValueError("at least two formal judges are required")

    formal_comparisons = [
        compare_judgments(left_id, formal[left_id], right_id, formal[right_id])
        for left_id, right_id in combinations(sorted(formal), 2)
    ]
    max_distance = max(
        (comparison.distance for comparison in formal_comparisons),
        default=0 if formal else None,
    )
    has_directional = any(
        comparison.classification == "directional"
        for comparison in formal_comparisons
    )
    has_material = any(
        comparison.classification == "material" for comparison in formal_comparisons
    )
    has_moderate = any(
        comparison.classification == "moderate" for comparison in formal_comparisons
    )
    exact = bool(formal) and len(set(formal.values())) == 1

    if len(formal) < required_formal_count:
        status: PanelStatus = "operationally_incomplete"
    elif has_directional:
        status = "directional_disagreement"
    elif has_material:
        status = "material_disagreement"
    elif has_moderate:
        status = "moderate_review"
    elif exact:
        status = "exact_convergence"
    else:
        status = "strong_proximity"
    accepted = status in {"exact_convergence", "strong_proximity"}

    external = {**supplemental, **anchors}
    external_comparisons = [
        compare_judgments(formal_id, formal_label, external_id, external_label)
        for formal_id, formal_label in sorted(formal.items())
        for external_id, external_label in sorted(external.items())
    ]
    external_material = any(
        item.classification in {"material", "directional"}
        for item in external_comparisons
    )
    neighborhood = sorted(set(formal.values()), key=_POSITION.__getitem__)
    return JudicialPanelAssessment(
        ordering=list(JUDICIAL_PROXIMITY_ORDER),
        required_formal_count=required_formal_count,
        formal_valid_count=len(formal),
        formal_status=status,
        formal_convergence_accepted=accepted,
        formal_labels=formal,
        formal_neighborhood=neighborhood,
        formal_max_distance=max_distance,
        provisional_exact_label=next(iter(formal.values())) if exact else None,
        formal_comparisons=formal_comparisons,
        supplemental_labels=supplemental,
        anchor_labels=anchors,
        external_comparisons=external_comparisons,
        external_material_or_directional_disagreement=external_material,
        limitations=[
            "Judicial distance is a calibration aid, not a report-severity or truth scale.",
            "Mixed retains its evidence-configuration meaning and uncertain remains an abstention.",
            "Adjacent convergence does not manufacture a single canonical factual label.",
            "Expert anchors and supplemental models remain opinions rather than gold labels.",
        ],
    )
