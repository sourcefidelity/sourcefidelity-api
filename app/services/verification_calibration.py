"""Fixed-corpus metrics for claim/source relationship calibration.

This module evaluates already-labelled cases.  It deliberately does not infer
ground truth, retrieve source material, or enable verification adjudication.
Abstentions remain visible and count against recall instead of disappearing
from the denominator.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from app.services.verification_evidence import RelationshipStatus


CALIBRATION_LABELS = (
    RelationshipStatus.SUPPORTS,
    RelationshipStatus.CONTRADICTS,
    RelationshipStatus.UNRELATED,
    RelationshipStatus.INSUFFICIENT_EVIDENCE,
)


class CalibrationDatasetError(ValueError):
    """The fixed evaluation dataset is incomplete or internally inconsistent."""


class CalibrationCaseResult(BaseModel):
    """One human-labelled case and one bounded predictor output."""

    case_id: str = Field(min_length=1, max_length=128)
    gold_status: RelationshipStatus
    predicted_status: RelationshipStatus

    @field_validator("gold_status")
    @classmethod
    def _gold_must_be_substantive(cls, value: RelationshipStatus):
        if value is RelationshipStatus.NOT_ASSESSED:
            raise ValueError("Gold labels cannot be not_assessed")
        return value


class ClassCalibrationMetrics(BaseModel):
    label: RelationshipStatus
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


class CalibrationReport(BaseModel):
    total_cases: int
    assessed_cases: int
    abstained_cases: int
    coverage: float
    correct_cases: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: list[ClassCalibrationMetrics]
    confusion: dict[str, dict[str, int]]


def evaluate_calibration(
    cases: Iterable[CalibrationCaseResult],
) -> CalibrationReport:
    """Measure one predictor on one unique, fully labelled fixed corpus.

    ``not_assessed`` predictions are abstentions.  They are included in the
    confusion matrix and create false negatives for the gold class, so a
    cautious predictor cannot report strong recall by silently skipping hard
    cases.
    """

    rows = list(cases)
    if not rows:
        raise CalibrationDatasetError("Calibration requires at least one case")
    identifiers = [row.case_id for row in rows]
    duplicates = sorted(
        case_id for case_id, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise CalibrationDatasetError(
            f"Calibration case IDs must be unique: {', '.join(duplicates[:3])}"
        )

    prediction_labels = (*CALIBRATION_LABELS, RelationshipStatus.NOT_ASSESSED)
    confusion = {
        gold.value: {predicted.value: 0 for predicted in prediction_labels}
        for gold in CALIBRATION_LABELS
    }
    for row in rows:
        confusion[row.gold_status.value][row.predicted_status.value] += 1

    per_class: list[ClassCalibrationMetrics] = []
    for label in CALIBRATION_LABELS:
        true_positive = confusion[label.value][label.value]
        false_positive = sum(
            confusion[other.value][label.value]
            for other in CALIBRATION_LABELS
            if other is not label
        )
        false_negative = sum(
            count
            for predicted, count in confusion[label.value].items()
            if predicted != label.value
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _harmonic_mean(precision, recall)
        per_class.append(
            ClassCalibrationMetrics(
                label=label,
                support=sum(confusion[label.value].values()),
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    total = len(rows)
    abstained = sum(
        row.predicted_status is RelationshipStatus.NOT_ASSESSED for row in rows
    )
    correct = sum(row.gold_status is row.predicted_status for row in rows)
    return CalibrationReport(
        total_cases=total,
        assessed_cases=total - abstained,
        abstained_cases=abstained,
        coverage=_safe_ratio(total - abstained, total),
        correct_cases=correct,
        accuracy=_safe_ratio(correct, total),
        macro_precision=sum(metric.precision for metric in per_class)
        / len(per_class),
        macro_recall=sum(metric.recall for metric in per_class) / len(per_class),
        macro_f1=sum(metric.f1 for metric in per_class) / len(per_class),
        per_class=per_class,
        confusion=confusion,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0
