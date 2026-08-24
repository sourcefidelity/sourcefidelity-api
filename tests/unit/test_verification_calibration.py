import pytest
from pydantic import ValidationError

from app.services.verification_calibration import (
    CalibrationCaseResult,
    CalibrationDatasetError,
    evaluate_calibration,
)
from app.services.verification_evidence import RelationshipStatus


def _case(case_id: str, gold: str, predicted: str) -> CalibrationCaseResult:
    return CalibrationCaseResult(
        case_id=case_id,
        gold_status=gold,
        predicted_status=predicted,
    )


def test_calibration_counts_abstention_against_recall() -> None:
    report = evaluate_calibration(
        [
            _case("a", "supports", "supports"),
            _case("b", "supports", "not_assessed"),
            _case("c", "contradicts", "supports"),
            _case("d", "unrelated", "unrelated"),
            _case(
                "e",
                "insufficient_evidence",
                "insufficient_evidence",
            ),
        ]
    )

    assert report.total_cases == 5
    assert report.assessed_cases == 4
    assert report.abstained_cases == 1
    assert report.coverage == pytest.approx(0.8)
    assert report.accuracy == pytest.approx(0.6)
    supports = next(
        metric
        for metric in report.per_class
        if metric.label is RelationshipStatus.SUPPORTS
    )
    assert supports.precision == pytest.approx(0.5)
    assert supports.recall == pytest.approx(0.5)
    assert supports.f1 == pytest.approx(0.5)
    assert report.confusion["supports"]["not_assessed"] == 1


def test_calibration_rejects_duplicate_case_ids() -> None:
    with pytest.raises(CalibrationDatasetError, match="must be unique"):
        evaluate_calibration(
            [
                _case("same", "supports", "supports"),
                _case("same", "supports", "supports"),
            ]
        )


def test_calibration_rejects_unlabelled_gold_status() -> None:
    with pytest.raises(ValidationError, match="Gold labels cannot be"):
        _case("a", "not_assessed", "not_assessed")


def test_calibration_requires_cases() -> None:
    with pytest.raises(CalibrationDatasetError, match="at least one"):
        evaluate_calibration([])
