"""A failure code with no detail costs an offline replay to diagnose.

The owner's 65-reference paper failed at verification with `ValidationError`
and nothing else. The class name alone does not say which model, which field or
which rule, so diagnosis required re-driving the stage against the job's own
sources. A pydantic error already carries exactly the safe part - model name,
field path, rule name - while the offending value, which is the only part that
could hold student or source text, is never read.
"""
import pytest
from pydantic import BaseModel, ValidationError

from app.log_safety import safe_exception_detail


class Example(BaseModel):
    reference_id: str
    authors: list[str] = []


def _validation_error(**kwargs):
    try:
        Example(**kwargs)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def test_a_validation_failure_names_the_field_and_the_rule():
    detail = safe_exception_detail(_validation_error(authors=["a"]))
    assert detail is not None
    assert "reference_id" in detail
    assert "missing" in detail


def test_the_offending_value_is_never_recorded():
    secret = "Student wrote this confidential sentence"
    detail = safe_exception_detail(_validation_error(reference_id=secret, authors=secret))
    assert detail is not None
    assert secret not in detail
    for word in secret.split():
        assert word not in detail


def test_an_exception_with_no_structured_errors_yields_nothing():
    assert safe_exception_detail(RuntimeError("plain failure")) is None
    assert safe_exception_detail(ValueError("another")) is None


def test_an_existing_bounded_detail_is_preserved():
    class Carrier(Exception):
        detail = "stored=body_prose; derived=reference_list; method=bm25_concept"

    assert safe_exception_detail(Carrier()) == Carrier.detail


def test_a_detail_outside_the_safe_shape_is_dropped_not_truncated():
    class Carrier(Exception):
        detail = "student wrote: “the regulator’s finding”"

    # Dropped entirely rather than trimmed: a truncated excerpt is still an
    # excerpt.
    assert safe_exception_detail(Carrier()) is None


def test_the_recorded_detail_is_bounded_in_length():
    class Wide(BaseModel):
        a: int
        b: int
        c: int
        d: int
        e: int

    try:
        Wide()
    except ValidationError as exc:
        detail = safe_exception_detail(exc)
    assert detail is not None and len(detail) <= 200
    # Only the leading few are reported; the rest would add no diagnostic value.
    assert detail.count(";") <= 2
