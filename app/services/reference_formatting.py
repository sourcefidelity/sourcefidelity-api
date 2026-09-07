"""Conservative, rule-specific reference-format evidence.

This layer never turns one observed rule into a whole-entry or whole-paper
style verdict. Each rule remains independently assessed or not assessed.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.services.reference_layout import ReferenceLayoutArtifact


REFERENCE_FORMATTING_VERSION = "reference-formatting-v1"
HANGING_INDENT_POINTS = 36.0
HANGING_INDENT_TOLERANCE_POINTS = 4.5


class ReferenceFormattingRuleResult(BaseModel):
    reference_id: str = Field(min_length=1)
    rule_id: Literal["reference_list_hanging_indent_0_5_in"]
    status: Literal["matches_rule", "difference", "not_assessed"]
    observed_points: float | None = None
    expected_points: float = HANGING_INDENT_POINTS
    reason_code: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_observation(self):
        if self.status == "not_assessed" and self.observed_points is not None:
            raise ValueError("An unassessed rule cannot carry an observation")
        if self.status != "not_assessed" and self.observed_points is None:
            raise ValueError("An assessed rule requires an observation")
        return self


class ReferenceFormattingAssessment(BaseModel):
    assessment_version: str = REFERENCE_FORMATTING_VERSION
    citation_format: Literal["apa", "mla"]
    status: Literal["partial", "not_assessed"]
    assessed_rule_ids: list[str] = Field(default_factory=list)
    results: list[ReferenceFormattingRuleResult] = Field(default_factory=list)
    result_counts: dict[str, int] = Field(default_factory=dict)
    primary_rule_sources: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        expected = dict(sorted(Counter(item.status for item in self.results).items()))
        if self.result_counts != expected:
            raise ValueError("Reference-formatting counts do not match results")
        assessed = any(item.status != "not_assessed" for item in self.results)
        if (self.status == "partial") != assessed:
            raise ValueError("Partial formatting status requires assessed evidence")
        return self


def assess_reference_formatting(
    layout: ReferenceLayoutArtifact,
) -> ReferenceFormattingAssessment:
    """Assess only the shared APA/MLA 0.5-inch hanging-indent rule."""
    sources = (
        [
            "https://www.apa.org/ed/precollege/psn/2020/09/apa-style-student-papers"
        ]
        if layout.citation_format == "apa"
        else [
            "https://style.mla.org/hanging-indents/",
            "https://style.mla.org/app/uploads/sites/3/2020/12/Formatting-a-Research-Paper_v3_-The-MLA-Style-Center.pdf",
        ]
    )
    results: list[ReferenceFormattingRuleResult] = []
    for entry in layout.entries:
        observed = entry.observed_hanging_indent_points
        if entry.mapping_status != "matched":
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="not_assessed",
                    reason_code="reference_layout_not_uniquely_matched",
                )
            )
        elif observed is None:
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="not_assessed",
                    reason_code="continuation_indent_not_observable",
                )
            )
        else:
            matches = (
                abs(observed - HANGING_INDENT_POINTS)
                <= HANGING_INDENT_TOLERANCE_POINTS
            )
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="matches_rule" if matches else "difference",
                    observed_points=observed,
                    reason_code=(
                        "observed_hanging_indent_within_tolerance"
                        if matches
                        else "observed_hanging_indent_outside_tolerance"
                    ),
                )
            )
    counts = dict(sorted(Counter(item.status for item in results).items()))
    assessed = any(item.status != "not_assessed" for item in results)
    return ReferenceFormattingAssessment(
        citation_format=layout.citation_format,
        status="partial" if assessed else "not_assessed",
        assessed_rule_ids=(
            ["reference_list_hanging_indent_0_5_in"] if assessed else []
        ),
        results=results,
        result_counts=counts,
        primary_rule_sources=sources,
        limitations=[
            "This assessment checks only the observable 0.5-inch hanging-indent rule; it is not a whole-entry or whole-paper style verdict.",
            "A one-line PDF entry has no visible continuation indentation and remains not assessed.",
            "Italics, spacing, heading placement, capitalization, punctuation, field order and bibliographic completeness remain not assessed.",
        ],
    )
