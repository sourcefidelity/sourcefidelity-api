"""Independent optional advice does not promote malformed required findings."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.services.passage_relevance import _Response, inspect_independent_display_response


def response():
    return {"assessments": [{"passage_id": "p1", "relevance": "partially_relevant",
        "confidence": "high", "evidence_role": "source_own_claim_or_finding"}],
        "display_observations": {"p1": {"basis": "direct_attribution",
            "claim_token_ranges": [[0, 2]], "source_sentence_ids": ["s000"]}}}


def test_valid_response_unchanged():
    raw = response()
    result, diagnostics = inspect_independent_display_response(raw)
    assert result == _Response.model_validate(raw)
    assert diagnostics == []


@pytest.mark.parametrize("defect", ["too_many_sentences", "unknown_field", "invalid_basis", "missing_ids"])
def test_invalid_optional_advice_omitted_whole_not_truncated(defect):
    raw = response()
    observation = raw["display_observations"]["p1"]
    if defect == "too_many_sentences":
        observation["source_sentence_ids"] = ["s000", "s001", "s002"]
    elif defect == "unknown_field":
        observation["PRIVATE_KEY"] = "PRIVATE_VALUE"
    elif defect == "invalid_basis":
        observation["basis"] = "PRIVATE_VALUE"
    else:
        observation.pop("source_sentence_ids")
    before = deepcopy(raw)
    with pytest.raises(ValidationError):
        _Response.model_validate(raw)  # Strict schema remains available for historical diagnostics.
    result, diagnostics = inspect_independent_display_response(raw)
    assert result.display_observations == {}
    assert len(result.assessments) == 1
    assert diagnostics == [{"reason": "invalid_optional_observation", "count": 1}]
    assert "PRIVATE" not in str(diagnostics)
    assert raw == before


@pytest.mark.parametrize("defect", ["required_field", "required_label", "required_extra", "top_level_extra"])
def test_invalid_required_findings_and_envelope_still_fail(defect):
    raw = response()
    if defect == "required_field":
        raw["assessments"][0].pop("relevance")
    elif defect == "required_label":
        raw["assessments"][0]["relevance"] = "supported"
    elif defect == "required_extra":
        raw["assessments"][0]["support"] = True
    else:
        raw["support"] = True
    with pytest.raises(ValidationError):
        inspect_independent_display_response(raw)


def test_unknown_candidate_not_echoed_or_accepted():
    raw = response()
    raw["display_observations"]["PRIVATE_CANDIDATE"] = raw["display_observations"]["p1"]
    result, diagnostics = inspect_independent_display_response(raw)
    assert set(result.display_observations) == {"p1"}
    assert diagnostics == [{"reason": "unbound_optional_observation", "count": 1}]


@pytest.mark.parametrize("container", [None, [], "PRIVATE_VALUE"])
def test_invalid_optional_container_is_not_a_required_findings_failure(container):
    raw = response()
    raw["display_observations"] = container
    result, diagnostics = inspect_independent_display_response(raw)
    assert result.display_observations == {}
    assert diagnostics == [{"reason": "invalid_optional_container", "count": 1}]
