"""Prospective producer grammar; no ordinary parser activation or salvage."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.services import passage_relevance as relevance


def record():
    return {"assessments": [{"passage_id": "p", "relevance": "partially_relevant",
        "confidence": "high", "evidence_role": "source_own_claim_or_finding",
        "rationale": "Useful for inspecting one part, not whole-claim support.",
        "basis": "direct_attribution", "claim_token_ranges": [[0, 2]],
        "source_sentence_ids": ["s001", "s002"]}]}


def test_valid_adapter_preserves_fields_without_mutating_input():
    raw = record()
    before = deepcopy(raw)
    parsed = relevance.inspect_single_record_response(raw, ["p"])
    assert raw == before
    assert parsed.assessments[0].relevance == "partially_relevant"
    assert parsed.display_observations["p"].source_sentence_ids == ["s001", "s002"]
    assert parsed.display_observations["p"].claim_token_ranges == [(0, 2)]
    with pytest.raises(ValidationError):
        relevance._Response.model_validate(raw)  # Ordinary grammar unchanged.


@pytest.mark.parametrize("field", list(record()["assessments"][0]))
def test_every_field_required(field):
    raw = record()
    raw["assessments"][0].pop(field)
    with pytest.raises(ValidationError):
        relevance.inspect_single_record_response(raw, ["p"])


@pytest.mark.parametrize("defect", ["oversized", "extra", "old_container", "duplicate", "missing", "invented", "boolean", "string_integer"])
def test_rejects_invalid_output_without_repair(defect):
    raw = record()
    item = raw["assessments"][0]
    if defect == "oversized":
        item["source_sentence_ids"].append("s003")
    elif defect == "extra":
        item["support"] = True
    elif defect == "old_container":
        raw["display_observations"] = {}
    elif defect == "duplicate":
        raw["assessments"].append(deepcopy(item))
    elif defect == "missing":
        raw["assessments"] = []
    elif defect == "invented":
        item["passage_id"] = "other"
    else:
        item["claim_token_ranges"] = [[True if defect == "boolean" else "0", 2]]
    with pytest.raises((ValueError, ValidationError)):
        relevance.inspect_single_record_response(raw, ["p"])


def test_prompt_changes_format_not_semantic_directions():
    system = relevance._system_prompt(None) + relevance._DISPLAY_PROMPT
    changed = relevance.single_record_relevance_prompt(system)
    assert "All eight fields" in changed
    for instruction in ["Unclear wording remains unresolved.",
                        "Sharing actors or a broad topic alone is not relevant.",
                        "Non-unclear observations must supply at least one range.",
                        "NOT support/contradiction judgments or new interpretations of the student."]:
        assert instruction in system and instruction in changed
    with pytest.raises(ValueError):
        relevance.single_record_relevance_prompt(changed)
    with pytest.raises(ValueError):
        relevance.single_record_relevance_prompt("unknown producer")
