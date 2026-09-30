"""GLM sentence evidence selection: strict fixed-ID binding, no salvage, fallbacks."""
import pytest

from app.services.evidence_sentence_selection import (
    SelectionResponseInvalid,
    PreparedRequest,
    SentenceRef,
    bind_response,
)


def _request():
    refs = {f"s{i}": SentenceRef(f"s{i}", "p1", 3, i * 10, i * 10 + 9, f"Sentence {i}.", "body_prose")
            for i in range(1, 5)}
    return PreparedRequest("sys", "user", refs, "f" * 64, 0)


def test_non_adjacent_choices_bind_in_the_models_order():
    chosen = bind_response({"selections": [{"sentence_id": "s4", "reason": "bears_on_statement"},
                                           {"sentence_id": "s1", "reason": "necessary_context"}]}, _request())
    assert [(ref.alias, reason) for ref, reason in chosen] == [("s4", "bears_on_statement"), ("s1", "necessary_context")]


def test_an_empty_selection_is_valid_and_means_nothing_chosen():
    assert bind_response({"selections": []}, _request()) == []


@pytest.mark.parametrize("raw", [
    {"selections": [{"sentence_id": "s9", "reason": "bears_on_statement"}]},          # invented ID
    {"selections": [{"sentence_id": "s1", "reason": "bears_on_statement"}] * 2},      # duplicate
    {"selections": [{"sentence_id": "s1", "reason": "supports"}]},                    # a verdict, not a reason
    {"selections": [{"sentence_id": "s1", "reason": "bears_on_statement", "text": "x"}]},
    {"selections": [], "verdict": "supported"},
    {"selections": [{"sentence_id": f"s{i % 4 + 1}", "reason": "bears_on_statement"} for i in range(9)]},
    "not json",
])
def test_any_contract_violation_rejects_the_whole_response(raw):
    with pytest.raises(SelectionResponseInvalid):
        bind_response(raw, _request())
