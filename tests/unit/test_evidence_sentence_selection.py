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


def test_a_passage_with_an_over_bound_sentence_is_skipped_not_fatal(monkeypatch):
    # 2026-10-18: a 30-reference paper failed its whole run when one source
    # passage held a "sentence" over 2,000 characters (_EvidenceSentenceBudgetExceeded).
    from types import SimpleNamespace
    from app.services import evidence_sentence_selection as selection
    from app.services import facet_evidence_judgment as facets

    def sentences(passage):
        if passage.passage_id == "long":
            raise facets._EvidenceSentenceBudgetExceeded
        return [SimpleNamespace(passage_start=0, passage_end=9, text="Sentence.")]

    monkeypatch.setattr(facets, "_passage_sentences", sentences)
    passage = lambda pid, start: SimpleNamespace(passage_id=pid, page_index=1, character_start=start,
                                                 passage_role="body_prose")
    artifact = SimpleNamespace(claim=SimpleNamespace(text="Claim.", citation_marker="(A, 2020)"),
                               passages=[passage("long", 0), passage("ok", 100)])
    prepared = selection.prepare_request(artifact)
    assert prepared.excluded_passages == 1
    assert [ref.passage_id for ref in prepared.sentences.values()] == ["ok"]
