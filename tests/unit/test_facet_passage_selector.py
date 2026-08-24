"""Facet-specific usefulness selector contract tests."""

import app.services.facet_passage_selector as selector
from app.services.facet_passage_selector import (
    SelectorFacet,
    SelectorSentence,
    classify_facet_sentence_usefulness,
)


def _inputs():
    return (
        [SelectorFacet(facet_id="facet-a", text="The policy reduced harm.", allowed_sentence_ids=["sentence-a", "sentence-b"])],
        [
            SelectorSentence(sentence_id="sentence-a", text="The policy reduced harm."),
            SelectorSentence(sentence_id="sentence-b", text="The policy was introduced in 2020."),
        ],
    )


def test_selector_restores_application_ids_and_preserves_shadow_boundary(monkeypatch):
    def fake_chat(_system, payload, **_kwargs):
        assert "facet-a" not in payload
        assert "sentence-a" not in payload
        assert '"candidate_as_written":"The policy may reduce harm."' in payload
        assert '"complete_citation_unit":"The policy may reduce harm (Smith, 2020)."' in payload
        return {
            "assessments": [
                {"facet_id": "f1", "sentence_id": "s1", "usefulness": "sufficient"},
                {"facet_id": "f1", "sentence_id": "s2", "usefulness": "topically_relevant_not_evidentiary"},
            ]
        }

    monkeypatch.setattr(selector, "chat_completion_json", fake_chat)
    facets, sentences = _inputs()
    result = classify_facet_sentence_usefulness(
        "The policy may reduce harm.",
        "The policy may reduce harm (Smith, 2020).",
        facets,
        sentences,
    )

    assert result.status == "complete"
    assert result.decision_applied is False
    assert [(item.facet_id, item.sentence_id) for item in result.assessments] == [
        ("facet-a", "sentence-a"),
        ("facet-a", "sentence-b"),
    ]


def test_selector_fails_closed_when_model_omits_a_pair(monkeypatch):
    monkeypatch.setattr(
        selector,
        "chat_completion_json",
        lambda *_args, **_kwargs: {
            "assessments": [
                {"facet_id": "f1", "sentence_id": "s1", "usefulness": "sufficient"}
            ]
        },
    )
    facets, sentences = _inputs()
    result = classify_facet_sentence_usefulness(
        "The policy may reduce harm.",
        "The policy may reduce harm (Smith, 2020).",
        facets,
        sentences,
    )

    assert result.status == "not_assessed"
    assert result.failure_code == "invalid_pair_coverage"
    assert result.assessments == []


def test_selector_rejects_unauthorized_sentence_ids_without_a_call(monkeypatch):
    called = False

    def fake_chat(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(selector, "chat_completion_json", fake_chat)
    facets, sentences = _inputs()
    facets[0].allowed_sentence_ids.append("invented")
    result = classify_facet_sentence_usefulness(
        "The policy may reduce harm.",
        "The policy may reduce harm (Smith, 2020).",
        facets,
        sentences,
    )

    assert result.status == "not_assessed"
    assert result.failure_code == "invalid_input"
    assert called is False


def test_selector_requires_parent_candidate_context(monkeypatch):
    called = False

    def fake_chat(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(selector, "chat_completion_json", fake_chat)
    facets, sentences = _inputs()
    result = classify_facet_sentence_usefulness("", "", facets, sentences)

    assert result.status == "not_assessed"
    assert result.failure_code == "invalid_input"
    assert called is False
