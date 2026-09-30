"""Judgment coaching notes (Phase D): application checks, retry, fallback, cache."""
import json

import pytest

from app.services.judgment_coaching import (
    COACHING_PROMPT_VERSION,
    TEMPLATE_NOTES,
    check_note,
    coach,
    coaching_input,
)
from app.services.llm_service import LLMCallFailure, LLMRoute
from app.services.providers import ProviderConfig

ROUTE = LLMRoute(arm_id="deepseek", model="deepseek-v4-flash", endpoint_host="api.deepseek.com",
                 client_factory=lambda: None, provider_config=ProviderConfig(name="d"))
CLAIM = "All teachers reported that written feedback always raised student motivation"
FACETS = {"g": {"kind": "candidate_as_written", "text": CLAIM, "material_to_aggregate": True},
          "q": {"kind": "quantity", "text": "All", "material_to_aggregate": True},
          "m": {"kind": "modality_or_frequency", "text": "always", "material_to_aggregate": True}}
SENTENCES = {"s1": "Most teachers in the sample said written feedback raised motivation.",
             "s2": "Motivation did not change for a third of students.", "s3": "Unrelated sentence."}


def _arm(arm, q="qualifies", evidence=("s1", "s2")):
    return {"arm_id": arm, "status": "valid", "label": "qualifies", "mappings": [
        {"facet_id": "g", "direction": "qualifies", "evidence_sentence_ids": list(evidence),
         "rationale": f"{arm}: only most teachers, not all."},
        {"facet_id": "q", "direction": q, "evidence_sentence_ids": [], "rationale": ""},
        {"facet_id": "m", "direction": "supports", "evidence_sentence_ids": [], "rationale": ""}]}


PANEL = {"arms": [_arm("deepseek"), _arm("glm", evidence=("s1", "s3")), _arm("qwen", q="none")]}


def _bound():
    return coaching_input("qualified", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES)


def test_only_evidence_two_judges_cited_and_majority_unsupported_facets_are_sent():
    payload, bound = _bound()
    assert [s["text"] for s in payload["evidence_sentences"]] == [SENTENCES["s1"], SENTENCES["s2"]]
    assert [f["text"] for f in payload["unsupported_facets"]] == ["All"]
    assert payload["result"] == "qualified" and "coaching_request" in payload


def _note(text, facet_ids=("f1",), sentence_ids=("s1",)):
    return {"note": text, "facet_ids": list(facet_ids), "sentence_ids": list(sentence_ids)}


def test_a_clean_note_passes_with_ids_restored():
    _, bound = _bound()
    checked, violations = check_note(_note(
        "The statement says “all teachers”, but the source's evidence says “most teachers in the "
        "sample”, so it supports only part of the statement (Rivera, 2019)."), bound)
    assert violations == [] and checked["facet_ids"] == ["q"] and checked["sentence_ids"] == ["s1"]


@pytest.mark.parametrize(("text", "violation"), [
    ("You could say that most teachers reported this instead.", "rewrite_wording"),
    ("Rewrite the sentence so it matches the evidence.", "rewrite_wording"),
    ("Find another source that shows every teacher agreed.", "other_sources"),
    ("Other studies such as Smith (2018) report the same result.", "other_sources"),
    ("See https://example.org for more.", "other_sources"),
    ("The source says “every single teacher agreed without exception”, so check it.", "unsupported_quotation"),
    ("x" * 700, "too_long"),
    ("Your whole claim (f1) is at issue; see s2.", "id_in_note"),
])
def test_each_rule_rejects_the_note(text, violation):
    _, bound = _bound()
    checked, violations = check_note(_note(text), bound)
    assert checked is None and violation in violations


def test_unknown_ids_are_rejected():
    _, bound = _bound()
    assert "unbound_evidence" in check_note(_note("Check the evidence.", sentence_ids=("s9",)), bound)[1]
    assert "unknown_facet" in check_note(_note("Check the evidence.", facet_ids=("f7",)), bound)[1]


def _call(responses, calls):
    def call(system_prompt, user_prompt, *, route, receipt, **_):
        calls.append(user_prompt)
        receipt.update(prompt_tokens=400, completion_tokens=80, reported_cost_usd=0.0001)
        item = responses[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item
    return call


def test_a_rejected_note_is_retried_once_with_the_reasons():
    calls = []
    result = coach("qualified", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE,
                   call=_call([_note("Rewrite it."), _note("The source reports most teachers, not all of them.")], calls))
    assert result["status"] == "model" and result["attempts"] == 2
    assert "rewrite_wording" in calls[1] and result["cost_usd"] == pytest.approx(0.0002)


def test_two_rejections_or_failures_fall_back_to_the_fixed_note():
    result = coach("contradicts", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE,
                   call=_call([_note("Find another source."), LLMCallFailure("timeout")], []))
    assert result["status"] == "template" and result["note"] == TEMPLATE_NOTES["contradicts"]
    assert "other_sources" in result["violations"] and "call_timeout" in result["violations"]


def test_a_cached_note_makes_no_call_and_supported_claims_get_none():
    store = {}
    first = coach("qualified", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE,
                  call=_call([_note("The source reports most teachers, not all of them.")], []),
                  cache_store=store.__setitem__)
    assert first["status"] == "model" and store
    def forbidden(*a, **k):
        raise AssertionError("called")
    again = coach("qualified", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE,
                  call=forbidden, cache_lookup=store.get)
    assert again["cached"] is True and again["note"] == first["note"] and again["version"] == COACHING_PROMPT_VERSION
    assert coach("supported", CLAIM, "", FACETS, PANEL, SENTENCES, route=ROUTE, call=forbidden) == {
        "status": "not_applicable"}


def test_malformed_panel_input_falls_back_without_a_call():
    def forbidden(*a, **k):
        raise AssertionError("called")
    result = coach("insufficient", CLAIM, "", FACETS, {"arms": [{"status": "valid", "mappings": [None]}]},
                   SENTENCES, route=ROUTE, call=forbidden)
    assert result["status"] == "template"



def test_with_one_judge_the_evidence_it_cited_is_sent():
    """Since 2026-09-27 one judge decides; requiring two citing judges sent no evidence at all."""
    payload, _ = coaching_input("qualified", CLAIM, "(Rivera, 2019)", FACETS,
                                {"arms": [_arm("zai_glm", q="none")]}, SENTENCES)
    assert [s["text"] for s in payload["evidence_sentences"]] == [SENTENCES["s1"], SENTENCES["s2"]]
    assert [f["text"] for f in payload["unsupported_facets"]] == ["All"]
    assert "judges" not in payload["result_meaning"]


def test_a_note_giving_advice_is_rejected_and_old_notes_keep_only_their_description():
    # Owner decision 2026-09-30: describe the judgment, no coaching.
    from app.services.judgment_coaching import without_advice
    _, bound = _bound()
    for advice in ("Check how many teachers the source reports.", "You should compare the numbers.",
                   "Consider revising the statement."):
        assert "advice" in check_note(_note(advice), bound)[1]
    old = ("The source says most teachers in the sample, not all teachers. Check how many teachers the "
           "source reports, and consider narrowing the statement.")
    assert without_advice(old) == "The source says most teachers in the sample, not all teachers."
