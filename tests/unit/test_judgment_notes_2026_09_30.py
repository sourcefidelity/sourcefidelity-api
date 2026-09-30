"""Owner decisions 2026-09-30: undecided notes by cause, the key Supports sentence, and the
judge's own attribution reading in the Judgment layout."""
import json

from app.services.judgment_coaching import (
    TEMPLATE_NOTES,
    UNDECIDED_NOTE_VERSION,
    check_note,
    coach,
    coaching_input,
)
from app.services.judgment_report import UNDECIDED_NOTE, UNDECIDED_WORDING_NOTE, judgment_result
from app.services.llm_service import LLMRoute
from app.services.providers import ProviderConfig

ROUTE = LLMRoute(arm_id="deepseek", model="deepseek-v4-flash", endpoint_host="api.deepseek.com",
                 client_factory=lambda: None, provider_config=ProviderConfig(name="d"))
PAYLOAD = {
    "claim": {"text": "Most teachers said feedback helped (Rivera, 2019)."},
    "verification_candidates": {"candidates": [{"candidate_id": "c1", "text": "Most teachers said feedback helped",
                                                "segments": [{"text": "Most teachers said feedback helped"}]}]},
    "facet_evidence_foundation": {
        "source_sentences": [
            {"sentence_id": "s1", "text": "Feedback <b>raised</b> motivation.", "passage_id": "p1",
             "passage_start": 0, "passage_end": 30},
            {"sentence_id": "s2", "text": "Most teachers agreed.", "passage_id": "p1",
             "passage_start": 31, "passage_end": 52},
            {"sentence_id": "s3", "text": "Teachers in the sample said feedback helped.", "passage_id": "p1",
             "passage_start": 53, "passage_end": 97}],
        "candidate_bundles": [{"candidate_id": "c1", "evidence_sentence_ids": ["s1", "s2", "s3"], "facets": [
            {"facet_id": "g", "kind": "candidate_as_written", "text": "Most teachers said feedback helped",
             "material_to_aggregate": True},
            {"facet_id": "q", "kind": "quantity", "text": "Most", "material_to_aggregate": True}]}]},
    "passages": [{"passage_id": "p1", "page_index": 2, "page_label": "7", "character_start": 500}],
}


def _row(state, reason, mappings, coaching=None):
    return {"display_state": state, "reason_code": reason, "candidate_id": "c1", "coaching": coaching,
            "panel": {"arms": [{"arm_id": "zai_glm", "status": "valid", "mappings": mappings}]}}


# ---- undecided notes -------------------------------------------------------------------

def test_an_unresolved_statement_gets_the_owner_wording_note():
    html = judgment_result(_row("not_judged", "judge_undecided", []), PAYLOAD, None)["html"]
    assert UNDECIDED_WORDING_NOTE == ("The model cannot decide if this statement is supported. "
                                      "It could not tell what the statement refers to.")
    assert UNDECIDED_WORDING_NOTE in html


def test_an_uncertain_reading_shows_its_checked_note_and_the_sentences_it_cites():
    mappings = [{"facet_id": "g", "direction": "uncertain", "evidence_sentence_ids": ["s2"], "rationale": "r"}]
    note = {"status": "model", "note": "It is unclear whether &most& refers to the sample.",
            "version": UNDECIDED_NOTE_VERSION}
    result = judgment_result(_row("not_judged", "judge_undecided", mappings, coaching=note), PAYLOAD, None)
    assert result["state"] == "undecided"
    assert "It is unclear whether &amp;most&amp; refers to the sample." in result["html"]
    assert UNDECIDED_WORDING_NOTE not in result["html"]
    assert [s["text"] for s in result["evidence"]] == ["Most teachers agreed."]


def test_an_uncertain_reading_without_a_note_keeps_the_fixed_note():
    mappings = [{"facet_id": "g", "direction": "uncertain", "evidence_sentence_ids": ["s2"], "rationale": "r"}]
    html = judgment_result(_row("not_judged", "judge_undecided", mappings), PAYLOAD, None)["html"]
    assert UNDECIDED_NOTE in html and UNDECIDED_WORDING_NOTE not in html


# ---- the key Supports sentence ---------------------------------------------------------

def test_supports_shows_the_sentence_most_supporting_readings_cite():
    mappings = [{"facet_id": "g", "direction": "supports", "evidence_sentence_ids": ["s1", "s2"], "rationale": ""},
                {"facet_id": "q", "direction": "supports", "evidence_sentence_ids": ["s2"], "rationale": ""}]
    html = judgment_result(_row("supported", "judge_supports", mappings), PAYLOAD, None)["html"]
    assert TEMPLATE_NOTES["supported"] in html
    assert '<ul class="evidence-sentences jw-key-evidence"><li data-key-evidence="2:531:552">' in html
    assert "<q>Most teachers agreed.</q>" in html and "raised" not in html
    assert '<span class="ev-page">p. 7</span>' in html


def test_a_tie_goes_to_the_whole_statement_readings_first_sentence_and_text_is_escaped():
    mappings = [{"facet_id": "g", "direction": "supports", "evidence_sentence_ids": ["s1", "s3"], "rationale": ""}]
    html = judgment_result(_row("supported", "judge_supports", mappings), PAYLOAD, None)["html"]
    assert "<q>Feedback &lt;b&gt;raised&lt;/b&gt; motivation.</q>" in html and "sample" not in html


def test_other_results_show_no_key_sentence():
    mappings = [{"facet_id": "g", "direction": "qualifies", "evidence_sentence_ids": ["s1"], "rationale": ""}]
    html = judgment_result(_row("qualified", "judge_qualifies", mappings), PAYLOAD, None)["html"]
    assert "jw-key-evidence" not in html


# ---- the undecided coaching note -------------------------------------------------------

CLAIM = "Most teachers said feedback helped"
FACETS = {"g": {"kind": "candidate_as_written", "text": CLAIM, "material_to_aggregate": True},
          "a": {"kind": "source_attribution", "text": CLAIM, "material_to_aggregate": True}}
SENTENCES = {"s1": "Teachers said feedback helped.", "s2": "The survey was run by the ministry."}
PANEL = {"arms": [{"arm_id": "zai_glm", "status": "valid", "mappings": [
    {"facet_id": "g", "direction": "supports", "evidence_sentence_ids": ["s1"], "rationale": "Content matches."},
    {"facet_id": "a", "direction": "uncertain", "evidence_sentence_ids": ["s2"],
     "rationale": "Unclear whether the cited author holds this view."}]}]}


def test_the_undecided_request_carries_the_uncertain_part_its_reason_and_its_evidence():
    payload, bound = coaching_input("undecided", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES)
    assert [f["facet_id"] for f in payload["unsupported_facets"]] == ["f1"]
    assert payload["judge_rationales"][0] == "Unclear whether the cited author holds this view."
    assert [s["text"] for s in payload["evidence_sentences"]] == [SENTENCES["s2"]]
    assert bound["state"] == "undecided"


def test_an_undecided_note_that_decides_the_result_is_rejected():
    _, bound = coaching_input("undecided", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES)
    for text in ("The source supports the statement.", "The evidence confirms the survey result."):
        note, violations = check_note({"note": text, "facet_ids": [], "sentence_ids": []}, bound)
        assert note is None and "decides_result" in violations
    for text in ("It is unclear whether the source supports the claim about who holds this view.",
                 "The source does not establish whether the author holds this view."):
        note, violations = check_note({"note": text, "facet_ids": [], "sentence_ids": []}, bound)
        assert note is not None, violations


def test_the_undecided_note_uses_its_own_prompt_version_and_falls_back_to_the_fixed_note():
    prompts = []

    def call(system, user, **kwargs):
        prompts.append(system)
        return {"note": "It is unclear whether the cited author holds this view.", "facet_ids": ["f1"],
                "sentence_ids": ["s1"]}
    result = coach("undecided", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE, call=call)
    assert result["status"] == "model" and result["version"] == UNDECIDED_NOTE_VERSION
    assert "could not decide" in prompts[0] and result["facet_ids"] == ["a"] and result["sentence_ids"] == ["s2"]

    def deciding(system, user, **kwargs):
        return {"note": "The source supports this statement.", "facet_ids": [], "sentence_ids": []}
    fallback = coach("undecided", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES, route=ROUTE, call=deciding)
    assert fallback["status"] == "template" and fallback["note"] == UNDECIDED_NOTE
    assert json.dumps(fallback)


def test_an_undecided_note_in_the_judges_vocabulary_is_rejected():
    _, bound = coaching_input("undecided", CLAIM, "(Rivera, 2019)", FACETS, PANEL, SENTENCES)
    for text in ("The judge could not settle the proposition.", "The sentence is unmarked document voice."):
        note, violations = check_note({"note": text, "facet_ids": [], "sentence_ids": []}, bound)
        assert note is None and "internal_terms" in violations


def test_split_answers_are_undecided_with_the_fixed_note_and_only_majority_answers_are_shown():
    split = {"display_state": "not_judged", "reason_code": "samples_split", "candidate_id": "c1",
             "panel": {"arms": [{"arm_id": "zai_glm", "status": "valid", "in_majority": False, "mappings": []}]}}
    result = judgment_result(split, PAYLOAD, None)
    assert result["state"] == "undecided" and UNDECIDED_NOTE in result["html"]
    assert UNDECIDED_WORDING_NOTE not in result["html"]
    row = _row("supported", "judge_supports",
               [{"facet_id": "g", "direction": "supports", "evidence_sentence_ids": ["s3"], "rationale": ""}])
    row["panel"]["arms"].append({"arm_id": "zai_glm:s2", "status": "valid", "in_majority": False, "mappings": [
        {"facet_id": "g", "direction": "contradicts", "evidence_sentence_ids": ["s1"], "rationale": ""}]})
    result = judgment_result(row, PAYLOAD, None)
    assert [s["text"] for s in result["evidence"]] == ["Teachers in the sample said feedback helped."]
