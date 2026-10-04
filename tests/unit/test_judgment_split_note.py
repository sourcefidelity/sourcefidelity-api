"""Owner decision 2026-10-03: answers with no shared result get a note about the statement
and the source, never about the answers differing."""
from app.services.judgment_coaching import (
    SPLIT_NOTE_VERSION,
    UNDECIDED_NOTE_VERSION,
    check_note,
    coach,
    coaching_input,
)
from app.services.judgment_report import UNDECIDED_NOTE, judgment_result
from app.services.llm_service import LLMRoute
from app.services.providers import ProviderConfig

ROUTE = LLMRoute(arm_id="deepseek", model="deepseek-v4-flash", endpoint_host="api.deepseek.com",
                 client_factory=lambda: None, provider_config=ProviderConfig(name="d"))
CLAIM = "The film is marked by long shots and thematic ambiguity"
FACETS = {"g": {"kind": "candidate_as_written", "text": CLAIM, "material_to_aggregate": True},
          "a": {"kind": "source_attribution", "text": CLAIM, "material_to_aggregate": True}}
SENTENCES = {"x1": "Slow cinema is defined by long shots.", "x2": "This film has no notably long shots.",
             "x3": "The film was released in 1979."}
PANEL = {"arms": [
    {"arm_id": "zai_glm", "status": "valid", "in_majority": False, "label": "contradicts", "mappings": [
        {"facet_id": "g", "direction": "contradicts", "evidence_sentence_ids": ["x2"],
         "rationale": "The source (s9, s10) defines slow cinema, but s11 says the film lacks long shots."}]},
    {"arm_id": "zai_glm:s2", "status": "invalid", "in_majority": False, "mappings": []},
    {"arm_id": "zai_glm:s3", "status": "valid", "in_majority": False, "label": "qualifies", "mappings": [
        {"facet_id": "g", "direction": "qualifies", "evidence_sentence_ids": ["x1", "x2"],
         "rationale": "Long shots define slow cinema; thematic ambiguity is never discussed."},
        {"facet_id": "a", "direction": "supports", "evidence_sentence_ids": ["x3"], "rationale": "Attributed."}]}]}


def test_the_split_request_uses_every_valid_answer_without_judge_aliases_or_its_result():
    payload, bound = coaching_input("split", CLAIM, "(Hess, 2020)", FACETS, PANEL, SENTENCES)
    assert payload["result"] == "undecided" and bound["state"] == "split"
    assert [f["text"] for f in payload["unsupported_facets"]] == [CLAIM]
    assert [s["text"] for s in payload["evidence_sentences"]] == [SENTENCES["x2"], SENTENCES["x1"]]
    assert payload["judge_rationales"][0] == ("The source defines slow cinema, but a source sentence says "
                                              "the film lacks long shots.")
    assert "contradicts" not in str(payload) and "qualifies" not in str(payload)


def test_a_split_note_about_the_answers_is_rejected():
    _, bound = coaching_input("split", CLAIM, "(Hess, 2020)", FACETS, PANEL, SENTENCES)
    for text in ("The answers disagreed about this statement.", "The readings differ on the long shots.",
                 "Two of the models found a problem.", "There was no majority."):
        note, violations = check_note({"note": text, "facet_ids": [], "sentence_ids": []}, bound)
        assert note is None and "mentions_answers" in violations, text
    note, violations = check_note({"note": "The source contradicts the statement.", "facet_ids": [],
                                   "sentence_ids": []}, bound)
    assert note is None and "decides_result" in violations
    ok = ("The source describes slow cinema by its long shots but says this film has no notably long shots, "
          "and it never discusses thematic ambiguity.")
    note, violations = check_note({"note": ok, "facet_ids": ["f1"], "sentence_ids": ["s1"]}, bound)
    assert note is not None, violations


def test_the_split_note_has_its_own_prompt_and_version_and_falls_back_to_the_fixed_note():
    prompts = []

    def call(system, user, **kwargs):
        prompts.append(system)
        return {"note": "The source never discusses thematic ambiguity.", "facet_ids": ["f1"], "sentence_ids": []}
    result = coach("split", CLAIM, "(Hess, 2020)", FACETS, PANEL, SENTENCES, route=ROUTE, call=call)
    assert result["status"] == "model" and result["version"] == SPLIT_NOTE_VERSION
    assert "never say how many there were" in prompts[0] and result["facet_ids"] == ["g"]

    def about_answers(system, user, **kwargs):
        return {"note": "The answers disagreed.", "facet_ids": [], "sentence_ids": []}
    fallback = coach("split", CLAIM, "(Hess, 2020)", FACETS, PANEL, SENTENCES, route=ROUTE, call=about_answers)
    assert fallback["status"] == "template" and fallback["note"] == UNDECIDED_NOTE


def _split_row(coaching):
    return {"display_state": "not_judged", "reason_code": "samples_split", "candidate_id": "c1",
            "coaching": coaching, "panel": PANEL}


def test_the_report_shows_a_checked_split_note_and_otherwise_the_fixed_note():
    note = {"status": "model", "version": SPLIT_NOTE_VERSION, "note": "The source never discusses ambiguity."}
    html = judgment_result(_split_row(note), {}, None)["html"]
    assert "The source never discusses ambiguity." in html and UNDECIDED_NOTE not in html
    for coaching in (None, {"status": "template", "note": UNDECIDED_NOTE},
                     {"status": "model", "version": UNDECIDED_NOTE_VERSION, "note": "Old note."}):
        html = judgment_result(_split_row(coaching), {}, None)["html"]
        assert UNDECIDED_NOTE in html and "Old note." not in html


def test_a_word_the_statement_itself_uses_is_not_about_the_answers():
    claim = "The film emphasises practical models over digital effects"
    _, bound = coaching_input("split", claim, "(Ochonicky, 2019)", {"g": {"kind": "candidate_as_written",
                              "text": claim, "material_to_aggregate": True}}, PANEL, SENTENCES)
    note, violations = check_note({"note": "The source never mentions practical models.", "facet_ids": [],
                                   "sentence_ids": []}, bound)
    assert note is not None, violations
    note, violations = check_note({"note": "The answers differ on practical models.", "facet_ids": [],
                                   "sentence_ids": []}, bound)
    assert note is None and "mentions_answers" in violations

