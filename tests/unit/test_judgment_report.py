"""Judgment result block in the one-layout window: reasoning and note, references as keys; escaped."""
from html import escape

import pytest

from app.services.judgment_report import (
    LABELS,
    candidate_paper_ranges,
    candidate_text,
    judgment_result,
)

PAYLOAD = {
    "claim": {"text": "Teachers <b>said</b> feedback helped (Rivera, 2019)."},
    "verification_candidates": {"candidates": [{
        "candidate_id": "c1", "text": "Teachers <b>said</b> feedback helped",
        "segments": [{"text": "Teachers <b>said</b> feedback helped", "paper_start": 100, "paper_end": 135,
                      "local_start": 0, "local_end": 35, "role": "clause"}]}]},
    "facet_evidence_foundation": {
        "source_sentences": [{"sentence_id": "s1", "text": "Feedback <script>x</script> raised motivation.",
                              "passage_id": "p1", "passage_start": 10, "passage_end": 56}],
        "candidate_bundles": [{"candidate_id": "c1", "evidence_sentence_ids": ["s1"], "facets": [
            {"facet_id": "f1", "kind": "candidate_as_written", "text": "Teachers said feedback helped",
             "material_to_aggregate": True}]}]},
    "passages": [{"passage_id": "p1", "page_index": 4, "page_label": "12", "character_start": 1000}],
}


def _row(state, reason, label="supports", coaching=None, rationale="The <i>source</i> agrees (s1)."):
    arm = {"arm_id": "zai_glm", "status": "valid", "label": label, "mappings": [
        {"facet_id": "f1", "direction": label, "evidence_sentence_ids": ["s1"], "rationale": rationale}]}
    return {"display_state": state, "reason_code": reason, "candidate_id": "c1",
            "panel": {"arms": [arm]}, "coaching": coaching}


def test_owner_approved_labels():
    assert LABELS == {"supported": "Supports", "qualified": "Qualified or Mixed", "contradicts": "Contradicts",
                      "insufficient": "Not Supported", "undecided": "LLM Undecided",
                      "not_judged": "Not Judged"}


def test_block_is_the_escaped_note_without_the_judges_reasoning():
    # Owner request 2026-09-28: the note, not the judge's reasoning.
    result = judgment_result(_row("qualified", "judge_qualifies",
                                  coaching={"status": "model", "note": "The <u>scope</u> differs.",
                                            "version": "judgment-coaching-v4"}), PAYLOAD, None)
    html = result["html"]
    assert result["label"] == "Qualified or Mixed" and result["state"] == "qualified"
    assert "The &lt;u&gt;scope&lt;/u&gt; differs." in html
    assert "source" not in html and "jw-why" not in html and "ev-ref" not in html
    assert "<script>" not in html
    # The label lives in the part's heading; the evidence list is the window's own.
    assert "jw-state" not in html and "<ol" not in html
    # The sentence the judge relied on is still returned for the evidence list.
    assert result["evidence"] == [{"key": "4:1010:1056", "text": "Feedback <script>x</script> raised motivation.",
                                   "page": "12"}]


@pytest.mark.parametrize("unwanted", ["experimental", "Experimental", "not a grade", "Citation", "What to check",
                                      "Evidence the judges", "Other judges", "Parts not supported", "judges agree"])
def test_no_explanatory_text_is_added(unwanted):
    result = judgment_result(_row("supported", "judge_supports", coaching={"status": "model", "note": "A note."}),
                             PAYLOAD, None)
    assert unwanted not in result["html"]


def test_not_judged_has_no_reasoning_and_a_retry_when_a_judge_failed():
    result = judgment_result({"display_state": "not_judged", "reason_code": "judge_failed", "candidate_id": "c1"},
                             PAYLOAD, None)
    assert result["label"] == "Not Judged" and "data-judgment-retry" in result["html"]
    assert "jw-why" not in result["html"] and result["evidence"] == []


def test_a_sentence_named_only_in_the_reasoning_is_still_returned():
    payload = {**PAYLOAD, "facet_evidence_foundation": {
        **PAYLOAD["facet_evidence_foundation"],
        "source_sentences": [*PAYLOAD["facet_evidence_foundation"]["source_sentences"],
                             {"sentence_id": "s2", "text": "Gomery disagrees.", "passage_id": "p1",
                              "passage_start": 60, "passage_end": 77}],
        "candidate_bundles": [{**PAYLOAD["facet_evidence_foundation"]["candidate_bundles"][0],
                               "evidence_sentence_ids": ["s1", "s2"]}]}}
    result = judgment_result(_row("contradicts", "judge_contradicts", label="contradicts",
                                  rationale="s1 agrees, but s2 (Gomery) disagrees; s9 is unknown."), payload, None)
    assert [e["key"] for e in result["evidence"]] == ["4:1010:1056", "4:1060:1077"]
    assert "s9 is unknown" not in result["html"]


def test_a_fixed_note_is_shown_in_the_current_approved_wording():
    from app.services.judgment_coaching import TEMPLATE_NOTES
    row = _row("contradicts", "judge_contradicts", label="contradicts",
               coaching={"status": "template", "note": "The judges found something."})
    html = judgment_result(row, PAYLOAD, None)["html"]
    assert escape(TEMPLATE_NOTES["contradicts"]) in html and "The judges found" not in html


def test_candidate_wording_and_paper_ranges_are_exact():
    assert candidate_text(PAYLOAD, "c1") == "Teachers <b>said</b> feedback helped"
    assert candidate_paper_ranges(PAYLOAD, "c1") == [(100, 135)]
    assert candidate_paper_ranges(PAYLOAD, "missing") == []


def test_a_statement_describing_research_is_judged_as_an_example_of_it():
    # Owner decision and wording 2026-10-09.
    from app.services.citation_structure import describes_research_literature as describes
    assert describes("Most research on harbour cranes focuses on safety (Ames, 2020; Bell, 2021).")
    assert describes("There is much less research investigating river ports, all published recently (Cole, 2023).")
    assert describes("Few studies have examined crane training (Ames, 2020).")
    assert not describes("Several studies show that cranes speed loading (Ames, 2020).")
    assert not describes("In most studies of ports, crews feel training is too short (Bell, 2021).")
    assert not describes("Research has shown that ports that examine their cranes save money (Ames, 2020).")
    assert not describes("The most recent work to feature the harbour is a film (Cole, 2020).")
    payload = {**PAYLOAD, "claim": {"text": "Most research on harbour cranes focuses on safety (Rivera, 2019)."}}
    line = ("Judged as an example of the research the statement describes; the statement&#x27;s overall "
            "picture of the research is the writer&#x27;s own.")
    assert line in judgment_result(_row("supported", "judge_supports"), payload, None)["html"]
    assert line not in judgment_result(_row("supported", "judge_supports"), PAYLOAD, None)["html"]


def test_only_a_research_description_gets_the_example_question():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_facet_evidence_judgment import _prepared
    from app.services import facet_evidence_judgment as judge
    source = "This study examines crane safety training in river ports."
    for claim, expected in (("Most research on river ports focuses on crane safety (Wood, 2012).", True),
                            ("River ports train crane crews in safety (Wood, 2012).", False)):
        prepared = judge.prepare_candidate_prompts(_prepared(source, claim, "(Wood, 2012)"), max_input_tokens=100_000)
        items = [i for i in prepared.items if isinstance(i, judge.PreparedCandidate)]
        assert items
        for item in items:
            assert (item.system_prompt == judge._EXAMPLE_OF_RESEARCH_PROMPT) is expected
            assert ('"example_of_described_research"' in item.user_prompt) is expected
            if not expected:
                assert item.system_prompt == judge._SYSTEM_PROMPT
