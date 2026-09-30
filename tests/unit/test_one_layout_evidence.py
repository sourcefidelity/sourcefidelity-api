"""One layout (owner decisions 2026-09-28): GLM-selected sentences are the citation's one evidence list."""
from types import SimpleNamespace

import pytest

from app.services.evidence_report import (
    EVIDENCE_LABEL_CSS,
    _member_label,
    _render_member,
    _with_sentence_evidence,
)
from app.services.evidence_sentence_selection import attach_sentence_evidence
from app.services.llm_service import LLMRoute
from app.services.providers import ProviderConfig
from tests.unit.test_facet_evidence_judgment import _prepared

ROUTE = LLMRoute(arm_id="zai_glm", model="glm-5.3-flash", endpoint_host="open.bigmodel.cn",
                 client_factory=lambda: None, provider_config=ProviderConfig(name="g"), max_output_tokens=8000)


def _artifact():
    return _prepared("Context sentence one discusses market structure. Licensing supports competition through "
                     "provider diversity. A final sentence closes the passage.",
                     "Licensing supports competition through provider diversity (Smith, 2020).", "(Smith, 2020)")


def _call(response):
    def call(system_prompt, user_prompt, *, receipt, **_):
        receipt.update(prompt_tokens=900, completion_tokens=40)
        return response
    return call


def test_glm_picks_are_bound_to_exact_spans():
    artifact = attach_sentence_evidence(_artifact(), route=ROUTE, call=_call(
        {"selections": [{"sentence_id": "s2", "reason": "bears_on_statement"}]}))
    selection = artifact.sentence_evidence
    assert selection.status == "selected" and selection.model == "glm-5.3-flash"
    item = selection.items[0]
    passage = next(p for p in artifact.passages if p.passage_id == item.passage_id)
    assert passage.text[item.passage_start:item.passage_end] == item.text
    assert "Licensing supports competition" in item.text


@pytest.mark.parametrize("response,status", [
    ({"selections": []}, "empty"),
    ({"selections": [{"sentence_id": "s99", "reason": "bears_on_statement"}]}, "invalid"),
    ({"selections": [{"sentence_id": "s1", "reason": "supports"}]}, "invalid"),
])
def test_empty_and_rejected_responses(response, status):
    assert attach_sentence_evidence(_artifact(), route=ROUTE, call=_call(response)).sentence_evidence.status == status


def test_without_a_glm_route_the_gate_display_is_kept():
    # Tests run with Z.ai not configured: no call, status says so.
    artifact = attach_sentence_evidence(_artifact(), call=lambda *a, **k: pytest.fail("no call expected"))
    assert artifact.sentence_evidence.status == "unavailable"


def _record(selection, passages):
    return SimpleNamespace(id="rec-1", report_payload={"sentence_evidence": selection, "passages": passages})


PASSAGES = [{"passage_id": "p1", "page_index": 4, "page_label": "12", "character_start": 1000}]


def test_selected_sentences_are_the_windows_collapsed_evidence_list():
    view = _with_sentence_evidence({"coverage_level": "full_text"}, _record(
        {"status": "selected", "items": [{"passage_id": "p1", "passage_start": 10, "passage_end": 40,
                                          "page_index": 4, "text": "Stu-\ndios sold <b>films</b>.",
                                          "reason": "bears_on_statement"}]}, PASSAGES), SimpleNamespace(reference_ids=["r"]))
    assert view["verification_report_id"] == "rec-1"
    assert view["evidence_sentences"] == [{"key": "4:1010:1040", "text": "Studios sold <b>films</b>.", "page": "12",
                                           "reason": "bears_on_statement"}]
    html = _render_member({**view, "source": {"author": "", "year": "", "title": "", "raw_reference": "Ref"},
                           "best_evidence": {"text": "OLD PASSAGE", "display_text": "OLD PASSAGE"}}, grouped=True)
    # Collapsed under "Evidence", unnumbered, page kept (owner request 2026-09-28).
    assert ('<details class="evidence-disclosure"><summary>Evidence</summary><ul class="evidence-sentences" '
            'data-evidence-list><li data-evidence-key="4:1010:1040"><span class="ev-page">p. 12</span>') in html
    assert "&lt;b&gt;films&lt;/b&gt;" in html and "OLD PASSAGE" not in html
    # The note comes before the evidence.
    assert html.index('data-judgment-record="rec-1"') < html.index('evidence-disclosure')


def test_an_empty_selection_uses_the_existing_wording():
    view = _with_sentence_evidence({"coverage_level": "full_text", "availability": ""},
                                   _record({"status": "empty", "items": []}, PASSAGES), SimpleNamespace(reference_ids=["r"]))
    assert view["evidence_sentences"] == []
    assert view["availability"].startswith("No clearly relevant passage was found. Check the source manually")


def test_without_a_selection_there_is_no_sentence_list():
    view = _with_sentence_evidence({"coverage_level": "full_text"},
                                   _record({"status": "unavailable"}, PASSAGES), SimpleNamespace(reference_ids=["r"]))
    assert "evidence_sentences" not in view


@pytest.mark.parametrize("level,kind", [("full_text", "Full Text Retrieved"), ("abstract_only", "Abstract Retrieved"),
                                        ("partial_text", "Limited Text Retrieved"), ("unavailable", "No Text Retrieved")])
def test_label_is_judgment_then_evidence_kind_underlined_as_one(level, kind):
    label = _member_label({"coverage_level": level, "verification_report_id": "rec-1"})
    assert f'<span class="evidence-kind kind-{level}">{kind}</span></span>' in label
    if level == "full_text":
        # Filled in, with its underline style, by the Judgment script when the result arrives.
        assert ('<span class="judgment-label" data-judgment-label="rec-1"><span class="judgment-part" hidden>'
                '</span><span class="judgment-sep" hidden> - </span>') in label
    else:
        assert ('<span class="judgment-label jk state-not_judged"><span class="judgment-part">Not Judged</span>'
                '<span class="judgment-sep"> - </span>') in label


def test_label_text_is_black():
    assert ".member-label,.member-label .evidence-kind{color:var(--ink)}" in EVIDENCE_LABEL_CSS


def test_a_film_citation_is_a_media_reference_whatever_its_boundary_status():
    label = _member_label({"status": "citation_not_assessed", "coverage_level": "unavailable",
                           "source": {"source_kind": "traditional_media", "raw_reference": "Burton, T. (2010). Alice."}})
    assert label == "Media Reference - Cannot Retrieve"


def test_citations_carry_no_evidence_kind_highlight():
    assert (".citation-overlay.member-target .source-highlight:not(.unverified-highlight):not(.academic-highlight)"
            "{fill-opacity:0}") in EVIDENCE_LABEL_CSS
