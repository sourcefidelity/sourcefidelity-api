import copy

import pytest

from app.services.evidence_window_experiment import inspect_window_choice
from app.services.evidence_context_experiment import Region, prepare_comparison
from app.services.verification_evidence import _SourcePage
from tests.unit.test_evidence_context_experiment import fixture, answer, bind


@pytest.mark.parametrize("choice,expected", [
    (None, "no_window_selected"),
    ("a1", "listed_window"),
    ("b1", "window_exceeds_supplied_sentences"),
    ("a2", "window_exceeds_supplied_sentences"),
    ("a0", "window_ineligible_boundary_or_length"),
    ("b0", "window_ineligible_boundary_or_length"),
    ("b01", "invalid_window_syntax"),
    ([], "invalid_window_syntax"),
])
def test_diagnostic_never_repairs_or_mutates(choice, expected):
    passage = {"text": "[s000] continued here. [s001] A complete sentence.",
               "source_sentences": ["s000", "s001"]}
    before = copy.deepcopy(passage)
    assert inspect_window_choice(passage, choice) == expected
    assert passage == before


def test_diagnostic_checks_original_inventory():
    with pytest.raises(ValueError, match="ambiguous_source_labels"):
        inspect_window_choice({"text": "[s000] A sentence.",
                               "source_sentences": ["s001"]}, "a0")


def test_fragment_failure_retains_original_type_and_message_without_text():
    _, claim, _, _ = fixture()
    text = "continued from the preceding page. A complete sentence follows."
    pages = [_SourcePage(0, "1", "The sentence begins"), _SourcePage(1, "2", text)]
    regions = [Region(1, 0, len(text), text, "body_prose")]
    request = prepare_comparison(claim=claim, source_title="Study", regions=regions,
                                 pages=pages, source_binding={"content": "bound"}, mode="diagnostic")
    with pytest.raises(ValueError, match="^extract_boundary$") as exc:
        bind(request, answer(), pages, claim)
    assert type(exc.value) is ValueError
    assert exc.value.reason == "fragment_boundary"
    # No automatic joining, endpoint clipping, or rejection of the intact sentence.
    result = bind(request, answer(sentence_ids=["s001"]), pages, claim)
    assert result["selected"][0]["text"] == "A complete sentence follows."
    assert result["selected"][0]["page_index"] == 1


def test_length_failure_is_separate_from_fragment_failure():
    _, claim, _, _ = fixture()
    text = "A " + "word " * 300 + "ends."
    pages = [_SourcePage(0, "1", text)]
    request = prepare_comparison(claim=claim, source_title="Study",
                                 regions=[Region(0, 0, len(text), text, "body_prose")],
                                 pages=pages, source_binding={"content": "bound"}, mode="diagnostic")
    with pytest.raises(ValueError) as exc:
        bind(request, answer(), pages, claim)
    assert exc.value.reason == "extract_length_exceeded"
