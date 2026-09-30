import copy

import pytest

from app.services.evidence_context_experiment import (
    Region, bind_comparison, enrich_regions, focus_regions, merge_regions, multiscale_regions,
    prepare_comparison,
)
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from app.services.verification_evidence import ClaimEvidence, _SourcePage, _SourceStructuralSpan


TEXT = "Studios controlled publicity for their stars. They controlled interviews and photographs. Other industries had different arrangements."


def fixture():
    pages = [_SourcePage(0, "1", TEXT)]
    claim = ClaimEvidence(claim_id="c", paper_version_id="p", text="Studios controlled publicity.",
                          claim_type="paraphrase")
    regions = [Region(0, 0, len(TEXT), TEXT, "body_prose", ("original-1",))]
    req = prepare_comparison(claim=claim, source_title="Stars", regions=regions, pages=pages,
                             source_binding={"content": "bound"}, mode="diagnostic")
    return pages, claim, regions, req


def answer(**changes):
    choice = dict(region_id="r000", sentence_ids=["s000"], purpose="primary", context_for=None,
                  why_useful="Describes the attributed relationship.")
    choice.update(changes)
    return {"selected": [choice]}


def bind(req, raw, pages, claim):
    return bind_comparison(req, raw, pages=pages, claim=claim, source_binding={"content": "bound"})


def test_exact_extract_and_empty_are_not_support():
    pages, claim, _, req = fixture()
    result = bind(req, answer(), pages, claim)
    assert result['selected'][0]['text'] == TEXT[:TEXT.index('.')+1]
    assert not result['source_support_assessed'] and not result['semantic_acceptance']
    assert bind(req, {"selected": []}, pages, claim)['outcome'] == "no_selection_from_supplied_inputs"


@pytest.mark.parametrize("changes", [
    {"region_id": "fabricated"}, {"sentence_ids": ["s009"]},
    {"sentence_ids": ["s000", "s002"]}, {"sentence_ids": ["s000", "s000"]},
    {"purpose": "necessary_context", "context_for": "r000"},
    {"purpose": "additional_material"}, {"context_for": "r099"},
    {"purpose": "supports"}, {"support": True},
])
def test_invalid_selection(changes):
    pages, claim, _, req = fixture()
    with pytest.raises(ValueError):
        bind(req, answer(**changes), pages, claim)


@pytest.mark.parametrize("change", ["request", "page", "claim", "scope"])
def test_stale_input(change):
    pages, claim, _, req = fixture()
    binding = {"content": "bound"}
    if change == "request":
        req['prompt'] += " altered"
    elif change == "page":
        pages = [_SourcePage(0, "1", TEXT + " Changed.")]
    elif change == "claim":
        claim = claim.model_copy(update={"text": "Another claim."})
    else:
        binding = {"content": "other"}
    with pytest.raises(ValueError):
        bind_comparison(req, answer(), pages=pages, claim=claim, source_binding=binding)


def test_merge_preserves_every_origin_and_coverage_without_gap():
    pages, _, _, _ = fixture()
    rows = [Region(0, 0, 50, TEXT[:50], "body_prose", ("a",)),
            Region(0, 30, 70, TEXT[30:70], "body_prose", ("b",)),
            Region(0, 71, len(TEXT), TEXT[71:], "body_prose", ("c",))]
    result = merge_regions(rows, pages)
    assert len(result) == 2 and result[0].text == TEXT[:70]
    assert result[0].origins == ("a", "b")


def test_merge_does_not_cross_role_or_page():
    pages = [_SourcePage(0, "1", TEXT), _SourcePage(1, "2", TEXT)]
    rows = [Region(i, 0, len(TEXT), TEXT, role) for i, role in
            [(0, "body_prose"), (0, "citation_notes"), (1, "body_prose")]]
    assert len(merge_regions(rows, pages)) == 3


def test_context_parent_exact_and_input_unchanged():
    pages, _, _, _ = fixture()
    start = TEXT.index("They")
    hit = Region(0, start, TEXT.index('.', start)+1, TEXT[start:TEXT.index('.', start)+1],
                 "body_prose", ("hit",))
    output = enrich_regions([hit], pages)
    assert output[0].text == TEXT and output[0].origins == hit.origins
    assert hit.start == start


def test_context_stays_within_structural_boundary():
    text = "Private publication metadata.\n\n" + TEXT
    n = text.index("Studios")
    page = _SourcePage(0, "1", text, (_SourceStructuralSpan(0, n-2, "publication_metadata"),))
    r = Region(0, n, n+43, text[n:n+43], "body_prose")
    assert enrich_regions([r], [page])[0].start == n


def test_excluded_roles_never_model_inputs():
    pages, claim, rows, _ = fixture()
    rows[0] = Region(0, 0, len(TEXT), TEXT, "publication_metadata")
    with pytest.raises(ValueError, match="excluded_role"):
        prepare_comparison(claim=claim, source_title="T", regions=rows, pages=pages,
                           source_binding={"bound": True}, mode="diagnostic")


def test_budget_abstains_without_pruning():
    pages, claim, rows, _ = fixture()
    original = copy.deepcopy(rows)
    with pytest.raises(LLMInputBudgetExceeded):
        prepare_comparison(claim=claim, source_title="T", regions=rows, pages=pages,
                           source_binding={"bound": True}, mode="diagnostic", max_input_tokens=20)
    assert rows == original


def test_bad_coordinates_rejected():
    pages, _, _, _ = fixture()
    with pytest.raises(ValueError, match="source_region_binding"):
        merge_regions([Region(0, 1, 10, TEXT[:9], "body_prose")], pages)


def test_multiscale_retains_protected_and_respects_budget():
    pages, _, rows, _ = fixture()
    protected = rows[:]
    result = multiscale_regions(pages, "studios publicity", rows, protected=protected, top_k=1)
    assert result == protected
    assert rows == protected


def test_redaction_precedes_labels():
    text = "Name: Example Student\nStudios controlled publicity."
    page = _SourcePage(0, "1", text)
    _, claim, _, _ = fixture()
    request = prepare_comparison(claim=claim, source_title="T",
        regions=[Region(0, 0, len(text), text, "body_prose")], pages=[page],
        source_binding={"bound": True}, mode="diagnostic")
    assert "Example Student" not in request['prompt']
    assert request['regions'][0]['text'] == text


def test_focusing_keeps_every_candidate_and_exact_offsets():
    text = "Gardens contain decorative flowers. Studios managed publicity and press interviews. Later paragraphs discuss unrelated matters."
    region = Region(0, 50, 50+len(text), text, "body_prose", ("one",))
    result = focus_regions([region, region], "studios publicity interviews", max_characters=55)
    assert len(result) == 2
    assert all(r.text == "Studios managed publicity and press interviews." for r in result)
    assert result[0].start == 50+text.index("Studios")
    assert result[0].origins == region.origins


def test_focusing_does_not_claim_fragment_is_complete():
    text = "A very long unfinished sentence " * 30
    r = Region(0, 0, len(text), text, "body_prose", ("one",))
    focused = focus_regions([r], "sentence", max_characters=50)
    assert len(focused) == 1 and len(focused[0].text) == 50
    _, claim, _, _ = fixture()
    page = _SourcePage(0, "1", text)
    request = prepare_comparison(claim=claim, source_title="T", regions=focused, pages=[page],
                                 source_binding={"content": "bound"}, mode="diagnostic")
    with pytest.raises(ValueError, match="extract_boundary"):
        bind(request, answer(), [page], claim)


def test_same_region_twice_retains_explicit_contract_failure():
    pages, claim, _, req = fixture()
    raw = answer()
    raw['selected'].append(answer(purpose="additional_material", sentence_ids=["s001"])['selected'][0])
    with pytest.raises(ValueError, match="selection_order"):
        bind(req, raw, pages, claim)


def test_necessary_context_must_have_preceding_selected_target():
    pages, claim, _, req = fixture()
    with pytest.raises(ValueError):
        bind(req, answer(purpose="necessary_context", context_for="r099"), pages, claim)


def test_source_fragment_cannot_be_repaired_by_model():
    pages, claim, _, req = fixture()
    raw = answer()
    raw['selected'][0]['text'] = "A generated replacement."
    with pytest.raises(ValueError):
        bind(req, raw, pages, claim)
