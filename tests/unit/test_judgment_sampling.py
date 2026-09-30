"""Repeat sampling for Phase E: keys, temperature, passage order and decision rules.

Fake judges only: no provider is contacted.
"""
import json
from dataclasses import replace

import pytest

from app.services import judgment_sampling as sampling
from app.services.judgment_panel import arm_cache_key
from app.services.llm_input_boundary import json_data_envelope
from test_judgment_panel import ROUTES, _fake_call, _prepared_candidate

DEEPSEEK = ROUTES[0]


def _with_groups(item, groups):
    """The candidate with a synthetic evidence list: group -> number of sentences."""
    data = json.loads(item.user_prompt)
    data["evidence_sentences"] = [
        {"sentence_id": f"S{g}{n}", "passage_group": g, "passage_sequence": n, "text": f"g{g} s{n}"}
        for g, count in groups.items() for n in range(1, count + 1)]
    return replace(item, user_prompt=json_data_envelope(data))


def _sample(spec, directions=None, seen=None, lookup=None):
    artifact, prepared, item = _prepared_candidate()
    call = _fake_call(directions or {"deepseek": "supports"}, seen=seen)
    return sampling.sample_arm(item, artifact, prepared.sentences, DEEPSEEK, "policy", spec,
                               cache_lookup=lookup, call=call), item


# ---- sample identity ------------------------------------------------------------

def test_sample_keys_never_equal_the_panel_key_and_differ_per_spec():
    _, _, item = _prepared_candidate()
    panel_key = arm_cache_key(DEEPSEEK, "policy", item)
    specs = [sampling.SampleSpec(0), sampling.SampleSpec(1), sampling.SampleSpec(0, temperature=0.7),
             sampling.SampleSpec(0, order_seed=1)]
    keys = {sampling.sample_cache_key(DEEPSEEK, "policy", item, s) for s in specs}
    assert len(keys) == len(specs)
    assert panel_key not in keys
    assert sampling.sample_cache_key(DEEPSEEK, "policy", item, specs[0]) == \
        sampling.sample_cache_key(DEEPSEEK, "policy", item, sampling.SampleSpec(0))


@pytest.mark.parametrize("kwargs", [{"index": -1}, {"index": 0, "temperature": 1.5},
                                    {"index": 0, "temperature": -0.1}])
def test_invalid_specs_are_refused(kwargs):
    with pytest.raises(ValueError):
        sampling.SampleSpec(**kwargs)


# ---- one sample -------------------------------------------------------------------

def test_a_sample_passes_its_temperature_and_the_unchanged_prompt():
    captured = {}

    def call(system_prompt, user_prompt, *, temperature, route, receipt, **_):
        captured.update(temperature=temperature, user=user_prompt, arm=route.arm_id)
        receipt.update(prompt_tokens=10, completion_tokens=5, reported_cost_usd=0.0001)
        return _fake_call({"deepseek": "supports"})(system_prompt, user_prompt, route=route, receipt=receipt)

    artifact, prepared, item = _prepared_candidate()
    result = sampling.sample_arm(item, artifact, prepared.sentences, DEEPSEEK, "policy",
                                 sampling.SampleSpec(2, temperature=0.7), call=call)
    assert captured == {"temperature": 0.7, "user": item.user_prompt, "arm": "deepseek"}
    assert (result.arm.status, result.arm.label, result.order_changed) == ("valid", "supports", False)
    assert result.arm.cache_key == sampling.sample_cache_key(DEEPSEEK, "policy", item, result.spec)


def test_a_cached_sample_makes_no_call_and_a_failure_is_recorded():
    seen = []
    first, _ = _sample(sampling.SampleSpec(0), seen=seen)
    cached, _ = _sample(sampling.SampleSpec(0), seen=seen, lookup=lambda key: first.arm.response)
    assert len(seen) == 1 and cached.arm.cached and cached.arm.label == "supports"
    failed, _ = _sample(sampling.SampleSpec(1), directions={"deepseek": {"mappings": "bad"}})
    assert (failed.arm.status, failed.arm.failure) == ("failed", "invalid_response")


def test_the_production_panel_is_unchanged_at_temperature_zero():
    temperatures = []

    def call(system_prompt, user_prompt, *, temperature, route, receipt, **_):
        temperatures.append(temperature)
        return _fake_call({"deepseek": "supports", "glm": "supports", "qwen": "supports"})(
            system_prompt, user_prompt, route=route, receipt=receipt)

    from app.services import judgment_panel as panel
    artifact, prepared, item = _prepared_candidate()
    panel.judge_candidate(item, artifact, prepared.sentences, ROUTES, "policy", call=call)
    assert temperatures == [0.0, 0.0, 0.0]


# ---- passage order ------------------------------------------------------------------

def test_reordering_moves_whole_passage_groups_and_keeps_every_sentence():
    _, _, item = _prepared_candidate()
    multi = _with_groups(item, {1: 2, 2: 1, 3: 3, 4: 1})
    original = json.loads(multi.user_prompt)["evidence_sentences"]
    moved = None
    for seed in range(10):
        candidate, changed = sampling.reorder_passages(multi, seed)
        if changed:
            moved = json.loads(candidate.user_prompt)
            break
    assert moved is not None
    sentences = moved["evidence_sentences"]
    assert sorted(map(json.dumps, sentences)) == sorted(map(json.dumps, original))
    groups = [s["passage_group"] for s in sentences]
    assert groups != [s["passage_group"] for s in original]
    # Each group stays contiguous and in its own reading order.
    assert all(groups.index(g) + groups.count(g) - 1 == len(groups) - 1 - groups[::-1].index(g)
               for g in set(groups))
    for g in set(groups):
        assert [s["passage_sequence"] for s in sentences if s["passage_group"] == g] == \
            sorted(s["passage_sequence"] for s in sentences if s["passage_group"] == g)
    # Everything outside the evidence list is byte-identical.
    rest = {k: v for k, v in moved.items() if k != "evidence_sentences"}
    assert rest == {k: v for k, v in json.loads(multi.user_prompt).items() if k != "evidence_sentences"}
    assert sampling.reorder_passages(multi, seed)[0].user_prompt == candidate.user_prompt


def test_a_single_passage_group_is_reported_unchanged():
    _, _, item = _prepared_candidate()
    single = _with_groups(item, {1: 3})
    same, changed = sampling.reorder_passages(single, 5)
    assert (same.user_prompt, changed) == (single.user_prompt, False)


def test_an_unchanged_reordering_is_sent_as_prepared_and_says_so():
    seen = []
    result, item = _sample(sampling.SampleSpec(0, order_seed=3), seen=seen)
    assert result.order_changed is False
    assert seen[0][2] == item.user_prompt


# ---- decision rules and summary --------------------------------------------------------

@pytest.mark.parametrize(("label", "state"), [
    ("supports", "supported"), ("qualifies", "qualified"), ("mixed", "qualified"),
    ("contradicts", "contradicts"), ("none", "insufficient"), ("uncertain", "not_judged"),
    (None, "not_judged")])
def test_single_judge_rule_has_no_disagreement_state(label, state):
    assert sampling.single_judge_state(label)[0] == state


def test_self_consistency_uses_the_panel_rule_over_repeats():
    assert sampling.self_consistency_state(["supports"] * 3) == ("supported", "agreed_supports")
    assert sampling.self_consistency_state(["supports", "none", "supports"])[1] == "judges_disagree"
    assert sampling.self_consistency_state(["supports", None, "supports"])[1] == "judge_failed"
    assert sampling.self_consistency_state(["none"] * 5, required=5)[0] == "insufficient"


def test_summary_compares_rules_with_the_panel():
    rows = [
        {"panel": ["supports"] * 3, "single": "supports", "repeats": ["supports"] * 3},
        {"panel": ["supports", "none", "supports"], "single": "supports",
         "repeats": ["supports", "none", "supports"]},
        {"panel": ["supports", "contradicts", "supports"], "single": "supports",
         "repeats": ["supports"] * 3},
        {"panel": ["none"] * 3, "single": "none", "repeats": ["none", "qualifies", "none"]},
    ]
    rules = {
        "panel": lambda r: sampling.self_consistency_state(r["panel"]),
        "single": lambda r: sampling.single_judge_state(r["single"]),
        "repeats": lambda r: sampling.self_consistency_state(r["repeats"]),
    }
    summary = sampling.summarize(rows, rules)
    assert summary["candidates"] == 4
    assert summary["rules"]["panel"]["not_judged_share"] == 0.5
    assert summary["rules"]["single"]["not_judged_share"] == 0.0
    single = summary["rules"]["single"]["versus_panel"]
    assert (single["panel_disagree"], single["rule_judged_where_panel_not"]) == (2, 2)
    repeats = summary["rules"]["repeats"]["versus_panel"]
    assert repeats["panel_disagree_and_rule_unstable"] == 1
    assert repeats["panel_agrees_but_rule_unstable"] == 1
    assert (repeats["both_judged"], repeats["both_judged_same_state"]) == (1, 1)


def test_a_two_provider_panel_uses_the_same_rule_with_two_required():
    from app.services.judgment_panel import panel_display_state
    assert panel_display_state({"deepseek": "supports", "glm": "qualifies"}, required=2)[:2] == \
        ("qualified", "agreed_qualified_or_mixed")
    assert panel_display_state({"deepseek": "supports", "glm": "none"}, required=2)[1] == "judges_disagree"
    assert panel_display_state({"deepseek": "supports"}, required=2)[1] == "judge_failed"
