"""Judgment panel (Phase B): routing, arm gates, stored-record input and panel states.

Fake judges only: no provider is contacted.
"""
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services import judgment_panel as panel
from app.services.facet_evidence_judgment import CandidateFacetFinding, prepare_candidate_prompts
from app.services.judge_arms import JudgePanelUnavailable, call_cost_usd, formal_judge_arms, policy_hash
from app.services.judgment_input import (
    JudgmentInputUnavailable,
    judgment_eligibility,
    prepare_from_payload,
)
from app.services.llm_service import LLMCallFailure, LLMRoute, chat_completion
from app.services.providers import ProviderConfig
from test_facet_evidence_judgment import _prepared

SOURCE = "Participants associated disabled people with normal or high intelligence."
CLAIM = "People with disabilities were viewed as intelligent (Wood, 2012)."


def _artifact():
    return _prepared(SOURCE, CLAIM, "(Wood, 2012)")


def _payload(artifact, **changes):
    payload = {
        "coverage": {**artifact.coverage.model_dump(mode="json"), "completeness_verdict": "complete"},
        "source_identity": artifact.source_identity.model_dump(mode="json"),
        "claim": {**artifact.claim.model_dump(mode="json"), "text_truncated": False, "text_sha256": "x"},
        "source_binding": None,
        "verification_candidates": artifact.verification_candidates.model_dump(mode="json"),
        "facet_evidence_foundation": artifact.facet_evidence_foundation.model_dump(mode="json"),
        "passages": [{"passage_id": p.passage_id, "page_index": p.page_index, "page_label": p.page_label}
                     for p in artifact.passages],
    }
    payload.update(changes)
    return payload


def _route(arm_id):
    return LLMRoute(arm_id=arm_id, model=f"{arm_id}-model", endpoint_host=f"{arm_id}.example",
                    client_factory=lambda: None,
                    provider_config=ProviderConfig(name=arm_id))


ROUTES = [_route("deepseek"), _route("glm"), _route("qwen")]


def _response(user_prompt, direction):
    payload = json.loads(user_prompt)
    sentence = payload["evidence_sentences"][0]["sentence_id"]
    evidence = [] if direction == "none" else [sentence]
    return {"context_resolution": "not_required", "mappings": [
        {"facet_id": f["facet_id"], "direction": direction, "confidence": "high",
         "evidence_sentence_ids": evidence, "rationale": "Fake judge.", "limitations": []}
        for f in payload["facets"]]}


def _fake_call(directions, seen=None, receipts=None):
    """directions: arm_id -> direction, or an exception to raise."""
    def call(system_prompt, user_prompt, *, route, receipt, **_):
        if seen is not None:
            seen.append((route.arm_id, system_prompt, user_prompt))
        receipt.update(receipts or {"prompt_tokens": 1000, "completion_tokens": 100,
                                    "reported_cost_usd": 0.001})
        outcome = directions[route.arm_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, dict) else _response(user_prompt, outcome)
    return call


def _prepared_candidate():
    artifact = _artifact()
    prepared = prepare_candidate_prompts(artifact, max_input_tokens=100_000)
    item = next(i for i in prepared.items if not isinstance(i, CandidateFacetFinding))
    return artifact, prepared, item


def _judge(directions, **kwargs):
    artifact, prepared, item = _prepared_candidate()
    return panel.judge_candidate(item, artifact, prepared.sentences, ROUTES, "policy",
                                 call=_fake_call(directions, **kwargs.pop("fake", {})), **kwargs)


# ---- display-state mapping (owner decision 7) ---------------------------------

@pytest.mark.parametrize(("labels", "state", "reason"), [
    ({"a": "supports", "b": "supports", "c": "supports"}, "supported", "agreed_supports"),
    ({"a": "supports", "b": "qualifies", "c": "supports"}, "qualified", "agreed_qualified_or_mixed"),
    ({"a": "mixed", "b": "mixed", "c": "mixed"}, "qualified", "agreed_qualified_or_mixed"),
    ({"a": "contradicts", "b": "contradicts", "c": "contradicts"}, "contradicts", "agreed_contradicts"),
    ({"a": "contradicts", "b": "mixed", "c": "contradicts"}, "contradicts", "agreed_contradicts"),
    ({"a": "none", "b": "none", "c": "none"}, "insufficient", "agreed_no_evidence"),
    ({"a": "none", "b": "uncertain", "c": "none"}, "not_judged", "judges_undecided"),
    ({"a": "qualifies", "b": "uncertain", "c": "qualifies"}, "not_judged", "judges_undecided"),
    ({"a": "supports", "b": "contradicts", "c": "supports"}, "not_judged", "judges_disagree"),
    ({"a": "supports", "b": "none", "c": "supports"}, "not_judged", "judges_disagree"),
    ({"a": "supports", "b": "supports"}, "not_judged", "judge_failed"),
])
def test_panel_display_states(labels, state, reason):
    got_state, got_reason, _ = panel.panel_display_state(labels)
    assert (got_state, got_reason) == (state, reason)


def test_most_serious_state_across_sources():
    assert panel.most_serious(["supported", "insufficient", "contradicts"]) == "insufficient"
    assert panel.most_serious(["supported", "not_judged", "qualified"]) == "qualified"
    assert panel.most_serious(["not_judged", "not_judged"]) == "not_judged"


# ---- the panel run -------------------------------------------------------------

def test_every_arm_gets_the_identical_prompt_and_agreement_is_supported():
    seen = []
    result = _judge({"deepseek": "supports", "glm": "supports", "qwen": "supports"}, fake={"seen": seen})
    assert result.display_state == "supported"
    assert {arm for arm, _, _ in seen} == {"deepseek", "glm", "qwen"}
    assert len({(system, user) for _, system, user in seen}) == 1
    assert result.assessment.formal_status == "exact_convergence"
    assert result.spend_usd == pytest.approx(0.003)


def test_a_failed_arm_is_not_judged_and_never_a_two_judge_result():
    result = _judge({"deepseek": "supports", "glm": LLMCallFailure("timeout"), "qwen": "supports"})
    assert (result.display_state, result.reason_code) == ("not_judged", "judge_failed")
    failed = [a for a in result.arms if a.status == "failed"]
    assert [a.failure for a in failed] == ["timeout"]


def test_an_invalid_response_counts_as_a_failed_arm():
    result = _judge({"deepseek": "supports", "glm": {"mappings": "not a list"}, "qwen": "supports"})
    assert result.reason_code == "judge_failed"
    assert [a.failure for a in result.arms if a.status == "failed"] == ["invalid_response"]


def test_directional_disagreement_is_not_judged():
    result = _judge({"deepseek": "supports", "glm": "contradicts", "qwen": "supports"})
    assert (result.display_state, result.reason_code) == ("not_judged", "judges_disagree")
    assert result.assessment.formal_status == "directional_disagreement"


def test_agreed_none_is_insufficient_pending_the_wider_search():
    result = _judge({"deepseek": "none", "glm": "none", "qwen": "none"})
    assert result.display_state == "insufficient"


def test_cache_hit_makes_no_call_and_reinterprets_the_stored_response():
    stored = {}
    first = _judge({"deepseek": "supports", "glm": "supports", "qwen": "supports"},
                   cache_store=lambda arm: stored.__setitem__(arm.cache_key, arm.response))
    assert len(stored) == 3
    boom = LLMCallFailure("rate_limited")
    second = _judge({"deepseek": boom, "glm": boom, "qwen": boom}, cache_lookup=stored.get)
    assert second.display_state == "supported" and all(a.cached for a in second.arms)
    assert second.spend_usd == 0


def test_failures_are_never_cached():
    stored = {}
    _judge({"deepseek": "supports", "glm": LLMCallFailure("timeout"), "qwen": "supports"},
           cache_store=lambda arm: stored.__setitem__(arm.arm_id, arm.response))
    assert set(stored) == {"deepseek", "qwen"}


def test_cache_key_changes_with_the_prompt_the_model_or_the_policy():
    _, _, item = _prepared_candidate()
    key = panel.arm_cache_key(ROUTES[0], "p1", item)
    assert key != panel.arm_cache_key(ROUTES[1], "p1", item)
    assert key != panel.arm_cache_key(ROUTES[0], "p2", item)
    changed = SimpleNamespace(system_prompt=item.system_prompt, user_prompt=item.user_prompt + " ")
    assert key != panel.arm_cache_key(ROUTES[0], "p1", changed)


def test_judgment_needs_at_least_one_judge():
    artifact, prepared, item = _prepared_candidate()
    with pytest.raises(ValueError):
        panel.judge_candidate(item, artifact, prepared.sentences, [], "p", call=_fake_call({}))


@pytest.mark.parametrize(("label", "state"), [("supports", "supported"), ("qualifies", "qualified"),
                                              ("mixed", "qualified"), ("contradicts", "contradicts"),
                                              ("none", "insufficient"), ("uncertain", "not_judged")])
def test_one_judge_label_is_the_result(label, state):
    assert panel.panel_display_state({"zai_glm": label}, required=1)[0] == state
    assert panel.panel_display_state({}, required=1)[:2] == ("not_judged", "judge_failed")


def test_a_single_judge_result_through_the_panel():
    artifact, prepared, item = _prepared_candidate()
    result = panel.judge_candidate(item, artifact, prepared.sentences, ROUTES[:1], "p",
                                   call=_fake_call({"deepseek": "contradicts"}))
    assert (result.display_state, result.reason_code) == ("contradicts", "judge_contradicts")


def test_a_transient_failure_is_retried_once(monkeypatch):
    monkeypatch.setattr(panel, "_RETRY_PAUSE_SECONDS", 0)
    monkeypatch.setattr(panel.settings, "JUDGMENT_SAMPLES", 1)
    calls = []
    def flaky(system_prompt, user_prompt, *, route, receipt, **_):
        calls.append(1)
        if len(calls) == 1:
            raise LLMCallFailure("timeout")
        return _response(user_prompt, "supports")
    artifact, prepared, item = _prepared_candidate()
    result = panel.judge_candidate(item, artifact, prepared.sentences, ROUTES[:1], "p", call=flaky)
    assert result.display_state == "supported" and len(calls) == 2


# ---- cost ------------------------------------------------------------------------

def test_cost_prefers_reported_then_tariff_then_ceiling():
    assert call_cost_usd({"reported_cost_usd": 0.002}) == (0.002, "provider_reported")
    ceiling = {"price_context": {"ceiling_usd_per_million": [0.15, 0.47]},
               "prompt_tokens": 1_000_000, "completion_tokens": 0}
    assert call_cost_usd(ceiling) == (pytest.approx(0.15), "ceiling_estimate")
    assert call_cost_usd({}) == (None, "unpriced")


# ---- arm gates ---------------------------------------------------------------------

def _synthetic_key(arm):
    return f"synthetic-{arm}-key-000"


def _settings(**changes):
    values = dict(JUDGMENT_JUDGES="deepseek,glm,qwen",
                  LLM_BASE_URL="https://api.deepseek.com/v1", LLM_API_KEY=_synthetic_key("ds"),
                  LLM_MODEL="deepseek-v4-flash", OPENROUTER_ENABLED=True,
                  OPENROUTER_API_KEY=_synthetic_key("or"), QWEN_ENABLED=True,
                  QWEN_API_KEY=_synthetic_key("qw"), QWEN_RETENTION_ACKNOWLEDGED=True)
    values.update(changes)
    return Settings(_env_file=None, **values)


def test_three_routes_with_their_privacy_and_thinking_controls():
    routes, snapshot = formal_judge_arms(_settings())
    by_arm = {route.arm_id: route for route in routes}
    assert list(by_arm) == ["deepseek", "glm", "qwen"]
    assert by_arm["deepseek"].extra_body == {"thinking": {"type": "disabled"}}
    provider = by_arm["glm"].extra_body["provider"]
    assert provider["data_collection"] == "deny" and provider["zdr"] is True
    assert provider["require_parameters"] is True
    assert by_arm["glm"].extra_body["reasoning"] == {"effort": "low", "exclude": True}
    assert by_arm["glm"].max_output_tokens == 8_000 and by_arm["deepseek"].max_output_tokens is None
    assert by_arm["qwen"].extra_body == {"enable_thinking": False}
    assert by_arm["qwen"].provider_config.json_mode is True     # not the local-Qwen config
    rendered = json.dumps(snapshot) + repr(routes)
    assert "synthetic-" not in rendered
    assert len(policy_hash(snapshot)) == 64


@pytest.mark.parametrize("changes", [dict(OPENROUTER_ENABLED=False), dict(QWEN_RETENTION_ACKNOWLEDGED=False),
                                     dict(LLM_BASE_URL="https://other.example/v1"), dict(QWEN_API_KEY=None)])
def test_any_refused_arm_makes_the_whole_panel_unavailable(changes):
    with pytest.raises(JudgePanelUnavailable):
        formal_judge_arms(_settings(**changes))


# ---- routed LLM calls ----------------------------------------------------------------

class _FakeClient:
    def __init__(self):
        self.calls = []
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                                model_extra={"cost": 0.0004})
        response = SimpleNamespace(model="z-ai/glm-5.3-flash-20260901", model_extra={"provider": "Z.AI"},
                                   usage=usage,
                                   choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: self.calls.append(kw) or response))


def test_routed_call_uses_only_its_own_client_body_and_records_provenance(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr("app.services.llm_service.get_client",
                        lambda: (_ for _ in ()).throw(AssertionError("default client used")))
    route = LLMRoute(arm_id="glm", model="z-ai/glm-5.3-flash", endpoint_host="openrouter.ai",
                     client_factory=lambda: fake, provider_config=ProviderConfig(name="glm"),
                     extra_body={"provider": {"zdr": True}})
    receipt = {}
    assert chat_completion("s", "u", route=route, receipt=receipt, max_tokens=50) == '{"ok": 1}'
    assert fake.calls[0]["model"] == "z-ai/glm-5.3-flash"
    assert fake.calls[0]["extra_body"] == {"provider": {"zdr": True}}
    assert receipt["returned_model"] == "z-ai/glm-5.3-flash-20260901"
    assert receipt["returned_provider"] == "Z.AI" and receipt["reported_cost_usd"] == 0.0004


# ---- stored-record input -------------------------------------------------------------

def test_stored_payload_rebuilds_the_byte_identical_prompt():
    artifact = _artifact()
    direct = prepare_candidate_prompts(artifact, max_input_tokens=100_000)
    _, stored = prepare_from_payload(_payload(artifact))
    direct_prompts = [i.user_prompt for i in direct.items if not isinstance(i, CandidateFacetFinding)]
    stored_prompts = [i.user_prompt for i in stored.items if not isinstance(i, CandidateFacetFinding)]
    assert direct_prompts and direct_prompts == stored_prompts


@pytest.mark.parametrize(("changes", "reason"), [
    (dict(coverage={"level": "abstract_only"}), "not_complete_full_text"),
    (dict(coverage={"level": "full_text", "completeness_verdict": "incomplete"}), "not_complete_full_text"),
    (dict(source_identity={"status": "uncertain"}), "identity_unconfirmed"),
    (dict(facet_evidence_foundation={"foundation_version": "old-v1", "status": "complete"}),
     "prepared_before_feature"),
])
def test_ineligible_records_say_why(changes, reason):
    payload = _payload(_artifact(), **changes)
    assert judgment_eligibility(payload).reason_code == reason
    with pytest.raises(JudgmentInputUnavailable) as raised:
        prepare_from_payload(payload)
    assert raised.value.reason_code == reason


def test_a_truncated_stored_claim_is_never_judged():
    artifact = _artifact()
    payload = _payload(artifact, claim={**artifact.claim.model_dump(mode="json"), "text_truncated": True})
    assert judgment_eligibility(payload).reason_code == "claim_text_truncated"


def test_the_zai_judge_reasons_at_low_effort_and_needs_verified_terms():
    configured = _settings(JUDGMENT_JUDGES="zai_glm", ZAI_ENABLED=True, ZAI_API_KEY=_synthetic_key("zai"),
                           ZAI_TERMS_VERIFIED_ON="2026-09-27")
    routes, snapshot = formal_judge_arms(configured)
    [route] = routes
    assert route.arm_id == "zai_glm" and route.model == "glm-5.3-flash" and route.endpoint_host == "api.z.ai"
    assert route.extra_body == {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    assert route.max_output_tokens == 8_000 and "synthetic-" not in json.dumps(snapshot) + repr(routes)
    with pytest.raises(JudgePanelUnavailable):
        formal_judge_arms(_settings(JUDGMENT_JUDGES="zai_glm", ZAI_ENABLED=True,
                                    ZAI_API_KEY=_synthetic_key("zai")))          # terms date missing


def test_notes_use_deepseek_whichever_model_judges():
    from app.services.judge_arms import coaching_route
    assert coaching_route(_settings(JUDGMENT_JUDGES="zai_glm")).arm_id == "deepseek"
    assert coaching_route(_settings(LLM_BASE_URL="https://other.example/v1")) is None


# ---- majority of repeated answers (owner decision 2026-09-30) -------------------

@pytest.mark.parametrize(("labels", "state", "reason", "flags"), [
    (["supports", "supports", "qualifies"], "supported", "judge_supports", [True, True, False]),
    (["qualifies", "mixed", "supports"], "qualified", "judge_qualifies", [True, True, False]),
    (["contradicts", None, "contradicts"], "contradicts", "judge_contradicts", [True, False, True]),
    (["uncertain", "uncertain", "none"], "not_judged", "judge_undecided", [True, True, False]),
    (["supports", "none", "contradicts"], "not_judged", "samples_split", [False, False, False]),
    (["supports", None, "none"], "not_judged", "samples_split", [False, False, False]),
    (["supports", None, None], "not_judged", "judge_failed", [False, False, False]),
])
def test_the_majority_of_one_judges_answers_is_shown(labels, state, reason, flags):
    assert panel.majority_display_state(labels) == (state, reason, flags)


def test_a_single_judge_answers_three_times_and_reuses_its_first_cached_answer(monkeypatch):
    monkeypatch.setattr(panel.settings, "JUDGMENT_SAMPLES", 3)
    answers = iter(["contradicts", "supports"])
    calls = []

    def call(system_prompt, user_prompt, *, route, receipt, **_):
        calls.append(1)
        return _response(user_prompt, next(answers))
    artifact, prepared, item = _prepared_candidate()
    first_key = panel.arm_cache_key(ROUTES[0], "p", item)
    cache = {first_key: _response(item.user_prompt, "supports")}
    stored = []
    result = panel.judge_candidate(item, artifact, prepared.sentences, ROUTES[:1], "p", call=call,
                                   cache_lookup=cache.get, cache_store=stored.append)
    assert len(calls) == 2 and result.display_state == "supported"
    assert [a.arm_id for a in result.arms] == ["deepseek", "deepseek:s2", "deepseek:s3"]
    assert [a.in_majority for a in result.arms] == [True, False, True]
    assert len({a.cache_key for a in result.arms}) == 3 and len(stored) == 2
