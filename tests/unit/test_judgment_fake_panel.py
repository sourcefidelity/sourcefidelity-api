"""Stand-in judges (JUDGMENT_FAKE_PANEL): no provider route, real contract, every state."""
from app.services import judgment_panel as panel
from app.services.judge_arms import formal_judge_arms, policy_hash
from app.services.judgment_fake_panel import _SCENARIOS
from app.services.llm_service import chat_completion_json
from test_judgment_panel import _prepared_candidate, _settings


def test_stand_in_panel_never_builds_a_provider_route():
    routes, snapshot = formal_judge_arms(_settings(JUDGMENT_FAKE_PANEL=True, OPENROUTER_ENABLED=False))
    assert snapshot["fake_panel"] is True
    assert {route.endpoint_host for route in routes} == {"stand-in.invalid"}


def test_stand_in_answers_pass_the_real_validation_at_no_cost():
    routes, snapshot = formal_judge_arms(_settings(JUDGMENT_FAKE_PANEL=True))
    artifact, prepared, item = _prepared_candidate()
    result = panel.judge_candidate(item, artifact, prepared.sentences, routes, policy_hash(snapshot),
                                   call=chat_completion_json)
    assert all(arm.status == "valid" for arm in result.arms)
    assert result.spend_usd == 0


def test_stand_in_scenarios_cover_every_display_state():
    states = {panel.panel_display_state(dict(s))[0] for s in _SCENARIOS}
    assert states == {"supported", "qualified", "contradicts", "insufficient", "not_judged"}
