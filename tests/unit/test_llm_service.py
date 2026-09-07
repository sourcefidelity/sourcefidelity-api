"""Structured-output integrity tests for the shared LLM boundary."""

from types import SimpleNamespace

import pytest

from app.services import llm_service


_TRUNCATED = '{"references":[{"title":"complete"},{"title":"unfinished"'


def _configure_truncated_responses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_completion(**kwargs) -> str:
        calls.append(kwargs["user_prompt"])
        return _TRUNCATED

    monkeypatch.setattr(
        llm_service,
        "get_provider_config",
        lambda _model=None: SimpleNamespace(json_mode=False),
    )
    monkeypatch.setattr(llm_service, "chat_completion", fake_completion)
    return calls


def test_truncated_json_fails_closed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _configure_truncated_responses(monkeypatch)

    with pytest.raises(RuntimeError, match="Failed to get valid JSON"):
        llm_service.chat_completion_json("system", "user", max_retries=1)

    assert len(calls) == 2


def test_partial_json_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _configure_truncated_responses(monkeypatch)

    result = llm_service.chat_completion_json(
        "system", "user", max_retries=1, allow_partial=True
    )

    assert result == {"references": [{"title": "complete"}]}
    assert len(calls) == 2
