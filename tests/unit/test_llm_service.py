"""Structured-output integrity tests for the shared LLM boundary."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, APIConnectionError, APIStatusError

from pydantic import SecretStr
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


@pytest.mark.parametrize("status,category", [
    (401, "authentication_failed"), (403, "authentication_failed"),
    (429, "rate_limited"), (503, "provider_server_error"),
    (400, "provider_request_rejected"),
    ("timeout", "timeout"), ("connection", "connection_failed"),
])
def test_provider_failure_retains_only_safe_category(monkeypatch, status, category):
    request = httpx.Request("POST", "https://example.test/private-endpoint")
    if status == "timeout":
        error = APITimeoutError(request=request)
    elif status == "connection":
        error = APIConnectionError(request=request)
    else:
        error = APIStatusError("PRIVATE RESPONSE", response=httpx.Response(status, request=request),
                               body={"secret": "PRIVATE RESPONSE"})
    calls = []
    def create(**kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(llm_service.settings, "LLM_API_KEY", SecretStr("test-only"))
    monkeypatch.setattr(llm_service, "get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(llm_service.LLMCallFailure) as caught:
        llm_service.chat_completion_json("system", "private student input")
    assert caught.value.category == category
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert len(calls) == 1


@pytest.mark.parametrize("response,category,expected_calls", [
    ("", "empty_response", 2), ("not JSON", "invalid_json", 3),
    (_TRUNCATED, "invalid_json", 3),
])
def test_json_failure_category_preserves_retry_limits(monkeypatch, response, category, expected_calls):
    calls = []
    def completion(**kwargs):
        calls.append(1)
        return response
    monkeypatch.setattr(llm_service, "chat_completion", completion)
    with pytest.raises(llm_service.LLMCallFailure) as caught:
        llm_service.chat_completion_json("system", "user")
    assert caught.value.category == category
    assert caught.value.attempts == expected_calls == len(calls)


def test_null_provider_content_is_an_empty_response(monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(1)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=None))])
    monkeypatch.setattr(llm_service.settings, "LLM_API_KEY", SecretStr("test-only"))
    monkeypatch.setattr(llm_service, "get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(llm_service.LLMCallFailure) as caught:
        llm_service.chat_completion_json("system", "user")
    assert caught.value.category == "empty_response"
    assert len(calls) == 2



# ------------------------------------------------ Qwen: API route vs local route
def test_qwen_api_route_is_distinguished_from_the_local_route() -> None:
    """The registry takes the first substring match, and "qwen" matched both.

    The local route (oMLX/Ollama) cannot use response_format; the cloud API
    can, but only with thinking off. Confusing the two would either break JSON
    output locally or exhaust every output budget on the API."""
    from app.services.providers import get_provider_config

    api = get_provider_config("qwen3.8-flash", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    assert api.name == "Qwen (Model Studio API)"
    assert api.json_mode is True
    assert api.reasoning_disable_body == {"enable_thinking": False}
    # The production model id alone is enough, even without a base URL.
    assert get_provider_config("qwen3.8-flash", "").name == "Qwen (Model Studio API)"
    # A workspace-scoped Model Studio host is the API route too.
    assert get_provider_config("qwen3.8-flash", "https://ws-1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1").json_mode is True

    local = get_provider_config("qwen2.5:7b", "http://localhost:11434/v1")
    assert local.name == "Qwen (local)"
    assert local.json_mode is False
    assert local.reasoning_disable_body is None
