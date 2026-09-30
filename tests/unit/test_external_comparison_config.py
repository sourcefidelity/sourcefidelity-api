"""Fail-closed configuration tests for development-only external comparisons."""

import pytest

from app.config import Settings
from app.services.external_comparison_config import (
    OPENROUTER_GLM53_FLASH_MODEL,
    openrouter_glm53_flash_policy,
)


def test_openrouter_comparison_calls_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_ENABLED", raising=False)

    isolated = Settings(_env_file=None)

    assert isolated.OPENROUTER_ENABLED is False
    assert isolated.OPENROUTER_MODEL == OPENROUTER_GLM53_FLASH_MODEL


def test_glm53_flash_supplemental_policy_is_private_and_price_bounded() -> None:
    isolated = Settings(
        _env_file=None,
        OPENROUTER_ENABLED=True,
        OPENROUTER_API_KEY="test-only",
    )

    assert openrouter_glm53_flash_policy(isolated) == {
        "data_collection": "deny",
        "zdr": True,
        "require_parameters": True,
        "max_price": {"prompt": 0.075, "completion": 0.25},
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"OPENROUTER_MODEL": "stealth/ox-alpha"}, "authorized GLM 5.3"),
        ({"OPENROUTER_ZDR": False}, "zero-data-retention"),
        ({"OPENROUTER_REQUIRE_PARAMETERS": False}, "parameter support"),
        ({"OPENROUTER_BASE_URL": "https://example.com/v1"}, "OpenRouter"),
    ],
)
def test_glm53_flash_supplemental_policy_rejects_unsafe_overrides(
    override: dict[str, object], message: str
) -> None:
    isolated = Settings(
        _env_file=None,
        OPENROUTER_ENABLED=True,
        OPENROUTER_API_KEY="test-only",
        **override,
    )

    with pytest.raises(RuntimeError, match=message):
        openrouter_glm53_flash_policy(isolated)



# ---------------------------------------------------------------- Qwen API arm
from app.services.external_comparison_config import (  # noqa: E402
    QWEN_API_EXTRA_BODY,
    QWEN_API_MODEL,
    qwen_api_client,
    qwen_api_policy,
)


def _qwen_settings(**override):
    base = dict(_env_file=None, QWEN_ENABLED=True, QWEN_API_KEY="test-only",
                QWEN_RETENTION_ACKNOWLEDGED=True)
    base.update(override)
    return Settings(**base)


def test_qwen_api_arm_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("QWEN_ENABLED", raising=False)
    isolated = Settings(_env_file=None)
    assert isolated.QWEN_ENABLED is False
    assert isolated.QWEN_RETENTION_ACKNOWLEDGED is False
    assert isolated.QWEN_MODEL == QWEN_API_MODEL
    with pytest.raises(RuntimeError, match="disabled"):
        qwen_api_policy(isolated)


def test_qwen_api_policy_records_terms_and_turns_thinking_off() -> None:
    policy = qwen_api_policy(_qwen_settings())
    assert policy["model"] == "qwen3.8-flash"
    assert policy["host"] == "dashscope-intl.aliyuncs.com"
    assert policy["training_use"] == "none_documented"
    assert policy["retention"] == "provider_stored_no_zero_retention_option"
    assert policy["terms_verified_on"] == "2026-09-24"
    # Thinking does not support structured output, so every JSON call is
    # made with it off.
    assert policy["extra_body"] == {"enable_thinking": False} == QWEN_API_EXTRA_BODY
    assert policy["max_price"] == {"prompt": 0.15, "completion": 0.47}


@pytest.mark.parametrize("base_url", [
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://ws-abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
])
def test_qwen_api_policy_accepts_each_model_studio_host(base_url) -> None:
    assert qwen_api_policy(_qwen_settings(QWEN_BASE_URL=base_url))["host"].endswith("aliyuncs.com")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"QWEN_API_KEY": None}, "credential"),
        ({"QWEN_MODEL": "qwen3.8-max"}, "authorized Qwen 3.8 Flash"),
        ({"QWEN_BASE_URL": "https://openrouter.ai/api/v1"}, "Model Studio"),
        ({"QWEN_BASE_URL": "https://evil.example/aliyuncs.com/v1"}, "Model Studio"),
        ({"QWEN_BASE_URL": "http://dashscope-intl.aliyuncs.com/compatible-mode/v1"}, "https"),
        ({"QWEN_RETENTION_ACKNOWLEDGED": False}, "zero-retention"),
        ({"QWEN_TERMS_VERIFIED_ON": "soon"}, "verification date"),
        ({"QWEN_MAX_PROMPT_USD_PER_MILLION": 0}, "prompt price ceiling"),
        ({"QWEN_MAX_COMPLETION_USD_PER_MILLION": -1}, "completion price ceiling"),
    ],
)
def test_qwen_api_policy_rejects_unsafe_overrides(override, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        qwen_api_policy(_qwen_settings(**override))


def test_qwen_api_client_is_gated_and_never_exposes_the_key() -> None:
    """A client cannot be built for a configuration the policy refuses."""
    with pytest.raises(RuntimeError):
        qwen_api_client(_qwen_settings(QWEN_RETENTION_ACKNOWLEDGED=False))
    client = qwen_api_client(_qwen_settings(QWEN_API_KEY="sk-secret-value"))
    assert str(client.base_url).startswith("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    assert "sk-secret-value" not in repr(client)
