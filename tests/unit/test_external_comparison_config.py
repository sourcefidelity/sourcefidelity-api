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
