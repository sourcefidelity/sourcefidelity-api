"""Fail-closed policy for optional external comparison opinions."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.config import Settings


OPENROUTER_GLM53_FLASH_MODEL = "z-ai/glm-5.3-flash"


def openrouter_glm53_flash_policy(settings: Settings) -> dict[str, Any]:
    """Return the bounded provider policy or reject unsafe configuration.

    GLM 5.3 Flash is a supplemental development opinion only. Callers must
    still validate the model's structured response locally and record the
    returned model/provider, cost, input hashes and authorization scope.
    """

    if not settings.OPENROUTER_ENABLED:
        raise RuntimeError("OpenRouter supplemental comparisons are disabled")
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter API credential is unavailable")
    if settings.OPENROUTER_MODEL != OPENROUTER_GLM53_FLASH_MODEL:
        raise RuntimeError("only the authorized GLM 5.3 Flash slug is enabled")
    host = (urlparse(settings.OPENROUTER_BASE_URL).hostname or "").casefold()
    if host != "openrouter.ai":
        raise RuntimeError("supplemental calls are restricted to OpenRouter")
    if settings.OPENROUTER_DATA_COLLECTION != "deny":
        raise RuntimeError("OpenRouter data collection must remain denied")
    if not settings.OPENROUTER_ZDR:
        raise RuntimeError("OpenRouter zero-data-retention routing is required")
    if not settings.OPENROUTER_REQUIRE_PARAMETERS:
        raise RuntimeError("OpenRouter parameter support must be enforced")
    if settings.OPENROUTER_MAX_PROMPT_USD_PER_MILLION <= 0:
        raise RuntimeError("OpenRouter prompt price ceiling must be positive")
    if settings.OPENROUTER_MAX_COMPLETION_USD_PER_MILLION <= 0:
        raise RuntimeError("OpenRouter completion price ceiling must be positive")
    return {
        "data_collection": "deny",
        "zdr": True,
        "require_parameters": True,
        "max_price": {
            "prompt": settings.OPENROUTER_MAX_PROMPT_USD_PER_MILLION,
            "completion": settings.OPENROUTER_MAX_COMPLETION_USD_PER_MILLION,
        },
    }
