"""Fail-closed policy for optional external comparison opinions."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.config import Settings, secret_value


OPENROUTER_GLM53_FLASH_MODEL = "z-ai/glm-5.3-flash"

# Production API id of the model ARCHITECTURE §7 names as Qwen3.8-Flash-Next;
# the "-Next" suffix denotes the open-weight preview checkpoint.
QWEN_API_MODEL = "qwen3.8-flash"
# The two classic DashScope hosts plus the per-workspace Model Studio hosts
# (`<workspace>.<region>.maas.aliyuncs.com`). Anything else is refused.
_QWEN_API_HOSTS = frozenset({"dashscope-intl.aliyuncs.com", "dashscope.aliyuncs.com"})
_QWEN_WORKSPACE_HOST = re.compile(r"^[a-z0-9-]+\.[a-z0-9-]+\.maas\.aliyuncs\.com$")
# Thinking is on by default for this model and does not support structured
# output, so every JSON call must turn it off.
QWEN_API_EXTRA_BODY = {"enable_thinking": False}


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


def qwen_api_policy(settings: Settings) -> dict[str, Any]:
    """Return the bounded Qwen API policy or reject unsafe configuration.

    Mirrors the OpenRouter gate. Where OpenRouter can be asked for zero data
    retention, this provider cannot: its documentation states inputs are not
    used for training but that call data is stored under regulation. The gate
    therefore requires an explicit acknowledgement rather than a routing flag,
    and records the date the terms were verified so a caller can see whether
    that verification is stale. Callers still validate the model's structured
    response locally and record model, cost, input hashes and scope.
    """

    if not settings.QWEN_ENABLED:
        raise RuntimeError("Qwen API comparisons are disabled")
    if not settings.QWEN_API_KEY:
        raise RuntimeError("Qwen API credential is unavailable")
    if settings.QWEN_MODEL != QWEN_API_MODEL:
        raise RuntimeError("only the authorized Qwen 3.8 Flash model is enabled")
    parsed = urlparse(settings.QWEN_BASE_URL)
    host = (parsed.hostname or "").casefold()
    if host not in _QWEN_API_HOSTS and not _QWEN_WORKSPACE_HOST.match(host):
        raise RuntimeError("Qwen calls are restricted to Alibaba Cloud Model Studio")
    if parsed.scheme != "https":
        raise RuntimeError("Qwen calls require https")
    if not settings.QWEN_RETENTION_ACKNOWLEDGED:
        raise RuntimeError(
            "the provider stores call data and offers no zero-retention option; "
            "set QWEN_RETENTION_ACKNOWLEDGED after reviewing its terms"
        )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", settings.QWEN_TERMS_VERIFIED_ON or ""):
        raise RuntimeError("Qwen terms verification date must be recorded")
    if settings.QWEN_MAX_PROMPT_USD_PER_MILLION <= 0:
        raise RuntimeError("Qwen prompt price ceiling must be positive")
    if settings.QWEN_MAX_COMPLETION_USD_PER_MILLION <= 0:
        raise RuntimeError("Qwen completion price ceiling must be positive")
    return {
        "provider": "alibaba_model_studio",
        "model": QWEN_API_MODEL,
        "host": host,
        "training_use": "none_documented",
        "retention": "provider_stored_no_zero_retention_option",
        "terms_verified_on": settings.QWEN_TERMS_VERIFIED_ON,
        "extra_body": dict(QWEN_API_EXTRA_BODY),
        "max_price": {
            "prompt": settings.QWEN_MAX_PROMPT_USD_PER_MILLION,
            "completion": settings.QWEN_MAX_COMPLETION_USD_PER_MILLION,
        },
    }


def qwen_api_client(settings: Settings):
    """Build an OpenAI-compatible client for the Qwen arm, gated by the policy.

    The policy runs first so a client can never exist for a configuration the
    gate would refuse. The credential goes only to the SDK; it is never logged
    or returned.
    """
    from openai import OpenAI

    qwen_api_policy(settings)
    return OpenAI(api_key=secret_value(settings.QWEN_API_KEY), base_url=settings.QWEN_BASE_URL)


_ZAI_HOSTS = frozenset({"api.z.ai", "open.bigmodel.cn"})


def zai_glm_policy(settings: Settings) -> dict[str, Any]:
    """Return the bounded Z.ai (GLM) policy or reject unsafe configuration.

    Z.ai's own API offers no zero-data-retention routing of the kind OpenRouter
    provided, so the owner records the date the provider's data terms were
    verified; the gate refuses without it. Callers still validate the model's
    structured response locally and record model, cost, input hashes and scope.
    """
    if not settings.ZAI_ENABLED:
        raise RuntimeError("Z.ai judgment calls are disabled")
    if not settings.ZAI_API_KEY:
        raise RuntimeError("Z.ai API credential is unavailable")
    if not settings.ZAI_MODEL.startswith("glm-"):
        raise RuntimeError("only GLM models are enabled on the Z.ai route")
    parsed = urlparse(settings.ZAI_BASE_URL)
    host = (parsed.hostname or "").casefold()
    if host not in _ZAI_HOSTS or parsed.scheme != "https":
        raise RuntimeError("Z.ai calls are restricted to its https API hosts")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", settings.ZAI_TERMS_VERIFIED_ON or ""):
        raise RuntimeError("Z.ai data terms verification date must be recorded")
    if settings.ZAI_MAX_PROMPT_USD_PER_MILLION <= 0 or settings.ZAI_MAX_COMPLETION_USD_PER_MILLION <= 0:
        raise RuntimeError("Z.ai price ceilings must be positive")
    return {
        "provider": "zai", "model": settings.ZAI_MODEL, "host": host,
        "terms_verified_on": settings.ZAI_TERMS_VERIFIED_ON,
        "reasoning_effort": settings.ZAI_REASONING_EFFORT,
        "max_price": {"prompt": settings.ZAI_MAX_PROMPT_USD_PER_MILLION,
                      "completion": settings.ZAI_MAX_COMPLETION_USD_PER_MILLION},
    }
