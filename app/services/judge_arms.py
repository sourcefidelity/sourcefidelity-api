"""The three formal Judgment panel arms (ARCHITECTURE §7, AGENTS arms table).

DeepSeek V4 Flash, GLM-5.3-Flash through OpenRouter and Qwen3.8-Flash through
Alibaba Cloud Model Studio each receive one identical prepared prompt. Each arm
is built only after its own policy gate passes; if any gate refuses, the panel
is unavailable as a whole and nothing is sent to any arm, because a two-judge
result is not the accepted panel.

No credential is returned, logged or stored: clients are built lazily inside
the route and the policy snapshot records hosts, models and routing flags only.
"""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

from app.config import Settings, secret_value, settings as default_settings
from app.services.external_comparison_config import (
    QWEN_API_EXTRA_BODY,
    openrouter_glm53_flash_policy,
    qwen_api_client,
    qwen_api_policy,
    zai_glm_policy,
)
from app.services.llm_service import LLMRoute
from app.services.providers import ProviderConfig, get_provider_config
from app.services.usage_cost import llm_cost, llm_price_context

PANEL_ARM_IDS = ("deepseek", "glm", "qwen")
_GLM_MAX_OUTPUT_TOKENS = 8_000
_DEEPSEEK_HOST = "api.deepseek.com"
_OPENROUTER_CONFIG = ProviderConfig(
    name="GLM (OpenRouter)", json_mode=True, json_object_required=True,
    input_batch_tokens=100_000, max_output_tokens=8192,
)


class JudgePanelUnavailable(RuntimeError):
    """A panel arm's policy refused; the reason names the arm, never a value."""

    def __init__(self, arm_id: str, reason: str):
        super().__init__(f"{arm_id}: {reason}")
        self.arm_id = arm_id
        self.reason = reason


def _deepseek_route(settings: Settings) -> tuple[LLMRoute, dict]:
    host = (urlsplit(settings.LLM_BASE_URL or "").hostname or "").casefold()
    if host != _DEEPSEEK_HOST:
        raise RuntimeError("the DeepSeek arm requires the configured DeepSeek endpoint")
    if not settings.LLM_API_KEY:
        raise RuntimeError("DeepSeek credential is unavailable")
    if not settings.LLM_MODEL.startswith("deepseek"):
        raise RuntimeError("the configured default model is not a DeepSeek model")
    config = get_provider_config(settings.LLM_MODEL)

    def client():
        from app.services.llm_service import get_client
        return get_client()

    route = LLMRoute(
        arm_id="deepseek", model=settings.LLM_MODEL, endpoint_host=host,
        client_factory=client, provider_config=config,
        extra_body=dict(config.reasoning_disable_body or {}),
        price_context=lambda model: llm_price_context(model, settings.LLM_BASE_URL or ""),
    )
    return route, {"model": settings.LLM_MODEL, "host": host, "thinking": "disabled"}


def _glm_route(settings: Settings) -> tuple[LLMRoute, dict]:
    routing = openrouter_glm53_flash_policy(settings)
    host = (urlsplit(settings.OPENROUTER_BASE_URL).hostname or "").casefold()
    cache: dict = {}

    def client():
        if "client" not in cache:
            from openai import OpenAI
            cache["client"] = OpenAI(api_key=secret_value(settings.OPENROUTER_API_KEY),
                                     base_url=settings.OPENROUTER_BASE_URL)
        return cache["client"]

    ceiling = [routing["max_price"]["prompt"], routing["max_price"]["completion"]]
    route = LLMRoute(
        arm_id="glm", model=settings.OPENROUTER_MODEL, endpoint_host=host,
        client_factory=client, provider_config=_OPENROUTER_CONFIG,
        # GLM-5.3-Flash reasoning is mandatory on OpenRouter (default effort
        # "max"; supported max/high/low). The 2026-09-25 smoke test spent all
        # 1,600 output tokens reasoning and returned no answer, so the arm
        # reasons at the lowest effort, the reasoning text is excluded from the
        # response, and the arm gets room to answer after reasoning.
        extra_body={"provider": routing, "reasoning": {"effort": "low", "exclude": True}},
        max_output_tokens=_GLM_MAX_OUTPUT_TOKENS,
        price_context=lambda model: {"pricing_basis": "provider_reported_cost",
                                     "ceiling_usd_per_million": ceiling, "model": model},
    )
    return route, {"model": settings.OPENROUTER_MODEL, "host": host, "provider_routing": routing}


def _qwen_route(settings: Settings) -> tuple[LLMRoute, dict]:
    policy = qwen_api_policy(settings)
    config = get_provider_config(settings.QWEN_MODEL, settings.QWEN_BASE_URL)
    if not config.name.startswith("Qwen (Model Studio"):
        raise RuntimeError("the Qwen arm resolved to a non-API provider configuration")
    cache: dict = {}

    def client():
        if "client" not in cache:
            cache["client"] = qwen_api_client(settings)
        return cache["client"]

    ceiling = [policy["max_price"]["prompt"], policy["max_price"]["completion"]]
    route = LLMRoute(
        arm_id="qwen", model=settings.QWEN_MODEL, endpoint_host=policy["host"],
        client_factory=client, provider_config=config,
        extra_body=dict(QWEN_API_EXTRA_BODY),
        price_context=lambda model: {"pricing_basis": "configured_ceiling",
                                     "ceiling_usd_per_million": ceiling, "model": model},
    )
    return route, {key: policy[key] for key in ("model", "host", "training_use", "retention",
                                                 "terms_verified_on", "extra_body", "max_price")}


def _zai_glm_route(settings: Settings) -> tuple[LLMRoute, dict]:
    policy = zai_glm_policy(settings)
    cache: dict = {}

    def client():
        if "client" not in cache:
            from openai import OpenAI
            cache["client"] = OpenAI(api_key=secret_value(settings.ZAI_API_KEY), base_url=settings.ZAI_BASE_URL)
        return cache["client"]

    ceiling = [policy["max_price"]["prompt"], policy["max_price"]["completion"]]
    route = LLMRoute(
        arm_id="zai_glm", model=settings.ZAI_MODEL, endpoint_host=policy["host"],
        client_factory=client, provider_config=_OPENROUTER_CONFIG,
        # Thinking cannot be disabled on GLM-5.3-Flash; lowest effort, room to answer.
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": settings.ZAI_REASONING_EFFORT},
        max_output_tokens=_GLM_MAX_OUTPUT_TOKENS,
        # The list price when one is on record; the ceiling bounds the estimate otherwise.
        price_context=lambda model: {"pricing_basis": "configured_ceiling",
                                     "ceiling_usd_per_million": ceiling, "model": model,
                                     **llm_price_context(model, settings.ZAI_BASE_URL or "")},
    )
    return route, policy


_BUILDERS = {"deepseek": _deepseek_route, "glm": _glm_route, "qwen": _qwen_route, "zai_glm": _zai_glm_route}
# Model setting behind each judge, for the notice's model list.
JUDGE_MODEL_SETTING = {"deepseek": "LLM_MODEL", "glm": "OPENROUTER_MODEL", "qwen": "QWEN_MODEL",
                       "zai_glm": "ZAI_MODEL"}


def configured_judges(settings: Settings = default_settings) -> list[str]:
    judges = [j.strip() for j in (settings.JUDGMENT_JUDGES or "").split(",") if j.strip()]
    return [j for j in judges if j in _BUILDERS] or ["zai_glm"]


def coaching_route(settings: Settings = default_settings) -> LLMRoute | None:
    """DeepSeek writes the notes whichever model judges; None if unavailable."""
    try:
        return _deepseek_route(settings)[0]
    except RuntimeError:
        return None


def formal_judge_arms(settings: Settings = default_settings) -> tuple[list[LLMRoute], dict]:
    """All three routes and a secret-free policy snapshot, or JudgePanelUnavailable."""
    if settings.JUDGMENT_FAKE_PANEL:
        from app.services.judgment_fake_panel import fake_judge_arms
        return fake_judge_arms()
    routes, snapshot = [], {}
    for arm_id in configured_judges(settings):
        try:
            route, policy = _BUILDERS[arm_id](settings)
        except RuntimeError as exc:
            raise JudgePanelUnavailable(arm_id, str(exc)) from None
        routes.append(route)
        snapshot[arm_id] = policy
    return routes, snapshot


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()


def call_cost_usd(receipt: dict) -> tuple[float | None, str]:
    """The best available cost for one call, and how it was obtained.

    Provider-reported cost first (OpenRouter), then the DeepSeek tariff table,
    then the configured price ceiling as an upper-bound estimate. Unpriced
    stays None, never zero.
    """
    reported = receipt.get("reported_cost_usd")
    if type(reported) in (int, float) and reported >= 0:
        return float(reported), "provider_reported"
    context = receipt.get("price_context") or {}
    if context.get("usd_per_million"):
        usage = {**context, **{k: receipt.get(k) for k in ("prompt_tokens", "completion_tokens")},
                 **{k: receipt.get(k) for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                    if receipt.get(k) is not None}}
        cost = llm_cost(usage)
        if cost is not None:
            return cost, "tariff"
    ceiling = context.get("ceiling_usd_per_million")
    prompt, completion = receipt.get("prompt_tokens"), receipt.get("completion_tokens")
    if (isinstance(ceiling, list) and len(ceiling) == 2
            and type(prompt) is int and type(completion) is int):
        return (prompt * ceiling[0] + completion * ceiling[1]) / 1_000_000, "ceiling_estimate"
    return None, "unpriced"
