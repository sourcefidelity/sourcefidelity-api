"""LLM provider configuration.

Maps LLM models to their provider-specific capabilities and constraints.
This is the single source of truth for model-agnostic behavior — all
model-specific workarounds live here, not scattered in the service code.

Each provider defines:
  - json_mode: whether the model supports response_format={"type": "json_object"}
  - json_object_required: whether JSON output MUST be a top-level object
    (DeepSeek requires {"key": [...]}, not bare [...])
  - json_mode_required_for_json: whether response_format must be explicitly
    set to get JSON output (DeepSeek returns empty without it)
  - input_batch_tokens: max input tokens per call before batching is needed
    (DeepSeek: ~1500; GPT-4: ~100000; Claude: ~200000)
  - max_output_tokens: max output tokens the model can produce
  - short_json_keys: whether to use abbreviated JSON keys to save output space
  - reasoning_disable_body: the extra_body payload to DISABLE the model's
    thinking/reasoning mode for low-stakes structured-output calls, or None
    when the provider does not expose reasoning control. Reasoning models
    (DeepSeek V4 Flash/Pro, OpenAI o-series) emit hidden "thinking" tokens
    before the answer; on large inputs the thinking phase can consume the
    entire max_tokens budget and produce an EMPTY response. Disabling
    thinking for low-stakes, structured passes (subject identification,
    reference parsing) trades some reasoning quality for reliability and
    speed. The verification judge (Phase 3.8 Stage 5) keeps thinking ON —
    high-stakes, low-volume. Per-call via chat_completion(disable_thinking=True).

Usage:
    from app.services.providers import get_provider_config
    config = get_provider_config()
    if config.json_mode:
        response_format = {"type": "json_object"}
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    name: str
    json_mode: bool = True
    json_object_required: bool = False
    json_mode_required_for_json: bool = False
    input_batch_tokens: int = 100000
    max_output_tokens: int = 8192
    short_json_keys: bool = False
    reasoning_disable_body: Optional[Dict[str, dict]] = None
    # Whether the provider accepts reasoning_effort ("low"/"high"/"max") via
    # extra_body. Guards the reasoning_effort param in chat_completion so that
    # swapping cloud models does NOT silently degrade — if False, the effort
    # string is dropped (with a log line) rather than sent to a provider that
    # may reject it or no-op it. Only the SUPPORT flag lives here; the effort
    # STRING is task-specific (MLA cleanup wants "low", the judge will want
    # thinking ON) and stays in the application caller. Note the effort string
    # maps per-model on the provider side (e.g. "low" on deepseek-v4-flash =
    # low effort, but "low" on deepseek-v4-pro maps to "high") — documented
    # per-call; re-validate if switching models within a provider.
    reasoning_effort_supported: bool = False


# Registry of known provider configurations.
# Keyed by a substring that appears in the model name or base URL.
_PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        name="DeepSeek",
        json_mode=True,
        json_object_required=True,  # {"key": [...]} not bare [...]
        json_mode_required_for_json=True,  # returns empty without response_format
        input_batch_tokens=1500,  # stops returning output above ~2000 tokens
        max_output_tokens=8192,
        short_json_keys=True,  # keep output small
        # V4 Flash/Pro are reasoning models — thinking tokens can exhaust the
        # max_tokens budget on large inputs and return empty. Disabling via
        # the OpenAI-compatible extra_body (DeepSeek thinking_mode docs).
        reasoning_disable_body={"thinking": {"type": "disabled"}},
        # V4 also supports reasoning_effort levels (low/high/max) via extra_body,
        # validated Aug 9: MLA cleanup with "low" fixes the reasoning-budget
        # empty-response problem (3/3 correct vs 0/3 empty on default).
        reasoning_effort_supported=True,
    ),
    "gpt": ProviderConfig(
        name="OpenAI GPT",
        json_mode=True,
        json_object_required=True,  # OpenAI also requires top-level object
        json_mode_required_for_json=False,  # works without response_format too
        input_batch_tokens=100000,
        max_output_tokens=16384,
        short_json_keys=False,  # plenty of output room
    ),
    "claude": ProviderConfig(
        name="Anthropic Claude",
        json_mode=False,  # Claude doesn't support response_format
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=200000,
        max_output_tokens=8192,
        short_json_keys=False,
    ),
    "o1": ProviderConfig(
        name="OpenAI o1",
        json_mode=False,  # o1 doesn't support response_format
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=100000,
        max_output_tokens=32768,
        short_json_keys=False,
    ),
    "llama": ProviderConfig(
        name="Ollama Llama (local)",
        json_mode=True,
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=4000,  # local models have smaller context
        max_output_tokens=4096,
        short_json_keys=True,  # smaller models benefit from compact output
    ),
    "qwen": ProviderConfig(
        name="Qwen (local)",
        json_mode=False,  # oMLX/Ollama response_format breaks Qwen JSON output; prompt alone works
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=4000,
        max_output_tokens=8192,
        short_json_keys=False,
    ),
    "mistral": ProviderConfig(
        name="Ollama Mistral (local)",
        json_mode=True,
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=4000,
        max_output_tokens=4096,
        short_json_keys=True,
    ),
}

# Default config for unknown models — conservative
_DEFAULT_CONFIG = ProviderConfig(
    name="Unknown",
    json_mode=True,
    json_object_required=True,  # safe default
    json_mode_required_for_json=False,
    input_batch_tokens=4000,  # conservative
    max_output_tokens=8192,
    short_json_keys=False,
)


def get_provider_config(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ProviderConfig:
    """Get the provider configuration for the current or specified model.

    Detection order:
    1. Match model name against known provider keys
    2. Match base_url against known providers (e.g., ollama)
    3. Fall back to default conservative config

    Args:
        model: Model name (default: settings.LLM_MODEL).
        base_url: Base URL (default: settings.LLM_BASE_URL).

    Returns:
        ProviderConfig for this model.
    """
    model = model or settings.LLM_MODEL or ""
    base_url = base_url or (settings.LLM_BASE_URL or "")
    combined = f"{model} {base_url}".lower()

    # Check each provider key
    for key, config in _PROVIDERS.items():
        if key in combined:
            return config

    # Check for Ollama via base_url (covers any local model)
    if "ollama" in base_url.lower() or "localhost:11434" in base_url.lower():
        return _PROVIDERS["llama"]  # treat as local, conservative

    logger.debug("Unknown LLM provider for model=%s, base_url=%s — using default", model, base_url)
    return _DEFAULT_CONFIG
