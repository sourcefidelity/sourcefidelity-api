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

Usage:
    from app.services.providers import get_provider_config
    config = get_provider_config()
    if config.json_mode:
        response_format = {"type": "json_object"}
"""

import logging
from dataclasses import dataclass
from typing import Optional

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
        name="Ollama Qwen (local)",
        json_mode=True,
        json_object_required=False,
        json_mode_required_for_json=False,
        input_batch_tokens=4000,
        max_output_tokens=4096,
        short_json_keys=True,
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
