"""LLM integration service.

Handles all LLM calls (OpenAI-compatible API).
Used for reference parsing, judgment, APA checking, etc.

Model-agnostic: provider-specific behavior (JSON mode, batching thresholds)
is configured via `app.services.providers.ProviderConfig`.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, List, Dict, Any

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from app.config import secret_value, settings
from app.services.providers import ProviderConfig, get_provider_config
from app.services.processing_metrics import record_llm_attempt, record_llm_usage

JSON_REPAIR_SUFFIX = "\n\nIMPORTANT: Output valid JSON only. No markdown, no explanation."

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_provider_cache: Optional[object] = None


@dataclass(frozen=True)
class LLMRoute:
    """One explicitly routed model endpoint (a Judgment panel arm).

    Carries its own client, model, provider config and request body extras, so
    a routed call never falls back to the default client or to model-name
    detection (a Qwen API model name would otherwise select the local-Qwen
    config). Built only by `judge_arms`, after that arm's policy gate passed.
    """

    arm_id: str
    model: str
    endpoint_host: str
    client_factory: Callable[[], Any]
    provider_config: ProviderConfig
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    price_context: Callable[[str], dict] = lambda model: {}
    # Output allowance for this arm when it must reason before answering.
    max_output_tokens: int | None = None


def _receipt_from_response(response, receipt: dict) -> None:
    """Numeric and identifying provenance only; never response text."""
    receipt["returned_model"] = str(getattr(response, "model", "") or "")[:120]
    extra = getattr(response, "model_extra", None) or {}
    provider = extra.get("provider") if isinstance(extra, dict) else None
    receipt["returned_provider"] = str(provider)[:80] if provider else None
    usage = getattr(response, "usage", None)
    for name in ("prompt_tokens", "completion_tokens", "total_tokens",
                 "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            receipt[name] = value
    receipt.update(cache_split(usage))
    usage_extra = getattr(usage, "model_extra", None) or {}
    cost = usage_extra.get("cost") if isinstance(usage_extra, dict) else None
    receipt["reported_cost_usd"] = float(cost) if type(cost) in (int, float) and cost >= 0 else None


def cache_split(usage) -> dict:
    """Cache-hit and miss prompt tokens for a response that lacks DeepSeek's fields:
    the OpenAI-style cached count when reported, otherwise no cache hit."""
    prompt = getattr(usage, "prompt_tokens", None)
    if (getattr(usage, "prompt_cache_hit_tokens", None) is not None
            or not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0):
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    cached = cached if isinstance(cached, int) and not isinstance(cached, bool) and 0 <= cached <= prompt else 0
    return {"prompt_cache_hit_tokens": cached, "prompt_cache_miss_tokens": prompt - cached}


class LLMCallFailure(RuntimeError):
    """Application-owned failure category; never retain provider response text."""

    def __init__(self, category: str, *, attempts: int = 1):
        allowed = {
            "not_configured", "timeout", "connection_failed", "rate_limited",
            "authentication_failed", "provider_server_error", "provider_request_rejected",
            "request_failed", "empty_response", "invalid_json",
        }
        self.category = category if category in allowed else "request_failed"
        self.attempts = attempts
        super().__init__(f"Failed to get valid JSON or complete LLM request: {self.category}")


def _request_failure_category(error: Exception) -> str:
    if isinstance(error, APITimeoutError):
        return "timeout"
    if isinstance(error, APIConnectionError):
        return "connection_failed"
    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            return "rate_limited"
        if error.status_code in {401, 403}:
            return "authentication_failed"
        return "provider_server_error" if error.status_code >= 500 else "provider_request_rejected"
    return "request_failed"


def get_client() -> OpenAI:
    """Get or create the OpenAI client singleton."""
    global _client
    if _client is None:
        client_kwargs = {"api_key": secret_value(settings.LLM_API_KEY)}
        if settings.LLM_BASE_URL:
            client_kwargs["base_url"] = settings.LLM_BASE_URL
        _client = OpenAI(**client_kwargs)
    return _client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    response_format: Optional[Dict[str, str]] = None,
    disable_thinking: bool = False,
    reasoning_effort: Optional[str] = None,
    route: Optional[LLMRoute] = None,
    receipt: Optional[dict] = None,
) -> str:
    """Send a chat completion request to the LLM.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User message.
        model: Model name (default from settings).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        disable_thinking: If True, disable the model's thinking/reasoning mode
            for this call (when the provider supports it — see
            ProviderConfig.reasoning_disable_body). Use for low-stakes structured
            passes where reasoning can exhaust the token budget and return empty.
            The verification judge should keep thinking ON.
        reasoning_effort: Optional reasoning effort level for reasoning models
            that support it (DeepSeek V4: "low"/"high"/"max"). Kept SEPARATE
            from disable_thinking: effort caps HOW MUCH the model reasons while
            keeping thinking on (preserves reasoning-quality benefits), whereas
            disable_thinking turns it off entirely. Ignored when
            disable_thinking=True (thinking off makes effort meaningless).
            Note: the effort string maps per-model on the provider side — e.g.
            "low" on deepseek-v4-flash = low effort, but "low" on deepseek-v4-pro
            maps to "high" (see DeepSeek thinking_mode docs). Per-call only.

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If LLM is not configured or request fails.
    """
    if route is None and not settings.LLM_API_KEY:
        raise LLMCallFailure("not_configured")

    if route is not None:
        # A routed arm uses only its own client, model and body extras.
        try:
            client = route.client_factory()
        except Exception:
            raise LLMCallFailure("not_configured") from None
        model = route.model
    else:
        client = get_client()
        model = model or settings.LLM_MODEL

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add response format if specified (for JSON mode)
    if response_format:
        kwargs["response_format"] = response_format

    # Reasoning control. Two independent knobs for reasoning models (DeepSeek V4,
    # OpenAI o-series), passed via extra_body (OpenAI-compatible passthrough):
    #   - disable_thinking: turn reasoning OFF entirely (binary). Guarded by
    #     ProviderConfig.reasoning_disable_body — no-op for providers without it.
    #   - reasoning_effort: cap HOW MUCH reasoning while keeping it ON
    #     ("low"/"high"/"max"). Guarded by ProviderConfig.reasoning_effort_supported
    #     — if False, the effort string is DROPPED (with a warning, so silent
    #     degradation on model swap is visible) rather than sent to a provider
    #     that may reject or no-op it. Ignored when thinking is disabled.
    extra_body: Dict[str, Any] = {}
    if route is not None:
        extra_body.update(route.extra_body)
    elif disable_thinking:
        config = get_provider_config(model)
        if config.reasoning_disable_body:
            extra_body.update(config.reasoning_disable_body)
            logger.debug("Thinking disabled for model=%s (extra_body=%s)", model, config.reasoning_disable_body)
        else:
            logger.debug("disable_thinking=True but model=%s has no reasoning_disable_body — no-op", model)
    elif reasoning_effort:
        config = get_provider_config(model)
        if config.reasoning_effort_supported:
            extra_body["reasoning_effort"] = reasoning_effort
            logger.debug("Reasoning effort=%s for model=%s", reasoning_effort, model)
        else:
            # Provider doesn't support effort levels — DROP rather than send.
            # This is the plug-and-play guard: swapping cloud models should not
            # silently send an unsupported param. Visible as a WARNING (not debug)
            # so degradation surfaces in logs instead of hiding as a quality drop.
            logger.warning(
                "reasoning_effort=%s requested but model=%s (%s) does not support "
                "effort levels — dropping. Call will run at the provider's default "
                "reasoning; re-validate caller if quality changes.",
                reasoning_effort, model, config.name,
            )
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        from urllib.parse import urlsplit
        from app.services.usage_cost import llm_price_context
        if route is not None:
            price_context = route.price_context(model)
            endpoint_host = route.endpoint_host
        else:
            price_context = llm_price_context(model, settings.LLM_BASE_URL or '')
            # Host only: never a path, query or credential. The OpenAI client's
            # default endpoint is used when no base URL is configured.
            endpoint_host = urlsplit(settings.LLM_BASE_URL).hostname if settings.LLM_BASE_URL else 'api.openai.com'
        record_llm_attempt(model=model, endpoint_host=endpoint_host)
        response = client.chat.completions.create(**kwargs)
        record_llm_usage(response.usage, price_context=price_context, model=model, endpoint_host=endpoint_host)
        if receipt is not None:
            _receipt_from_response(response, receipt)
            receipt["price_context"] = dict(price_context or {})
        content = response.choices[0].message.content
        logger.debug(
            "LLM response: model=%s, tokens=%d",
            model,
            response.usage.total_tokens if response.usage else 0,
        )
        return content or ""
    except Exception as e:
        logger.error("LLM request failed (type=%s)", type(e).__name__)
        raise LLMCallFailure(_request_failure_category(e)) from None


def _salvage_truncated_json(text: str) -> Any | None:
    """Attempt to recover complete data from truncated JSON output.

    When an LLM stops mid-generation (finish_reason='length' or a network cut),
    the output is invalid JSON but often contains complete, usable objects before
    the truncation point. This function tries to close the JSON structure at the
    last complete object boundary and parse what's recoverable.

    Works for the common shape {"references": [{...}, {...}, {partial...}]},
    which is exactly what reference parsing produces. Also handles the simpler
    {"key": "value", partial... case.

    Args:
        text: The raw (potentially truncated) LLM response text.

    Returns:
        Parsed dict if salvage succeeded, None otherwise.
    """
    text = (text or "").strip()
    if not text:
        return None

    # Already valid JSON — nothing to salvage
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 1: find the last complete array element ("},") and close the array.
    # Handles {"references": [{...}, {...}, {truncated
    last_complete_obj = text.rfind("},")
    if last_complete_obj > 0:
        salvaged = text[: last_complete_obj + 1]  # include the closing }
        # Find the array opening to know we need to close it
        if '"references"' in salvaged or salvaged.rstrip().endswith("}"):
            # Close array + object: }]}  (covers {"references":[{...} )
            # But only close what's actually open. Count brackets.
            open_arrays = salvaged.count("[") - salvaged.count("]")
            open_objects = salvaged.count("{") - salvaged.count("}")
            candidate = salvaged + ("]" * max(open_arrays, 0)) + ("}" * max(open_objects, 0))
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 2: last complete string value before a truncation.
    # Handles {"key": "value", "key2": "partial
    last_quote = text.rfind('",')
    if last_quote > 0:
        salvaged = text[: last_quote + 1]  # include the closing "
        open_objects = salvaged.count("{") - salvaged.count("}")
        candidate = salvaged + ("}" * max(open_objects, 0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    max_retries: int = 2,
    disable_thinking: bool = False,
    reasoning_effort: Optional[str] = None,
    allow_partial: bool = False,
    route: Optional[LLMRoute] = None,
    receipt: Optional[dict] = None,
) -> Dict[str, Any]:
    """Send a chat completion request expecting JSON output.

    Robust to the three failure modes any LLM can produce on structured output,
    with failure-type-aware retry (avoids wasting time on identical retries that
    will fail identically):

      1. Truncated JSON (partial output, e.g. model stopped mid-generation) —
         RETRY and fail closed by default. A caller must explicitly opt into a
         visibly partial result with ``allow_partial=True``.
      2. Empty response (model returned nothing) — RETRY ONCE. Empty responses are
         often transient (rate-limit, momentary glitch); a single retry usually
         succeeds. Don't burn all max_retries on them.
      3. Malformed JSON (non-JSON text) — RETRY with a "JSON only" hint.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User message.
        model: Model name (default from settings).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        max_retries: Maximum number of retries on malformed/empty responses.
        disable_thinking: If True, disable the model's thinking/reasoning mode.
            See chat_completion() for rationale.
        reasoning_effort: Optional reasoning effort level ("low"/"high"/"max").
            See chat_completion() for rationale and the per-model-mapping caveat.
            Ignored when disable_thinking=True.

    Returns:
        Parsed JSON dictionary. Partial salvage is returned only when explicitly
        requested with ``allow_partial=True``.

    Raises:
        RuntimeError: If LLM is not configured or all retries + salvage fail.
    """
    config = route.provider_config if route is not None else get_provider_config(model)
    response_format = {"type": "json_object"} if config.json_mode else None

    empty_attempts = 0
    for attempt in range(max_retries + 1):
        try:
            response_text = chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                disable_thinking=disable_thinking,
                reasoning_effort=reasoning_effort,
                **({"route": route} if route is not None else {}),
                **({"receipt": receipt} if receipt is not None else {}),
            )
            if receipt is not None:
                receipt["attempts"] = attempt + 1

            # Parse JSON — success path
            data = json.loads(response_text.strip())
            return data

        except json.JSONDecodeError:
            is_empty = not (response_text and response_text.strip())
            logger.warning(
                "JSON parse failed (attempt %d/%d, %s; response_chars=%d)",
                attempt + 1,
                max_retries + 1,
                "empty" if is_empty else "truncated/malformed",
                len(response_text or ""),
            )

            # A partial response is not a complete structured judgment. Keep it
            # only for an explicitly partial-tolerant caller, and only after the
            # configured complete-response attempts are exhausted.
            if not is_empty:
                salvaged = _salvage_truncated_json(response_text)
                # salvaged may be a dict ({"references": [...]}) or a bare list
                # ([...]). Accept either, as long as it has recoverable content.
                if isinstance(salvaged, dict):
                    has_content = bool(salvaged.get("references")) or any(
                        isinstance(v, list) and v for v in salvaged.values()
                    )
                    n_items = sum(
                        len(v) for v in salvaged.values() if isinstance(v, list)
                    )
                elif isinstance(salvaged, list):
                    has_content = bool(salvaged)
                    n_items = len(salvaged)
                else:
                    has_content = False
                    n_items = 0

                if has_content and allow_partial and attempt == max_retries:
                    logger.info(
                        "Returning explicitly permitted partial JSON (%d items)",
                        n_items,
                    )
                    return salvaged
                if attempt < max_retries:
                    user_prompt = user_prompt + JSON_REPAIR_SUFFIX

            # Empty response — allow ONE retry (transient), then stop retrying.
            # Burning all max_retries on identical empty responses wastes minutes.
            else:
                empty_attempts += 1
                if empty_attempts > 1:
                    logger.warning(
                        "Empty LLM response on %d attempts — stopping retries to "
                        "avoid wasting time; raising for caller fallback.",
                        empty_attempts,
                    )
                    break

        except LLMCallFailure:
            raise
        except Exception as e:
            raise LLMCallFailure(_request_failure_category(e)) from None

    raise LLMCallFailure(
        "empty_response" if is_empty else "invalid_json", attempts=attempt + 1
    ) from None


def _supports_json_mode(model: Optional[str] = None) -> bool:
    """Check if the model supports JSON mode (delegates to ProviderConfig).

    Kept for backward compatibility with existing callers.
    """
    return get_provider_config(model).json_mode


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------


def batch_process(
    items: List[Any],
    batch_size: int,
    process_fn,
    on_batch_complete=None,
) -> List[Any]:
    """Process items in batches.

    Args:
        items: List of items to process.
        batch_size: Number of items per batch.
        process_fn: Function to process a batch, takes list of items, returns list of results.
        on_batch_complete: Optional callback after each batch (batch_index, batch_results).

    Returns:
        Flattened list of all results.
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)

        if on_batch_complete:
            on_batch_complete(i // batch_size, batch_results)

    return results
