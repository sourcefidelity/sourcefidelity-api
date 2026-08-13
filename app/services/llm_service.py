"""LLM integration service.

Handles all LLM calls (OpenAI-compatible API).
Used for reference parsing, judgment, APA checking, etc.

Model-agnostic: provider-specific behavior (JSON mode, batching thresholds)
is configured via `app.services.providers.ProviderConfig`.
"""

import json
import logging
from typing import Optional, List, Dict, Any

from openai import OpenAI

from app.config import settings
from app.services.providers import get_provider_config

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_provider_cache: Optional[object] = None


def get_client() -> OpenAI:
    """Get or create the OpenAI client singleton."""
    global _client
    if _client is None:
        client_kwargs = {"api_key": settings.LLM_API_KEY}
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
    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not configured. Set it in .env or environment.")

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
    if disable_thinking:
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
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        logger.debug(
            "LLM response: model=%s, tokens=%d",
            model,
            response.usage.total_tokens if response.usage else 0,
        )
        return content
    except Exception as e:
        logger.error("LLM request failed: %s", e)
        raise RuntimeError(f"LLM request failed: {e}") from e


def _salvage_truncated_json(text: str) -> Optional[Dict[str, Any]]:
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
) -> Dict[str, Any]:
    """Send a chat completion request expecting JSON output.

    Robust to the three failure modes any LLM can produce on structured output,
    with failure-type-aware retry (avoids wasting time on identical retries that
    will fail identically):

      1. Truncated JSON (partial output, e.g. model stopped mid-generation) —
         SALVAGE IMMEDIATELY. Retrying truncates identically; the complete objects
         before the cut are valid, so recover them and return on the first occurrence.
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
        Parsed JSON dictionary (complete or salvaged-partial).

    Raises:
        RuntimeError: If LLM is not configured or all retries + salvage fail.
    """
    config = get_provider_config(model)
    response_format = {"type": "json_object"} if config.json_mode else None

    last_error = None
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
            )

            # Parse JSON — success path
            data = json.loads(response_text.strip())
            return data

        except json.JSONDecodeError as e:
            last_error = e
            is_empty = not (response_text and response_text.strip())
            logger.warning(
                "JSON parse failed (attempt %d/%d, %s): %s\nResponse: %s",
                attempt + 1,
                max_retries + 1,
                "empty" if is_empty else "truncated/malformed",
                e,
                response_text[:500] if response_text else "N/A",
            )

            # Truncated/malformed (has content) — try salvage IMMEDIATELY.
            # Retrying a truncation produces the same truncation; the partial
            # data is already valid, so recover it now instead of wasting calls.
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

                if has_content:
                    logger.info(
                        "Salvaged %d items from truncated JSON (attempt %d/%d) — "
                        "partial result returned, caller may flag for review.",
                        n_items,
                        attempt + 1,
                        max_retries + 1,
                    )
                    return salvaged
                # Salvage failed (genuinely malformed, not truncation) — retry once
                # with a JSON-only hint, then give up.
                if attempt < max_retries:
                    user_prompt = (
                        f"{user_prompt}\n\nIMPORTANT: Output valid JSON only. "
                        "No markdown, no explanation."
                    )

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

        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

    raise RuntimeError(
        f"Failed to get valid JSON after {attempt + 1} attempts: {last_error}"
    )


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
