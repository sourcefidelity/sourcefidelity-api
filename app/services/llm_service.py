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
) -> str:
    """Send a chat completion request to the LLM.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User message.
        model: Model name (default from settings).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        response_format: Optional response format (e.g., {"type": "json_object"}).

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


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Send a chat completion request expecting JSON output.

    Automatically retries on JSON parse failures.
    Uses provider-specific JSON mode settings from ProviderConfig.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User message.
        model: Model name (default from settings).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        max_retries: Number of retries on JSON parse failure.

    Returns:
        Parsed JSON dictionary.

    Raises:
        RuntimeError: If LLM is not configured or all retries fail.
    """
    config = get_provider_config(model)

    # Use JSON mode based on provider config
    response_format = {"type": "json_object"} if config.json_mode else None

    last_error = None
    response_text = ""
    for attempt in range(max_retries + 1):
        try:
            response_text = chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            # Parse JSON
            data = json.loads(response_text.strip())
            return data

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s\nResponse: %s",
                attempt + 1,
                max_retries + 1,
                e,
                response_text[:500] if response_text else "N/A",
            )
            if attempt < max_retries:
                # Add hint to retry
                user_prompt = f"{user_prompt}\n\nIMPORTANT: Output valid JSON only. No markdown, no explanation."

        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

    raise RuntimeError(f"Failed to get valid JSON after {max_retries + 1} attempts: {last_error}")


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
