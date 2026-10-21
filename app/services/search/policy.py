"""Immutable search-policy identifiers; never reinterpret legacy records."""

from app.config import settings

LEGACY_SEARCH_POLICY = "configured-search-v1"
API_FIRST_SEARCH_POLICY = "api-first-search-v2"
# Brave plus one second required web provider (SEARCH_SECOND_REQUIRED_PROVIDER),
# or DeepSeek alone (SEARCH_DEEPSEEK_ONLY). A stored record is judged by the
# providers it names, so records made with Exa stay valid after the setting
# changes, and the reverse.
SECOND_REQUIRED_WEB_PROVIDERS = frozenset({"exa", "deepseek"})
SOLE_REQUIRED_WEB_PROVIDERS = frozenset({"deepseek"})
REQUIRED_WEB_PROVIDER_NAMES = frozenset({"brave"}) | SECOND_REQUIRED_WEB_PROVIDERS
def required_api_providers() -> tuple[str, ...]:
    """Brave and the configured second provider, or DeepSeek alone, read at call time."""
    if settings.SEARCH_DEEPSEEK_ONLY:
        return ("deepseek",)
    return ("brave", settings.SEARCH_SECOND_REQUIRED_PROVIDER)


def is_required_web_provider_set(names) -> bool:
    """True when a record names Brave and exactly one accepted second provider,
    or DeepSeek alone."""
    named = frozenset(names or ())
    if named == SOLE_REQUIRED_WEB_PROVIDERS:
        return True
    return len(named) == 2 and "brave" in named and bool(named & SECOND_REQUIRED_WEB_PROVIDERS)
