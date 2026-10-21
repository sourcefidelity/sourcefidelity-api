"""Search provider factory.

Selects configured search providers.
Mirrors the retrieval factory pattern (retrieval/__init__.py).

Usage:
    from app.services.search import get_search_provider
    provider = get_search_provider()
    if provider:
        results = provider.search('"Some Title" filetype:pdf')

Bing Search API is intentionally absent: Microsoft retired the standalone
service on August 11, 2025. Consumer-site scraping is not an API replacement,
which is also why the DuckDuckGo adapter was removed on 2026-09-22.
"""

import logging
from typing import Optional

from app.config import secret_value, settings
from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


def get_search_provider(provider_name: str | None = None) -> Optional[SearchProvider]:
    """Get a named or configured search provider, or None if unavailable.

    Missing configuration returns None. The versioned orchestrator records
    missing required APIs as incomplete; absence of a key is not an empty search.
    """
    provider_name = (provider_name or settings.SEARCH_PROVIDER or "").lower().strip()

    if not provider_name:
        return None

    if provider_name == "google":
        if not settings.GOOGLE_SEARCH_API_KEY or not settings.GOOGLE_SEARCH_CSE_ID:
            logger.warning(
                "SEARCH_PROVIDER=google but GOOGLE_SEARCH_API_KEY or "
                "GOOGLE_SEARCH_CSE_ID not set — web search disabled"
            )
            return None
        from app.services.search.google import GoogleCustomSearch
        return GoogleCustomSearch(
            secret_value(settings.GOOGLE_SEARCH_API_KEY),
            settings.GOOGLE_SEARCH_CSE_ID,
        )

    if provider_name == "brave":
        if settings.SEARCH_POLICY_VERSION == "configured-search-v1" and not settings.BRAVE_SEARCH_RETENTION_PERMITTED:
            logger.warning("Brave transient discovery requires API-first orchestration; legacy retained-output route unavailable")
            return None
        if not settings.BRAVE_SEARCH_API_KEY:
            logger.warning(
                "SEARCH_PROVIDER=brave but BRAVE_SEARCH_API_KEY not set — web search disabled"
            )
            return None
        from app.services.search.brave import BraveSearch
        return BraveSearch(secret_value(settings.BRAVE_SEARCH_API_KEY))

    if provider_name == "duckduckgo":
        # Removed 2026-09-22. The adapter used DuckDuckGo's unofficial HTML
        # endpoint, which its terms do not permit. The official Instant Answer
        # API is not a substitute: it returns no ranked results for reference
        # titles. Refused by name so an existing configuration fails loudly
        # rather than silently resolving to no provider.
        logger.warning(
            "SEARCH_PROVIDER=duckduckgo is no longer supported — the "
            "unofficial HTML endpoint it used is not permitted by "
            "DuckDuckGo's terms. Configure brave, exa, deepseek or tavily."
        )
        return None

    if provider_name == "tavily":
        if not settings.TAVILY_API_KEY:
            logger.warning("SEARCH_PROVIDER=tavily but TAVILY_API_KEY not set — web search disabled")
            return None
        from app.services.search.tavily import TavilySearch
        return TavilySearch(secret_value(settings.TAVILY_API_KEY))

    if provider_name == "exa":
        if not settings.EXA_API_KEY:
            logger.warning("SEARCH_PROVIDER=exa but EXA_API_KEY not set — web search disabled")
            return None
        from app.services.search.exa import ExaSearch
        return ExaSearch(secret_value(settings.EXA_API_KEY))

    if provider_name == "deepseek":
        # The Search API shares the DeepSeek model key, so it is offered only
        # when the configured model endpoint is DeepSeek's own.
        from urllib.parse import urlparse
        if not settings.LLM_API_KEY or urlparse(settings.LLM_BASE_URL or "").netloc != "api.deepseek.com":
            logger.warning(
                "SEARCH_PROVIDER=deepseek needs LLM_API_KEY with LLM_BASE_URL at "
                "api.deepseek.com — web search disabled"
            )
            return None
        from app.services.search.deepseek import DeepSeekSearch
        return DeepSeekSearch(secret_value(settings.LLM_API_KEY))

    logger.warning("Unknown SEARCH_PROVIDER=%s — web search disabled", provider_name)
    return None
