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

    if provider_name == "searxng":
        if not settings.SEARXNG_URL:
            logger.warning(
                "SEARCH_PROVIDER=searxng but SEARXNG_URL not set — web search disabled. "
                "Self-host: docker run -d -p 8080:8080 searxng/searxng"
            )
            return None
        from app.services.search.searxng import SearXNGSearch
        return SearXNGSearch(
            settings.SEARXNG_URL,
            request_timeout_seconds=settings.SEARXNG_REQUEST_TIMEOUT_SECONDS,
            engine_timeout_seconds=settings.SEARXNG_ENGINE_TIMEOUT_SECONDS,
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

    if provider_name == "brightdata":
        if not settings.BRIGHTDATA_API_TOKEN or not settings.BRIGHTDATA_SERP_ZONE:
            logger.warning(
                "SEARCH_PROVIDER=brightdata but BRIGHTDATA_API_TOKEN or "
                "BRIGHTDATA_SERP_ZONE not set — web search disabled"
            )
            return None
        from app.services.search.brightdata import BrightDataSearch
        return BrightDataSearch(
            secret_value(settings.BRIGHTDATA_API_TOKEN),
            settings.BRIGHTDATA_SERP_ZONE,
            engine=settings.BRIGHTDATA_SERP_ENGINE,
            timeout_seconds=settings.BRIGHTDATA_TIMEOUT_SECONDS,
            language=settings.BRIGHTDATA_SEARCH_LANGUAGE,
            region=settings.BRIGHTDATA_SEARCH_REGION,
        )

    if provider_name == "duckduckgo":
        # Removed 2026-09-22. The adapter used DuckDuckGo's unofficial HTML
        # endpoint, which its terms do not permit. The official Instant Answer
        # API is not a substitute: it returns no ranked results for reference
        # titles. Refused by name so an existing configuration fails loudly
        # rather than silently resolving to no provider.
        logger.warning(
            "SEARCH_PROVIDER=duckduckgo is no longer supported — the "
            "unofficial HTML endpoint it used is not permitted by "
            "DuckDuckGo's terms. Configure brave, exa, brightdata or searxng."
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

    logger.warning("Unknown SEARCH_PROVIDER=%s — web search disabled", provider_name)
    return None
