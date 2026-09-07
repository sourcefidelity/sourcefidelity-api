"""Search provider factory.

Selects configured search providers.
Mirrors the retrieval factory pattern (retrieval/__init__.py).

Usage:
    from app.services.search import get_search_provider
    provider = get_search_provider()
    if provider:
        results = provider.search('"Some Title" filetype:pdf')

Bing Search API is intentionally absent: Microsoft retired the standalone
service on August 11, 2025. Consumer-site scraping is not an API replacement.
"""

import logging
from typing import Optional

from app.config import settings
from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


def get_search_provider(provider_name: str | None = None) -> Optional[SearchProvider]:
    """Get a named or configured search provider, or None if unavailable.

    Returns None silently (not an error) — web search is an optional fallback.
    The retrieval chain works without it; search just improves coverage for
    hard-to-find sources.
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
            settings.GOOGLE_SEARCH_API_KEY,
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
        if not settings.BRAVE_SEARCH_API_KEY:
            logger.warning(
                "SEARCH_PROVIDER=brave but BRAVE_SEARCH_API_KEY not set — web search disabled"
            )
            return None
        from app.services.search.brave import BraveSearch
        return BraveSearch(settings.BRAVE_SEARCH_API_KEY)

    if provider_name == "duckduckgo":
        # No API key needed — uses the free HTML endpoint (unofficial)
        from app.services.search.duckduckgo import DuckDuckGoSearch
        return DuckDuckGoSearch()

    if provider_name == "tavily":
        if not settings.TAVILY_API_KEY:
            logger.warning("SEARCH_PROVIDER=tavily but TAVILY_API_KEY not set — web search disabled")
            return None
        from app.services.search.tavily import TavilySearch
        return TavilySearch(settings.TAVILY_API_KEY)

    if provider_name == "exa":
        if not settings.EXA_API_KEY:
            logger.warning("SEARCH_PROVIDER=exa but EXA_API_KEY not set — web search disabled")
            return None
        from app.services.search.exa import ExaSearch
        return ExaSearch(settings.EXA_API_KEY)

    logger.warning("Unknown SEARCH_PROVIDER=%s — web search disabled", provider_name)
    return None
