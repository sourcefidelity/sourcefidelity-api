"""Abstract base class for web search providers.

Pluggable search interface — Google Custom Search, Bing Web Search, or a
future Baidu implementation. Used by the web-search PDF retrieval adapter
to find open-access PDFs for sources the academic-DB chain couldn't locate.

Follows the same pluggable pattern as retrieval adapters (retrieval/base.py):
each provider implements search(), the factory selects based on config.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single web search result."""
    url: str           # The result URL
    title: str         # Page title from search result
    snippet: str       # Short text snippet from search result
    is_pdf: bool = False  # True if the URL ends in .pdf or the result indicates a PDF


class SearchProvider(ABC):
    """Abstract web search provider.

    Implementations: GoogleCustomSearch, BingWebSearch.
    Selected via settings.SEARCH_PROVIDER in the factory (search/__init__.py).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search the web and return results.

        Args:
            query: The search query string.
            num_results: Max results to return (provider may cap lower).

        Returns:
            List of SearchResult. Empty list on error (no exception —
            callers handle gracefully, search is a best-effort fallback).
        """
        ...
