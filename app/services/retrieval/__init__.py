"""Retrieval source package.

Exposes the retrieval source ABC/result and a factory that builds sources
in the priority order configured by RETRIEVAL_SOURCES.
"""

from app.services.retrieval.base import RetrievalSource, RetrievalResult
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.semantic_scholar import SemanticScholarRetriever
from app.services.retrieval.core import CoreRetriever
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.retrieval.gutenberg import GutenbergRetriever
from app.services.retrieval.wikisource import WikisourceRetriever
from app.services.retrieval.elsevier import ElsevierRetriever
from app.services.retrieval.web_search import WebSearchRetriever

from app.config import settings

__all__ = [
    "RetrievalSource",
    "RetrievalResult",
    "get_retrieval_sources",
]


# Maps config names to their retriever classes. Unknown names in
# RETRIEVAL_SOURCES are ignored, which allows future custom sources to
# be added without breaking older configs.
_RETRIEVER_CLASSES: dict[str, type[RetrievalSource]] = {
    "openalex": OpenAlexRetriever,
    "semantic_scholar": SemanticScholarRetriever,
    "core": CoreRetriever,
    "crossref": CrossrefRetriever,
    "gutenberg": GutenbergRetriever,
    "wikisource": WikisourceRetriever,
    "elsevier": ElsevierRetriever,
    # Web-search fallback: searches Google/Bing for source PDFs after the
    # academic-DB chain fails. Only active when SEARCH_PROVIDER is configured.
    # Add "web_search" to RETRIEVAL_SOURCES in .env to enable.
    "web_search": WebSearchRetriever,
}


def get_retrieval_sources() -> list[RetrievalSource]:
    """Return configured retrieval sources in priority order."""
    names = [s.strip().lower() for s in settings.RETRIEVAL_SOURCES.split(",") if s.strip()]
    sources: list[RetrievalSource] = []
    for name in names:
        cls = _RETRIEVER_CLASSES.get(name)
        if cls is not None:
            sources.append(cls())
    return sources
