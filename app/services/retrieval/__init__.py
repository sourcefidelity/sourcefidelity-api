"""Retrieval source package.

Exposes the retrieval source ABC/result and a factory that builds sources
in the priority order configured by RETRIEVAL_SOURCES.
"""

from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
    SourceRepresentation,
)
from app.services.retrieval.canonical_work import (
    CanonicalWorkGraph,
    IdentityAssessment,
    assess_work_identity,
    canonicalize_location_url,
)
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.semantic_scholar import SemanticScholarRetriever
from app.services.retrieval.core import CoreRetriever
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.retrieval.gutenberg import GutenbergRetriever
from app.services.retrieval.wikisource import WikisourceRetriever
from app.services.retrieval.elsevier import ElsevierRetriever
from app.services.retrieval.web_search import WebSearchRetriever

from app.config import settings
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy

__all__ = [
    "AcquisitionLocation",
    "RepresentationKind",
    "RetrievalSource",
    "RetrievalResult",
    "SourceRepresentation",
    "CanonicalWorkGraph",
    "IdentityAssessment",
    "assess_work_identity",
    "canonicalize_location_url",
    "get_retrieval_sources",
    "installed_retrieval_providers",
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
    # Web-search fallback: searches configured discovery providers after the
    # academic-DB chain fails. Only active when SEARCH_PROVIDER is configured.
    # Add "web_search" to RETRIEVAL_SOURCES in .env to enable.
    "web_search": WebSearchRetriever,
}


def installed_retrieval_providers() -> dict[str, dict]:
    """Return non-secret metadata for trusted, installed adapter code."""
    installed: dict[str, dict] = {}
    for name, cls in _RETRIEVER_CLASSES.items():
        installed[name] = {
            "name": name,
            "capabilities": sorted(getattr(cls, "capabilities", frozenset())),
            "documentation_url": getattr(cls, "documentation_url", None),
        }
    return installed


def get_retrieval_sources() -> list[RetrievalSource]:
    """Return configured retrieval sources in priority order."""
    names = [s.strip().lower() for s in settings.RETRIEVAL_SOURCES.split(",") if s.strip()]
    sources: list[RetrievalSource] = []
    for name in names:
        cls = _RETRIEVER_CLASSES.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown retrieval provider {name!r}; install a trusted adapter "
                "before enabling it"
            )
        policy = provider_policy(name, getattr(cls, "default_policy", ProviderPolicy()))
        if policy.enabled:
            sources.append(cls())
    return sources
