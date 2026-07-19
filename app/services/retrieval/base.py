"""Retrieval source interface and result type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """The result of a retrieval attempt."""

    source_name: str
    success: bool
    metadata: dict | None = None  # raw metadata from source
    full_text: bytes | None = None  # PDF or HTML bytes
    full_text_url: str | None = None  # URL to full text (if not directly downloadable)
    abstract: str | None = None  # abstract text (for paywalled sources)
    doi: str | None = None
    title: str | None = None
    year: str | None = None
    authors: list[str] = field(default_factory=list)
    error: str | None = None


class RetrievalSource(ABC):
    """Abstract retrieval source."""

    name: str = "base"

    @abstractmethod
    def search_by_doi(self, doi: str) -> RetrievalResult:
        """Search for a document by DOI."""
        ...

    @abstractmethod
    def search_by_title_author(
        self,
        title: str,
        author: str | None = None,
    ) -> RetrievalResult:
        """Search for a document by title and optional author."""
        ...

    def download_full_text(self, result: RetrievalResult) -> RetrievalResult:
        """Download full text from a result that has a full_text_url.

        Override in subclasses that support direct download.
        Base implementation returns the result unchanged.
        """
        return result
