"""Retrieval source interface and result type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RepresentationKind(str, Enum):
    """Machine-actionable representation kinds with distinct validation paths."""

    PDF = "pdf"
    HTML = "html"
    XML = "xml"
    EPUB = "epub"
    PLAIN_TEXT = "plain_text"
    ABSTRACT = "abstract"
    METADATA = "metadata"


@dataclass(frozen=True)
class AcquisitionLocation:
    """One possible place and format from which a work can be acquired."""

    url: str
    provider: str
    media_type: str | None = None
    representation_kind: RepresentationKind | None = None
    landing_page_url: str | None = None
    host_type: str | None = None
    version: str | None = None
    license: str | None = None
    access_type: str | None = None
    intended_application: str | None = None
    is_best: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SourceRepresentation:
    """A typed acquired representation; never assume opaque bytes are a PDF."""

    kind: RepresentationKind
    media_type: str
    content: bytes
    source_url: str | None = None
    original_kind: RepresentationKind | None = None
    charset: str | None = None
    completeness: str = "not_assessed"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """The result of a retrieval attempt."""

    source_name: str
    success: bool
    metadata: dict | None = None  # raw metadata from source
    # Transitional compatibility fields. New code must prefer representation
    # and locations; these remain until every consumer is migrated.
    full_text: bytes | None = None
    full_text_url: str | None = None
    representation: SourceRepresentation | None = None
    locations: list[AcquisitionLocation] = field(default_factory=list)
    abstract: str | None = None  # abstract text (for paywalled sources)
    doi: str | None = None
    title: str | None = None
    year: str | None = None
    authors: list[str] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self) -> None:
        """Bridge old callers without silently labelling non-PDF bytes as PDF."""
        if self.representation is not None and self.full_text is None:
            self.full_text = self.representation.content
        elif self.full_text is not None and self.representation is None:
            kind = (
                RepresentationKind.PDF
                if self.full_text.startswith(b"%PDF-")
                else RepresentationKind.PLAIN_TEXT
            )
            media_type = "application/pdf" if kind is RepresentationKind.PDF else "text/plain"
            self.representation = SourceRepresentation(
                kind=kind,
                media_type=media_type,
                content=self.full_text,
                source_url=self.full_text_url,
            )
        if self.full_text_url and not self.locations:
            self.locations.append(
                AcquisitionLocation(
                    url=self.full_text_url,
                    provider=self.source_name,
                    representation_kind=(
                        self.representation.kind if self.representation else None
                    ),
                    media_type=(
                        self.representation.media_type if self.representation else None
                    ),
                    is_best=True,
                )
            )

    def set_representation(self, representation: SourceRepresentation) -> None:
        self.representation = representation
        self.full_text = representation.content


class RetrievalSource(ABC):
    """Abstract retrieval source."""

    name: str = "base"
    capabilities: frozenset[str] = frozenset()
    documentation_url: str | None = None

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
