"""Retrieval source interface and result type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


# The leading authors are the identifying ones - a citation and every matching
# rule use them - and `ExpectedBibliographicFields` keeps this many. Adapters
# that trim their own author lists trim to the same number, so no adapter can
# silently deliver less than the identity comparison would have kept.
OBSERVED_AUTHOR_LIMIT = 64

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
    # A normalized derivative may retain its immutable original separately.
    # The parent never becomes verification evidence merely because the
    # derivative was accepted.
    parent_representation: SourceRepresentation | None = None
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


# Catalogues of books hold monographs and their parts, and nothing else.
BOOK_SOURCE_KINDS = frozenset({"monograph", "edited_collection", "book_section"})
# Scholarly article and grey-literature indexes. They hold articles, papers,
# reports and theses, not books; a book is settled against a catalogue.
SCHOLARLY_PAPER_KINDS = frozenset(
    {"journal_article", "conference_paper", "book_review", "report", "thesis"})


class RetrievalSource(ABC):
    """Abstract retrieval source."""

    name: str = "base"
    capabilities: frozenset[str] = frozenset()
    documentation_url: str | None = None

    # Bibliographic kinds this adapter can plausibly return, or None when it
    # is not restricted by kind. Declared here so routing is checkable on the
    # class instead of hand-written per provider in the resolver, which is how
    # a book catalogue came to be queried for journal articles: measured over
    # the 2026-09-23 corpus run, Open Library produced 9 candidates on
    # journal-article references and **not one was plausible**.
    #
    # An unknown expected kind always passes: ignorance about a reference is
    # not a reason to narrow its search.
    supported_source_kinds: frozenset[str] | None = None

    # Whether this route must complete before the application may say a
    # reference could not be found. `True` is the SAFE direction: a required
    # route that fails forces `search_incomplete`, which blocks a potentially
    # fabricated reference finding. Marking an adapter `False` removes that
    # block, so it is a judgment about evidence, never a performance tweak.
    #
    # Until 2026-09-23 this was derived from `deferred`, which actually means
    # "batches DOI prefetch" -- an unrelated property -- so every adapter but
    # `semantic_scholar` was required by default and nobody had decided it.
    # `None` means undeclared; `tests/unit/test_required_route_declaration.py`
    # requires every registered adapter to state a value.
    required_for_search_completion: bool | None = None

    @classmethod
    def blocks_search_completion(cls) -> bool:
        """Resolve the declaration, falling back to the old derivation."""
        if cls.required_for_search_completion is not None:
            return cls.required_for_search_completion
        return not getattr(cls, "deferred", False)

    @classmethod
    def handles_source_kind(cls, kind: str | None) -> bool:
        """Whether this adapter is worth calling for this expected kind."""
        if cls.supported_source_kinds is None:
            return True
        normalized = (kind or "unknown").strip().casefold() or "unknown"
        return normalized == "unknown" or normalized in cls.supported_source_kinds

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
