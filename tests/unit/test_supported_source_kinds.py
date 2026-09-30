"""Adapters declare the bibliographic kinds they can plausibly return.

Until 2026-09-23 routing by kind was hand-written per provider inside the
resolver, which is why one public-domain path had a year bound and its sibling
did not, and why a book catalogue was queried for journal articles. Measured
over the 11-paper corpus run, Open Library produced **9 candidates on
journal-article references and not one was plausible** — noise the identity
rules then had to reject.

`unknown` always passes. Not knowing what a reference is cannot be a reason to
search less for it.
"""
import pytest

from app.services.retrieval import _RETRIEVER_CLASSES
from app.services.retrieval.base import BOOK_SOURCE_KINDS, RetrievalSource


def test_book_catalogues_declare_book_kinds() -> None:
    for name in ("open_library", "gutenberg", "wikisource"):
        assert _RETRIEVER_CLASSES[name].supported_source_kinds == BOOK_SOURCE_KINDS


@pytest.mark.parametrize("name", ["open_library", "gutenberg", "wikisource"])
@pytest.mark.parametrize("kind", ["journal_article", "webpage", "report", "thesis"])
def test_a_book_catalogue_is_not_consulted_for_a_non_book(name, kind) -> None:
    assert _RETRIEVER_CLASSES[name].handles_source_kind(kind) is False


@pytest.mark.parametrize("name", ["open_library", "gutenberg", "wikisource"])
@pytest.mark.parametrize("kind", ["monograph", "edited_collection", "book_section"])
def test_a_book_catalogue_is_consulted_for_a_book(name, kind) -> None:
    assert _RETRIEVER_CLASSES[name].handles_source_kind(kind) is True


@pytest.mark.parametrize("kind", ["unknown", "", None, "  ", "UNKNOWN"])
def test_an_unknown_kind_reaches_every_adapter(kind) -> None:
    """Fail open on ignorance: an unparsed reference must not be searched less."""
    for cls in _RETRIEVER_CLASSES.values():
        assert cls.handles_source_kind(kind) is True


def test_an_undeclared_adapter_is_unrestricted() -> None:
    """Silence means no restriction, so adding an adapter changes nothing."""
    assert RetrievalSource.supported_source_kinds is None
    assert RetrievalSource.handles_source_kind("journal_article") is True


def test_kind_matching_ignores_case_and_surrounding_space() -> None:
    cls = _RETRIEVER_CLASSES["open_library"]

    assert cls.handles_source_kind("  Monograph  ") is True
    assert cls.handles_source_kind("JOURNAL_ARTICLE") is False


def test_crossref_and_openalex_stay_unrestricted() -> None:
    """The two broad registries hold articles, chapters, books and reports."""
    for name in ("crossref", "openalex"):
        assert _RETRIEVER_CLASSES[name].supported_source_kinds is None


@pytest.mark.parametrize("name", ["core", "semantic_scholar", "eric", "elsevier", "datacite"])
@pytest.mark.parametrize("kind", ["monograph", "edited_collection", "webpage", "news_article", "video"])
def test_paper_indexes_are_not_consulted_for_books_or_web_media(name, kind) -> None:
    """Owner decision 2026-09-24 (reference-verification-v1): only indexes that
    hold the kind of work are queried.

    Measured before narrowing, over every stored job: of 64 distinct book
    references and 66 web/media references, CORE, Semantic Scholar and DataCite
    confirmed 22 between them, and none was the only confirmation; a catalogue,
    Crossref or OpenAlex confirmed each too. Elsevier keeps book chapters.
    """
    assert _RETRIEVER_CLASSES[name].handles_source_kind(kind) is False


@pytest.mark.parametrize("name", ["core", "semantic_scholar", "eric", "elsevier", "datacite"])
@pytest.mark.parametrize("kind", ["journal_article", "conference_paper", "report", "thesis"])
def test_paper_indexes_are_consulted_for_papers(name, kind) -> None:
    assert _RETRIEVER_CLASSES[name].handles_source_kind(kind) is True


def test_a_source_without_the_method_is_still_consulted() -> None:
    """Routing must not require every source object to derive from the base.

    The resolver reads this through `getattr`, as it already does for
    `capabilities`. A duck-typed source keeps working unchanged.
    """
    from app.services.source_resolver import _any_source_kind

    class Duck:
        name = "duck"
        capabilities = {"metadata_only_search"}

    assert getattr(Duck(), "handles_source_kind", _any_source_kind)("monograph") is True
    assert _any_source_kind("journal_article") is True
