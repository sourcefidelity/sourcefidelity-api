"""Public-domain catalogs are not asked for modern books.

Measured over the 11-paper corpus run, 2026-09-23. Gutenberg and Wikisource
were each queried 20 times, for books published **1978 to 2023** — the
earliest was Bork 1978 — found nothing, and together spent 615 seconds, 9.5%
of all attempt time. The kind gate was working: neither was ever called for a
journal article. The missing gate was the year. `_is_public_domain_front_route`
already bounds the *prioritised* route at 1900; the fallback checked only kind.
"""
import pytest

from app.services.source_resolver import SourceResolver
from app.services.source_type import SourceKindAssessment

BOOK = SourceKindAssessment("monograph", "high", ())
ARTICLE = SourceKindAssessment("journal_article", "high", ())
UNKNOWN = SourceKindAssessment("unknown", "low", ())


@pytest.mark.parametrize("year", ["1978", "1985", "2013", "2023"])
def test_a_modern_book_does_not_reach_the_public_domain_catalogs(year) -> None:
    assert SourceResolver._public_domain_fallback_allowed(BOOK, year) is False


@pytest.mark.parametrize("year", ["1601", "1890", "1930"])
def test_an_old_book_still_reaches_them(year) -> None:
    assert SourceResolver._public_domain_fallback_allowed(BOOK, year) is True


def test_an_unknown_year_still_reaches_them() -> None:
    """No date is not evidence of recency, so the route stays open."""
    assert SourceResolver._public_domain_fallback_allowed(BOOK, None) is True
    assert SourceResolver._public_domain_fallback_allowed(BOOK, "") is True
    assert SourceResolver._public_domain_fallback_allowed(UNKNOWN, "n.d.") is True


def test_the_kind_gate_still_applies_independently() -> None:
    """A journal article never reaches them, whatever its year."""
    assert SourceResolver._public_domain_fallback_allowed(ARTICLE, "1890") is False
    assert SourceResolver._public_domain_fallback_allowed(ARTICLE, None) is False


def test_the_bound_sits_at_the_public_domain_line_not_an_arbitrary_date() -> None:
    assert SourceResolver._PUBLIC_DOMAIN_FALLBACK_LATEST_YEAR == 1930
    assert SourceResolver._public_domain_fallback_allowed(BOOK, "1930") is True
    assert SourceResolver._public_domain_fallback_allowed(BOOK, "1931") is False
