import pytest

from app.services.source_type import (
    SourceKindAssessment,
    classify_content_source_kind,
    classify_html_source_kind,
    classify_provider_source_kind,
    classify_reference_source_kind,
    compare_source_kinds,
    is_archive_source,
    is_library_locator_url,
    is_traditional_media,
)


@pytest.mark.parametrize(
    "reference",
    [
        "Miyazaki, H. (Director). (2001). Spirited Away [Film].",
        "Bowie, D. (1977). Low [Album].",
        "Smith, A. (Host). (2025). Episode title [Podcast episode].",
    ],
)
def test_traditional_media_is_routed_away_from_academic_search(reference: str) -> None:
    assert is_traditional_media(reference) is True


def test_scholarly_work_is_not_misclassified_as_media() -> None:
    reference = "Smith, J. (2024). Film audiences and platform governance. Media Studies, 8(2)."

    assert is_traditional_media(reference) is False


@pytest.mark.parametrize(
    "reference",
    [
        "University Archive, Special Collections, Box 4, Folder 2.",
        "Author papers, unpublished manuscript, 1982.",
    ],
)
def test_physical_archives_are_detected(reference: str) -> None:
    assert is_archive_source(reference) is True


def test_archived_web_url_is_not_misrouted_as_physical_archive() -> None:
    assert is_archive_source(
        "Organization. Page title. https://web.archive.org/web/20200101/example.org"
    ) is False


def test_library_locator_detection() -> None:
    assert is_library_locator_url(
        "https://search-ebscohost-com.proxy.example.edu/login.aspx?db=example"
    ) is True
    assert is_library_locator_url("https://www.jstor.org/stable/j.ctv941t3g.8") is True
    assert is_library_locator_url("https://books.google.com/books?id=example") is True
    assert is_library_locator_url("https://example.org/news/article") is False
    assert is_library_locator_url(
        "https://scholarship.law.columbia.edu/books/63"
    ) is False


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (
            "Wu, T. (2010). The master switch: The rise and fall of information empires. Knopf.",
            "monograph",
        ),
        (
            "Khan, L. (2012). Book review: The master switch. Journal of Law, 4(2), 10-14.",
            "book_review",
        ),
        (
            "Lee, J. (2024). Platform governance. Media Studies, 8(2), 44-61.",
            "journal_article",
        ),
        (
            "Agency. (2024). Annual findings (Report No. 42). https://agency.example/report",
            "report",
        ),
        (
            "Reporter, R. (2024). Headline [News article]. Example News. https://news.example/item",
            "news_article",
        ),
        (
            "Creator. (2024). Explainer [Video]. YouTube. https://youtube.com/watch?v=1",
            "video",
        ),
        (
            "Wu, T. (2018). The curse of bigness: Antitrust in the new gilded age "
            "(Vol. 15). Columbia Global Reports. https://example.org/books/63",
            "monograph",
        ),
        (
            "OECD. (2021). OECD economic outlook, volume 2021 issue 1. "
            "OECD Publishing. https://doi.org/10.1787/edfbca02-en",
            "report",
        ),
        (
            "Vickers, M. R. (2005). Business ethics and the HR role: Past, present, "
            "and future. Human Resource Planning, 28 (1). "
            "https://www.proquest.com/trade-journals/example",
            "journal_article",
        ),
    ],
)
def test_reference_material_supplies_bounded_work_type(reference: str, expected: str) -> None:
    assessment = classify_reference_source_kind(reference)

    assert assessment.kind == expected
    assert assessment.confidence == "high"
    assert assessment.evidence


def test_bare_url_remains_low_confidence_generic_webpage() -> None:
    assessment = classify_reference_source_kind(
        "Organization. Page title. https://example.org/page"
    )

    assert assessment.kind == "webpage"
    assert assessment.confidence == "low"


def test_provider_type_conflict_rejects_book_review_as_book() -> None:
    observed = classify_provider_source_kind({"message": {"type": "journal-article"}})
    compatibility = compare_source_kinds(
        SourceKindAssessment("monograph", "high", ("publisher structure",)),
        observed,
    )

    assert observed.kind == "journal_article"
    assert compatibility.verdict == "incompatible"


def test_broad_provider_article_type_can_represent_explicit_book_review() -> None:
    compatibility = compare_source_kinds(
        SourceKindAssessment("book_review", "high", ("explicit book review",)),
        classify_provider_source_kind({"message": {"type": "journal-article"}}),
    )

    assert compatibility.verdict == "compatible"


def test_acquired_content_explicitly_identifies_book_review() -> None:
    assessment = classify_content_source_kind(
        "BOOK REVIEW\nThe Master Switch, by Tim Wu\nReviewed by Lina Khan"
    )

    assert assessment.kind == "book_review"
    assert assessment.confidence == "high"


@pytest.mark.parametrize(
    "front_matter",
    [
        (
            "International Journal of Cultural Policy ISSN: 1028-6632. "
            "Journal homepage. To cite this article: Nobuko Kawashima (2016)."
        ),
        (
            "PERCEPTIONS, Spring 2018, Volume XXIII, Number 1, pp. 95-120. "
            "İbrahim AKBAŞ"
        ),
    ],
)
def test_journal_front_matter_outranks_incidental_report_language(
    front_matter: str,
) -> None:
    assessment = classify_content_source_kind(
        front_matter + " A cited report number appears later in the article."
    )

    assert assessment.kind == "journal_article"
    assert assessment.confidence == "high"


def test_html_structured_metadata_distinguishes_news_from_generic_page() -> None:
    assessment = classify_html_source_kind(
        '<html><head><script type="application/ld+json">'
        '{"@type":"NewsArticle"}</script></head><body>Story</body></html>',
        "https://news.example/story",
    )

    assert assessment.kind == "news_article"
    assert assessment.confidence == "high"
