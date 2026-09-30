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


@pytest.mark.parametrize('container',[
    '*Policy Studies, 12*(3), 45-67.',
    '*Policy Studies*, 12, 45-67.',
    '*Policy Studies*, 12(3), 45-67.',
])
def test_literal_emphasis_does_not_hide_journal_structure(container):
    raw=f'Writer, A. (2020). An article title. {container} https://example.org/article'
    result=classify_reference_source_kind(raw)
    assert result.kind=='journal_article'
    assert result.confidence=='high'


@pytest.mark.parametrize('tail',[
    '*An emphasized phrase*', '*A journal, 12*',
    '*A broken marker, 12(3), 45-67.',
])
def test_marked_journal_cue_requires_complete_structure(tail):
    from app.services.source_type import _MARKED_JOURNAL_STRUCTURE_RE
    assert _MARKED_JOURNAL_STRUCTURE_RE.search(tail) is None


def test_marked_journal_cue_does_not_override_explicit_media_role():
    raw='Writer, A. (Director). (2020). An unusual title [Film]. *Media Studies, 12*(3), 45-67.'
    assert classify_reference_source_kind(raw).kind=='traditional_media'


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
def test_media_role_word_in_article_title_is_not_a_credit():
    from app.services.source_type import is_traditional_media
    assert not is_traditional_media("Writer, A. (2020). Creator's perspective. Journal, 6(9), 12–19.")
    assert is_traditional_media('Writer, A. (Director). (2020). A film.')


def test_marked_journal_title_can_contain_commas():
    from app.services.ref_field_extractor import _apa_title
    assert _apa_title('A complete article title. *Media, Language & Society*, 42(4), 637-654.') == 'A complete article title'
    assert _apa_title('A title with *emphasis, and punctuation*.') == 'A title with *emphasis, and punctuation*'


def test_unclassified_reference_is_searchable_but_media_is_not():
    """The review gate turns on this predicate.  An unclassified reference must
    reach a search, because author/title/year settle identity without a kind;
    a film or a tweet must not, because an absent index match is silence."""
    from app.services.source_type import (
        BOOK_CATALOGUE_KINDS,
        is_bibliographically_searchable,
    )
    for kind in ("unknown", None, "", "journal_article", "monograph",
                 "edited_collection", "book_section", "thesis", "report",
                 "conference_paper", "book_review"):
        assert is_bibliographically_searchable(kind), kind
    for kind in ("film", "television_series", "webpage", "news_article",
                 "blog_post", "social_media_post", "video", "podcast_episode",
                 "traditional_media", "archival_source", "personal_communication",
                 "interview", "artwork", "album"):
        assert not is_bibliographically_searchable(kind), kind
    # An unclassified reference must not inherit a book's evidence requirement
    # on a guess; only named book kinds carry the catalogue obligation.
    assert "unknown" not in BOOK_CATALOGUE_KINDS
    assert BOOK_CATALOGUE_KINDS == {"monograph", "edited_collection", "book_section"}


class TestInstitutionalAndIdentifierBearingReferences:
    """Two classifier gaps found on the owner's own paper (job `17e3b5a8`).

    A university language policy was left `unknown`, which
    `bounded-reference-review-v9` reviews, and it drew a
    `potentially_fabricated_reference` finding. A body publishing a document
    about itself is not deposited in a bibliographic index, so an absent match
    is silence. Whether the document is still reachable online does not change
    that: going offline does not make it newly searchable.

    The converse gap ran the other way. A reference whose only web marker was a
    `doi.org` link classified as `webpage`, which is excluded from review
    entirely - so a fabricated reference carrying an invented DOI was never
    reviewed at all. A DOI says the work is registered and resolvable; it is a
    more specific marker than a bare URL, not a weaker one.
    """

    def test_a_self_published_institutional_document_is_not_searchable(self):
        from app.services.source_type import (
            classify_reference_source_kind,
            is_bibliographically_searchable,
        )

        for raw in (
            "Northfield State University (2015). NSU language policy.",
            "Ministry of Education (2019). National curriculum framework.",
            "UNESCO Institute for Statistics (2020). Global education monitoring.",
        ):
            assessment = classify_reference_source_kind(raw)
            assert assessment.kind == "webpage", raw
            assert not is_bibliographically_searchable(assessment.kind), raw

    def test_an_organizational_author_does_not_override_a_real_work_type(self):
        """The rule requires the absence of every stronger marker."""
        from app.services.source_type import classify_reference_source_kind

        cases = {
            "American Psychiatric Association (2013). Diagnostic and statistical "
            "manual of mental disorders (5th ed.). American Psychiatric Publishing.": "monograph",
            "OECD (2020). Education at a glance 2020: OECD indicators.": "report",
            "University of Oxford committee member. (2001). A study. "
            "Journal of Things, 4(2), 1-20.": "journal_article",
        }
        for raw, expected in cases.items():
            assert classify_reference_source_kind(raw).kind == expected, raw

    def test_a_personal_author_is_never_institutional(self):
        from app.services.source_type import _institutional_self_published

        assert not _institutional_self_published(
            "Berg, S. V., & Forsyth, P. (2007). The economic analysis of regulation."
        )
        assert not _institutional_self_published(
            "Khan, L. (2017). Amazon's antitrust paradox."
        )

    def test_a_doi_link_is_not_a_bare_webpage_url(self):
        from app.services.source_type import (
            classify_reference_source_kind,
            is_bibliographically_searchable,
        )

        raw = "Smith, J. (2020). Invented title. https://doi.org/10.9999/fake.2020"
        assessment = classify_reference_source_kind(raw)
        assert assessment.kind != "webpage"
        # The point of the change: it must reach the review rather than be
        # silently excluded.
        assert is_bibliographically_searchable(assessment.kind)

    def test_an_ordinary_web_resource_is_still_a_webpage(self):
        from app.services.source_type import classify_reference_source_kind

        assert (
            classify_reference_source_kind(
                "Someone, A. (2022). A page about things. https://example.org/page"
            ).kind
            == "webpage"
        )


def test_a_book_reference_keeps_its_type_despite_a_trailing_file_share_url() -> None:
    """A link to a copy does not turn a book into a web page.

    A canonical work record created 2026-09-05 carries `work_type='webpage'`
    for "Belton, J. (2013). American cinema/American culture. New York:
    McGraw-Hill. Retrieved from https://pan.baidu.com/..." — a monograph filed
    as a web page because the reference ends in a URL. The classifier was
    repaired afterwards and the terminal publisher clause now decides, but
    nothing pinned that, so this fixes the behaviour in place. The stored
    record is deliberately left alone: `_canonical_work` restricts its
    title/author/year fallback to book kinds precisely so old webpage
    classifications are not mutated.
    """
    assessment = classify_reference_source_kind(
        "Belton, J. (2013). American cinema/American culture. New York: "
        "McGraw-Hill. Retrieved from https://pan.baidu.com/s/1h7JvFLwr?pwd=1111",
        title="American cinema/American culture",
        url="https://pan.baidu.com/s/1h7JvFLwr?pwd=1111",
    )

    assert assessment.kind == "monograph"
    assert assessment.confidence == "high"


def test_a_publisher_clause_outranks_a_trailing_url_for_any_imprint() -> None:
    assessment = classify_reference_source_kind(
        "Bordwell, D. (1985). The Classical Hollywood Cinema. London: Routledge. "
        "Retrieved from https://example.org/copy.pdf",
        title="The Classical Hollywood Cinema",
        url="https://example.org/copy.pdf",
    )

    assert assessment.kind == "monograph"


def test_a_url_still_decides_when_no_publisher_clause_supports_a_book() -> None:
    """Without the structure, a trailing URL is the only evidence there is."""
    assessment = classify_reference_source_kind(
        "CRTC. (2016). Communications Monitoring Report. "
        "Retrieved from https://www.crtc.gc.ca",
        title="Communications Monitoring Report",
        url="https://www.crtc.gc.ca",
    )

    assert assessment.kind == "webpage"
    assert assessment.confidence == "low"
