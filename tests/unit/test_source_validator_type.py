import fitz
import pytest

from app.services.source_validator import (
    _detect_nonwork_listing,
    _normalize_identity_text,
    validate_ocr_derivative_text,
    validate_retrieved_pdf,
)


def _pdf(text: str, metadata: dict[str, str] | None = None) -> bytes:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(60, 60, 552, 732), text, fontsize=11)
    if metadata:
        document.set_metadata(metadata)
    payload = document.tobytes()
    document.close()
    return payload


_ACCESSIBILITY_DOCUMENT = (
    "Morgan, Understanding Cinema, 2013\n"
    "Our Commitment to Accessibility\n"
    "Our digital content is tested against WCAG AA. VPATs are available.\n"
    "This report reflects the accessibility level of the content available "
    "within our learning platform as it relates to this title.\n"
    "Content Feature | Definitions | Status and Alternatives\n"
    "Electronic book: partially supports keyboard navigation.\n"
    "Related book identifier: 10.1234/example-book"
)


@pytest.mark.parametrize("source_kind", ["unknown", "webpage", "monograph"])
def test_accessibility_document_is_not_book_even_with_matching_doi(source_kind) -> None:
    result = validate_retrieved_pdf(
        _pdf(_ACCESSIBILITY_DOCUMENT),
        expected_title="Understanding Cinema",
        expected_author="Morgan",
        expected_year="2013",
        expected_doi="10.1234/example-book",
        expected_source_kind=source_kind,
        skip_completeness=True,
    )
    assert not result.accept
    assert result.identity_confidence == "rejected"
    assert "accessibility or conformance document" in result.reason


def test_ocr_accessibility_document_cannot_be_admitted_as_book() -> None:
    result = validate_ocr_derivative_text(
        _ACCESSIBILITY_DOCUMENT,
        expected_title="Understanding Cinema",
        expected_author="Morgan",
        expected_doi="10.1234/example-book",
        completeness="complete",
        page_count=3,
    )
    assert not result.accept
    assert result.identity_confidence == "rejected"
    assert "accessibility or conformance document" in result.reason


@pytest.mark.parametrize("removed", [
    "Our Commitment to Accessibility",
    "Our digital content is tested against WCAG AA. VPATs are available.",
    "This report reflects the accessibility level of the content available "
    "within our learning platform as it relates to this title.",
])
def test_accessibility_purpose_guard_requires_all_independent_cues(removed) -> None:
    assert _detect_nonwork_listing(
        _ACCESSIBILITY_DOCUMENT.replace(removed, ""), "Understanding Cinema"
    ) is None


def test_explicitly_cited_accessibility_report_remains_eligible() -> None:
    result = validate_retrieved_pdf(
        _pdf(_ACCESSIBILITY_DOCUMENT),
        expected_title="Our Commitment to Accessibility",
        expected_author="Morgan",
        expected_year="2013",
        skip_completeness=True,
    )
    assert result.accept
    assert result.identity_confidence == "high"


def test_accessibility_scholarship_is_not_rejected_for_shared_vocabulary() -> None:
    text = (
        "Understanding Cinema Accessibility\nMorgan, 2013\n"
        "This study evaluates a publisher accessibility statement against WCAG.\n"
        "The VPAT describes keyboard navigation and assistive technologies.\n"
        "We compare these claims with observations of readers using the book."
    )
    result = validate_retrieved_pdf(
        _pdf(text), expected_title="Understanding Cinema Accessibility",
        expected_author="Morgan", expected_year="2013", skip_completeness=True,
    )
    assert result.accept


def test_book_review_pdf_is_rejected_for_monograph_even_with_matching_fields() -> None:
    payload = _pdf(
        "BOOK REVIEW\n\n"
        "The Master Switch: The Rise and Fall of Information Empires\n"
        "by Tim Wu (2010)\n\n"
        "Reviewed by Lina Khan\n\n"
        "This review discusses the book and repeatedly names its author and title."
    )

    result = validate_retrieved_pdf(
        payload,
        expected_title="The Master Switch: The Rise and Fall of Information Empires",
        expected_author="Wu, Tim",
        expected_year="2010",
        expected_source_kind="monograph",
        expected_source_kind_confidence="high",
        expected_source_kind_evidence=("book-publisher citation structure",),
        skip_completeness=True,
    )

    assert result.accept is False
    assert result.identity_confidence == "rejected"
    assert result.observed_source_kind == "book_review"
    assert result.source_kind_verdict == "incompatible"
    assert "type conflict" in result.reason.casefold()


def test_cv_publication_list_cannot_satisfy_article_identity() -> None:
    payload = _pdf(
        "(Abridged) C.V. of Jeremy Shtern, PhD\n\n"
        "EDUCATION\nUniversity of Example, PhD, 2010\n\n"
        "ACADEMIC EMPLOYMENT\nFull Professor, 2022-present\n\n"
        "PUBLICATIONS\nAkanbi, O., Hill, S., & Shtern, J. (2023). "
        "Platform Governance: The Antitrust Option. Canadian Journal of "
        "Communication 48(2), 361-380."
    )

    result = validate_retrieved_pdf(
        payload,
        expected_title="Platform governance: The antitrust option",
        expected_author="Akanbi, O., Hill, S., & Shtern, J.",
        expected_year="2023",
        expected_source_kind="journal_article",
        expected_source_kind_confidence="high",
        skip_completeness=True,
    )

    assert result.accept is False
    assert result.identity_confidence == "rejected"
    assert "curriculum vitae or publication list" in result.reason


def test_award_listing_cannot_satisfy_named_article_identity() -> None:
    payload = _pdf(
        "2009 AWARD WINNERS\n\n"
        "RAY AND PAT BROWNE AWARD FOR SINGLE AUTHORED WORK\n"
        "An Unrelated Winning Book\n\n"
        "RUSSEL B. NYE AWARD FOR OUTSTANDING ARTICLE\n"
        "Ashley Elaine York, From Chick Flicks to Millennial Blockbusters: "
        "Spinning Female-Driven Narratives into Franchises\n\n"
        "WILLIAM E. BRIGMAN AWARD FOR OUTSTANDING GRADUATE STUDENT PAPER"
    )

    result = validate_retrieved_pdf(
        payload,
        expected_title=(
            "From chick flicks to millennial blockbusters: Spinning "
            "female-driven narratives into franchises"
        ),
        expected_author="York, A. E.",
        expected_year="2010",
        expected_source_kind="journal_article",
        expected_source_kind_confidence="high",
        skip_completeness=True,
    )

    assert result.accept is False
    assert result.identity_confidence == "rejected"
    assert "awards or contents listing" in result.reason


def test_same_author_later_pdf_citing_target_cannot_satisfy_identity() -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text(
        (60, 80),
        "A Later Study of Digital Television Services",
        fontsize=16,
    )
    page.insert_text((60, 112), "Jonathan Hardy", fontsize=11)
    page.insert_text((60, 136), "2017", fontsize=11)
    page.insert_textbox(
        fitz.Rect(60, 260, 552, 720),
        "This later paper discusses the earlier work.\n\n"
        "References\n"
        "Hardy, Jonathan. Cross-media promotion. 2010.",
        fontsize=10,
    )
    payload = document.tobytes()
    document.close()

    result = validate_retrieved_pdf(
        payload,
        expected_title="Cross-media promotion",
        expected_author="Hardy, Jonathan",
        expected_year="2010",
        expected_source_kind="unknown",
        expected_source_kind_confidence="unknown",
        skip_completeness=True,
    )

    assert result.accept is False
    assert result.identity_confidence != "high"


def test_prominent_working_paper_year_conflict_blocks_published_identity() -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text(
        (60, 80),
        "Vertical Integration During the Hollywood Studio Era",
        fontsize=16,
    )
    page.insert_text((60, 112), "F. Andrew Hanssen", fontsize=11)
    page.insert_text((60, 136), "Working paper - April 4, 2008", fontsize=11)
    page.insert_textbox(
        fitz.Rect(60, 260, 552, 720),
        "A later reference list may still contain the published 2010 citation.",
        fontsize=10,
    )
    payload = document.tobytes()
    document.close()

    result = validate_retrieved_pdf(
        payload,
        expected_title="Vertical Integration During the Hollywood Studio Era",
        expected_author="Hanssen, F. Andrew",
        expected_year="2010",
        expected_source_kind="journal_article",
        expected_source_kind_confidence="high",
        skip_completeness=True,
    )

    assert result.accept is False
    assert result.identity_confidence == "medium"


def test_identity_matching_folds_diacritics_and_quote_variants() -> None:
    payload = _pdf(
        "PERCEPTIONS, Spring 2018, Volume XXIII, Number 1, pp. 95-120.\n"
        "Ibrahim AKBAS\n"
        "A “Cool’’ Approach to Japanese Foreign Policy: Linking Anime to "
        "International Relations\n"
    )

    result = validate_retrieved_pdf(
        payload,
        expected_title=(
            "A “Cool” Approach to Japanese Foreign Policy: Linking Anime to "
            "International Relations"
        ),
        expected_author="Akbas, I",
        expected_year="2018",
        expected_source_kind="journal_article",
        expected_source_kind_confidence="high",
        skip_completeness=True,
    )

    assert result.accept is True
    assert result.identity_confidence == "high"
    assert result.observed_source_kind == "journal_article"
    assert _normalize_identity_text("İbrahim AKBAŞ") == "ibrahim akbas"


def test_pdf_metadata_supplies_canonical_report_title_and_doi() -> None:
    payload = _pdf(
        "OECD Economic Outlook\n109\nMAY 2021",
        metadata={
            "title": "OECD Economic Outlook, Volume 2021 Issue 1 (EN)",
            "author": "OECD",
            "subject": "https://doi.org/10.1787/edfbca02-en",
        },
    )

    result = validate_retrieved_pdf(
        payload,
        expected_doi="10.1787/edfbca02-en",
        expected_title="OECD economic outlook, volume 2021 issue 1",
        expected_author="OECD",
        expected_year="2021",
        expected_source_kind="report",
        expected_source_kind_confidence="high",
        skip_completeness=True,
        skip_text_quality=True,
    )

    assert result.accept is True
    assert result.identity_confidence == "high"


def test_unrelated_pdf_metadata_does_not_create_identity_match() -> None:
    payload = _pdf(
        "A different report issued in 2021",
        metadata={
            "title": "Different report",
            "author": "Different agency",
            "subject": "https://doi.org/10.9999/different",
        },
    )

    result = validate_retrieved_pdf(
        payload,
        expected_doi="10.1787/edfbca02-en",
        expected_title="OECD economic outlook, volume 2021 issue 1",
        expected_author="OECD",
        expected_year="2021",
        expected_source_kind="report",
        expected_source_kind_confidence="high",
        skip_completeness=True,
        skip_text_quality=True,
    )

    assert result.accept is False
    assert result.identity_confidence != "high"


def test_matching_metadata_doi_requires_visible_front_matter_support() -> None:
    payload = _pdf(
        "Unrelated technical appendix with no matching work identity.",
        metadata={
            "title": "Unrelated technical appendix",
            "author": "Different agency",
            "subject": "https://doi.org/10.1787/edfbca02-en",
        },
    )

    result = validate_retrieved_pdf(
        payload,
        expected_doi="10.1787/edfbca02-en",
        expected_title="OECD economic outlook, volume 2021 issue 1",
        expected_author="OECD",
        expected_year="2021",
        expected_source_kind="report",
        expected_source_kind_confidence="high",
        skip_completeness=True,
        skip_text_quality=True,
    )

    assert result.accept is False
    assert result.identity_confidence != "high"


def test_shared_title_prefix_cannot_hide_different_report_year() -> None:
    payload = _pdf(
        "OECD Economic Outlook\nJUNE 2026\nPrior editions include 2021.",
        metadata={
            "title": "OECD Economic Outlook, Volume 2026 Issue 1 (EN)",
            "author": "OECD",
            "subject": "https://doi.org/10.1787/2d1956f0-en",
        },
    )

    result = validate_retrieved_pdf(
        payload,
        expected_doi="10.1787/edfbca02-en",
        expected_title="OECD economic outlook, volume 2021 issue 1",
        expected_author="OECD",
        expected_year="2021",
        expected_source_kind="report",
        expected_source_kind_confidence="high",
        skip_completeness=True,
        skip_text_quality=True,
    )

    assert result.accept is False
    assert result.identity_confidence == "medium"
