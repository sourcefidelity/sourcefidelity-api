"""Regression tests for fail-closed instructor source identity checks."""

import fitz
import pytest

from app.services.pdf_verifier import verify_instructor_upload


def _pdf_bytes(*lines: tuple[str, float]) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 72.0
    for text, size in lines:
        page.insert_text((72.0, y), text, fontsize=size)
        y += size + 12.0
    data = document.tobytes()
    document.close()
    return data


def test_unrelated_pdf_is_not_accepted_from_supplied_metadata_alone() -> None:
    data = _pdf_bytes(
        ("A Completely Different Study of Marine Sediments", 18.0),
        ("Taylor Researcher", 12.0),
        ("Published 2024", 11.0),
    )

    verified, reasons = verify_instructor_upload(
        data,
        provided_title="Effects of Machine Translation on Academic Writing",
        provided_author="Jordan Scholar",
        provided_year="2021",
    )

    assert verified is False
    assert "insufficient_identity_evidence" in reasons


def test_title_and_author_corroboration_accepts_matching_pdf() -> None:
    title = "Effects of Machine Translation on Academic Writing"
    data = _pdf_bytes(
        (title, 18.0),
        ("Jordan Scholar and Morgan Writer", 12.0),
        ("Published 2021", 11.0),
    )

    verified, reasons = verify_instructor_upload(
        data,
        provided_title=title,
        provided_author="Jordan Scholar",
        provided_year="2021",
    )

    assert verified is True
    assert "title_match" in reasons
    assert "author_match" in reasons


def test_embedded_doi_conflict_rejects_otherwise_similar_pdf() -> None:
    title = "Effects of Machine Translation on Academic Writing"
    data = _pdf_bytes(
        (title, 18.0),
        ("Jordan Scholar", 12.0),
        ("https://doi.org/10.1234/actual-work", 11.0),
    )

    verified, reasons = verify_instructor_upload(
        data,
        provided_doi="10.1234/different-work",
        provided_title=title,
        provided_author="Jordan Scholar",
    )

    assert verified is False
    assert reasons == ["doi_conflict"]


@pytest.mark.parametrize("title_page,author,accepted", [(4, "Jordan Scholar", True), (6, "Jordan Scholar", False), (4, "Unrelated Writer", False)])
def test_book_front_matter_identity_remains_bounded(title_page, author, accepted):
    title = "Embodied cinema: Stars and society"
    document = fitz.open()
    for number in range(1, 7):
        page = document.new_page()
        if number == title_page:
            page.insert_text((72, 72), "Embodied cinema\nStars and society\nJordan Scholar", fontsize=16)
    content = document.tobytes()
    document.close()

    verified, reasons = verify_instructor_upload(content, provided_title=title, provided_author=author)
    assert verified is accepted
    if accepted:
        assert "title_match" in reasons and "author_match" in reasons
    else:
        assert "insufficient_identity_evidence" in reasons
