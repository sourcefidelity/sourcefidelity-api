"""Regression tests for source-type-aware PDF completeness evidence."""

import fitz

from app.services.completeness_checker import (
    COMPLETE,
    INCOMPLETE,
    UNCERTAIN,
    check_completeness,
)
from app.services.source_validator import _check_completeness


def test_uncropped_raster_avoids_image_hash_and_xref_work(monkeypatch):
    from app.services.completeness_checker import _signal_rendered_raster_coverage
    with fitz.open(stream=_raster_pdf_with_horizontal_crop(ink_in_hidden_bands=True),filetype='pdf') as doc:
        for page in doc:
            page.set_cropbox(page.mediabox)
        data=doc.tobytes()
    def forbidden(*a,**k):
        raise AssertionError('Uncropped page must not request image xref hashes')
    monkeypatch.setattr(fitz.Page,'get_image_rects',forbidden)
    result=_signal_rendered_raster_coverage(data)
    assert result['vote'] is None
    assert result['detail']=='Rendered raster coverage: no material ink-bearing crop found'


def test_cropped_raster_still_uses_original_pixel_check(monkeypatch):
    from app.services.completeness_checker import _signal_rendered_raster_coverage
    data=_raster_pdf_with_horizontal_crop(ink_in_hidden_bands=True)
    original=fitz.Page.get_image_rects
    calls=[]
    def observed(*a,**k):
        calls.append(True)
        return original(*a,**k)
    monkeypatch.setattr(fitz.Page,'get_image_rects',observed)
    result=_signal_rendered_raster_coverage(data)
    assert calls and result['vote']==INCOMPLETE


def _pdf(pages: list[str], *, page_numbers: list[int] | None = None) -> bytes:
    document = fitz.open()
    for index, text in enumerate(pages):
        page = document.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(72, 72, 540, 700), text, fontsize=11)
        if page_numbers is not None:
            page.insert_text((500, 30), str(page_numbers[index]), fontsize=10)
    payload = document.tobytes()
    document.close()
    return payload


def _raster_pdf_with_horizontal_crop(*, ink_in_hidden_bands: bool) -> bytes:
    source = fitz.open()
    for _index in range(4):
        page = source.new_page(width=612, height=792)
        page.draw_rect(page.rect, color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_textbox(
            fitz.Rect(100, 80, 512, 712),
            "Readable retained text. " * 80,
            fontsize=10,
        )
        if ink_in_hidden_bands:
            for y_pos in range(80, 712, 18):
                page.draw_rect(
                    fitz.Rect(8, y_pos, 64, y_pos + 8),
                    color=(0, 0, 0),
                    fill=(0, 0, 0),
                )
                page.draw_rect(
                    fitz.Rect(548, y_pos, 604, y_pos + 8),
                    color=(0, 0, 0),
                    fill=(0, 0, 0),
                )

    raster = fitz.open()
    for source_page in source:
        pixmap = source_page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        page = raster.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pixmap)
        page.set_cropbox(fitz.Rect(72, 0, 540, 792))
    payload = raster.tobytes()
    raster.close()
    source.close()
    return payload


def test_article_bibliographic_endnotes_are_terminal_evidence():
    from app.services.completeness_checker import _signal_back_matter
    notes = "Notes\n1. A. Author (1998). First work.\n2. B. Author (2001). Second work.\n3. C. Author (2010). Third work."
    payload = _pdf(["Article opening", "Article body", notes])
    assert _signal_back_matter(payload, 3, document_kind="article")["vote"] == COMPLETE
    assert _signal_back_matter(payload, 3, document_kind="book")["vote"] is None
    partial = _pdf(["Selected pages", "Article body", notes])
    assert check_completeness(partial, document_kind="article").verdict != COMPLETE


def test_journal_references_and_notes_heading_is_terminal_evidence():
    # A science-journal article's "REFERENCES AND NOTES" (judged limited text, 2026-10-07).
    from app.services.completeness_checker import _signal_back_matter
    tail = "REFERENCES AND NOTES\n1. A. Author, First work (Press, 1999)."
    payload = _pdf(["Article opening", "Article body", "Results", tail])
    assert _signal_back_matter(payload, 4, document_kind="article")["vote"] == COMPLETE


def test_bare_or_nonbibliographic_notes_do_not_establish_completeness():
    from app.services.completeness_checker import _signal_back_matter
    for notes in ["Notes", "Notes\n1. First observation\n2. Second observation\n3. Third observation"]:
        assert _signal_back_matter(_pdf([notes]), 1, document_kind="article")["vote"] is None


def test_article_advertised_range_detects_two_page_excerpt():
    payload = _pdf(
        [
            "Proceedings of Example Research, pages 12076-12100\n\n"
            "Article title\n\nAbstract and opening text.",
            "The article continues, but this representation stops here.",
        ]
    )

    report = check_completeness(
        payload,
        document_kind="article",
        external_lookup=False,
    )

    assert report.verdict == INCOMPLETE
    assert any("advertised pp. 12076-12100" in signal for signal in report.signals)


def test_metadata_page_range_detects_excerpt_without_first_page_notice():
    payload = _pdf(["Opening text.", "The representation stops here."])

    report = check_completeness(
        payload,
        document_kind="article",
        expected_page_range=(100, 124),
        external_lookup=False,
    )

    assert report.verdict == INCOMPLETE
    assert any("provided metadata" in signal for signal in report.signals)


def test_complete_article_range_and_reference_heading_are_affirmative():
    payload = _pdf(
        [
            "Example Journal, pages 100-102\n\nArticle title and abstract.",
            "The analysis continues on the second page.",
            "References\n\nSmith, A. (2020). Complete reference entry.",
        ]
    )

    report = check_completeness(
        payload,
        document_kind="article",
        external_lookup=False,
    )

    assert report.verdict == COMPLETE


def test_statistical_sample_and_cited_page_range_are_not_partial_markers():
    payload = _pdf(
        [
            "This article studies a representative sample. Prior work appears "
            "in Example (2019, pp. 1-322).",
            "References\n\nExample, A. (2019). Complete reference entry.",
        ]
    )

    report = check_completeness(
        payload,
        document_kind="article",
        external_lookup=False,
    )

    assert report.verdict == COMPLETE
    assert not any("partial-document marker found" in signal for signal in report.signals)
    assert any("Advertised page range: unavailable" in signal for signal in report.signals)


def test_late_printed_pagination_applies_to_book_not_chapter():
    payload = _pdf(
        ["Book extract page one.", "Book extract page two.", "Book extract page three."],
        page_numbers=[14, 15, 16],
    )

    book = check_completeness(
        payload,
        document_kind="book",
        external_lookup=False,
    )
    chapter = check_completeness(
        payload,
        document_kind="chapter",
        external_lookup=False,
    )

    assert book.verdict == INCOMPLETE
    assert chapter.verdict == UNCERTAIN


def test_short_book_without_affirmative_evidence_abstains():
    payload = _pdf(["A complete pamphlet can legitimately be one page long."])

    report = check_completeness(
        payload,
        document_kind="book",
        external_lookup=False,
    )

    assert report.verdict == UNCERTAIN


def test_book_uses_metadata_for_automatically_extracted_isbn(monkeypatch):
    payload = _pdf(
        [
            "Example Book\nCopyright page\nISBN 978-0-306-40615-7",
            "Chapter one begins.",
            "The sample stops.",
        ]
    )
    calls = []

    def fake_lookup(*, isbn, title, author):
        calls.append((isbn, title, author))
        if isbn:
            return {"pages": 240, "source": "Google Books"}
        return None

    monkeypatch.setattr(
        "app.services.completeness_checker._lookup_google_books",
        fake_lookup,
    )
    monkeypatch.setattr(
        "app.services.completeness_checker._lookup_open_library",
        lambda **_kwargs: None,
    )

    report = check_completeness(
        payload,
        title="Example Book",
        author="Rivera, Alex",
        document_kind="book",
    )

    assert report.verdict == INCOMPLETE
    assert any(call[0] == "9780306406157" for call in calls)
    assert any("Automatic ISBN extraction" in signal for signal in report.signals)


def test_extracted_isbn_is_not_sent_without_cited_title(monkeypatch):
    payload = _pdf(["Copyright page\nISBN 978-0-306-40615-7"])
    calls = []

    def fake_lookup(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "app.services.completeness_checker._lookup_google_books",
        fake_lookup,
    )
    monkeypatch.setattr(
        "app.services.completeness_checker._lookup_open_library",
        lambda **_kwargs: None,
    )

    report = check_completeness(payload, document_kind="book")

    assert report.verdict == UNCERTAIN
    assert all(call["isbn"] is None for call in calls)


def test_in_bounds_bookmarks_are_not_proof_of_completeness():
    payload = _pdf(["First chapter.", "Second chapter.", "Third chapter."])
    source = fitz.open(stream=payload, filetype="pdf")
    source.set_toc([[1, "Chapter 1", 1], [1, "Chapter 2", 2]])
    with_toc = source.tobytes()
    source.close()

    report = check_completeness(
        with_toc,
        document_kind="unknown",
        external_lookup=False,
    )

    assert report.verdict == UNCERTAIN
    assert any("not proof" in signal for signal in report.signals)


def test_repeated_crop_of_ink_bearing_raster_is_incomplete():
    report = check_completeness(
        _raster_pdf_with_horizontal_crop(ink_in_hidden_bands=True),
        document_kind="unknown",
        external_lookup=False,
    )

    assert report.verdict == INCOMPLETE
    assert any("hides ink-bearing content" in signal for signal in report.signals)


def test_crop_of_blank_scanner_margins_is_not_incomplete():
    report = check_completeness(
        _raster_pdf_with_horizontal_crop(ink_in_hidden_bands=False),
        document_kind="unknown",
        external_lookup=False,
    )

    assert report.verdict == UNCERTAIN


def test_parse_failure_returns_uncertain_instead_of_raising():
    report = check_completeness(
        b"<html>not a pdf</html>",
        document_kind="unknown",
        external_lookup=False,
    )

    assert report.verdict == UNCERTAIN
    assert report.n_up_layout is None


def test_source_validator_wrapper_preserves_logical_page_count():
    payload = _pdf(["Page one.", "Page two.", "References\nSmith (2020). Entry."])

    verdict, page_count = _check_completeness(
        payload,
        document_kind="article",
    )

    assert verdict == "complete"
    assert page_count == 3
