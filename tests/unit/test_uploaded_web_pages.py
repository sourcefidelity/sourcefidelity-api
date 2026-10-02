"""Uploaded PDFs of web pages (owner decision 2026-10-01): near title match with the author,
and the web-page completeness rule instead of book/article rules."""
import fitz

from app.services.pdf_verifier import verify_instructor_upload
from app.services.web_completeness import uploaded_page_completeness


def _pdf(heading: str, byline: str, words: int) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 60), heading)
    page.insert_text((50, 80), byline)
    body = ("Review text " * (words // 2)).split()
    for row in range(0, len(body), 12):
        page.insert_text((50, 100 + (row // 12) * 10 % 680), " ".join(body[row:row + 12]))
    return document.tobytes()


def test_a_web_page_title_near_match_needs_the_author():
    pdf = _pdf("Star Trek: The Motion Picture", "Roger Ebert", 60)
    title = "Star Trek: The Motion Picture movie review [Online]"
    assert not verify_instructor_upload(pdf, provided_title=title, provided_author="Ebert, R")[0]
    assert verify_instructor_upload(pdf, provided_title=title, provided_author="Ebert, R", web_page=True)[0]
    assert not verify_instructor_upload(pdf, provided_title=title, provided_author="Smith, J", web_page=True)[0]


def test_uploaded_web_page_completeness():
    assert uploaded_page_completeness("word " * 700, "webpage")["verdict"] == "complete"
    assert uploaded_page_completeness("word " * 700 + " continue reading", "webpage")["reason"] == "cut_off_signal"
    assert uploaded_page_completeness("word " * 50, "webpage")["reason"] == "length_does_not_fit_kind"
    assert uploaded_page_completeness("word " * 700, "monograph")["reason"] == "not_a_web_page"
