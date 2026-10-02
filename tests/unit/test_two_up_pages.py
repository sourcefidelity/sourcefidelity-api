"""Facing book pages scanned on one sheet are split (`two-up-split-v1`, owner request 2026-10-02)."""
import fitz

from app.routers.report import _stated_page_range
from app.services.two_up_pages import detect_two_up, split_two_up

WORDS = " ".join(f"word{n}" for n in range(60))


def pdf(two_up: bool, sheets: int = 3) -> bytes:
    document = fitz.open()
    for sheet in range(sheets):
        if two_up:
            page = document.new_page(width=780, height=600)
            page.insert_textbox(fitz.Rect(40, 40, 360, 560), f"Left page {2 * sheet + 22}. {WORDS}", fontsize=9)
            page.insert_textbox(fitz.Rect(420, 40, 740, 560), f"Right page {2 * sheet + 23}. {WORDS}", fontsize=9)
        else:
            page = document.new_page(width=540, height=700)
            page.insert_textbox(fitz.Rect(40, 40, 500, 660), f"Page {sheet + 1}. {WORDS} {WORDS}", fontsize=9)
    data = document.tobytes()
    document.close()
    return data


def test_a_two_up_scan_is_split_into_book_pages_in_order():
    original = pdf(True)
    assert detect_two_up(original)
    data, record = split_two_up(original)
    assert record["policy_version"] == "two-up-split-v1" and record["sheets_split"] == 3
    with fitz.open(stream=data, filetype="pdf") as document:
        texts = [page.get_text() for page in document]
    assert len(texts) == 6
    assert texts[0].startswith("Left page 22") and texts[1].startswith("Right page 23") and texts[5].startswith("Right page 27")
    assert "Right page" not in texts[0]


def test_an_ordinary_pdf_is_left_unchanged():
    assert split_two_up(pdf(False)) is None


def test_the_reference_states_the_range_an_upload_is_checked_against():
    assert _stated_page_range({"raw_reference": "Lee, K. (2020). A chapter. In A. Editor (Ed.), Book (pp. 64–86). Press."}) == (64, 86)
    assert _stated_page_range({"raw_reference": "Lee, K. (2014). An article. Journal, 53(3), 152–177."}) == (152, 177)
    assert _stated_page_range({"raw_reference": "Lee, K. (1985). A chapter. In Book (A. Editor, Ed.)."}) == (None, None)
