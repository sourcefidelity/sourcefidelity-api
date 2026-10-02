"""Split a scan with two book pages on each PDF page (`two-up-split-v1`, owner request 2026-10-02).

A photocopied or scanned chapter often puts facing book pages side by side on
one landscape sheet (Bordwell's chapter: ten sheets showing pages 22-23,
24-25 ...). Page counts, printed page numbers and quotation locators then
describe sheets, not book pages, so completeness cannot be judged. When most
sheets are landscape and their text sits in two columns with an empty gutter
down the middle, each sheet is split into its left and right halves before any
identity, completeness or text check runs. Anything else is left unchanged.
"""
from __future__ import annotations

import hashlib

import fitz

POLICY = "two-up-split-v1"
_LANDSCAPE_RATIO = 1.15       # width / height of a sheet holding two pages
_GUTTER_SHARE = 0.06          # half-width of the middle band, as a share of the width
_MAX_CROSSING_SHARE = 0.03    # words that may cross the middle band
_MIN_SIDE_SHARE = 0.2         # each half must hold at least this share of the words
_MIN_WORDS = 40               # a sheet with less text says nothing about layout


def _two_up_sheet(page) -> bool | None:
    """True or False for a sheet with enough text; None when it cannot tell."""
    rect = page.rect
    if rect.width < rect.height * _LANDSCAPE_RATIO:
        return False
    words = page.get_text("words")
    if len(words) < _MIN_WORDS:
        return None
    middle = rect.x0 + rect.width / 2
    band = rect.width * _GUTTER_SHARE
    crossing = sum(1 for w in words if w[0] < middle - band / 4 and w[2] > middle + band / 4)
    left = sum(1 for w in words if w[2] <= middle)
    right = sum(1 for w in words if w[0] >= middle)
    return (crossing <= _MAX_CROSSING_SHARE * len(words)
            and min(left, right) >= _MIN_SIDE_SHARE * len(words))


def _gutter(page) -> float:
    """The x position of the widest word-free vertical band in the middle
    third of the sheet; the sheet's centre when there is none."""
    rect = page.rect
    low, high = rect.x0 + rect.width / 3, rect.x0 + 2 * rect.width / 3
    spans = sorted((max(w[0], low), min(w[2], high)) for w in page.get_text("words") if w[2] > low and w[0] < high)
    best, cursor, best_width = rect.x0 + rect.width / 2, low, 0.0
    for start, end in spans + [(high, high)]:
        if start - cursor > best_width:
            best, best_width = (cursor + start) / 2, start - cursor
        cursor = max(cursor, end)
    return best


def detect_two_up(content: bytes) -> bool:
    """Most readable sheets are landscape with two separate text columns of pages."""
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            verdicts = [_two_up_sheet(page) for page in document]
    except Exception:  # noqa: BLE001 - an unreadable file is left to the ordinary checks
        return False
    readable = [v for v in verdicts if v is not None]
    return len(readable) >= 2 and sum(readable) >= 0.8 * len(readable)


def split_two_up(content: bytes) -> tuple[bytes, dict] | None:
    """The PDF with each two-page sheet split into two pages, and its record; or None."""
    if not detect_two_up(content):
        return None
    with fitz.open(stream=content, filetype="pdf") as source, fitz.open() as target:
        split = 0
        for index, page in enumerate(source):
            rect = page.rect
            if _two_up_sheet(page) is False:
                target.insert_pdf(source, from_page=index, to_page=index)
                continue
            cut = _gutter(page)
            for clip in (fitz.Rect(rect.x0, rect.y0, cut, rect.y1),
                         fitz.Rect(cut, rect.y0, rect.x1, rect.y1)):
                new = target.new_page(width=clip.width, height=clip.height)
                new.show_pdf_page(new.rect, source, index, clip=clip)
            split += 1
        data = target.tobytes(garbage=3, deflate=True)
    return data, {"policy_version": POLICY, "original_sha256": hashlib.sha256(content).hexdigest(),
                  "split_sha256": hashlib.sha256(data).hexdigest(), "sheets_split": split}
