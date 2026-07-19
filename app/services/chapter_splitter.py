"""Chapter splitter for edited collections.

Detects whether a book PDF is a monograph or an edited collection,
then splits edited collections into individual chapter PDFs based on
the PDF's table of contents. Monographs are left whole.
"""

import logging
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from app.services.page_layout import detect_n_up

logger = logging.getLogger(__name__)

# A TOC entry that looks like a chapter: a leading number (optionally followed
# by "." or ")") then a title. e.g. "1 A Brief History...", "12. Conclusion".
# The separator may be punctuation OR just whitespace (publishers vary).
_CHAPTER_NUM_RE = re.compile(r"^\s*(\d{1,3})[\.\)]?\s+\S")

# Editor markers that signal an edited collection.
# Markers that unambiguously indicate an edited collection. We deliberately
# EXCLUDE bare "editor:" / "editors:" / "volume editor" because those appear on
# the series page of monographs as "Series Editor:" / "General Editor:" credits,
# which do NOT mean the book is an edited collection. Only the standalone
# "edited by" / "edited and" phrasings indicate a true edited volume.
_EDITOR_MARKERS = ("edited by", "edited and")


@dataclass
class ChapterInfo:
    """Metadata for a single chapter."""

    title: str
    page_start: int  # 1-based page number (human convention)
    page_end: int  # 1-based, inclusive
    author: str | None = None


def is_edited_collection(file_bytes: bytes) -> bool:
    """Determine if a PDF is an edited collection.

    Detection relies on the "edited by" / "edited and" markers in the front
    matter. These are the only reliable signal: the TOC structure of a
    monograph (chapters with numbered sections) is indistinguishable from that
    of an edited volume (chapters by different authors), so a TOC-based
    heuristic produces false positives on monographs with rich tables of
    contents.

    When the front matter is absent (e.g. a scan missing the title page), this
    returns False — the book is treated as a monograph and stored whole. An
    instructor who knows it's an edited collection can force chapter splitting
    via the upload flow in a future enhancement.

    Returns True if this should be split into chapters.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        text = ""
        for page in doc[:8]:
            text += page.get_text()
    finally:
        doc.close()

    text_lower = text.lower()
    return any(marker in text_lower for marker in _EDITOR_MARKERS)


def _select_chapter_entries(toc: list) -> list:
    """From a TOC, return the entries that represent chapters.

    Chapters are numbered entries (e.g. "1 Introduction", "2. Theory").
    Front matter ("Cover", "Title page"), Parts, and the Index are
    excluded by requiring a leading number.
    """
    return [e for e in toc if _CHAPTER_NUM_RE.match(e[1])]


def split_into_chapters(file_bytes: bytes) -> list[tuple[ChapterInfo, bytes]]:
    """Split an edited collection PDF into individual chapters.

    Uses PyMuPDF's TOC to find numbered chapter boundaries. Each chapter
    is extracted as its own PDF bytes, spanning from its start page up to
    (but not including) the next chapter's start page.

    Returns:
        List of (ChapterInfo, chapter_pdf_bytes) tuples. Empty list if
        the PDF has no usable TOC or too few chapters.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        toc = doc.get_toc()
        if not toc:
            logger.warning("No TOC in PDF, cannot split")
            return []

        chapters = _select_chapter_entries(toc)
        if len(chapters) < 2:
            logger.info("Too few numbered chapters (%d) to split", len(chapters))
            return []

        total_pages = len(doc)

        # N-up handling: TOC page numbers refer to BOOK pages (logical), but
        # PyMuPDF indexes CAPTURED pages. For a 2-up scan, divide book-page
        # numbers by pages_per_sheet to get the captured-page index.
        layout = detect_n_up(file_bytes)
        pages_per_sheet = layout.pages_per_sheet  # 1 (normal) or 2 (2-up)

        results: list[tuple[ChapterInfo, bytes]] = []
        author_pattern = re.compile(
            r"^([A-Z][A-Za-z.\-\u2019']+(?:\s+[A-Z][A-Za-z.\-\u2019']+)+)",
            re.MULTILINE,
        )

        for i, entry in enumerate(chapters):
            _, title, start_page_1based = entry
            # Convert 1-based BOOK page → 0-based CAPTURED-page index.
            start_idx = max(0, (start_page_1based - 1) // pages_per_sheet)

            # End page: start of next chapter minus one (1-based, inclusive).
            if i + 1 < len(chapters):
                next_start = chapters[i + 1][2]
                end_page_1based = max(start_page_1based, next_start - 1)
            else:
                end_page_1based = layout.logical_pages  # last chapter → logical end
            end_idx = min(total_pages - 1, (end_page_1based - 1) // pages_per_sheet)
            end_idx = min(total_pages - 1, end_page_1based - 1)

            # Extract the chapter as its own PDF (insert_pdf uses 0-based,
            # inclusive to_page).
            chapter_doc = fitz.open()
            chapter_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)
            chapter_bytes = chapter_doc.tobytes()
            chapter_doc.close()

            # Try to find an author line on the chapter's first page.
            author: str | None = None
            try:
                probe = fitz.open(stream=chapter_bytes, filetype="pdf")
                if len(probe) > 0:
                    first_page_text = probe[0].get_text()
                    author_match = author_pattern.search(first_page_text)
                    if author_match:
                        author = author_match.group(1).strip()
                probe.close()
            except Exception:
                pass  # author extraction is best-effort

            info = ChapterInfo(
                title=title.strip(),
                page_start=start_page_1based,
                page_end=end_page_1based,
                author=author,
            )
            results.append((info, chapter_bytes))
            logger.info(
                "Split chapter %d/%d: %s (pp. %d-%d, author=%s)",
                i + 1,
                len(chapters),
                info.title[:60],
                info.page_start,
                info.page_end,
                info.author,
            )

        return results
    finally:
        doc.close()
