"""N-up PDF layout detection.

Some scanned PDFs place two (or four) book pages on a single captured PDF
page — a "2-up" or "4-up" layout. This breaks naive page counting: a 500-page
book scanned 2-up appears as ~250 PDF pages.

This module detects N-up layouts and reports the TRUE logical page count,
which the completeness checker and chapter splitter rely on.

Detection signals (combined; all three are required for a confident 2-up
verdict):
  1. Landscape orientation (width > height) on most sampled pages.
  2. A text-block "gutter" at the page midline: blocks cluster into a
     left half and a right half with a clear gap between them.
  3. Two consecutive page-number tokens per captured page (e.g. 246 and
     247), confirming each captured page holds two book pages.

For 4-up, the same logic extends to four quadrants, but 4-up is rare in
academic scanning; the primary real-world case this targets is 2-up.
"""

import logging
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# A standalone page-number-like token (1-4 digits) on its own line.
_PAGENUM_RE = re.compile(r"(?m)^\s*(\d{1,4})\s*$")

# Landscape ratio threshold: width must exceed height by this factor.
_LANDSCAPE_RATIO = 1.05

# Minimum fraction of sampled pages that must be landscape to call it N-up.
_LANDSCAPE_MIN_FRAC = 0.8

# Gutter: a block is "left" if its midpoint < page_mid - gutter_pad,
# "right" if midpoint > page_mid + gutter_pad. gutter_pad as fraction of width.
_GUTTER_PAD = 0.03

# Minimum text-block count on each side to count it as a real column.
_MIN_BLOCKS_PER_SIDE = 1

# Text-quality classification thresholds.
# A page with image coverage above this fraction is a full-page scan image.
_FULL_PAGE_IMAGE_THRESHOLD = 0.90
# If this fraction of sampled pages have substantial text, the PDF has a text layer.
_HAS_TEXT_FRACTION = 0.5

# Text-quality verdict constants.
PURE_SCAN = "pure_scan"      # No text layer — unusable without OCR.
SCAN_OCR = "scan_ocr"        # Text layer over full-page images — OCR may have errors.
DIGITAL = "digital"          # Born-digital — text is authoritative.


@dataclass
class TextQuality:
    """Result of text-quality classification.

    verdict is one of: PURE_SCAN, SCAN_OCR, DIGITAL.
    - PURE_SCAN: no extractable text (needs OCR before use).
    - SCAN_OCR:  has text, but it's an OCR layer over scanned page images —
                 may contain recognition errors. Citation exact-match results
                 from this text should be treated with lower confidence.
    - DIGITAL:   born-digital PDF; text layer is authoritative.
    """

    verdict: str
    image_coverage: float  # mean fraction of page covered by images
    text_fraction: float  # fraction of sampled pages with substantial text
    sampled_pages: int


def classify_text_quality(file_bytes: bytes, sample_size: int = 8) -> TextQuality:
    """Classify a PDF's text quality into pure_scan / scan_ocr / digital.

    Signals:
      - Image coverage: a full-page image under the text layer indicates a
        scanned+OCRed PDF (born-digital PDFs have no full-page image).
      - Text presence: pages with ~0 extractable characters indicate no text
        layer at all (a pure scan needing OCR).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        n = len(doc)
        if n == 0:
            return TextQuality(PURE_SCAN, 0.0, 0.0, 0)

        indices = _sample_indices(n, sample_size)
        coverages: list[float] = []
        text_pages = 0

        for idx in indices:
            page = doc[idx]
            rect = page.rect
            page_area = rect.width * rect.height

            # Text presence
            text_len = len(page.get_text().strip())
            if text_len > 50:  # substantial text, not stray artifacts
                text_pages += 1

            # Image coverage
            img_area = 0.0
            for img in page.get_images():
                try:
                    for r in page.get_image_rects(img[0]):
                        img_area = max(img_area, abs(r.width * r.height))
                except Exception:
                    pass
            coverages.append(img_area / page_area if page_area else 0.0)
    finally:
        doc.close()

    sampled = len(indices)
    mean_coverage = sum(coverages) / sampled if sampled else 0.0
    text_fraction = text_pages / sampled if sampled else 0.0

    if text_fraction < _HAS_TEXT_FRACTION:
        # Little/no extractable text → pure scan
        return TextQuality(PURE_SCAN, mean_coverage, text_fraction, sampled)
    if mean_coverage >= _FULL_PAGE_IMAGE_THRESHOLD:
        # Text present BUT a full-page image underlies it → scan + OCR
        return TextQuality(SCAN_OCR, mean_coverage, text_fraction, sampled)
    # Text present, no full-page image → born-digital
    return TextQuality(DIGITAL, mean_coverage, text_fraction, sampled)


@dataclass
class NUpLayout:
    """Result of N-up layout analysis.

    Attributes:
        is_n_up: True if the PDF appears to be an N-up scan.
        pages_per_sheet: 1 (normal), 2 (2-up), or 4 (4-up).
        captured_pages: The number of pages PyMuPDF reports (len(doc)).
        logical_pages: The estimated number of book pages = captured × pages_per_sheet.
        confidence: "high" | "medium" | "low"
        signals: Human-readable list of which signals fired.
    """

    is_n_up: bool
    pages_per_sheet: int
    captured_pages: int
    logical_pages: int
    confidence: str
    signals: list[str]


def detect_n_up(file_bytes: bytes, sample_size: int = 8) -> NUpLayout:
    """Detect whether a PDF is an N-up scan and compute the logical page count.

    Args:
        file_bytes: Raw PDF bytes.
        sample_size: Number of pages to sample (even spaced) for analysis.

    Returns:
        NUpLayout describing the detected layout.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        n = len(doc)
        if n == 0:
            return NUpLayout(False, 1, 0, 0, "low", ["empty document"])

        # Sample evenly-spaced pages (avoid front/back matter skewing results).
        sample_indices = _sample_indices(n, sample_size)

        landscape_hits = 0
        gutter_hits = 0
        pagenum_pair_hits = 0
        signals: list[str] = []

        for idx in sample_indices:
            page = doc[idx]
            rect = page.rect

            # Signal 1: landscape orientation
            if rect.width > rect.height * _LANDSCAPE_RATIO:
                landscape_hits += 1

            # Signal 2 + 3: text-block layout and page-number pairs
            has_gutter, has_pair = _analyze_page_blocks(page)
            if has_gutter:
                gutter_hits += 1
            if has_pair:
                pagenum_pair_hits += 1
    finally:
        doc.close()

    sampled = len(sample_indices)
    landscape_frac = landscape_hits / sampled if sampled else 0
    gutter_frac = gutter_hits / sampled if sampled else 0
    pagenum_frac = pagenum_pair_hits / sampled if sampled else 0

    # Verdict: require landscape + gutter to call it 2-up. Page-number pairs
    # raise confidence but aren't required (OCR text may not be clean).
    is_2up = landscape_frac >= _LANDSCAPE_MIN_FRAC and gutter_frac >= 0.5

    if is_2up:
        pages_per_sheet = 2
        confidence = "high" if pagenum_frac >= 0.5 else "medium"
        signals = [
            f"landscape on {landscape_hits}/{sampled} sampled pages",
            f"center gutter detected on {gutter_hits}/{sampled} sampled pages",
        ]
        if pagenum_frac > 0:
            signals.append(f"consecutive page-number pairs on {pagenum_pair_hits}/{sampled} sampled pages")
        logical = n * pages_per_sheet
        return NUpLayout(True, pages_per_sheet, n, logical, confidence, signals)

    # Not N-up
    signals = [
        f"landscape on {landscape_hits}/{sampled} sampled pages",
        f"center gutter on {gutter_hits}/{sampled} sampled pages",
    ]
    return NUpLayout(False, 1, n, n, "high", signals)


def _sample_indices(total: int, sample_size: int) -> list[int]:
    """Return evenly-spaced page indices, avoiding the very first/last pages."""
    if total <= sample_size:
        # Use all but guard against 0
        return list(range(total))
    # Sample from the middle 80% to avoid cover/blank/end pages
    start = max(1, total // 10)
    end = total - start
    span = end - start
    if span <= 0:
        return list(range(total))
    step = span / sample_size
    return [min(end - 1, int(start + i * step)) for i in range(sample_size)]


def _analyze_page_blocks(page) -> tuple[bool, bool]:
    """Analyze a single page for a center gutter and consecutive page-number pairs.

    Returns:
        (has_gutter, has_pagenum_pair)
    """
    rect = page.rect
    mid_x = rect.width / 2
    pad = rect.width * _GUTTER_PAD
    left_threshold = mid_x - pad
    right_threshold = mid_x + pad

    try:
        d = page.get_text("dict")
    except Exception:
        return False, False

    left_blocks = 0
    right_blocks = 0
    block_texts: list[str] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        # Use the block's bounding box midpoint
        bbox = block.get("bbox")
        if not bbox:
            continue
        block_mid_x = (bbox[0] + bbox[2]) / 2
        # Collect text for page-number analysis
        text = _block_text(block)
        if text.strip():
            block_texts.append(text.strip())
        if block_mid_x < left_threshold:
            left_blocks += 1
        elif block_mid_x > right_threshold:
            right_blocks += 1

    has_gutter = (
        left_blocks >= _MIN_BLOCKS_PER_SIDE
        and right_blocks >= _MIN_BLOCKS_PER_SIDE
    )

    # Page-number pair detection: find standalone numeric tokens, check for
    # a consecutive pair (n, n+1).
    has_pair = _has_consecutive_pagenum_pair(block_texts)

    return has_gutter, has_pair


def _block_text(block: dict) -> str:
    """Concatenate all span text in a dict-format block."""
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return " ".join(parts)


def _has_consecutive_pagenum_pair(block_texts: list[str]) -> bool:
    """Check if the page contains two consecutive integer page numbers.

    Looks for standalone numeric tokens and checks whether any two are
    consecutive (e.g., 246 and 247), which indicates 2 book pages per sheet.
    """
    numbers: list[int] = []
    for text in block_texts:
        # Match lines that are purely a number (page-number markers)
        for m in _PAGENUM_RE.finditer(text):
            try:
                numbers.append(int(m.group(1)))
            except ValueError:
                continue
        # Also catch numbers that are the entire short block (<=4 chars)
        stripped = text.strip()
        if stripped.isdigit() and len(stripped) <= 4:
            try:
                numbers.append(int(stripped))
            except ValueError:
                continue

    if len(numbers) < 2:
        return False
    # Dedupe and sort
    nums = sorted(set(numbers))
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] == 1:
            return True
    return False
