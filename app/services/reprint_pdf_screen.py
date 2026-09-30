"""Safety-first, byte-bound publication-history screening (not admission).

Internal callers must authorize processing before calling. No endpoint or
automatic verified-edition producer is exposed by this module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Literal

import fitz
from pydantic import BaseModel, ConfigDict, Field

from app.services.edition_statement_verifier import (
    MAX_PAGE_CHARACTERS, MAX_TOTAL_CHARACTERS, PublicationPage, ReprintStatementFinding,
    verify_reprint_history,
)
from app.services.file_safety import inspect_uploaded_pdf, SafetyVerdict


class BlankPageRender(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    page_index: int = Field(ge=0, le=5)
    render_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    dpi: Literal[250] = 250


def _blank_page_render(page: fitz.Page, index: int) -> BlankPageRender | None:
    """Prove a bounded rendered viewport white, never infer blankness from OCR.

    No near-white tolerance: even one nonwhite sample retains the unknown page.
    This does not claim anything about invisible/off-page PDF objects.
    """
    scale = 250 / 72
    width = math.ceil(page.rect.width * scale) + 2
    height = math.ceil(page.rect.height * scale) + 2
    if width <= 0 or height <= 0 or width * height > 8_000_000:
        return None
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                          alpha=False, colorspace=fitz.csRGB)
    samples = pix.samples
    if not samples or samples.count(b'\xff') != len(samples):
        return None
    return BlankPageRender(page_index=index,
                           render_sha256=hashlib.sha256(pix.tobytes('png')).hexdigest())


class PDFReprintScreen(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    status: Literal['screened', 'not_assessed']
    reason: str
    representation_sha256: str
    finding: ReprintStatementFinding | None = None
    unassessed_page_indexes: tuple[int, ...] = ()
    blank_page_renders: tuple[BlankPageRender, ...] = ()


def screen_reprint_pdf(content: bytes, *, submitted_reference_sha256: str,
                       cited_year: str, retrieved_year: str) -> PDFReprintScreen:
    """Inspect safety first, then extract at most six pages without truncation.

    Computes the representation hash itself and never accepts supplied text,
    safety verdicts, or a caller's source hash as a substitute for inspected
    bytes. Empty/image-only pages and over-budget extraction abstain; no OCR,
    network search, model request, source storage or attestation is performed.
    """
    digest = hashlib.sha256(content).hexdigest()
    blank_renders = []

    def abstain(reason, unassessed_page_indexes=()):
        return PDFReprintScreen(status='not_assessed', reason=reason,
                               representation_sha256=digest,
                               unassessed_page_indexes=unassessed_page_indexes,
                               blank_page_renders=tuple(blank_renders))

    try:
        safety = inspect_uploaded_pdf(content)
    except Exception:
        return abstain('safety_inspection_unavailable')
    if safety.verdict != SafetyVerdict.CLEAN:
        return abstain('safety_not_clean')
    try:
        with fitz.open(stream=content, filetype='pdf') as doc:
            pages = []
            unassessed = []
            for index in range(min(6, len(doc))):
                page = doc[index]
                text = page.get_text()
                # A blank page is benign; an image-only page may contain an
                # unobserved edition warning, so do not silently skip it.
                if not text.strip():
                    if page.get_images() or page.get_drawings():
                        blank = _blank_page_render(page, index)
                        if blank is None:
                            unassessed.append(index)
                        else:
                            blank_renders.append(blank)
                    continue
                if len(text) > MAX_PAGE_CHARACTERS:
                    return abstain('publication_page_text_budget_exceeded', (index,))
                pages.append(PublicationPage(pdf_page_index=index, text=text))
            if unassessed:
                return abstain('publication_text_unavailable', tuple(unassessed))
            if not pages:
                return abstain('publication_text_unavailable')
            if sum(len(p.text) for p in pages) > MAX_TOTAL_CHARACTERS:
                return abstain('publication_text_budget_exceeded')
    except Exception:
        return abstain('publication_extraction_incomplete')
    finding = verify_reprint_history(
        pages, representation_sha256=digest,
        submitted_reference_sha256=submitted_reference_sha256,
        cited_year=cited_year, retrieved_year=retrieved_year)
    return PDFReprintScreen(status='screened', reason='publication_history_only',
                           representation_sha256=digest, finding=finding,
                           blank_page_renders=tuple(blank_renders))
