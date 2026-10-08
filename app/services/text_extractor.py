"""Text extraction from PDF and DOCX files.

Text extraction module.
Supports pdfplumber, PyMuPDF, and python-docx backends.
"""

import io
import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF
from docx import Document

logger = logging.getLogger(__name__)


class TextExtractionError(Exception):
    """Raised when text extraction fails."""


TEXT_EXTRACTION_QUALIFICATION_VERSION = "paper-text-qualification-v2"


@dataclass(frozen=True)
class TextCandidate:
    backend: str
    text: str
    sha256: str
    word_count: int
    suspicious_unbroken_token_count: int
    max_unbroken_token_length: int
    replacement_character_count: int
    # Native line boundaries, separate from authoritative semantic text.
    layout_text: str = ""

    @property
    def suspicious_rate(self) -> float:
        return self.suspicious_unbroken_token_count / max(self.word_count, 1)


@dataclass(frozen=True)
class QualifiedTextExtraction:
    selected: TextCandidate
    candidates: tuple[TextCandidate, ...]
    selection_reason: str

    def candidate(self, backend: str) -> TextCandidate | None:
        return next((item for item in self.candidates if item.backend == backend), None)

    def evidence(self) -> dict:
        return {
            "version": TEXT_EXTRACTION_QUALIFICATION_VERSION,
            "selected_backend": self.selected.backend,
            "selected_sha256": self.selected.sha256,
            "selection_reason": self.selection_reason,
            "candidates": [
                {
                    "backend": item.backend,
                    "sha256": item.sha256,
                    "layout_sha256": hashlib.sha256(item.layout_text.encode()).hexdigest() if item.layout_text else None,
                    "character_count": len(item.text),
                    "word_count": item.word_count,
                    "suspicious_unbroken_token_count": item.suspicious_unbroken_token_count,
                    "max_unbroken_token_length": item.max_unbroken_token_length,
                    "replacement_character_count": item.replacement_character_count,
                }
                for item in self.candidates
            ],
        }


def _clean_text(text: str) -> str:
    """Remove invisible Unicode characters that garble extracted text.

    PDF extractors (especially pdfplumber) sometimes insert zero-width spaces
    (\u200b), zero-width joiners (\u200d), and other invisible Unicode
    characters between visible characters. These break sentence splitting,
    citation regex, and LLM text comprehension.

    Also collapses excessive whitespace from the removal.

    PDF line-wrap reconstruction: PDF text extraction emits a newline at every
    visual line wrap, so a single paragraph becomes many "paragraphs". Real
    paragraph breaks are the double-newlines the extractors already emit (and
    that DOCX extraction produces natively). We join single newlines into
    spaces while preserving ``\\n\\n`` paragraph boundaries, so downstream
    paragraph splitters see real paragraphs. This is a no-op for DOCX (which
    has no single-newline line-wraps). Hyphenated line-breaks ("represen-\\n
    tation") are de-hyphenated; other single-newlines become a single space.
    """
    import re

    # Remove zero-width characters
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner
    text = text.replace("\ufeff", "")  # byte order mark / zero-width no-break space
    text = text.replace("\u2060", "")  # word joiner
    text = text.replace("\u00ad", "-")  # preserve a rendered discretionary hyphen

    # Collapse whitespace left behind by zero-width removals
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ newlines to 2

    # PDF line-wrap reconstruction (see docstring). We join ONLY mid-sentence
    # line-wraps, not reference-section line breaks. A newline is treated as a
    # line-wrap (joined into a space) only when a lowercase letter precedes it
    # and a lowercase letter follows (allowing optional whitespace on either
    # side — pdfplumber sometimes inserts a leading space or residual zero-width
    # chars at the start of the wrapped line). This preserves:
    #   - Reference-list line breaks (each ref on its own line, typically starts
    #     with an uppercase author name or [n] marker — not "lowercase\nlowercase")
    #   - Heading lines ("References", "Works Cited") that stand alone on a line
    #   - Paragraph breaks (\n\n, untouched)
    # The de-hyphenation step runs first because "represen-\ntation" ends with a
    # lowercase letter before \n and the join would merge it wrongly otherwise.
    # A line-end hyphen may be lexical (e.g. a compound). Preserve it in
    # authoritative paper wording; retrieval/quotation normalization owns
    # any explicitly labelled dehyphenated comparison alternative.
    text = re.sub(r"-\n[ \t]*(?=[a-z])", "-", text)
    text = re.sub(r"(?<=[a-z])[^\S\r\n]*\n[^\S\r\n]*(?=[a-z])", " ", text)
    # Tidy any double spaces introduced by the join
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def extract_from_pdf_pdfplumber(file_path: str) -> str:
    """Extract text from PDF using pdfplumber (good for structured PDFs)."""
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return join_pdf_pages(_strip_repeated_page_marginals(pages))
    except Exception as e:
        raise TextExtractionError(f"pdfplumber failed: {e}")


def extract_from_pdf_pymupdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (good fallback)."""
    try:
        doc = fitz.open(file_path)
        pages = _strip_repeated_page_marginals([page.get_text() for page in doc])
        doc.close()
        return join_pdf_pages(pages)
    except Exception as e:
        raise TextExtractionError(f"PyMuPDF failed: {e}")


def extract_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise TextExtractionError(f"DOCX extraction failed: {e}")


def extract_text(file_path: str, preferred_backend: str = "pdfplumber") -> str:
    """
    Extract text from a PDF or DOCX file.

    Args:
        file_path: Path to the file.
        preferred_backend: "pdfplumber" or "pymupdf" (PDF only).

    Returns:
        Extracted plain text.

    Raises:
        TextExtractionError if all extraction methods fail.
    """
    path = Path(file_path)
    if not path.exists():
        raise TextExtractionError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # Try preferred backend first, fall back to alternative
        backends = []
        if preferred_backend == "pdfplumber":
            backends = [extract_from_pdf_pdfplumber, extract_from_pdf_pymupdf]
        else:
            backends = [extract_from_pdf_pymupdf, extract_from_pdf_pdfplumber]

        last_error = None
        for backend in backends:
            try:
                text = backend(file_path)
                if text.strip():
                    return _clean_text(text)
            except TextExtractionError as e:
                last_error = e
                logger.warning(
                    "Backend %s failed (type=%s)",
                    backend.__name__,
                    type(e).__name__,
                )

        raise TextExtractionError(
            "All PDF extraction backends failed"
        ) from last_error

    elif suffix == ".docx":
        return _clean_text(extract_from_docx(file_path))
    else:
        raise TextExtractionError(f"Unsupported file type: {suffix}")


def extract_qualified_text_from_bytes(
    content: bytes,
    filename: str,
    preferred_backend: str = "pdfplumber",
) -> QualifiedTextExtraction:
    """Run every available PDF text route and select a quality-qualified result.

    A nonempty extraction is not sufficient: missing word spaces can leave the
    text technically populated while corrupting citation boundaries and the
    report. The preferred backend remains stable when it is clean; otherwise a
    materially cleaner alternate is selected. Both candidate hashes and
    text-free quality measurements remain available for audit and exact reuse.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        try:
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            candidate = _text_candidate("python-docx", _clean_text("\n\n".join(paragraphs)))
            if not candidate.text.strip():
                raise TextExtractionError("DOCX contained no extractable text")
            return QualifiedTextExtraction(
                selected=candidate,
                candidates=(candidate,),
                selection_reason="only_available_backend",
            )
        except TextExtractionError:
            raise
        except Exception as exc:
            raise TextExtractionError(f"DOCX extraction failed: {exc}") from exc
    if suffix != ".pdf":
        raise TextExtractionError(f"Unsupported file type: {suffix}")

    candidates: list[TextCandidate] = []
    failures: list[str] = []
    extractors = (
        ("pdfplumber", _extract_pdfplumber_bytes),
        ("pymupdf", _extract_pymupdf_bytes),
    )
    for backend, extractor in extractors:
        try:
            layout_text = extractor(content)
            text = _clean_text(layout_text)
            if text.strip():
                candidates.append(_text_candidate(backend, text, layout_text=layout_text))
            else:
                failures.append(f"{backend}: empty")
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}")
            logger.warning(
                "%s from bytes failed (type=%s)", backend, type(exc).__name__
            )
    if not candidates:
        raise TextExtractionError(
            "All PDF extraction backends failed: " + ", ".join(failures)
        )

    preferred = next(
        (item for item in candidates if item.backend == preferred_backend),
        candidates[0],
    )
    selected = preferred
    reason = "preferred_backend_quality_acceptable"
    alternatives = [item for item in candidates if item is not preferred]
    if alternatives:
        best = min(alternatives, key=_quality_order)
        preferred_bad = (
            preferred.suspicious_rate > 0.005
            or preferred.max_unbroken_token_length > 80
            or preferred.replacement_character_count > 0
        )
        materially_cleaner = (
            best.suspicious_rate + 0.002 < preferred.suspicious_rate
            and best.max_unbroken_token_length < preferred.max_unbroken_token_length
            and best.replacement_character_count <= preferred.replacement_character_count
        )
        if preferred_bad and materially_cleaner:
            selected = best
            reason = "alternate_materially_cleaner_than_preferred"
        elif _native_spacing_recovery(preferred, best):
            selected = best
            reason = "alternate_recovers_native_word_boundaries"
    return QualifiedTextExtraction(
        selected=selected,
        candidates=tuple(candidates),
        selection_reason=reason,
    )


def _native_spacing_recovery(preferred: TextCandidate, alternate: TextCandidate) -> bool:
    """Select a native transcription, never guess spaces inside student words.

    Long shared URLs can mask spacing damage in the older maximum-token test.
    Require near-complete agreement of physical lines without whitespace and a
    substantial recovery of word boundaries, not simply more extracted text.
    """
    if not preferred.layout_text or not alternate.layout_text:
        return False
    if alternate.word_count < preferred.word_count * 1.25:
        return False
    if alternate.replacement_character_count > preferred.replacement_character_count:
        return False
    lines = [Counter(re.sub(r"\s+", "", line) for line in item.layout_text.splitlines())
             for item in (preferred, alternate)]
    shared = sum(len(line) * count for line, count in (lines[0] & lines[1]).items())
    totals = [sum(len(line) * count for line, count in counts.items()) for counts in lines]
    return min(totals) >= 200 and shared >= .95 * max(totals)


def extract_text_from_bytes(
    content: bytes, filename: str, preferred_backend: str = "pdfplumber"
) -> str:
    """
    Extract text from file bytes (e.g., from an upload).

    Args:
        content: Raw file bytes.
        filename: Original filename (to determine file type).
        preferred_backend: PDF extraction backend.

    Returns:
        Extracted plain text.
    """
    return extract_qualified_text_from_bytes(
        content,
        filename,
        preferred_backend=preferred_backend,
    ).selected.text


def _extract_pdfplumber_bytes(content: bytes) -> str:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return join_pdf_pages(_strip_repeated_page_marginals(pages), bold_lines=_bold_pdf_lines(content))


def _extract_pymupdf_bytes(content: bytes) -> str:
    document = fitz.open(stream=content, filetype="pdf")
    try:
        return join_pdf_pages(
            _strip_repeated_page_marginals([page.get_text() for page in document]),
            bold_lines=_bold_pdf_lines(content),
        )
    finally:
        document.close()


def _bold_pdf_lines(content: bytes) -> frozenset[str]:
    """Printed lines set wholly in a bold font, whitespace-normalised."""
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception:  # noqa: BLE001 - layout evidence is optional
        return frozenset()
    lines = set()
    try:
        for page in document:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if spans and all(int(s.get("flags", 0)) & 16 or re.search(r"bold|black", s.get("font", ""), re.I)
                                     for s in spans):
                        lines.add(" ".join("".join(s["text"] for s in spans).split()))
    finally:
        document.close()
    return frozenset(lines)


def join_pdf_pages(pages: list[str], *, bold_lines: frozenset[str] = frozenset()) -> str:
    """Pages as paragraphs, except a sentence the page break cuts: a page ending
    without sentence punctuation before one that opens in lower case
    ("This part" / "of her experience", paper 9, 2026-10-04) joins with a space."""
    text = ""
    for page in pages:
        if not text:
            text = page
            continue
        if re.search(r"[A-Za-z,;]\s*$", text) and re.match(r"\s*[a-z]", page):
            # Move the cut sentence's start to the next page, so each page keeps
            # its size (a page is one paragraph to the citation reader).
            boundary = max((m.end() for m in re.finditer(r"[.!?][\"”’')]*\s+", text)), default=-1)
            if boundary > 0:
                text = text[:boundary].rstrip() + "\n\n" + " ".join(text[boundary:].split()) + " " + page.lstrip()
            else:
                text = text.rstrip() + " " + page.lstrip()
        else:
            text = text + "\n\n" + page
    return isolate_pdf_tables(isolate_pdf_headings(text, bold_lines=bold_lines))


_FUNCTION_WORDS = frozenset({"the", "of", "to", "and", "a", "an", "is", "are", "was", "were", "for", "that", "it",
                             "in", "on", "with", "because", "they", "this", "be", "by", "as", "not", "their", "do"})


def isolate_pdf_tables(text: str) -> str:
    """A table with no ruling lines reads as one run of short fragments after its
    "Table N" caption, straight into the next prose (the owner's article: a
    marking rubric captured as part of a citation, 2026-10-07). In a paragraph
    opening with such a caption, the first line that reads as prose (a capital
    start, eight or more words, four or more function words, a sentence break
    within three lines) begins a new paragraph."""
    paragraphs = text.split("\n\n")
    caption = re.compile(r"\s*Table\s+\d+\b")
    for index, paragraph in enumerate(paragraphs):
        # The caption may already stand alone (a heading line), the rows after it.
        after_caption = index > 0 and caption.match(paragraphs[index - 1]) and "\n" not in paragraphs[index - 1].strip()
        if not (caption.match(paragraph) or after_caption):
            continue
        lines = paragraph.split("\n")
        for at in range(0 if after_caption else 1, len(lines)):
            line = lines[at].strip()
            words = re.findall(r"[A-Za-z’'-]+", line)
            function = sum(w.casefold() in _FUNCTION_WORDS for w in words)
            ahead = " ".join(lines[at:at + 3])
            if (line[:1].isupper() and len(words) >= 8 and function >= 4
                    and re.search(r"[a-z)][.!?][\"”’)]*(?:\s+[A-Z]|\s*$)", ahead)):
                paragraphs[index] = "\n".join(lines[:at]) + "\n\n" + "\n".join(lines[at:])
                break
    return "\n\n".join(paragraphs)


_REFERENCE_LIST_HEADING = re.compile(r"(?im)^\s*(?:references|reference list|bibliography|works cited)\s*$")


_KEYWORDS_LINE = re.compile(r"(?i)^(?:key\s*words?|index\s+terms)\s*[:：—–-]\s*\S")


def isolate_pdf_headings(text: str, *, bold_lines: frozenset[str] = frozenset()) -> str:
    """A heading line the PDF prints without a blank line after it is not the
    start of the next sentence ("Conceptualisation of Identity and
    Representation" / "Identity is …", paper 9, 2026-10-04): a short,
    title-cased line with no closing punctuation, after a finished sentence
    and before a capitalised line, becomes its own paragraph. The reference
    list is left as it is."""
    end = len(text)
    heading = None
    for heading in _REFERENCE_LIST_HEADING.finditer(text):
        pass
    if heading is not None:
        end = heading.start()
    lines = text[:end].split("\n")
    out = list(lines)
    # A keywords line is front matter, not the start of the first sentence, and
    # a heading may follow it (the owner's article, 2026-10-07).
    keywords = {i for i, line in enumerate(lines) if _KEYWORDS_LINE.match(line.strip())}
    for i in keywords:
        out[i] = "\n" + lines[i] + "\n"
    for i in range(1, len(lines) - 1):
        if i in keywords:
            continue
        line, before, after = lines[i].strip(), lines[i - 1].strip(), lines[i + 1].strip()
        if i - 1 in keywords:
            before = "."
            # One title-cased word right after the keywords is the first heading.
            if (re.fullmatch(r"(?:\d+\.?\s+)?[A-Z][A-Za-z'’-]{3,}", line) and after[:1].isupper()):
                out[i] = "\n" + lines[i] + "\n"
                continue
        words = re.findall(r"[A-Za-z][\w'’-]*", line)
        long_words = [w for w in words if len(w) >= 4]
        # A line set wholly in bold is a heading in any case ("Performance and
        # influencing factors", Franchise 2, 2026-10-07); otherwise title case.
        bold = " ".join(line.split()) in bold_lines
        if (not 2 <= len(words) <= (15 if bold else 12) or len(long_words) < 2 or re.search(r"[.!?:;,]$", line)
                or not re.search(r"[.!?][\"”’')]*$", before) or not after[:1].isupper()
                or (not bold and sum(w[0].isupper() for w in long_words) < 0.75 * len(long_words))):
            continue
        out[i] = "\n" + lines[i] + "\n"
    return "\n".join(out) + text[end:]


# A publication footer: a licence statement, a bare address, or a journal
# name with its issue date (the owner's article, page 1, 2026-10-07).
_LICENCE_LINE = re.compile(
    r"(?i)\b(?:CC[ -]BY(?:[ -][A-Z]{2})*|creative\s*commons|creativecommons\.org|open access under|"
    r"licen[cs]ed under|all rights reserved)\b|©|\bcopyright\b")
_FOOTER_LINE = re.compile(
    r"(?i)^(?:(?:[A-Z][\w&:'’,.-]*\s+){0,8}?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Spring|Summer|Autumn|Fall|Winter)?\s*\d{4}\s+)?"
    r"(?:https?://\S+|doi:\s*\S+)(?:\s+(?:https?://\S+))*$"
    r"|^(?:[A-Z][\w&:'’-]*\s+){1,8}?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}(?:\s+https?://\S+)?$")


def _strip_publication_footer(page: str) -> str:
    """A page's closing licence block is not paper text: from its first footer
    line among the last four lines, when one of them states a licence."""
    lines = page.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    tail = nonempty[-4:]

    def licence(line: str) -> bool:
        # A short statement, not a sentence of the paper that cites a source.
        return bool(_LICENCE_LINE.search(line)) and len(line.split()) <= 14 and not re.search(r"\(\D*\d{4}", line)

    if not any(licence(lines[index]) for index in tail):
        return page
    cut = None
    for index in reversed(tail):
        line = lines[index].strip()
        if licence(line) or _FOOTER_LINE.match(line):
            cut = index
        else:
            break
    if cut is None:
        return page
    return "\n".join(lines[:cut])


def _strip_repeated_page_marginals(pages: list[str]) -> list[str]:
    """Remove only repeated running headers/footers from PDF semantic text.

    A page number or running title can otherwise land inside a citation sentence
    when extraction joins page text.  Detection is deliberately page-zone and
    repetition bound: a line must occur in the first/last three non-empty lines
    on a material number of pages.  Unique headings and body lines are retained.
    """
    pages = [_strip_publication_footer(page) for page in pages]
    if len(pages) < 3:
        return pages
    page_lines = [page.splitlines() for page in pages]
    signatures_by_page: list[set[str]] = []
    for lines in page_lines:
        nonempty = [line.strip() for line in lines if line.strip()]
        marginal = [*nonempty[:3], *nonempty[-3:]]
        signatures_by_page.append(
            {signature for line in marginal if (signature := _marginal_signature(line))}
        )
    counts: dict[str, int] = {}
    for signatures in signatures_by_page:
        for signature in signatures:
            counts[signature] = counts.get(signature, 0) + 1
    threshold = max(3, math.ceil(len(pages) * 0.2))
    repeated = {signature for signature, count in counts.items() if count >= threshold}
    if not repeated:
        return pages

    cleaned: list[str] = []
    for lines in page_lines:
        nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
        marginal_indexes = set(nonempty_indexes[:3]) | set(nonempty_indexes[-3:])
        retained = [
            line
            for index, line in enumerate(lines)
            if not (
                index in marginal_indexes
                and _marginal_signature(line) in repeated
            )
        ]
        cleaned.append("\n".join(retained))
    return cleaned


def _marginal_signature(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line).strip().casefold()
    if not normalized or len(normalized) > 220:
        return ""
    normalized = re.sub(r"\d+", "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _text_candidate(backend: str, text: str, *, layout_text: str = "") -> TextCandidate:
    words = text.split()
    lengths = [len(word) for word in words]
    return TextCandidate(
        backend=backend,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        word_count=len(words),
        suspicious_unbroken_token_count=sum(length > 40 for length in lengths),
        max_unbroken_token_length=max(lengths, default=0),
        replacement_character_count=text.count("\ufffd"),
        layout_text=layout_text,
    )


def _quality_order(candidate: TextCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.suspicious_rate,
        candidate.replacement_character_count,
        candidate.max_unbroken_token_length,
        -candidate.word_count,
    )
