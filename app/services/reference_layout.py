"""Layout-preserving, text-free evidence for paper reference entries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import io
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Literal

import fitz
from docx import Document
from docx.text.run import Run
from docx.oxml.ns import qn
from pydantic import BaseModel, Field, model_validator

from app.services.schemas import ParsedReference


REFERENCE_LAYOUT_VERSION = "reference-layout-v1"
_MIN_MATCH_SCORE = 0.72


class ReferenceEntryLayout(BaseModel):
    """Bounded layout measurements for one parsed reference entry."""

    reference_id: str = Field(min_length=1)
    reference_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_status: Literal["matched", "ambiguous", "not_matched"]
    match_confidence: float = Field(ge=0.0, le=1.0)
    location_indexes: list[int] = Field(default_factory=list)
    line_count: int = Field(ge=0)
    first_line_x_points: float | None = None
    continuation_x_median_points: float | None = None
    observed_hanging_indent_points: float | None = None
    italic_character_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    bold_character_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    text_style_spans: list["ReferenceTextStyleSpan"] = Field(default_factory=list)
    style_binding_version: Literal['reference-style-binding-v1'] | None = None
    style_observation_ranges: list["ReferenceStyleRange"] = Field(default_factory=list)
    rectangles: list["ReferenceLayoutRectangle"] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mapping(self):
        if self.mapping_status != "matched" and (
            self.location_indexes
            or self.line_count
            or self.first_line_x_points is not None
            or self.continuation_x_median_points is not None
            or self.observed_hanging_indent_points is not None
            or self.italic_character_fraction is not None
            or self.bold_character_fraction is not None
            or self.text_style_spans
            or self.style_observation_ranges
            or self.style_binding_version
            or self.rectangles
        ):
            raise ValueError("Unmatched layout entries cannot carry measurements")
        if self.style_observation_ranges and not self.style_binding_version:
            raise ValueError('Style observations require a versioned binding')
        return self


class ReferenceStyleRange(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_range(self):
        if self.end <= self.start:
            raise ValueError('Style range must be nonempty')
        return self


class ReferenceLayoutRectangle(BaseModel):
    page_index: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_rectangle(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Reference layout rectangle must have positive area")
        return self


class ReferenceTextStyleSpan(BaseModel):
    """Text-free submitted-style coordinates in one retained raw reference."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    italic: bool = False
    bold: bool = False

    @model_validator(mode="after")
    def validate_span(self):
        if self.end <= self.start or not (self.italic or self.bold):
            raise ValueError("Reference style span must be nonempty and styled")
        return self


class ReferenceLayoutArtifact(BaseModel):
    """Text-free layout evidence derived from one immutable paper upload."""

    layout_version: str = REFERENCE_LAYOUT_VERSION
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    extraction_backend: Literal["pymupdf_dict", "python_docx"]
    location_kind: Literal["page", "paragraph"]
    citation_format: Literal["apa", "mla"]
    status: Literal["complete", "partial", "not_assessed"]
    heading_status: Literal["matched", "inferred_from_first_reference", "not_matched"]
    heading_page_index: int | None = Field(default=None, ge=0)
    reference_count: int = Field(ge=0)
    matched_reference_count: int = Field(ge=0)
    entries: list[ReferenceEntryLayout] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.reference_count != len(self.entries):
            raise ValueError("Reference layout count does not match entries")
        if self.matched_reference_count != sum(
            item.mapping_status == "matched" for item in self.entries
        ):
            raise ValueError("Matched reference count does not match entries")
        return self


@dataclass(frozen=True)
class _LayoutLine:
    text: str
    location_index: int
    x0: float
    continuation_x0: float | None
    italic_characters: int
    bold_characters: int
    styled_characters: int
    style_spans: tuple[tuple[int, int, bool, bool], ...] = ()
    style_complete: bool = True
    indent_complete: bool = True
    bbox: tuple[float, float, float, float] | None = None


def extract_reference_layout_from_bytes(
    content: bytes,
    filename: str,
    *,
    references: list[ParsedReference],
    citation_format: str,
) -> ReferenceLayoutArtifact:
    """Bind parsed references to physical layout without retaining their text."""
    if citation_format not in {"apa", "mla"}:
        raise ValueError("Reference layout supports APA and MLA papers only")
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf":
        return _extract_pdf(content, references, citation_format)
    if suffix == ".docx":
        return _extract_docx(content, references, citation_format)
    raise ValueError("Reference layout supports PDF and DOCX papers only")


def _extract_pdf(
    content: bytes,
    references: list[ParsedReference],
    citation_format: str,
) -> ReferenceLayoutArtifact:
    lines: list[_LayoutLine] = []
    document = fitz.open(stream=content, filetype="pdf")
    try:
        for page_index, page in enumerate(document):
            payload = page.get_text("dict", sort=True)
            for block in payload.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text, style_spans = _styled_text(
                        (
                            span.get("text", ""),
                            _pdf_span_italic(span),
                            _pdf_span_bold(span),
                        )
                        for span in spans
                    )
                    if not text:
                        continue
                    # Page-number-only margin lines are not bibliography text.
                    # Keep numeric lines in the content area, which may be locators.
                    if (text.isdecimal() and spans and
                            (max(s['bbox'][3] for s in spans) < page.rect.height * .06
                             or min(s['bbox'][1] for s in spans) > page.rect.height * .94)):
                        continue
                    styled = sum(len(span.get("text", "")) for span in spans)
                    italic = sum(
                        len(span.get("text", ""))
                        for span in spans
                        if _pdf_span_italic(span)
                    )
                    bold = sum(
                        len(span.get("text", ""))
                        for span in spans
                        if _pdf_span_bold(span)
                    )
                    lines.append(
                        _LayoutLine(
                            text=text,
                            location_index=page_index,
                            x0=round(min(span["bbox"][0] for span in spans), 3),
                            continuation_x0=None,
                            italic_characters=italic,
                            bold_characters=bold,
                            styled_characters=styled,
                            style_spans=style_spans,
                            style_complete=all('flags' in span and bool(span.get('font')) for span in spans),
                            bbox=(
                                round(min(span["bbox"][0] for span in spans), 3),
                                round(min(span["bbox"][1] for span in spans), 3),
                                round(max(span["bbox"][2] for span in spans), 3),
                                round(max(span["bbox"][3] for span in spans), 3),
                            ),
                        )
                    )
    finally:
        document.close()
    return _build_artifact(
        content=content,
        media_type="application/pdf",
        backend="pymupdf_dict",
        citation_format=citation_format,
        references=references,
        lines=lines,
        style_limitation=(
            "PDF font flags are observed rendering metadata; they do not identify which bibliographic field should carry a style."
        ),
    )


def _extract_docx(
    content: bytes,
    references: list[ParsedReference],
    citation_format: str,
) -> ReferenceLayoutArtifact:
    document = Document(io.BytesIO(content))
    lines: list[_LayoutLine] = []
    for paragraph_index, paragraph in enumerate(document.paragraphs):
        # Native hyperlink runs belong to the visible paragraph too. Omitting
        # them loses URL text and lowers otherwise exact entry mappings.
        runs = [run for part in paragraph.iter_inner_content()
                for run in ([part] if isinstance(part, Run) else part.runs)]
        text, style_spans = _styled_text(
            (run.text, bool(_docx_run_italic(run)), bool(_docx_run_bold(run)))
            for run in runs
        )
        if not text:
            continue
        paragraph_format = paragraph.paragraph_format
        left_indent = _length_points(
            paragraph_format.left_indent
            if paragraph_format.left_indent is not None
            else paragraph.style.paragraph_format.left_indent
        )
        first_line_indent = _length_points(
            paragraph_format.first_line_indent
            if paragraph_format.first_line_indent is not None
            else paragraph.style.paragraph_format.first_line_indent
        )
        first_x = left_indent + first_line_indent
        styled = sum(len(run.text) for run in runs)
        italic = sum(len(run.text) for run in runs if _docx_run_italic(run))
        bold = sum(len(run.text) for run in runs if _docx_run_bold(run))
        lines.append(
            _LayoutLine(
                text=text,
                location_index=paragraph_index,
                x0=round(first_x, 3),
                continuation_x0=round(left_indent, 3),
                italic_characters=italic,
                bold_characters=bold,
                styled_characters=styled,
                style_spans=style_spans,
                style_complete=all(_docx_run_italic(run) is not None and _docx_run_bold(run) is not None for run in runs),
            )
        )
        if '\n' in text:
            original = lines.pop()
            offset = 0
            for fragment in text.split('\n'):
                stop = offset + len(fragment)
                spans = tuple((max(a, offset)-offset, min(b, stop)-offset, i, bld)
                              for a, b, i, bld in style_spans if a < stop and b > offset)
                if fragment.strip():
                    lines.append(_LayoutLine(
                        text=fragment, location_index=paragraph_index,
                        x0=original.x0, continuation_x0=None,
                        italic_characters=sum(b-a for a,b,i,_ in spans if i),
                        bold_characters=sum(b-a for a,b,_,bold in spans if bold),
                        styled_characters=len(fragment), style_spans=spans,
                        style_complete=original.style_complete, indent_complete=False,
                    ))
                offset = stop + 1
    return _build_artifact(
        content=content,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        backend="python_docx",
        citation_format=citation_format,
        references=references,
        lines=lines,
        style_limitation=(
            "DOCX font observations resolve document defaults, paragraph/base and character styles, and direct overrides. Unsupported style contexts remain unknown. Soft-break fragments do not establish reference-level hanging indentation."
        ),
        docx_paragraph_mode=True,
    )


def _build_artifact(
    *,
    content: bytes,
    media_type: str,
    backend: str,
    citation_format: str,
    references: list[ParsedReference],
    lines: list[_LayoutLine],
    style_limitation: str,
    docx_paragraph_mode: bool = False,
) -> ReferenceLayoutArtifact:
    location_kind = "paragraph" if docx_paragraph_mode else "page"
    headings = (
        {"reference", "references", "reference list", "reference section"}
        if citation_format == "apa"
        else {"works cited", "work cited", "works consulted", "sources", "bibliography"}
    )
    heading_indexes = [
        index for index, line in enumerate(lines) if _normalize(line.text) in headings
    ]
    inferred_start = None
    if not heading_indexes and references:
        inferred_start = _infer_first_reference_start(references[0], lines)
    if not heading_indexes and inferred_start is None:
        return ReferenceLayoutArtifact(
            content_sha256=hashlib.sha256(content).hexdigest(),
            media_type=media_type,
            extraction_backend=backend,
            location_kind=location_kind,
            citation_format=citation_format,
            status="not_assessed",
            heading_status="not_matched",
            reference_count=len(references),
            matched_reference_count=0,
            entries=[_unmatched(reference, "reference_heading_not_matched") for reference in references],
            limitations=[
                "The physical reference-section heading could not be matched; no layout finding is permitted."
            ],
        )

    heading_index = heading_indexes[-1] if heading_indexes else None
    candidate_lines = (
        lines[heading_index + 1 :] if heading_index is not None else lines[inferred_start:]
    )
    entries, matched = _match_entries(
        references,
        candidate_lines,
        style_limitation=style_limitation,
        docx_paragraph_mode=docx_paragraph_mode,
    )
    status = "complete" if matched == len(references) else "partial"
    return ReferenceLayoutArtifact(
        content_sha256=hashlib.sha256(content).hexdigest(),
        media_type=media_type,
        extraction_backend=backend,
        location_kind=location_kind,
        citation_format=citation_format,
        status=status,
        heading_status=(
            "matched" if heading_index is not None else "inferred_from_first_reference"
        ),
        heading_page_index=(
            lines[heading_index].location_index
            if heading_index is not None and location_kind == "page"
            else None
        ),
        reference_count=len(references),
        matched_reference_count=matched,
        entries=entries,
        limitations=[
            style_limitation,
            "Measurements describe the submitted rendering; no APA or MLA conformance conclusion is applied.",
            *(
                ["The physical section boundary was inferred from a high-confidence full match to the first parsed reference."]
                if heading_index is None
                else []
            ),
        ],
    )


def _infer_first_reference_start(
    reference: ParsedReference,
    lines: list[_LayoutLine],
) -> int | None:
    target = _normalize(reference.raw_ref)
    candidates: list[tuple[float, int]] = []
    for start in range(len(lines)):
        for end in range(start + 1, min(len(lines), start + 12) + 1):
            candidate = _normalize(" ".join(line.text for line in lines[start:end]))
            if candidate:
                candidates.append((SequenceMatcher(None, target, candidate).ratio(), start))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < 0.85:
        return None
    best_score, best_start = candidates[0]
    second_distinct = max(
        (score for score, start in candidates[1:] if start != best_start),
        default=0.0,
    )
    return best_start if second_distinct < best_score - 0.015 else None


def _match_entries(
    references: list[ParsedReference],
    lines: list[_LayoutLine],
    *,
    style_limitation: str,
    docx_paragraph_mode: bool,
) -> tuple[list[ReferenceEntryLayout], int]:
    results: list[ReferenceEntryLayout] = []
    cursor = 0
    for reference in references:
        target = _normalize(reference.raw_ref)
        candidates: list[tuple[float, int, int]] = []
        max_start = min(len(lines), cursor + 30)
        for start in range(cursor, max_start):
            max_window = 3 if docx_paragraph_mode else 12
            for end in range(start + 1, min(len(lines), start + max_window) + 1):
                candidate = _normalize(" ".join(line.text for line in lines[start:end]))
                if not candidate:
                    continue
                score = SequenceMatcher(None, target, candidate).ratio()
                candidates.append((score, start, end))
        candidates.sort(reverse=True)
        best = candidates[0] if candidates else None
        if best is None or best[0] < _MIN_MATCH_SCORE:
            results.append(_unmatched(reference, "reference_layout_text_not_matched"))
            continue
        score, start, end = best
        # A unique complete normalized entry is stronger than a near match
        # elsewhere (e.g. the same author/title/URL with a different date).
        # Multiple exact occurrences still compete and remain unplaced.
        if score == 1.0:
            candidates = [item for item in candidates if item[0] == 1.0]
        second_score = max(
            (item[0] for item in candidates[1:] if item[1] != start
             # A lower-scoring strict superset of an exact span is not a
             # competing occurrence (e.g. the same entry plus a page number).
             # Distinct exact occurrences still remain ambiguous.
             and not (score == 1.0 and item[0] < 1.0
                      and item[1] < start and item[2] >= end)),
            default=0.0,
        )
        if second_score >= score - 0.015:
            results.append(
                _unmatched(
                    reference,
                    "reference_layout_text_match_ambiguous",
                    status="ambiguous",
                    confidence=score,
                )
            )
            continue
        selected = lines[start:end]
        cursor = end
        first_x = selected[0].x0
        continuation_x = [line.x0 for line in selected[1:]]
        if not continuation_x and selected[0].continuation_x0 is not None:
            continuation_x = [selected[0].continuation_x0]
        continuation_median = (
            round(statistics.median(continuation_x), 3) if continuation_x else None
        )
        hanging = (
            round(continuation_median - first_x, 3)
            if continuation_median is not None
            else None
        )
        if not all(line.indent_complete for line in selected):
            continuation_median = hanging = None
        styled = sum(line.styled_characters for line in selected)
        text_style_spans, observed_ranges = _bind_style_evidence(reference.raw_ref, selected)
        results.append(
            ReferenceEntryLayout(
                reference_id=reference.reference_id,
                reference_text_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
                mapping_status="matched",
                match_confidence=round(score, 6),
                location_indexes=list(
                    dict.fromkeys(line.location_index for line in selected)
                ),
                line_count=len(selected),
                first_line_x_points=first_x,
                continuation_x_median_points=continuation_median,
                observed_hanging_indent_points=hanging,
                italic_character_fraction=(
                    round(sum(line.italic_characters for line in selected) / styled, 6)
                    if styled
                    else None
                ),
                bold_character_fraction=(
                    round(sum(line.bold_characters for line in selected) / styled, 6)
                    if styled
                    else None
                ),
                text_style_spans=text_style_spans,
                style_binding_version='reference-style-binding-v1',
                style_observation_ranges=observed_ranges,
                rectangles=[
                    ReferenceLayoutRectangle(
                        page_index=line.location_index,
                        x0=line.bbox[0],
                        y0=line.bbox[1],
                        x1=line.bbox[2],
                        y1=line.bbox[3],
                    )
                    for line in selected
                    if line.bbox is not None
                ],
                limitations=[style_limitation],
            )
        )
    return results, sum(item.mapping_status == "matched" for item in results)


def _unmatched(
    reference: ParsedReference,
    reason: str,
    *,
    status: Literal["ambiguous", "not_matched"] = "not_matched",
    confidence: float = 0.0,
) -> ReferenceEntryLayout:
    return ReferenceEntryLayout(
        reference_id=reference.reference_id,
        reference_text_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
        mapping_status=status,
        match_confidence=round(confidence, 6),
        line_count=0,
        limitations=[reason],
    )


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _styled_text(
    pieces,
) -> tuple[str, tuple[tuple[int, int, bool, bool], ...]]:
    # ``pieces`` is commonly a generator; materialize once before both passes.
    pieces = list(pieces)
    raw = "".join(value for value, _italic, _bold in pieces)
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    text = raw.strip()
    spans = []
    cursor = 0
    for value, italic, bold in pieces:
        piece_start, piece_end = cursor, cursor + len(value)
        cursor = piece_end
        start = max(piece_start, leading)
        end = min(piece_end, trailing)
        if start < end and (italic or bold):
            spans.append((start - leading, end - leading, italic, bold))
    return text, tuple(spans)


def _bind_style_spans(
    raw_reference: str,
    lines: list[_LayoutLine],
) -> list[ReferenceTextStyleSpan]:
    return _bind_style_evidence(raw_reference, lines)[0]


def _bind_style_evidence(raw_reference: str, lines: list[_LayoutLine]):
    """Map observed font runs onto the retained reference without guessing fields."""
    observed_parts = []
    observed_styles: list[tuple[bool, bool]] = []
    observed_known = []
    for line_index, line in enumerate(lines):
        if line_index:
            observed_parts.append(" ")
            observed_styles.append((False, False))
            observed_known.append(False)
        observed_parts.append(line.text)
        flags = [(False, False)] * len(line.text)
        for start, end, italic, bold in line.style_spans:
            for index in range(max(0, start), min(len(flags), end)):
                flags[index] = (italic, bold)
        observed_styles.extend(flags)
        observed_known.extend([line.style_complete] * len(line.text))
    observed = "".join(observed_parts)
    if not observed or not raw_reference:
        return [], []
    raw_flags = [(False, False)] * len(raw_reference)
    known = [False] * len(raw_reference)
    matcher = SequenceMatcher(None, raw_reference, observed, autojunk=False)
    for tag, raw_start, raw_end, observed_start, observed_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(raw_end - raw_start):
            raw_flags[raw_start + offset] = observed_styles[observed_start + offset]
            known[raw_start + offset] = observed_known[observed_start + offset]
    result = []
    start = 0
    while start < len(raw_flags):
        italic, bold = raw_flags[start]
        if not (italic or bold):
            start += 1
            continue
        end = start + 1
        while end < len(raw_flags) and raw_flags[end] == (italic, bold):
            end += 1
        result.append(
            ReferenceTextStyleSpan(
                start=start,
                end=end,
                italic=italic,
                bold=bold,
            )
        )
        start = end
    ranges = []
    index = 0
    while index < len(known):
        if not known[index]:
            index += 1
            continue
        end = index + 1
        while end < len(known) and known[end]:
            end += 1
        ranges.append(ReferenceStyleRange(start=index,end=end))
        index = end
    return result, ranges


def reference_span_style_observed(entry: ReferenceEntryLayout, raw_reference: str, start: int, end: int) -> bool:
    """Whether every non-space character has bound font observations, not a style verdict."""
    if (entry.style_binding_version != 'reference-style-binding-v1'
            or entry.reference_text_sha256 != hashlib.sha256(raw_reference.encode()).hexdigest()
            or not 0 <= start < end <= len(raw_reference)
            or not raw_reference[start:end].strip()):
        return False
    return all(any(r.start <= i < r.end for r in entry.style_observation_ranges)
        for i in range(start,end) if not raw_reference[i].isspace())


def _pdf_span_italic(span: dict) -> bool:
    return bool(int(span.get("flags", 0)) & 2) or bool(
        re.search(r"italic|oblique", span.get("font", ""), re.IGNORECASE)
    )


def _pdf_span_bold(span: dict) -> bool:
    return bool(int(span.get("flags", 0)) & 16) or bool(
        re.search(r"bold|black|semibold", span.get("font", ""), re.IGNORECASE)
    )


def _docx_run_italic(run) -> bool | None:
    return _docx_toggle(run, 'italic', 'i')


def _docx_run_bold(run) -> bool | None:
    return _docx_toggle(run, 'bold', 'b')


def _docx_toggle(run, attribute: str, tag: str) -> bool | None:
    direct = getattr(run, attribute)
    if direct is not None:
        return bool(direct)
    # Bounded paragraph/character style cascade, including document defaults.
    # Cycles or unresolved style references are unknown, never plain text.
    styles = run.part.styles
    value = False
    defaults = styles.element.find(qn('w:docDefaults'))
    if defaults is not None:
        node = defaults.find('.//' + qn('w:' + tag))
        if node is not None:
            value = node.get(qn('w:val'), '1').lower() not in {'0','false','off'}
    paragraph = run._r.getparent()
    while paragraph is not None and paragraph.tag != qn('w:p'):
        paragraph = paragraph.getparent()
    if paragraph is None:
        return None
    ppr = paragraph.find(qn('w:pPr'))
    if ppr is not None and ppr.find(qn('w:numPr')) is not None:
        return None  # Numbering/style interactions need separate acceptance.
    known_style_ids = {element.get(qn('w:styleId')) for element in styles.element}
    if paragraph.style and paragraph.style not in known_style_ids:
        return None
    if run._r.style and run._r.style not in known_style_ids:
        return None
    paragraph_style = run.part.get_style(paragraph.style, 1)
    for style in (paragraph_style, run.style):
        seen = set()
        while style is not None:
            if style.style_id in seen or len(seen) >= 64:
                return None
            seen.add(style.style_id)
            style_ppr = style.element.find(qn('w:pPr'))
            if style_ppr is not None and style_ppr.find(qn('w:numPr')) is not None:
                return None
            if getattr(style.font, attribute) is True:
                value = not value
            # An explicit unresolved basedOn cannot silently inherit normal.
            base_id = style.element.basedOn_val
            base = style.base_style
            if base_id and base is None:
                return None
            style = base
    return value


def _length_points(value) -> float:
    return 0.0 if value is None else float(value.pt)
