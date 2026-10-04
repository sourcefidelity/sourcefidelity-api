"""Stable, text-free citation anchors on a fixed PDF presentation surface."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import fitz
from pydantic import BaseModel, Field, model_validator

from app.services.schemas import InTextCitation


PRESENTATION_ANCHOR_VERSION = "presentation-anchor-v3"


class PresentationRectangle(BaseModel):
    page_index: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float


class CitationPresentationAnchor(BaseModel):
    anchor_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    citation_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passage_start: int = Field(ge=0)
    passage_end: int = Field(gt=0)
    semantic_paragraph_index: int = Field(ge=0)
    paragraph_context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mapping_status: str
    mapping_method: str | None = None
    localization_level: str
    match_count: int = Field(ge=0)
    page_indexes: list[int] = Field(default_factory=list)
    rectangles: list[PresentationRectangle] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mapping(self):
        if self.localization_level == "exact_rectangle" and (
            self.mapping_status != "matched"
            or self.match_count != 1
            or not self.rectangles
            or not self.mapping_method
        ):
            raise ValueError("Matched citation anchors require one bounded surface match")
        if self.localization_level == "page_only" and (
            not self.page_indexes or self.rectangles
        ):
            raise ValueError("Page-only anchors require pages and cannot carry rectangles")
        if self.localization_level in {"structural_only", "semantic_only"} and (
            self.page_indexes or self.rectangles
        ):
            raise ValueError("Non-page anchors cannot carry presentation geometry")
        return self


class PresentationAnchorArtifact(BaseModel):
    anchor_version: str = PRESENTATION_ANCHOR_VERSION
    presentation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str
    citation_count: int = Field(ge=0)
    matched_citation_count: int = Field(ge=0)
    page_localized_citation_count: int = Field(ge=0)
    structurally_localized_citation_count: int = Field(ge=0)
    anchors: list[CitationPresentationAnchor] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_anchor_ids(self):
        anchor_ids = [item.anchor_id for item in self.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("Presentation anchor IDs must be unique")
        return self


def bind_citations_to_pdf(
    pdf_bytes: bytes,
    *,
    citations: list[InTextCitation],
    paragraphs: list[str] | None = None,
    body_text: str | None = None,
) -> PresentationAnchorArtifact:
    """Bind exact citation-unit tokens to PDF words without retaining paper prose."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    words: list[
        tuple[
            str,
            int,
            int,
            int,
            tuple[float, float, float, float],
            bool,
            bool,
        ]
    ] = []
    try:
        for page_index, page in enumerate(document):
            for item in _sentence_boundary_words(page):
                token = _token(item[4])
                if token:
                    page_height = float(page.rect.height)
                    margin_noise = (
                        float(item[3]) <= 75.0
                        or float(item[1]) >= page_height - 48.0
                    )
                    words.append(
                        (
                            token,
                            page_index,
                            int(item[5]),
                            int(item[6]),
                            tuple(round(float(value), 3) for value in item[:4]),
                            str(item[4]).rstrip().endswith("-"),
                            margin_noise,
                        )
                    )
    finally:
        document.close()
    surface_tokens = [item[0] for item in words]
    unique_citations: list[InTextCitation] = []
    seen_locations: set[tuple[int, int, str, int]] = set()
    for citation in citations:
        if citation.passage_start < 0 or citation.passage_end <= citation.passage_start:
            continue
        key = (
            citation.passage_start,
            citation.passage_end,
            hashlib.sha256(citation.text.encode("utf-8")).hexdigest(),
            citation.paragraph_index,
        )
        if key in seen_locations:
            continue
        seen_locations.add(key)
        unique_citations.append(citation)
    anchors = [
        _bind_one(
            surface_tokens,
            words,
            citation,
            paragraph=(
                paragraphs[citation.paragraph_index]
                if paragraphs is not None
                and 0 <= citation.paragraph_index < len(paragraphs)
                else None
            ),
            body_text=body_text,
        )
        for citation in unique_citations
    ]
    matched = sum(anchor.mapping_status == "matched" for anchor in anchors)
    page_localized = sum(bool(anchor.page_indexes) for anchor in anchors)
    structurally_localized = sum(
        anchor.localization_level == "structural_only" for anchor in anchors
    )
    return PresentationAnchorArtifact(
        presentation_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        status=(
            "complete" if matched == len(anchors)
            else "partial" if matched
            else "not_assessed"
        ),
        citation_count=len(anchors),
        matched_citation_count=matched,
        page_localized_citation_count=page_localized,
        structurally_localized_citation_count=structurally_localized,
        anchors=anchors,
        limitations=[
            "Exact citation geometry uses unique normalized text. Duplicate detections of the same exact paper span are collapsed to one surface anchor. Unique exact paragraph context may disambiguate a citation or establish page-only localization; unresolved geometry never suppresses semantic evidence."
        ],
    )


def _bind_one(
    surface_tokens,
    words,
    citation: InTextCitation,
    *,
    paragraph: str | None,
    body_text: str | None = None,
) -> CitationPresentationAnchor:
    target = [_token(value) for value in re.findall(r"\S+", citation.text)]
    target = [value for value in target if value]
    matches, mapping_method = (
        _token_matches(surface_tokens, words, target) if target else ([], None)
    )
    digest = hashlib.sha256(citation.text.encode("utf-8")).hexdigest()
    anchor_id = hashlib.sha256(
        f"{citation.passage_start}:{citation.passage_end}:{digest}".encode("ascii")
    ).hexdigest()
    paragraph_digest = (
        hashlib.sha256(paragraph.encode("utf-8")).hexdigest() if paragraph else None
    )
    paragraph_match = None
    if paragraph:
        paragraph_target = [_token(value) for value in re.findall(r"\S+", paragraph)]
        paragraph_target = [value for value in paragraph_target if value]
        paragraph_matches, _ = _token_matches(
            surface_tokens, words, paragraph_target
        ) if paragraph_target else ([], None)
        if len(paragraph_matches) == 1:
            paragraph_match = paragraph_matches[0]
            if len(matches) > 1:
                bounded = [
                    match for match in matches
                    if paragraph_match[0] <= match[0] and match[1] <= paragraph_match[1]
                ]
                if len(bounded) == 1:
                    matches = bounded
                    mapping_method = "paragraph_context_disambiguation"
    if len(matches) > 1 and body_text:
        # A passage the paper repeats word for word (paper 4 pasted one
        # paragraph twice): the k-th copy in the text is the k-th on the page,
        # when the counts agree (2026-10-02).
        words_pattern = r"\s+".join(re.escape(value) for value in citation.text.split())
        copies = [m.start() for m in re.finditer(words_pattern, body_text)] if words_pattern else []
        own = [k for k, at in enumerate(copies) if at <= citation.passage_start < at + len(citation.text) + 8]
        if len(copies) == len(matches) and len(own) == 1:
            matches = [sorted(matches)[own[0]]]
            mapping_method = "body_order_disambiguation"
    if len(matches) != 1:
        paragraph_pages = (
            _page_indexes(words[paragraph_match[0] : paragraph_match[1]])
            if paragraph_match is not None
            else []
        )
        return CitationPresentationAnchor(
            anchor_id=anchor_id,
            citation_text_sha256=digest,
            passage_start=citation.passage_start,
            passage_end=citation.passage_end,
            semantic_paragraph_index=citation.paragraph_index,
            paragraph_context_sha256=paragraph_digest,
            mapping_status="ambiguous" if len(matches) > 1 else "not_matched",
            mapping_method=mapping_method,
            localization_level=(
                "page_only" if paragraph_pages
                else "structural_only" if paragraph_digest
                else "semantic_only"
            ),
            match_count=len(matches),
            page_indexes=paragraph_pages,
        )
    selected = words[matches[0][0] : matches[0][1]]
    if mapping_method in {"normalized_tokens_with_margin_skip", "compact_alphanumeric_with_margin_skip"}:
        # The skipped running header/footer words occupy the same contiguous
        # surface slice as the page-spanning body text. They must not become
        # citation rectangles merely because they were bypassed for matching.
        selected = [item for item in selected if not item[6]]
    rectangles = _rectangles(selected)
    page_indexes = list(dict.fromkeys(item[0] for item in rectangles))
    return CitationPresentationAnchor(
        anchor_id=anchor_id,
        citation_text_sha256=digest,
        passage_start=citation.passage_start,
        passage_end=citation.passage_end,
        semantic_paragraph_index=citation.paragraph_index,
        paragraph_context_sha256=paragraph_digest,
        mapping_status="matched",
        mapping_method=mapping_method,
        localization_level="exact_rectangle",
        match_count=1,
        page_indexes=page_indexes,
        rectangles=[
            PresentationRectangle(
                page_index=item[0],
                x0=item[3][0],
                y0=item[3][1],
                x1=item[3][2],
                y1=item[3][3],
            )
            for item in rectangles
        ],
    )


def _rectangles(selected):
    rectangles = []
    for _, page_index, block_index, line_index, box, _hyphenated, _margin_noise in selected:
        if rectangles and rectangles[-1][0:3] == (page_index, block_index, line_index):
            prior = rectangles[-1]
            rectangles[-1] = (
                page_index,
                block_index,
                line_index,
                (
                    min(prior[3][0], box[0]),
                    min(prior[3][1], box[1]),
                    max(prior[3][2], box[2]),
                    max(prior[3][3], box[3]),
                ),
            )
        else:
            rectangles.append((page_index, block_index, line_index, box))
    return rectangles


def _page_indexes(selected) -> list[int]:
    return list(dict.fromkeys(item[1] for item in selected))


def _token_matches(surface_tokens, words, target) -> tuple[list[tuple[int, int]], str | None]:
    """Find exact sequences while permitting visible layout-only interruptions.

    Running headers and footers sit between two body fragments when a citation
    sentence crosses a PDF page boundary. They are presentation structure, not
    part of the semantic sentence, so a bounded run of margin words may be
    skipped while the complete normalized citation token sequence must still
    match exactly.
    """
    matches: list[tuple[int, int]] = []
    used_margin_skip = False
    for start in range(len(surface_tokens)):
        surface_index = start
        target_index = 0
        margin_skip_count = 0
        while surface_index < len(surface_tokens) and target_index < len(target):
            if surface_tokens[surface_index] == target[target_index]:
                surface_index += 1
                target_index += 1
                continue
            if (
                words[surface_index][5]
                and surface_index + 1 < len(surface_tokens)
                and surface_tokens[surface_index] + surface_tokens[surface_index + 1]
                == target[target_index]
            ):
                surface_index += 2
                target_index += 1
                continue
            if (
                target_index > 0
                and words[surface_index][6]
                and margin_skip_count < 32
            ):
                surface_index += 1
                margin_skip_count += 1
                continue
            break
        if target_index == len(target):
            matches.append((start, surface_index))
            used_margin_skip = used_margin_skip or margin_skip_count > 0
    if matches:
        return matches, (
            "normalized_tokens_with_margin_skip"
            if used_margin_skip
            else "normalized_tokens"
        )

    # PDF extractors can divide one visible word into several tokens (or join
    # tokens) even when the rendered characters are identical. Permit a second
    # exact mapping only when the complete normalized alphanumeric sequence
    # begins and ends on extracted-word boundaries.
    compact_target = "".join(target)
    if len(compact_target) < 20:
        return [], None

    def compact(indexes):
        surface = "".join(surface_tokens[i] for i in indexes)
        starts: dict[int, int] = {}
        ends: dict[int, int] = {}
        offset = 0
        for index in indexes:
            starts[offset] = index
            offset += len(surface_tokens[index])
            ends[offset] = index + 1
        found_matches = []
        cursor = 0
        while True:
            found = surface.find(compact_target, cursor)
            if found < 0:
                break
            finish = found + len(compact_target)
            if found in starts and finish in ends:
                found_matches.append((starts[found], ends[finish]))
            cursor = found + 1
        return found_matches

    compact_matches = compact(range(len(surface_tokens)))
    if compact_matches:
        return compact_matches, "compact_alphanumeric"
    # A paragraph crossing a page break carries the running header and page
    # number between its halves; the compact reading skips them as the
    # token reading does (paper 7, 2026-10-04).
    body_indexes = [i for i in range(len(surface_tokens)) if not words[i][6]]
    compact_matches = compact(body_indexes)
    return compact_matches, ("compact_alphanumeric_with_margin_skip" if compact_matches else None)


def _token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("\u00ad", "")
    return "".join(character for character in normalized if character.isalnum())


def reading_order(words):
    """Words of one PDF line left to right. The y-then-x sort puts a word set a
    point higher ("2020)." in paper 9) at the start of its line; PyMuPDF's own
    block and line numbers keep it in its line (2026-10-04)."""
    first: dict = {}
    for index, word in enumerate(words):
        first.setdefault((word[5], word[6]), index)
    return sorted(words, key=lambda word: (first[(word[5], word[6])], word[0]))


def _sentence_boundary_words(page):
    """Split a missing-space sentence join using exact PDF character boxes.

    Never estimate a partial word's width or accept an alphanumeric prefix.
    Unrecoverable character geometry leaves the original conservative word.
    """
    chars = None
    for word in reading_order(page.get_text('words', sort=True)):
        boundaries = [m.end() for m in re.finditer(r'[.!?](?=[A-Z])', word[4])]
        if not boundaries:
            yield word
            continue
        if chars is None:
            chars = [c for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                     for span in line.get('spans', []) for c in span.get('chars', [])]
        box = fitz.Rect(word[:4])
        observed = [c for c in chars if box.contains(fitz.Point((c['bbox'][0]+c['bbox'][2])/2,
                                                               (c['bbox'][1]+c['bbox'][3])/2))]
        if ''.join(c['c'] for c in observed) != word[4]:
            yield word
            continue
        start = 0
        for end in boundaries + [len(word[4])]:
            pieces = observed[start:end]
            rect = fitz.Rect(pieces[0]['bbox'])
            for c in pieces[1:]:
                rect |= fitz.Rect(c['bbox'])
            yield (*rect, word[4][start:end], *word[5:])
            start = end
