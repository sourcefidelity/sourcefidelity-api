"""Sentence splitting utility (regex-based, no NLTK dependency).

Splits text into sentences while respecting common abbreviations that would
cause false splits (Dr., Mr., i.e., e.g., U.S., etc.). Preserves paragraph
structure — paragraphs are split first (on double newlines), then each
paragraph is split into sentences.
"""

import re

# Abbreviations that end with a period but don't end a sentence.
# Matched case-insensitively before a period + space.
_ABBREVIATIONS = re.compile(
    r"\b("
    r"Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Hon|Capt|Lt|Sgt|Col|Gen|Bros"
    r"|i\.e|e\.g|etc|vs|cf|ca"
    r"|a\.m|p\.m|sec|min|hr"
    r"|B\.A|M\.A|Ph\.D|B\.S|M\.S|M\.D|Ed\.D|J\.D"
    r"|U\.S|U\.K|E\.U|U\.N|D\.C|N\.Y|L\.A"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r"|No|Vol|pp|ed|eds|trans|repr"
    r"|p"
    r")\.\s",
    re.IGNORECASE,
)

# Sentence-ending pattern: . ! ? (optionally followed by closing quote/paren).
# A closing double quotation mark directly followed by a capital also ends a
# sentence: students drop the space ("individual.\u201dDifferent", 2026-10-14).
_SENTENCE_END = re.compile(r"([.!?]+[\"'\u201d\u2019\)]*)(?:\s+|(?<=\)\.)(?=[A-Z])|(?<=[\u201d\"])(?=[A-Z]))")

# Han, kana and Hangul sentences end at \u3002\uff01\uff1f, a closing quotation mark or
# bracket staying with its sentence; any spacing after it (none, a space or a
# line break) belongs to neither sentence. The one rule every splitter uses:
# group 1 ends the sentence, the match's end starts the next (2026-10-14).
CJK_SENTENCE_END = re.compile(r"([\u3002\uff01\uff1f]+[\u201d\u2019\u300d\u300f\uff09\"')]*)(\s*)")


def split_cjk(text: str) -> list[str]:
    """A text cut at Han, kana and Hangul sentence ends (stripped pieces)."""
    pieces, start = [], 0
    for match in CJK_SENTENCE_END.finditer(text):
        if match.end(1) >= len(text.rstrip()):
            break
        pieces.append(text[start:match.end(1)].strip())
        start = match.end()
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def split_sentences(text: str) -> list[str]:
    """Split text into a list of sentences.

    Handles abbreviations, decimal numbers, and ellipses.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviations: replace ALL periods in matched abbreviations with
    # a placeholder so they don't trigger sentence-end detection.
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "\x00"), text)

    # Protect decimal numbers (3.14, p. 45)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x01" + m.group(2), protected)

    # Protect ellipses
    protected = protected.replace("...", "\x02")
    # Question/exclamation punctuation inside a title or quoted clause is
    # not a sentence boundary when the same sentence explicitly continues.
    protected = re.sub(
        r'([?!])(?=(?:[”"]\s+(?:and|but|while)\b|[ \t]+\([^()\n]{1,120}(?:19|20)\d{2}[^()\n]*\)[ \t]+[a-z]))',
        lambda m: '\x03' if m[1] == '?' else '\x04', protected)
    # A trailing secondary citation belongs to the preceding question or
    # exclamation, even when the student omitted quotation marks. Do not
    # attach a new parenthetical sentence or cross a paragraph boundary.
    protected = re.sub(
        r'([?!])(?=(?![ \t\n]*\n[ \t]*\n)[ \t\n]+\(as cited in [^()\n]{1,120},\s*(?:18|19|20)\d{2}[a-z]?\)\.)',
        lambda m: '\x03' if m[1] == '?' else '\x04', protected, flags=re.I)
    # A quotation's own citation after its closing mark ends that sentence:
    # “…?” (Author, 1930, p. 4). (owner review 2026-10-14, Harlow citation 5).
    protected = re.sub(
        r'([.?!])(?=[”"’][ \t]+\([^()\n]{1,120}(?:1[6-9]|20)\d{2}[^()\n]*\)[.!?])',
        lambda m: {'?': '\x03', '!': '\x04', '.': '\x00'}[m[1]], protected)

    # Protect "et al." — it's followed by a parenthetical, not a sentence end
    protected = re.sub(r"(et al)\.(?=\s*\()", lambda m: m.group(1) + "\x00", protected, flags=re.IGNORECASE)

    # Protect a middle initial inside a conventional person name. PDF text
    # frequently exposes "Edward C. Judson" as ordinary prose; treating the
    # initial as a sentence end truncates the citation unit at "Edward C.".
    protected = re.sub(
        r"\b([A-Z][a-z]{1,30}\s+[A-Z])\.(?=\s+[A-Z][a-z]{1,30}\b)",
        lambda m: m.group(1) + "\x00",
        protected,
    )

    # Split on sentence-ending punctuation followed by whitespace
    parts = _SENTENCE_END.split(protected)

    # Reassemble: odd-indexed parts are punctuation, even are text
    sentences: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if i + 1 < len(parts):
            chunk += parts[i + 1]
            i += 2
        else:
            i += 1

        # Restore placeholders
        chunk = chunk.replace("\x00", ".").replace("\x01", ".").replace("\x02", "...")
        chunk = chunk.replace('\x03', '?').replace('\x04', '!')
        chunk = chunk.strip()
        if chunk:
            sentences.extend(split_cjk(chunk))

    return sentences


def split_paragraphs_and_sentences(text: str) -> list[list[str]]:
    """Split text into paragraphs, then each paragraph into sentences.

    Returns:
        A list of paragraphs, where each paragraph is a list of sentence strings.
    """
    if not text or not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    result: list[list[str]] = []
    for para in paragraphs:
        para = para.strip()
        if para:
            sentences = split_sentences(para)
            if sentences:
                result.append(sentences)
    return result


# Source text keeps exact offsets: one rule for every source-side splitter
# (the evidence foundation, patchwriting), so a sentence has one identity
# everywhere (rules audit 2026-10-18: the two had drifted, and neither knew
# "Dr." or "et al." were not sentence ends).
_SOURCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-Z0-9\"'‘“(])|\n{2,}")
_SOURCE_NOT_AN_END = re.compile(
    r"(?:\b[A-Z]\.|\b(?:Dr|Mr|Mrs|Ms|Prof|St|Rev|vs|cf|ca|pp|Vol|vol|eds|trans|repr|Ph\.D|i\.e|e\.g|U\.S|U\.K)\."
    r"|\bet al\.|\bp\.)[\"'’”)]*\s*$")


def source_sentence_spans(text: str, extra_cuts: list[tuple[int, int]] = ()) -> list[tuple[int, int]]:
    """Exact (start, end) sentence spans of source text.

    A boundary is a sentence end followed by a capital, digit, quotation mark
    or bracket, or a blank line; an initial or a common abbreviation before it
    is not an end. Han, kana and Hangul sentence ends cut whatever follows.
    `extra_cuts` are (start, end) intervals that end the sentence before them
    and belong to no sentence (a front-matter label line).
    """
    cuts = [(m.start(), m.end(), True) for m in _SOURCE_BOUNDARY.finditer(text)]
    cuts += [(m.end(1), m.end(), False) for m in CJK_SENTENCE_END.finditer(text)
             if m.end(1) < len(text.rstrip())]
    cuts += [(start, end, False) for start, end in extra_cuts]
    spans, start = [], 0
    for cut_start, cut_end, checked in sorted(cuts):
        if cut_start < start:
            continue
        if checked and _SOURCE_NOT_AN_END.search(text, max(0, cut_start - 12), cut_start):
            continue
        piece = _trimmed(text, start, cut_start)
        if piece:
            spans.append(piece)
        start = cut_end
    piece = _trimmed(text, start, len(text))
    if piece:
        spans.append(piece)
    return spans


def _trimmed(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None
