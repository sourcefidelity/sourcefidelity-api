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
    r"Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Hon|Capt|Lt|Sgt|Col|Gen"
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

# Sentence-ending pattern: . ! ? (optionally followed by closing quote/paren)
_SENTENCE_END = re.compile(r"([.!?]+[\"'\u201d\u2019\)]*)\s+")


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

    # Protect "et al." — it's followed by a parenthetical, not a sentence end
    protected = re.sub(r"(et al)\.(?=\s*\()", lambda m: m.group(1) + "\x00", protected, flags=re.IGNORECASE)

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
        chunk = chunk.strip()
        if chunk:
            sentences.append(chunk)

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
