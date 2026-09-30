"""Recognize a contents listing supplied in place of an abstract.

Some catalog and publisher metadata records put a book's table of contents in
the field a scholarly service exposes as the abstract. The report then presents
chapter headings as if they were a summary of the work, which tells the reader
nothing about the cited content and looks like retrieved evidence.

The test here is structural and deterministic: a contents listing enumerates
headings, while an abstract is prose. It deliberately requires several agreeing
signals, because wrongly discarding a real abstract costs the reader more than
showing an unhelpful one.
"""
import re

POLICY_VERSION = "abstract-contents-listing-v1"

# "1. Overview", "2, The Company Town", "10. Strategic Behavior". A decimal
# number inside prose ("increased 3.5 percent") never matches, because the
# digits must be followed by a separator and a capitalized word.
_NUMBERED_HEADING_RE = re.compile(r"(?<![\d.,])(\d{1,2})[.,]\s+(?=[A-Z])")

# "Part I", "part ii", "Chapter IV", and bare roman enumerations "II. MARKET".
_PART_RE = re.compile(
    r"\b(?:part|chapter|section|book|volume)\s+(?:[ivxlc]+|\d{1,2})\b", re.IGNORECASE
)
_ROMAN_HEADING_RE = re.compile(r"(?<![A-Za-z])([IVX]{1,5})\.\s+(?=[A-Z])")

_MATTER_WORDS = (
    "acknowledgment", "acknowledgement", "table of contents", "contents",
    "preface", "foreword", "afterword", "epilogue", "prologue", "index",
    "bibliography", "glossary", "appendix", "notes on contributors",
    "about the author", "further reading", "list of illustrations",
    "list of figures", "list of tables", "endnotes",
)

# Prose runs on connectives; a heading list does not.
_FUNCTION_WORD_RE = re.compile(
    r"\b(?:the|this|that|these|those|is|are|was|were|has|have|had|which|"
    r"while|because|however|although|we|our|they|their|it|its|argues|shows|"
    r"finds|examines|suggests|paper|article|study|chapter's)\b",
    re.IGNORECASE,
)


def _front_matter_hits(text: str) -> int:
    lowered = text.casefold()
    return sum(1 for word in _MATTER_WORDS if word in lowered)


def describe(text: str) -> dict:
    """Report the structural signals without deciding, for inspection."""
    value = " ".join(str(text or "").split())
    words = value.split()
    numbered = len(_NUMBERED_HEADING_RE.findall(value))
    roman = len(_ROMAN_HEADING_RE.findall(value)) + len(_PART_RE.findall(value))
    matter = _front_matter_hits(value)
    function_words = len(_FUNCTION_WORD_RE.findall(value))
    density = function_words / len(words) if words else 0.0
    return {
        "policy_version": POLICY_VERSION,
        "numbered_headings": numbered,
        "part_markers": roman,
        "front_matter_cues": matter,
        "function_word_density": round(density, 4),
        "word_count": len(words),
    }


def looks_like_contents_listing(text: str) -> bool:
    """True when the supplied text enumerates headings instead of summarizing.

    Prose density is the guard. An abstract that happens to number its findings
    ("We report three results. 1. ... 2. ...") keeps a normal share of function
    words and is not rejected.
    """
    signals = describe(text)
    if signals["word_count"] < 12:
        return False
    enumerated = signals["numbered_headings"]
    parts = signals["part_markers"]
    matter = signals["front_matter_cues"]
    structural = parts + matter
    # Chapter titles are made of ordinary words, so a long contents listing can
    # reach prose-like function-word density. Where the enumeration itself is
    # unambiguous, do not let that density rescue it.
    if enumerated >= 6 or (enumerated >= 3 and structural >= 2):
        return True
    if signals["function_word_density"] >= 0.11:
        return False
    return enumerated >= 4 or (enumerated >= 2 and structural >= 1) or structural >= 4
