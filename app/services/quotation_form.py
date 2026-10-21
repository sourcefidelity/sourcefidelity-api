"""Is a citation only a quotation, or a claim with quoted words in it?

Owner rule (2026-10-12): a citation that is only a quotation -- the quoted
words plus an attribution or signal phrase -- needs no Judgment; quotation and
locator checks still run. A statement that embeds quoted words in the
student's own claim gets both. Deterministic and conservative: anything
uncertain is ``embedded``, so both checks run.

Classes: ``quotation_only``, ``embedded``, ``title`` (every quoted span is a
reference-list title, so nothing is quoted) and ``none`` (no quotation marks).
"""
from __future__ import annotations

import re

QUOTATION_FORM_VERSION = "quotation-form-v1"

# Sentence-opening connectives (owner list).
_OPENING_CONNECTIVES = re.compile(
    r"^\W*(?:in\s+addition|moreover|furthermore|firstly|however)\b\W*", re.IGNORECASE)
# Words a signal frame may hold besides names and the reporting verb:
# "According to X", "As X puts it", "In X's words", "X was called ... and ...".
_FRAME_WORDS = frozenset({
    "according", "to", "as", "in", "the", "a", "an", "of", "by", "and", "or", "that", "who",
    "also", "she", "he", "they", "it", "this", "these", "her", "his", "their", "its",
    "words", "et", "al", "author", "authors",
})
_COPULAS = frozenset({"is", "are", "was", "were", "be", "been"})
# A small closed reporting-verb list. Any other word outside the quotation is
# the student's own predicate, so the quotation is embedded in a claim.
_REPORTING_VERBS = frozenset({
    "states", "stated", "argues", "argued", "writes", "wrote", "notes", "noted",
    "describes", "described", "calls", "called", "defines", "defined", "puts",
    "says", "said", "claims", "claimed", "observes", "observed", "explains", "explained",
    "asserts", "asserted", "contends", "contended", "remarks", "remarked",
    "termed", "labelled", "labeled",
})
# Reporting verbs a narrative quotation's author may take ("As Smith (2020)
# emphasized, “…”"): one list for the quotation form and the locator rule
# (2026-10-18; the locator rule's own shorter list missed "emphasized").
NARRATIVE_REPORTING_VERBS = _REPORTING_VERBS | frozenset({
    "emphasizes", "emphasized", "emphasises", "emphasised", "suggests", "suggested",
    "highlights", "highlighted", "points", "pointed", "reports", "reported", "maintains", "maintained",
    "insists", "insisted", "acknowledges", "acknowledged", "warns", "warned", "concludes", "concluded",
})
# Only these follow a copula in a signal frame ("was called", "is described as").
_PASSIVE_REPORTING = frozenset({"called", "described", "defined", "termed", "labelled", "labeled"})
_MAX_CONTENT_WORDS = 2

_DOUBLE_OPEN, _DOUBLE_CLOSE = "“", "”"
_SINGLE_OPEN, _SINGLE_CLOSE = "‘", "’"


class _Unbalanced(ValueError):
    pass


def _double_spans(text: str) -> list[tuple[int, int]]:
    spans, start, straight = [], None, None
    for index, char in enumerate(text):
        if char == _DOUBLE_OPEN:
            if start is not None:
                raise _Unbalanced
            start = index
        elif char == _DOUBLE_CLOSE:
            if start is None:
                raise _Unbalanced
            spans.append((start, index + 1))
            start = None
        elif char == '"':
            if start is not None and straight:
                spans.append((start, index + 1))
                start, straight = None, None
            elif start is None:
                start, straight = index, True
            else:
                raise _Unbalanced
    if start is not None:
        raise _Unbalanced
    return spans


def _single_spans(text: str, outside: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Single-quoted spans outside double quotes. A closing mark with nothing
    open is an apostrophe ("authors’ view"); an opening mark left open is
    unbalanced."""
    inside = lambda i: any(a <= i < b for a, b in outside)
    spans, start = [], None
    for index, char in enumerate(text):
        if char not in {"'", _SINGLE_OPEN, _SINGLE_CLOSE} or inside(index):
            continue
        before = text[index - 1] if index else " "
        after = text[index + 1] if index + 1 < len(text) else " "
        if before.isalnum() and after.isalnum():
            continue  # don't, Smith’s
        opening = char == _SINGLE_OPEN or (char == "'" and not before.isalnum() and not after.isspace())
        if opening and start is None and not (char == "'" and before in ".,;:!?"):
            start = index
        elif start is not None and not before.isspace():
            spans.append((start, index + 1))
            start = None
    if start is not None:
        raise _Unbalanced
    return spans


def quoted_spans(text: str) -> list[tuple[int, int]]:
    """Every quoted span, marks included: double quotes, and single quotes
    outside them (an apostrophe is not a quote). The one reading of "what is
    quoted" for the quotation form, the quotation check and the exact-quote
    search (rules audit 2026-10-18; owner decision: single-quoted quotations
    get the quotation and locator checks). Unbalanced marks fall back to
    complete double-quoted spans."""
    text = str(text or "")
    try:
        double = _double_spans(text)
        return sorted(double + _single_spans(text, double))
    except _Unbalanced:
        return [m.span() for m in re.finditer(r"“[^”]{2,}”|\"[^\"]{2,}\"", text)]


def quoted_words(text: str, *, marker_spans=(), titles=()) -> list[tuple[int, int]]:
    """Quoted spans that quote someone: not a reference-list title and not
    inside a citation marker (a quoted title used as the marker)."""
    known_titles = {_normal(t) for t in titles if len(_normal(t).split()) >= 2}
    return [(a, b) for a, b in quoted_spans(text)
            if _normal(text[a + 1:b - 1]) not in known_titles
            and not any(start <= a and b <= end for start, end in marker_spans)]


def claim_type_for(claim, references) -> str:
    """A statement quoting someone is a quotation; one whose quoted words are
    all titles is not; a block quotation without marks stays one."""
    text = str(getattr(claim, "text", "") or "")
    markers = [(m.local_start, m.local_end) for m in getattr(claim, "citation_markers", []) or []]
    titles = [getattr(r, "title", "") for r in references or ()]
    if quoted_words(text, marker_spans=markers, titles=titles):
        return "quotation"
    if quoted_spans(text):
        return "paraphrase"   # every quoted span is a title
    return claim.claim_type


def _normal(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


def classify_quotation_form(text: str, *, marker_spans=(), names=(), titles=()) -> str:
    """``quotation_only``, ``embedded``, ``title`` or ``none`` for one citation.

    ``marker_spans`` are the citation markers' (start, end) in ``text``;
    ``names`` the cited works' author names (removed with the markers);
    ``titles`` the reference-list titles.
    """
    text = str(text or "")
    if not re.search(r"[\"“”‘']|’(?!\w)", text):
        return "none"
    try:
        double = _double_spans(text)
        spans = sorted(double + _single_spans(text, double))
    except _Unbalanced:
        return "embedded"
    if not spans:
        return "none"
    known_titles = {_normal(t) for t in titles if len(_normal(t).split()) >= 2}
    quoted = [(a, b) for a, b in spans if _normal(text[a + 1:b - 1]) not in known_titles]
    if not quoted:
        return "title"
    if not any(re.search(r"[^\W\d_]", text[a + 1:b - 1]) for a, b in quoted):
        return "embedded"
    # Everything outside the quotations and the citation markers.
    keep = list(text)
    for start, end in [*quoted, *marker_spans]:
        for index in range(max(0, start), min(len(keep), end)):
            keep[index] = " "
    outside = "".join(keep)
    outside = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", outside)
    if re.search(r"[()\[\]]", outside):
        return "embedded"
    outside = _OPENING_CONNECTIVES.sub(" ", outside)
    name_words = {w for name in names for w in re.findall(r"[^\W\d_]{2,}", str(name or ""))
                  if w[:1].isupper()}
    words = [re.sub(r"[’']s$", "", w) for w in re.findall(r"[^\W\d_]+(?:[’'][^\W\d_]+)*", outside)]
    words = [w for w in words if w and w not in name_words]
    content = 0
    for index, word in enumerate(words):
        lowered = word.casefold()
        if lowered in _FRAME_WORDS:
            continue
        if lowered in _COPULAS:
            following = [w.casefold() for w in words[index + 1:] if w.casefold() != "also"]
            if not following or following[0] not in _PASSIVE_REPORTING:
                return "embedded"
            continue
        if lowered not in _REPORTING_VERBS:
            return "embedded"
        content += 1
    return "quotation_only" if content <= _MAX_CONTENT_WORDS else "embedded"


def claim_quotation_form(claim, references) -> str:
    """Classify a stored citation claim against the paper's reference list."""
    linked = {getattr(r, "reference_id", None): r for r in references or ()}
    names = [getattr(linked.get(rid), "author", "") for rid in getattr(claim, "reference_ids", []) or []]
    return classify_quotation_form(
        claim.text,
        marker_spans=[(m.local_start, m.local_end) for m in getattr(claim, "citation_markers", []) or []],
        names=names,
        titles=[getattr(r, "title", "") for r in references or ()],
    )
