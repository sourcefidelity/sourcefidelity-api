"""Page-level text-quality check for extracted source text (owner request 2026-09-28).

An embedded OCR layer, or a PDF with broken font encoding, can yield text that
reads "tl1e", "witl1", "fom1s", "I-lollywood". Evidence built from it misleads
the judge and the student and weakens retrieval. This module scores one page's
text; `page_ocr_repair` re-OCRs the pages it flags, and a page that stays
damaged makes the source unusable.

The score is the share of checkable words that show typical text damage,
using the SCOWL word lists the image installs (`/usr/share/dict`):

- an unknown word that becomes an English word under one OCR character
  confusion ("tlie" -> "the", "wouJd" -> "would", "irnportant" -> "important");
- a fragment left by a dropped ligature ("ndings" for "findings", "e ciency");
- a word that mixes letters and digits ("tl1e", "fom1s").

Unknown words that are merely unknown (jargon, names, URL parts, other
languages) do not count: on clean born-digital pages they are common, and a
plain unknown-word share was measured to flag 131 clean pages while missing
most of a damaged scan. Only lowercase words are checked.

A page is assessed only when it is plainly English prose: enough checkable
words and enough English function words. Other pages are `not_assessed`, never
damaged, so a French source is not rejected for being French.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

TEXT_QUALITY_VERSION = "page-text-quality-v1"
WORD_LIST_PATHS = ("/usr/share/dict/american-english-large", "/usr/share/dict/british-english-large")
# Calibrated 2026-09-28 on all 3,134 assessed pages of the 57 stored PDFs: clean
# born-digital pages score 0 (median), Thompson's damaged scan pages 1-7%, and
# 41 of 42 sampled flagged pages score under 1% after local re-OCR. See STATE.
DAMAGED_WORD_SHARE = 0.01
MIN_CHECKED_WORDS = 50
MIN_FUNCTION_WORD_SHARE = 0.15

_FUNCTION_WORDS = frozenset(
    "the of and to in a is that for it as with was on be by are this which from or at an not have has "
    "were their but its they these also can been more than such other into".split())
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")
# Links and identifiers are not prose: their fragments are neither words nor damage.
_NOT_PROSE = re.compile(r"(?:https?://|www\.|doi:|10\.\d{4,9}/)\S*|\S+\.(?:com|org|edu|net|gov|html?|pdf|aspx?|php)\S*|\S*[%=?&_/@#]\S*",
                        re.IGNORECASE)
# Bound prefixes and abbreviations that the word lists hold only inside longer words.
_ALSO_KNOWN = frozenset("doi pre non post anti sub inter multi co etc vol eds pp ibid al cf esp approx fig".split())
_LINE_BREAK_HYPHEN = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([a-z])")
# Letter-digit mixes that are ordinary: 1920s, 21st, MP3, 3D, p2.
_ORDINARY_MIXED = re.compile(r"^(\d{2,4}s|\d+(st|nd|rd|th)|[A-Z]+\d+[A-Za-z]?|\d+[A-Z]+|\d+[a-z]{1,2}|[a-z]\d{1,2})$")

PageStatus = Literal["clean", "damaged", "not_assessed"]


class WordListUnavailable(RuntimeError):
    """The word lists are missing, so no page can be assessed."""


@dataclass(frozen=True)
class PageTextQuality:
    status: PageStatus
    damaged_share: float | None
    checked_words: int
    reason: str


@lru_cache(maxsize=1)
def _word_list() -> tuple[frozenset[str], str]:
    words: set[str] = set()
    digest = hashlib.sha256()
    for path in WORD_LIST_PATHS:
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        digest.update(data)
        for line in data.decode("utf-8", errors="replace").splitlines():
            word = line.strip().casefold()
            if word:
                words.add(word[:-2] if word.endswith("'s") else word)
    if not words:
        raise WordListUnavailable("No English word list is installed.")
    return frozenset(words), digest.hexdigest()


def word_list_sha256() -> str:
    return _word_list()[1]


def _known(word: str, words: frozenset[str]) -> bool:
    word = word.replace("’", "'")
    if word.endswith("'s"):
        word = word[:-2]
    word = word.strip("'")
    if word in words or word in _ALSO_KNOWN:
        return True
    # Common inflections the lists may hold only in base form.
    for suffix, replacement in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                                ("ly", ""), ("ies", "y"), ("ied", "y")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and word[: -len(suffix)] + replacement in words:
            return True
    return False


# One-step OCR confusions, applied at a single position of an unknown word.
_CONFUSIONS = (("li", "h"), ("ll", "h"), ("l1", "h"), ("1", "l"), ("I", "l"), ("J", "l"), ("j", "i"),
               ("U", "ll"), ("rn", "m"), ("m", "rn"), ("vv", "w"), ("cl", "d"), ("c", "e"), ("e", "c"),
               ("0", "o"), ("5", "s"), ("ii", "u"), ("ru", "hi"), ("l", "h"), ("l", "i"), ("i", "l"),
               ("r", "t"), ("h", "b"), ("b", "h"), ("f", "t"), ("t", "f"),
               ("11", "n"), ("1", "i"), ("1", "r"), ("0", "o"), ("5", "s"), ("8", "B"))
_LIGATURES = ("fi", "fl", "ff", "ffi", "ffl")


def _confusable(token: str, words: frozenset[str]) -> bool:
    for wrong, right in _CONFUSIONS:
        start = token.find(wrong)
        while start != -1:
            candidate = (token[:start] + right + token[start + len(wrong):]).casefold()
            if len(candidate) >= 2 and _known(candidate, words):
                return True
            start = token.find(wrong, start + 1)
    return False


def confirmable(token: str) -> bool:
    """Can an independent OCR reading confirm this suspect word as printed?

    Only a plain lowercase word ("cel"). Never a letter-digit mix ("tl1e"), a word
    with capitals inside ("wouJd"), or one that corrects to a function word
    ("tlie" -> "the"): printed text almost never contains those.
    """
    if not token.isalpha() or not token.islower():
        return False
    words, _ = _word_list()
    for wrong, right in _CONFUSIONS:
        start = token.find(wrong)
        while start != -1:
            if (token[:start] + right + token[start + len(wrong):]).casefold() in _FUNCTION_WORDS:
                return False
            start = token.find(wrong, start + 1)
    return True


def _ligature_gap(previous: str, token: str, words: frozenset[str]) -> bool:
    return any(_known(lig + token, words) or (previous and _known(previous + lig + token, words))
               for lig in _LIGATURES)


def _scan(text: str, confirmed: frozenset[str]) -> tuple[int, int, list[str]]:
    """(checked words, English function words, damaged words) for one page."""
    words, _ = _word_list()
    joined = _NOT_PROSE.sub(" ", _LINE_BREAK_HYPHEN.sub(r"\1\2", text))
    checked = function = 0
    damaged: list[str] = []
    previous = ""
    for token in _TOKEN.findall(joined):
        has_digit = any(c.isdigit() for c in token)
        has_letter = any(c.isalpha() for c in token)
        prior, previous = previous, (token.casefold() if token.isalpha() else "")
        if has_digit and has_letter:
            # Only a mix that reads as a word after an OCR confusion ("tl1e" -> "the"),
            # not a code or identifier ("task1", "491b55f9").
            if not _ORDINARY_MIXED.match(token) and token[0].islower() and _confusable(token, words):
                checked += 1
                if token not in confirmed:
                    damaged.append(token)
            continue
        if not has_letter or not token[0].islower() or len(token) < 3 and token not in _FUNCTION_WORDS:
            continue
        lowered = token.casefold()
        checked += 1
        if lowered in _FUNCTION_WORDS:
            function += 1
        elif (token not in confirmed and not _known(lowered, words)
              and (_confusable(token, words) or _ligature_gap(prior, lowered, words))):
            damaged.append(token)
    return checked, function, damaged


def damaged_words(text: str) -> list[str]:
    """The words on a page that the check counts as damage, in order."""
    return _scan(text, frozenset())[2]


def assess_page(text: str, confirmed: frozenset[str] = frozenset()) -> PageTextQuality:
    """Score one page. Raises WordListUnavailable if the lists are missing.

    `confirmed` holds words an independent OCR reading of the same page also
    produced ("cel", "cels" on an animation page): they are on the page, so
    they are not damage.
    """
    checked, function, damaged = _scan(text, confirmed)
    if checked < MIN_CHECKED_WORDS:
        return PageTextQuality("not_assessed", None, checked, "too_few_words")
    if function / checked < MIN_FUNCTION_WORD_SHARE:
        return PageTextQuality("not_assessed", None, checked, "not_english_prose")
    share = round(len(damaged) / checked, 4)
    if share >= DAMAGED_WORD_SHARE:
        return PageTextQuality("damaged", share, checked, "damaged_word_share")
    return PageTextQuality("clean", share, checked, "damaged_word_share")


_READING_HYPHEN = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z]+)")


def readable_text(text: str) -> str:
    """Source text for reading: line-break hyphens joined, line breaks as spaces.

    "stu-\ndio" becomes "studio"; "well-\nknown" keeps its hyphen when both
    halves are words and the joined form is not. Without the word lists every
    lowercase continuation is joined. Meaning-preserving and mechanical; the
    exact source text is kept wherever evidence is bound.
    """
    try:
        words = _word_list()[0]
    except WordListUnavailable:
        words = None

    def join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if words is not None and not _known((left + right).casefold(), words) \
                and _known(left.casefold(), words) and _known(right.casefold(), words):
            return f"{left}-{right}"
        return left + right
    joined = _READING_HYPHEN.sub(join, text)
    return re.sub(r"[ \t]*\n+[ \t]*", " ", joined).strip()
