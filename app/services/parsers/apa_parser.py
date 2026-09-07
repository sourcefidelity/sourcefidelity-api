"""APA (7th edition) citation parser.

Handles reference section extraction and splitting for APA format,
refactored into a :class:`BaseParser` subclass.
"""

import re
from typing import List

from app.services.parsers.base_parser import BaseParser


# Pattern matching the start of an APA reference line.
#
# APA format: Author(s) (Year). Title.
#
# Examples matched:
#   Bordwell, D. (2006). Title...
#   Whish, R., & Bailey, D. (2021). Title...
#   Ministry of Commerce. (n.d.). Title...
#   OECD. (2021). Title...
#   1. Bordwell, D. (2006). Title...
#
# Key structural markers that distinguish refs from body text:
# - Authors have commas or periods before the year (e.g., "D. (2006)" or "Commerce. (n.d.)")
# - Body text like "Section 6 of the regulations (2016)..." lacks this structure

_APA_AUTHOR_ONLY_START = re.compile(
    r'^\s*(?:\d+[\.\)]?\s+)?'
    r'[^\W\d_][^,\n]{1,80},\s*(?:[A-Z]\.?){1,6}(?:\s*,|\s*$)'
)

_APA_DATE = re.compile(
    r"(?<!\d)(?:(?:19|20)\d{2}[a-z]?|n\.d\.)(?!\d)",
    re.IGNORECASE,
)

_APA_PERSONAL_AUTHOR_CUE = re.compile(
    r",\s*(?:[A-Z](?:[-'\u2019][A-Z])?\.?)(?:\s*[A-Z]\.?)*"
)

_APA_START = re.compile(
    r'^\s*'
    r'(?:\d+[\.\)]?\s+)?'                                 # optional number
    r'(?:'
    r'(?:[A-Z]|\[|[\u4e00-\u9fff\u3400-\u4dbf])'          # ordinary complete first line
    r'[^\n]{0,400}?'
    r'[.,]?\s*\(?(?:(?:19|20)\d{2}[a-z]?|n\.d\.)\)?[.,\s]'
    r'|'
    # A long author list may wrap before its year.  A leading surname plus
    # initials is still a safe new-entry boundary even on that first line.
    r'[^\W\d_][^,\n]{1,80},\s*(?:[A-Z]\.?){1,6}(?:\s*,|\s*$)'
    r')',
)


class ApaParser(BaseParser):
    """Parser for APA (7th edition) references."""

    HEADINGS = [
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?references?',
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?reference list',
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?reference section',
    ]

    REF_START_PATTERN = _APA_START

    @classmethod
    def _starts_new_reference(cls, stripped: str, current: List[str]) -> bool:
        if not super()._starts_new_reference(stripped, current):
            return False
        if current and not _APA_DATE.search(" ".join(current)):
            # A wrapped APA author list can occupy several extracted lines
            # before the publication date appears.  Until that required date
            # has been seen, another author-shaped line completes the current
            # entry instead of starting a second one.
            return False
        date_match = _APA_DATE.search(stripped)
        if date_match:
            prefix = re.sub(
                r"^\s*\d+[.)]?\s+", "", stripped[: date_match.start()]
            ).strip().rstrip("(").rstrip()
            closing_index = date_match.end()
            date_is_apa_field = bool(
                date_match.start() > 0
                and stripped[date_match.start() - 1] == "("
                and closing_index < len(stripped)
                and stripped[closing_index] == ")"
                and stripped[closing_index + 1 :].lstrip().startswith((".", ","))
            )
            has_explicit_author_structure = bool(
                prefix.startswith("[")
                or prefix.endswith((".", "]"))
                or _APA_PERSONAL_AUTHOR_CUE.search(prefix)
                # PDF extraction can omit the separator before the date in a
                # long institutional author.  A long leading identity phrase
                # is materially different from a short work-title/year
                # continuation and remains an admissible entry boundary.
                or len(prefix) >= 60
            )
            if (
                current
                and date_is_apa_field
                and not has_explicit_author_structure
                and not cls._block_ends_cleanly(" ".join(current))
            ):
                return False
            has_author_structure = bool(
                has_explicit_author_structure or date_is_apa_field
            )
            if not has_author_structure:
                # A work title containing a date (for example a film title
                # followed by its release year) is a continuation, not a new
                # APA entry.  This structural rule replaces title-specific
                # exceptions.
                return False
        return True

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @classmethod
    def split_references(cls, raw_text: str) -> List[str]:
        """Split an APA reference section into individual reference strings.

        Uses :meth:`_merge_lines` to merge multi-line references, then
        applies APA-specific fallbacks when the pattern-based merge
        produces too few results.
        """
        if not raw_text:
            return []

        # Strip the heading line if present
        lines = raw_text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if re.match(
                r'(?i)^(?:references|bibliography|works cited)\s*$',
                line.strip(),
            ):
                start_idx = i + 1
                break
        lines = lines[start_idx:]

        # Primary merge
        refs = cls._merge_lines(lines)

        # Fallback 1: numbered-list split
        if len(refs) <= 1 and len(lines) > 1:
            numbered = re.compile(r'^\s*\d+[\.\)]\s+')
            raw = []
            for line in lines:
                s = line.strip()
                if s:
                    if numbered.match(s):
                        raw.append(s)
                    elif raw:
                        raw[-1] = raw[-1] + ' ' + s
            if len(raw) > 1:
                refs = raw

        # Fallback 2: single long string → split at author-year boundaries
        if len(refs) == 1 and len(refs[0]) > 500:
            pat = re.compile(
                r'[A-Z][a-z]+(?:[.,]\s*[A-Z]\.?)?\s*,'
                r'\s*(?:\()?(?:19|20)\d{2}|n\.d\.\)?'
            )
            matches = list(pat.finditer(refs[0]))
            if len(matches) > 1:
                starts = [m.start() for m in matches] + [len(refs[0])]
                refs = [
                    refs[0][starts[i]:starts[i+1]].strip()
                    for i in range(len(starts) - 1)
                ]

        # Final cleanup with continuation merging.  ``_merge_lines`` already
        # establishes entry boundaries.  A block that does not itself have a
        # structurally credible APA start is therefore a wrapped continuation,
        # not text to accept or discard according to a list of publishers,
        # titles, or phrases observed in one paper.
        cleaned = []
        for ref in refs:
            ref = re.sub(r'^\d+\.\s*', '', ref)
            ref = re.sub(r'\s+', ' ', ref).strip()

            if not ref:
                continue

            if cleaned and (
                not _APA_DATE.search(ref)
                or not cls._starts_new_reference(ref, [])
            ):
                cleaned[-1] = cleaned[-1] + ' ' + ref
            else:
                cleaned.append(ref)

        return cleaned
