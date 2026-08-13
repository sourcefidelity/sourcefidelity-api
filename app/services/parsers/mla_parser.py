"""MLA (9th edition) citation parser.

Key differences from APA:

* Heading is "Works Cited" (not "References").
* Author is "Lastname, Firstname." (full first name, not initials).
* Year appears near the *end*, after the publisher.
* In-text citations use "(Author PageNum)" not "(Author, Year)".
"""

import re
from typing import List

from app.services.parsers.base_parser import BaseParser


# MLA references start with "Lastname, Firstname." followed by a title
# (not a parenthesised year — that's the APA differentiator).
_MLA_START = re.compile(
    r'^\s*'
    r'(?:\d+[\.\)]\s+)?'                         # optional number prefix: "3. " or "1) "
    r'[A-Z][a-z]+(?:[\'-][A-Z][a-z]+)?\s*,\s*'  # Lastname,
    r'[A-Z][a-z]+'                               # Firstname (full, not initial)
    r'(?:\s+[A-Z]\.)?'                          # optional middle initial
    r'[.,]\s+'                                    # period OR comma after author (student papers often omit the period)
    r'(?!\(\d{4}[a-z]?\)|\(n\.d\.\))'           # NOT followed by APA year
)

# MLA references can also start with a quoted title (when no author)
_MLA_TITLE_START = re.compile(
    r'^\s*'
    r'(?:\d+[\.\)]\s+)?'                         # optional number prefix
    r'[\u201c"]'                                  # opening quote (curly or straight)
)


class MlaParser(BaseParser):
    """Parser for MLA (9th edition) references."""

    HEADINGS = [
        r'works cited',
        r'work cited',
        r'works consulted',
        r'sources',
        r'bibliography',
    ]

    REF_START_PATTERN = _MLA_START

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @classmethod
    def split_references(cls, raw_text: str) -> List[str]:
        """Split an MLA reference section into individual reference strings.

        Uses :meth:`_merge_lines` (inherited from BaseParser) to merge
        multi-line references, then applies MLA-specific fallbacks when
        the pattern-based merge produces too few results.

        MLA references start with ``Lastname, Firstname.`` or, when no
        author, with a quoted/italicised title. The year appears near
        the end (after publisher), not in parentheses after the author.
        """
        if not raw_text:
            return []

        # Strip the heading line if present
        lines = raw_text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if re.match(
                r'(?i)^(?:works cited|work cited|works consulted|bibliography|sources)\s*$',
                line.strip(),
            ):
                start_idx = i + 1
                break
        lines = lines[start_idx:]

        # NOTE: MLA student papers often have double-spacing (blank lines between
        # every line, including continuation lines). The _merge_lines blank-line
        # splitter fragments these, producing over-counts. Attempting to collapse
        # blank lines by heuristic (period-ending, etc.) fails because MLA refs
        # contain periods throughout — can't distinguish "end of reference" from
        # "end of title within reference." The LLM cleanup path (_mla_cleanup_split)
        # solves this by understanding reference structure. This native splitter
        # serves as a fallback for when the LLM is unavailable, accepting some
        # over-counting as the trade-off.

        # Primary merge using inherited _merge_lines (uses REF_START_PATTERN)
        refs = cls._merge_lines(lines)

        # Also catch author-less references (start with quote) that _merge_lines
        # may have missed because they don't match _MLA_START.
        # Re-process: split the raw lines again, this time also treating
        # title-start lines as new-reference markers.
        if len(refs) < 2 and len(lines) > 1:
            refs = cls._merge_lines_mla(lines)

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

        # Final cleanup — merge continuation fragments back into their parent
        # reference. MLA papers often have blank lines between every line
        # (double-spaced), which the _merge_lines blank-line splitter fragments.
        # A continuation fragment is one that does NOT start a new MLA reference
        # (no author pattern, no title-quote start) and the previous reference
        # didn't end cleanly (no period+URL, no "Print."/"Web." terminator).
        cleaned = []
        for ref in refs:
            ref = re.sub(r'^\d+[\.\)]\s*', '', ref)  # strip numbering
            ref = re.sub(r'\s+', ' ', ref).strip()

            if not ref:
                continue

            # Is this a new reference or a continuation fragment?
            is_new_ref = bool(
                _MLA_START.match(ref)
                or _MLA_TITLE_START.match(ref)
            )

            # URL fragment: always a continuation (MLA uses <URL> format, and
            # URL endings like "path-segment/>." split off from the URL start)
            if re.match(r'^\s*<?https?://', ref) and cleaned:
                cleaned[-1] = cleaned[-1] + ' ' + ref
                continue
            # URL-ending fragment (e.g. "use-across-social-media-platforms/>.")
            if re.match(r'^\s*[\w/-]+/>\.?\s*$', ref) and cleaned:
                cleaned[-1] = cleaned[-1] + ' ' + ref
                continue

            if is_new_ref or not cleaned:
                cleaned.append(ref)
            else:
                # Continuation fragment — check if previous ended cleanly
                prev = cleaned[-1].rstrip()
                # MLA references typically end with "Print." "Web." a URL/DOI,
                # or a period+page-range. If the previous didn't end cleanly,
                # this fragment is a continuation of it.
                ends_cleanly = bool(re.search(
                    r'(?:\.\s*$|Print\.\s*$|Web\.\s*$|>\s*$|'
                    r'https?://\S+\s*$|doi\.org/\S+\s*$|'
                    r'\d{4}\.\s*$|pp?\.\s*\d+)',  # year. or pp. N
                    prev,
                ))
                if not ends_cleanly:
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                else:
                    cleaned.append(ref)

        return cleaned

    @classmethod
    def _merge_lines_mla(cls, lines: List[str]) -> List[str]:
        """Merge lines for MLA, treating both author-start and title-start
        (quoted) lines as new-reference markers.

        This handles author-less MLA references that start with a title
        in quotation marks — common for websites, films, and org-authored
        sources.
        """
        blocks: List[str] = []
        current: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(' '.join(current))
                    current = []
                continue

            is_new_ref = (
                (cls.REF_START_PATTERN and cls.REF_START_PATTERN.match(stripped))
                or (
                    _MLA_TITLE_START.match(stripped)
                    and not cls._CONTINUATION_PATTERN.match(stripped)
                )
            )

            if is_new_ref and current:
                blocks.append(' '.join(current))
                current = [stripped]
            elif is_new_ref:
                current = [stripped]
            else:
                if current:
                    current.append(stripped)
                else:
                    current = [stripped]

        if current:
            blocks.append(' '.join(current))

        return blocks
