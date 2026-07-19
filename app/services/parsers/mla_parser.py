"""MLA (9th edition) citation parser.

Key differences from APA:

* Heading is "Works Cited" (not "References").
* Author is "Lastname, Firstname." (full first name, not initials).
* Year appears near the *end*, after the publisher.
* In-text citations use "(Author PageNum)" not "(Author, Year)".

The regex-based split_references() is not yet implemented — MLA
reference extraction uses the LLM-first path exclusively for now.
REF_START_PATTERN is defined so format auto-detection works.
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
    # Splitting (NOT IMPLEMENTED — MLA uses LLM-first only)
    # ------------------------------------------------------------------
    @classmethod
    def split_references(cls, raw_text: str) -> List[str]:
        """Split an MLA reference section into individual references.

        Not implemented: MLA reference extraction uses the LLM-first
        path exclusively.  Regex splitting will be added later if a
        cost-saving fallback is needed.
        """
        raise NotImplementedError(
            "MLA regex splitting is not implemented. "
            "Use extract_and_parse_references(use_llm_split=True)."
        )
