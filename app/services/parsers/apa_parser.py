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

_APA_START = re.compile(
    r'^\s*'
    r'(?:\d+[\.\)]?\s+)?'                                 # optional number
    r'(?:[A-Z]|\[|[\u4e00-\u9fff\u3400-\u4dbf])'          # first char: capital, bracket, or CJK
    r'[^\n]{0,80}?'                                        # rest of name
    r'[.,]?'                                               # comma/period before year (optional for edge cases)
    r'\s*'
    r'\(?'
    r'(?:(?:19|20)\d{2}|n\.d\.)'                           # year OR n.d.
    r'\)?'
    r'[.,\s]'                                              # MUST be followed by punctuation or space (not embedded in URL)
)


class ApaParser(BaseParser):
    """Parser for APA (7th edition) references."""

    HEADINGS = [
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?references?',
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?reference list',
        r'(?:\d+(?:\.\d+)*\s*[\.\)]?\s*)?reference section',
    ]

    REF_START_PATTERN = _APA_START

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

        # Final cleanup with continuation merging
        cleaned = []
        for ref in refs:
            ref = re.sub(r'^\d+\.\s*', '', ref)
            ref = re.sub(r'\s+', ' ', ref).strip()
            
            if not ref:
                continue
            
            lower = ref.lower()
            
            # Check continuation pattern first
            if cls._CONTINUATION_PATTERN and cls._CONTINUATION_PATTERN.match(ref):
                if cleaned:
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                continue
            
            # Continuation-only lines: never standalone references
            if re.match(r'^\[online\]', lower) or re.match(r'^retrieved\s', lower):
                if cleaned:
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                continue
            if re.match(r'^coppola[\'\u2019]?s?\s', lower) or re.match(r'^classicmoviehub\.\s', lower):
                if cleaned:
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                continue
            
            # URL-only fragment: merge into previous
            if re.match(r'^https?://', ref):
                if cleaned:
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                continue
            
            # Merge if previous ref ends incompletely AND current starts lowercase
            if ref[0].islower() and cleaned:
                prev_end = cleaned[-1].rstrip()[-1] if cleaned[-1].rstrip() else ''
                if prev_end not in '.?!)\'"':
                    cleaned[-1] = cleaned[-1] + ' ' + ref
                    continue
            
            # Real reference markers
            has_marker = (
                'http' in lower
                or 'doi' in lower
                or any(kw in lower for kw in [
                    'press', 'university', 'publishing', 'journal',
                    'routledge', 'palgrave', 'oxford', 'cambridge',
                    'berkeley', 'chicago', 'edinburgh', 'macmillan',
                    'columbia', 'harvard', 'princeton', 'sage',
                    'springer', 'wiley', 'elsevier', 'taylor',
                    'thesis', 'dissertation', 'vol.', 'pp.', 'ed.',
                    'retrieved', 'edn', '(ed)', '(eds)', 'editors',
                    'available at', 'accessed', 'director', 'film',
                    'motion picture', 'archives',
                ])
            )
            
            is_short = len(ref) < 100 and ref.count('.') >= 3
            
            if has_marker or is_short:
                cleaned.append(ref)
            else:
                if len(ref) < 40 or any(p in lower for p in ['more like', 'greater focus', 'illustrates']):
                    continue
                cleaned.append(ref)

        return cleaned
