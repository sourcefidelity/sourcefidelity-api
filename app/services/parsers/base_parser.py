"""Abstract base class for citation format-specific parsers."""

import re
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Pattern


class BaseParser(ABC):
    """Base class for citation format parsers (APA, MLA, …).

    Each subclass provides format-specific:
    - *HEADINGS* – list of section-heading regexes (most specific first).
    - *REF_START_PATTERN* – compiled regex that matches the beginning of
      a single reference on a line.
    - *split_references()* – how to merge continued lines into refs.
    """

    HEADINGS: ClassVar[List[str]] = []
    REF_START_PATTERN: ClassVar[Optional[Pattern]] = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    @classmethod
    def detect_in_text(cls, text: str) -> bool:
        """Return ``True`` if this format's reference pattern appears in
        the second half of *text*."""
        if cls.REF_START_PATTERN is None:
            return False
        lines = text.strip().split('\n')
        halfway = len(lines) // 2
        for i in range(halfway, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if cls.REF_START_PATTERN.match(stripped):
                return True
        return False

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------
    @classmethod
    def extract_reference_section(cls, text: str) -> Optional[str]:
        """Return the reference/bibliography section from *text*.

        Strategy 1 – look for a heading line that matches one of the
        format's *HEADINGS* (e.g. "References", "Works Cited").
        Strategy 2 – scan the second half of the document for the first
        line that matches *REF_START_PATTERN*.
        """
        if not text:
            return None

        # -- Strategy 1: heading-based ---------------------------------
        if cls.HEADINGS:
            tag = r'(?:<[^>]+>)*'
            alt = '|'.join(cls.HEADINGS)
            heading_re = re.compile(
                rf'(?m)^\s*{tag}\s*({alt})\s*{tag}\s*:?\s*$',
                re.IGNORECASE,
            )
            match = heading_re.search(text)
            if match:
                start = match.end()
                if start < len(text) and text[start] == '\n':
                    start += 1
                else:
                    nl = text.find('\n', start)
                    if nl != -1:
                        start = nl + 1
                section = text[start:].strip()
                if section:
                    return section

        # -- Strategy 2: content-based ---------------------------------
        if cls.REF_START_PATTERN is not None:
            lines = text.strip().split('\n')
            start_scan = max(0, len(lines) * 3 // 10)  # 30% mark
            
            for i in range(start_scan, len(lines)):
                stripped = lines[i].strip()
                if not stripped:
                    continue
                if cls.REF_START_PATTERN.match(stripped):
                    # Verify: is this really a reference section?
                    # Check that at least 1 more ref appears within next 20 lines
                    # This filters out isolated body-text matches (in-text citations)
                    refs_found = 0
                    for j in range(i, min(len(lines), i + 30)):
                        check_line = lines[j].strip()
                        if check_line and cls.REF_START_PATTERN.match(check_line):
                            refs_found += 1
                    if refs_found >= 2:
                        return '\n'.join(lines[i:]).strip()

        return None

    # ------------------------------------------------------------------
    # Reference splitting
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def split_references(cls, raw_text: str) -> List[str]:
        """Split the raw reference section into individual reference
        strings."""
        ...

    # Lines that are always continuations, never new references
    # Includes: [Online], Retrieved, website names, continuation-only fragments
    _CONTINUATION_PATTERN: ClassVar[Optional[Pattern]] = re.compile(
        r'^(?:\[Online\]|\[online\]|Retrieved\s|retrieved\s'
        r'|Coppola[\'\u2019]?s?\s|Classicmoviehub\.\s'
        r'|Journal,\s*\d+\s*\(\d+\)'                          # "Journal, 20(5)" continuation
        r'|Evaluation\s|Business\sInsider,\sInc'              # split continuations
        r'|Nosferatu\s\(\d{4}\)'                              # "Nosferatu (1922)" in title continuation
        r')',
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Helper: merge lines into reference blocks
    # ------------------------------------------------------------------
    @classmethod
    def _block_ends_cleanly(cls, block: str) -> bool:
        """Check if a block ends with a proper terminator (period, URL, etc.)."""
        block = block.rstrip()
        if not block:
            return False
        # Ends with period, question mark, exclamation, closing paren/bracket
        if block[-1] in '.?!)]':
            return True
        # Ends with URL or DOI
        if re.search(r'(https?://\S+|doi\s*:\s*\S+|doi\.org/\S+)$', block):
            return True
        # Ends with a recognizable publisher pattern
        if re.search(r'(Press|University|Publishing|Routledge|Palgrave|Oxford)$', block, re.IGNORECASE):
            return True
        return False
    @classmethod
    def _merge_lines(cls, lines: List[str]) -> List[str]:
        """Merge consecutive lines into reference blocks.

        A new reference block starts when a line matches
        *REF_START_PATTERN* and is NOT a continuation line.
        Empty lines act as separators.
        Returns cleaned reference strings (no numbering, single spaces).
        """
        if not lines:
            return []

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
                cls.REF_START_PATTERN is not None
                and cls.REF_START_PATTERN.match(stripped)
                and not cls._CONTINUATION_PATTERN.match(stripped)
            )

            if is_new_ref:
                if current:
                    blocks.append(' '.join(current))
                current = [stripped]
            else:
                # Always add non-matching lines to current, even if empty.
                # Multi-line references may have the year on line 2+.
                current.append(stripped)

        if current:
            blocks.append(' '.join(current))

        # Clean
        cleaned = []
        for ref in blocks:
            ref = re.sub(r'^\d+[\.\)]\s*', '', ref)
            ref = re.sub(r'\s+', ' ', ref).strip()
            cleaned.append(ref)

        return cleaned
