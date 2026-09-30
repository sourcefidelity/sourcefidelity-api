"""Citation format parser registry."""

import re

from app.services.parsers.apa_parser import ApaParser as _ApaParser
from app.services.parsers.base_parser import BaseParser as _BaseParser
from app.services.parsers.mla_parser import MlaParser as _MlaParser

# Order matters for detection: more format-specific / stricter parsers
# should come first so they get first crack at identification.
_PARSER_REGISTRY: list[type[_BaseParser]] = [
    _MlaParser,
    _ApaParser,
]


def detect_format(text: str) -> type[_BaseParser]:
    """Detect the citation format used in *text*.

    Iterates through registered parsers and returns the first one whose
    :meth:`BaseParser.detect_in_text` returns ``True``.  Falls back to
    APA if nothing matches.
    """
    # An explicit APA reference heading plus repeated author/date entries is
    # stronger than an incidental MLA-shaped line elsewhere in the paper.
    heading = re.search(r'(?im)^\s*references?(?:\s+list|\s+section)?\s*:?\s*$', text)
    mla_heading = re.search(r'(?im)^\s*works?\s+cited\s*:?\s*$', text)
    if heading and not mla_heading:
        section = _ApaParser.extract_reference_section(text) or ''
        starts = sum(bool(re.match(
            r'^\s*(?:[-•]\s+)?[^\n]{1,180}\((?:19|20)\d{2}[a-z]?\)[.,\s]', line
        )) for line in section.splitlines())
        if starts >= 2:
            return _ApaParser
    for parser_cls in _PARSER_REGISTRY:
        if parser_cls.detect_in_text(text):
            return parser_cls
    return _ApaParser  # sensible default
