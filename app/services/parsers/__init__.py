"""Citation format parser registry."""

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
    for parser_cls in _PARSER_REGISTRY:
        if parser_cls.detect_in_text(text):
            return parser_cls
    return _ApaParser  # sensible default
