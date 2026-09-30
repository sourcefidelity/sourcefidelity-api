"""Secondary citation from the GLM-selected evidence sentences (owner request 2026-09-29).

A source is flagged when every sentence GLM chose as bearing on the student's
statement attributes that content to another work: an in-text citation to
authors other than the source's own ("(Boltanski and Chiapello, 1999)",
"Grabher (2002) argues", "according to Smith"). The student may then be citing
a secondary source for another author's idea. Deterministic and inspectable:
the flag names the sentences and the attributed authors. It says what the
source's sentences do, never that the student meant anything; an attribution
the patterns miss leaves the source unflagged.
"""
from __future__ import annotations

import re

SECONDARY_CITATION_VERSION = "glm-sentence-secondary-citation-v1"
_YEAR = r"(?:1[5-9]|20)\d{2}[a-z]?"
_SURNAME = r"[A-Z][A-Za-z'’\-]+(?:\s+(?:de|van|von|der|la|le)\s+[A-Z][A-Za-z'’\-]+)?"
_PARENTHETICAL = re.compile(rf"\(([^()]*?\b{_YEAR}\b[^()]*)\)")
_NARRATIVE = re.compile(
    rf"\b({_SURNAME})(?:\s+(?:and|&)\s+({_SURNAME})|\s+et\s+al\.?)?\s*[\(\[]\s*{_YEAR}")
_ACCORDING = re.compile(rf"\b[Aa]ccording\s+to\s+({_SURNAME})")
_NOT_NAMES = frozenset({"see", "e.g", "eg", "cf", "p", "pp", "in", "and", "et", "al", "ibid", "also", "the",
                        "chapter", "figure", "table", "vol", "no", "ed", "eds", "trans"})


def _parenthetical_names(inner: str) -> list[str]:
    names = []
    for part in re.split(r";", inner):
        if not re.search(rf"\b{_YEAR}\b", part):
            continue
        for name in re.findall(r"[^\W\d_][\w'’\-]+", part.split(",")[0] if "," in part else part):
            if name[0].isupper() and name.casefold().strip(".") not in _NOT_NAMES:
                names.append(name)
    return names


def attributed_names(sentence: str) -> list[str]:
    """Authors a sentence attributes its content to, in order."""
    names: list[str] = []
    for match in _PARENTHETICAL.finditer(sentence):
        names.extend(_parenthetical_names(match.group(1)))
    for match in _NARRATIVE.finditer(sentence):
        names.extend(n for n in match.groups() if n)
    names.extend(match.group(1) for match in _ACCORDING.finditer(sentence))
    return list(dict.fromkeys(n for n in names if n.casefold() not in _NOT_NAMES))


def _own_authors(source: dict) -> set[str]:
    text = " ".join(str(source.get(key) or "") for key in ("author", "raw_reference"))[:300]
    head = text.split("(", 1)[0]
    return {name.casefold() for name in re.findall(r"[A-Z][A-Za-z'’\-]+", head)}


def secondary_citation(evidence_sentences: list[dict], source: dict) -> dict | None:
    """The flag, or None: every bearing sentence attributes to other authors."""
    bearing = [s for s in evidence_sentences or [] if s.get("reason") == "bears_on_statement"]
    if not bearing:
        return None
    own = _own_authors(source or {})
    attributed = []
    for sentence in bearing:
        others = [n for n in attributed_names(str(sentence.get("text") or "")) if n.casefold() not in own]
        if not others:
            return None
        attributed.append((sentence.get("key"), others))
    return {"version": SECONDARY_CITATION_VERSION,
            "sentence_keys": [key for key, _ in attributed],
            "attributed_to": list(dict.fromkeys(n for _, names in attributed for n in names))}
