"""Local, exact-span inventory of citation-unit structural phenomena.

The inventory is deliberately recall-oriented and non-authoritative.  A
detected phenomenon may trigger an audit or a conservative exact facet, but it
never rewrites student text, creates evidence, or decides a relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from app.services.source_attribution import find_external_attribution_cues


STRUCTURE_INVENTORY_VERSION = "citation-structure-inventory-v1"

StructureKind = Literal[
    "coordination",
    "shared_qualifier_coordination",
    "trailing_interpretive_result",
    "causal_relation",
    "comparison_or_coverage",
    "negation",
    "modality_or_frequency",
    "temporal_or_conditional_scope",
    "relative_or_subordinate_clause",
    "source_attribution",
    "anaphoric_dependency",
    "student_stance",
]


@dataclass(frozen=True)
class StructuralPhenomenon:
    kind: StructureKind
    start: int
    end: int
    text: str
    confidence: Literal["high", "medium"]
    intended_use: Literal["audit_only", "candidate_boundary", "material_facet"]
    trigger: str


_CAUSAL_PARTICIPLES = (
    "allowing",
    "causing",
    "creating",
    "leading",
    "making",
    "reducing",
    "resulting",
    "stifling",
    "supporting",
    "increasing",
    "achieving",
)
_INTERPRETIVE_RESULT_PARTICIPLES = (
    "demonstrating",
    "highlighting",
    "implying",
    "indicating",
    "marking",
    "reflecting",
    "revealing",
    "signaling",
    "signalling",
    "suggesting",
    "underscoring",
)
TRAILING_PARTICIPIAL_BOUNDARY_PATTERN = re.compile(
    r",\s*(?P<verb>"
    + "|".join((*_CAUSAL_PARTICIPLES, *_INTERPRETIVE_RESULT_PARTICIPLES))
    + r")\b",
    re.IGNORECASE,
)
_INTERPRETIVE_RESULT_PATTERN = re.compile(
    r"(?:^\s*|,\s*)(?P<verb>"
    + "|".join(_INTERPRETIVE_RESULT_PARTICIPLES)
    + r")\b[^,;.!?]{0,240}",
    re.IGNORECASE,
)
_SHARED_QUALIFIER_COORDINATION = re.compile(
    r"\b(?P<qualifier>a\s+large\s+number\s+of|large\s+numbers\s+of|many|"
    r"numerous|several|multiple|few|most)\s+"
    r"(?P<left>[^,;.!?]{1,100}?)\s+(?P<connector>and|or)\s+"
    r"(?P<right>[^,;.!?]{1,100})(?=$|[,;.!?])",
    re.IGNORECASE,
)

# High-precision finite predicates used only to expose exact shared-predicate
# object components. This is not a general proposition parser: both component
# facets retain the exact subject/predicate prefix and one exact coordinated
# object from the student's sentence.
_SHARED_OBJECT_PREDICATES = {
    "affect", "affects", "allow", "allows", "cause", "causes", "create",
    "creates", "damage", "damages", "derail", "derails", "describe",
    "describes", "discourage", "discourages", "enable", "enables",
    "encourage", "encourages", "establish", "establishes", "harm", "harms",
    "impair", "impairs", "increase", "increases", "limit", "limits",
    "maintain", "maintains", "prevent", "prevents", "promote", "promotes",
    "protect", "protects", "provide", "provides", "reduce", "reduces",
    "restrict", "restricts", "show", "shows", "support", "supports",
    "undermine", "undermines",
}
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

_AUDIT_PATTERNS: tuple[
    tuple[StructureKind, re.Pattern[str], Literal["high", "medium"], str], ...
] = (
    (
        "causal_relation",
        re.compile(
            r"\b(?:because|due\s+to|therefore|thereby|consequently|"
            r"caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in)\b",
            re.IGNORECASE,
        ),
        "high",
        "causal_lexeme",
    ),
    (
        "comparison_or_coverage",
        re.compile(
            r"\b(?:more|less|higher|lower|greater|better|worse)\b[^,;.]{0,50}\bthan\b|"
            r"\b(?:overlooks?|overlooked|omits?|omitted|ignores?|ignored|"
            r"focus(?:es|ed)?\s+(?:more|mainly|primarily))\b",
            re.IGNORECASE,
        ),
        "high",
        "comparison_or_coverage_lexeme",
    ),
    (
        "negation",
        re.compile(r"\b(?:not|no|never|neither|without|cannot|can't|didn't|doesn't)\b", re.IGNORECASE),
        "high",
        "negation_lexeme",
    ),
    (
        "modality_or_frequency",
        re.compile(
            r"\b(?:can|could|may|might|must|should|would|always|never|usually|"
            r"often|sometimes|rarely|generally|typically|constantly|repeatedly)\b",
            re.IGNORECASE,
        ),
        "medium",
        "modality_or_frequency_lexeme",
    ),
    (
        "temporal_or_conditional_scope",
        re.compile(
            r"\b(?:before|after|during|since|until|throughout|between|unless|"
            r"only\s+if|only\s+when|provided\s+that|regardless\s+of)\b",
            re.IGNORECASE,
        ),
        "medium",
        "temporal_or_conditional_lexeme",
    ),
    (
        "relative_or_subordinate_clause",
        re.compile(r"(?:^|,)\s*(?:which|who|where|while|whereas|although|because|with)\b", re.IGNORECASE),
        "high",
        "clause_connector",
    ),
    (
        "anaphoric_dependency",
        re.compile(r"\b(?:this|these|those|they|it|such)\b", re.IGNORECASE),
        "medium",
        "anaphoric_lexeme",
    ),
    (
        "student_stance",
        re.compile(
            r"\b(?:I\s+(?:argue|believe|agree|disagree|suggest|contend)|"
            r"in\s+my\s+view|we\s+(?:argue|believe|suggest|contend))\b",
            re.IGNORECASE,
        ),
        "high",
        "explicit_student_stance",
    ),
)


def inspect_citation_structure(text: str) -> list[StructuralPhenomenon]:
    """Return stable exact-span audit signals without making semantic claims."""
    findings: list[StructuralPhenomenon] = []
    for match in re.finditer(r"\b(?:and|or|but)\b", text, re.IGNORECASE):
        findings.append(
            _finding(
                "coordination",
                match.start(),
                match.end(),
                text,
                "medium",
                "audit_only",
                "coordination_lexeme",
            )
        )
    for match in _SHARED_QUALIFIER_COORDINATION.finditer(text):
        findings.append(
            _finding(
                "shared_qualifier_coordination",
                match.start(),
                match.end(),
                text,
                "high",
                "material_facet",
                "explicit_qualifier_with_two_content_heads",
            )
        )
    for cue in find_external_attribution_cues(text):
        findings.append(
            _finding(
                "source_attribution",
                cue.start,
                cue.end,
                text,
                "medium",
                "audit_only",
                f"reporting_template:{cue.family}",
            )
        )
    for match in _INTERPRETIVE_RESULT_PATTERN.finditer(text):
        findings.append(
            _finding(
                "trailing_interpretive_result",
                match.start("verb"),
                match.end(),
                text,
                "high",
                "material_facet",
                "trailing_interpretive_participle",
            )
        )
    for kind, pattern, confidence, trigger in _AUDIT_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                _finding(
                    kind,
                    match.start(),
                    match.end(),
                    text,
                    confidence,
                    "audit_only",
                    trigger,
                )
            )
    unique = {
        (finding.kind, finding.start, finding.end, finding.trigger): finding
        for finding in findings
    }
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.kind))


def interpretive_result_spans(text: str) -> list[tuple[int, int]]:
    """Exact spans for high-confidence trailing interpretive/result clauses."""
    return [
        (match.start("verb"), match.end())
        for match in _INTERPRETIVE_RESULT_PATTERN.finditer(text)
    ]


def shared_qualifier_coordination_spans(
    text: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return exact left/right content obligations under one explicit qualifier."""
    pairs = []
    for match in _SHARED_QUALIFIER_COORDINATION.finditer(text):
        left = _trim_span(text, match.start("left"), match.end("left"))
        right = _trim_span(text, match.start("right"), match.end("right"))
        if left[0] < left[1] and right[0] < right[1]:
            pairs.append((left, right))
    return pairs


def shared_predicate_component_spans(
    text: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return exact prefix+object spans for a narrow coordinated object.

    Example shape: ``The acts derail [media freedom] and [public access]``.
    Ambiguous serial lists, clause coordination, multiple connectors,
    punctuation boundaries and missing finite predicates are excluded.
    """
    connectors = list(re.finditer(r"\b(?:and|or)\b", text, re.IGNORECASE))
    if len(connectors) != 1 or re.search(r"[,;]", text):
        return []
    connector = connectors[0]
    tokens = [token for token in _WORD_TOKEN.finditer(text[: connector.start()])]
    predicate = next(
        (
            token
            for token in tokens
            if token.group(0).casefold() in _SHARED_OBJECT_PREDICATES
            and token.start() > 0
        ),
        None,
    )
    if predicate is None:
        return []
    prefix = _trim_span(text, 0, predicate.end())
    left = _trim_span(text, predicate.end(), connector.start())
    right = _trim_span(text, connector.end(), len(text))
    if not all(start < end for start, end in (prefix, left, right)):
        return []
    if not (1 <= _substantive_words(text[left[0]:left[1]]) <= 18):
        return []
    if not (1 <= _substantive_words(text[right[0]:right[1]]) <= 18):
        return []
    # A second finite predicate in either object normally signals clause
    # coordination, not two objects governed by one exact predicate.
    for start, end in (left, right):
        words = {
            token.group(0).casefold()
            for token in _WORD_TOKEN.finditer(text[start:end])
        }
        if words & _SHARED_OBJECT_PREDICATES:
            return []
    return [(prefix, left), (prefix, right)]


def _finding(kind, start, end, text, confidence, intended_use, trigger):
    return StructuralPhenomenon(
        kind=kind,
        start=start,
        end=end,
        text=text[start:end],
        confidence=confidence,
        intended_use=intended_use,
        trigger=trigger,
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;."):
        end -= 1
    return start, end


def _substantive_words(text: str) -> int:
    ignored = {"a", "an", "and", "or", "the", "to", "of", "in", "on", "for"}
    return sum(
        token.group(0).casefold() not in ignored
        for token in _WORD_TOKEN.finditer(text)
    )
