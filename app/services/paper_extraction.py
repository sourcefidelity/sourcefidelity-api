"""Application-level orchestration for paper reference/citation extraction.

This module joins the already-validated extraction services without taking on
upload persistence, source retrieval, verification judgment, or reporting.
Keeping that boundary explicit lets the application preserve stable identities
and auditable rejected spans before later pipeline stages are implemented.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from app.services.citation_extractor import SignalConfig, extract_citations
from app.services.antecedent_resolver import resolve_claim_antecedents
from app.services.discourse_context import resolve_claim_discourse_dependencies
from app.services.claim_atomizer import (
    AtomizationArtifact,
    atomize_claim_unit,
    claim_evidence_from_atom,
)
from app.services.parsers import detect_format
from app.services.parsers.apa_parser import ApaParser
from app.services.parsers.mla_parser import MlaParser
from app.services.reference_parser import (
    extract_and_parse_references,
    extract_reference_section,
)
from app.services.schemas import CitationMarkerMember, InTextCitation, ParsedReference
from app.services.sentence_splitter import split_sentences
from app.services.verification_evidence import (
    ClaimContextSegment,
    ClaimEvidence,
    claim_evidence_from_citation,
)


class PaperExtractionError(ValueError):
    """Raised when a paper cannot produce a safe extraction artifact."""


class ClaimBoundaryRejection(BaseModel):
    """An accepted citation that could not safely become a claim unit."""

    passage_start: int
    passage_end: int
    citation_marker: str
    reason: str


class PaperExtractionArtifact(BaseModel):
    """Typed, paper-local output consumed by later verification stages.

    The full paper and bibliography section are deliberately not copied into
    this result. References retain their source strings and citations retain
    only their bounded passages, identities, coordinates, and audit outcomes.
    """

    paper_version_id: str
    citation_format: str
    references: list[ParsedReference] = Field(default_factory=list)
    citations: list[InTextCitation] = Field(default_factory=list)
    rejected_citations: list[InTextCitation] = Field(default_factory=list)
    citation_claims: list[ClaimEvidence] = Field(default_factory=list)
    atomizations: list[AtomizationArtifact] = Field(default_factory=list)
    eligible_atomic_claims: list[ClaimEvidence] = Field(default_factory=list)
    claim_boundary_rejections: list[ClaimBoundaryRejection] = Field(
        default_factory=list
    )
    body_character_count: int = 0
    reference_section_character_count: int = 0


def extract_paper_evidence(
    paper_text: str,
    *,
    paper_version_id: str,
    format_hint: Optional[str] = None,
    use_llm_boundaries: bool = True,
    use_llm_atomizer: bool = True,
    use_llm_reference_fallback: bool = True,
    signals: Optional[SignalConfig] = None,
) -> PaperExtractionArtifact:
    """Extract stable references and citation spans from one paper version.

    ``paper_version_id`` is mandatory because it anchors deterministic
    reference IDs. LLM boundary extraction uses only the supported ``cite``
    route, which applies the shared masking, data framing, and prompt budget.
    Rejected model spans remain in a separate audit collection and cannot flow
    into verification as accepted evidence. Every accepted, uniquely linked
    citation is converted to a stable claim unit and passed through the
    attributed atom boundary. Complex or unresolved units remain inspectable
    but cannot enter source-relationship verification.
    """

    version_id = paper_version_id.strip()
    if not version_id:
        raise PaperExtractionError("paper_version_id is required")
    if not paper_text or not paper_text.strip():
        raise PaperExtractionError("paper_text is empty")

    parser = detect_format(paper_text) if format_hint is None else None
    citation_format = (format_hint or _format_name(parser)).strip().lower()
    if citation_format not in {"apa", "mla"}:
        raise PaperExtractionError(f"Unsupported citation format: {citation_format}")
    parser = parser or {"apa": ApaParser, "mla": MlaParser}[citation_format]

    reference_section = extract_reference_section(paper_text, citation_format)
    if not reference_section:
        raise PaperExtractionError("No reference section found")

    body_text = _body_before_reference_section(
        paper_text,
        reference_section,
        parser,
    )
    if not body_text:
        raise PaperExtractionError("Paper body is empty before the reference section")

    references = extract_and_parse_references(
        reference_section,
        format_hint=citation_format,
        use_regex_first=True,
        use_llm_fallback=use_llm_reference_fallback,
        paper_version_id=version_id,
    )
    if not references:
        raise PaperExtractionError("Reference section contained no parseable references")

    extracted = extract_citations(
        body_text,
        references,
        format_hint=citation_format,
        use_llm_boundaries=use_llm_boundaries,
        signals=signals or SignalConfig.all_off(),
        extractor="cite",
    )
    accepted = [citation for citation in extracted if citation.drop_reason is None]
    rejected = [citation for citation in extracted if citation.drop_reason is not None]
    accepted, ungrouped_multi_marker = _group_multi_marker_units(accepted)
    rejected.extend(ungrouped_multi_marker)
    accepted, duplicate_detections = _consolidate_duplicate_citation_units(accepted)
    rejected.extend(duplicate_detections)
    citation_claims: list[ClaimEvidence] = []
    atomizations: list[AtomizationArtifact] = []
    eligible_atomic_claims: list[ClaimEvidence] = []
    claim_boundary_rejections: list[ClaimBoundaryRejection] = []
    for citation in accepted:
        try:
            parent_claim = claim_evidence_from_citation(
                citation,
                paper_version_id=version_id,
                antecedent_context=bounded_antecedent_context(
                    body_text,
                    citation.passage_start,
                ),
            )
            parent_claim = resolve_claim_antecedents(body_text, parent_claim)
            parent_claim = resolve_claim_discourse_dependencies(parent_claim)
        except ValueError as exc:
            claim_boundary_rejections.append(
                ClaimBoundaryRejection(
                    passage_start=citation.passage_start,
                    passage_end=citation.passage_end,
                    citation_marker=citation.citation_marker,
                    reason=str(exc),
                )
            )
            continue
        citation_claims.append(parent_claim)
        atomization = atomize_claim_unit(
            parent_claim,
            use_llm=use_llm_atomizer,
        )
        atomizations.append(atomization)
        eligible_atomic_claims.extend(
            claim_evidence_from_atom(parent_claim, atom)
            for atom in atomization.eligible_atoms
        )

    return PaperExtractionArtifact(
        paper_version_id=version_id,
        citation_format=citation_format,
        references=references,
        citations=accepted,
        rejected_citations=rejected,
        citation_claims=citation_claims,
        atomizations=atomizations,
        eligible_atomic_claims=eligible_atomic_claims,
        claim_boundary_rejections=claim_boundary_rejections,
        body_character_count=len(body_text),
        reference_section_character_count=len(reference_section),
    )


def _consolidate_duplicate_citation_units(
    citations: list[InTextCitation],
) -> tuple[list[InTextCitation], list[InTextCitation]]:
    """Keep one claim when narrative and parenthetical markers cover one unit.

    Reference identity, exact paper coordinates, and exact attributed text own
    the citation-unit identity. A parenthetical marker is preferred as the
    controlling representation because removing it preserves any substantive
    narrative attribution at the beginning of the sentence. Other detections
    remain inspectable in the rejected audit rather than becoming duplicate
    downstream claims.
    """
    grouped: dict[tuple, list[tuple[int, InTextCitation]]] = {}
    for index, citation in enumerate(citations):
        key = (
            tuple(citation.reference_ids),
            citation.passage_start,
            citation.passage_end,
            citation.text,
        )
        grouped.setdefault(key, []).append((index, citation))

    kept: list[tuple[int, InTextCitation]] = []
    duplicates: list[tuple[int, InTextCitation]] = []
    for detections in grouped.values():
        winner_index, winner = max(
            detections,
            key=lambda item: (
                item[1].marker_type == "parenthetical",
                bool(item[1].page_number),
                -item[0],
            ),
        )
        kept.append((winner_index, winner))
        duplicates.extend(
            (
                index,
                citation.model_copy(
                    update={"drop_reason": "duplicate_marker_same_citation_unit"}
                ),
            )
            for index, citation in detections
            if index != winner_index
        )
    kept.sort(key=lambda item: item[0])
    duplicates.sort(key=lambda item: item[0])
    return [citation for _, citation in kept], [citation for _, citation in duplicates]


_COLLECTIVE_REPORTING_SUFFIX = re.compile(
    r"^\s*(?:(?:each|both|all|collectively|together)\s+)?"
    r"(?:argue|claim|conclude|demonstrate|describe|discuss|explain|find|"
    r"identify|note|observe|point\s+out|report|show|state|suggest)\b",
    re.IGNORECASE,
)
_COLLECTIVE_PREFIX = re.compile(r"^\s*(?:both\s+)?$", re.IGNORECASE)
_COORDINATED_MARKER_GAP = re.compile(
    r"^\s*(?:,\s*|,?\s*(?:and|or|&)\s*)$",
    re.IGNORECASE,
)


def _group_multi_marker_units(
    citations: list[InTextCitation],
) -> tuple[list[InTextCitation], list[InTextCitation]]:
    """Resolve high-confidence multi-marker units and reject the remainder.

    A compound parenthetical marker is already represented by one shared raw
    marker and remains eligible. Separately detected sources are either split
    at an explicit semicolon boundary, grouped when coordinated narrative
    markers jointly govern one reporting predicate, or rejected. No semantic
    source scope is inferred from mere proximity.
    """
    grouped: dict[tuple[int, int, str], list[tuple[int, InTextCitation]]] = {}
    for index, citation in enumerate(citations):
        key = (citation.passage_start, citation.passage_end, citation.text)
        grouped.setdefault(key, []).append((index, citation))

    replacements: dict[int, list[InTextCitation]] = {}
    rejected_indexes: set[int] = set()
    consumed_indexes: set[int] = set()
    for detections in grouped.values():
        reference_ids = {
            reference_id
            for _index, citation in detections
            for reference_id in citation.reference_ids
        }
        marker_identities = {
            (
                citation.citation_marker,
                citation.marker_start,
                citation.marker_end,
            )
            for _index, citation in detections
        }
        if len(reference_ids) <= 1 or len(marker_identities) <= 1:
            continue
        resolved = _split_source_specific_clauses(detections)
        if resolved is None:
            resolved = _group_collective_narrative_markers(detections)
        indexes = {index for index, _citation in detections}
        if resolved is None:
            rejected_indexes.update(indexes)
            continue
        first_index = min(indexes)
        replacements[first_index] = resolved
        consumed_indexes.update(indexes)

    kept: list[InTextCitation] = []
    for index, citation in enumerate(citations):
        if index in replacements:
            kept.extend(replacements[index])
        elif index not in consumed_indexes and index not in rejected_indexes:
            kept.append(citation)
    rejected = [
        citation.model_copy(
            update={"drop_reason": "multi_marker_citation_unit_requires_grouping"}
        )
        for index, citation in enumerate(citations)
        if index in rejected_indexes
    ]
    return kept, rejected


def _split_source_specific_clauses(
    detections: list[tuple[int, InTextCitation]],
) -> list[InTextCitation] | None:
    """Split only when semicolons give every marker its own exact clause."""
    text = detections[0][1].text
    if ";" not in text:
        return None
    marker_spans = _located_marker_detections(detections)
    if marker_spans is None:
        return None
    raw_clauses: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r";", text):
        raw_clauses.append((start, match.start()))
        start = match.end()
    raw_clauses.append((start, len(text)))
    clauses = [
        trimmed
        for raw_start, raw_end in raw_clauses
        if (trimmed := _trim_span(text, raw_start, raw_end)) is not None
    ]
    assignments: list[tuple[tuple[int, int], tuple[int, int, InTextCitation]]] = []
    for clause in clauses:
        members = [
            marker
            for marker in marker_spans
            if clause[0] <= marker[0] and marker[1] <= clause[1]
        ]
        if members:
            if len(members) != 1:
                return None
            assignments.append((clause, members[0]))
    if len(assignments) != len(marker_spans) or len(assignments) < 2:
        return None

    split: list[InTextCitation] = []
    for (clause_start, clause_end), (marker_start, marker_end, citation) in assignments:
        clause_text = text[clause_start:clause_end]
        split.append(
            citation.model_copy(
                update={
                    "text": clause_text,
                    "passage_start": citation.passage_start + clause_start,
                    "passage_end": citation.passage_start + clause_end,
                    "citation_markers": [
                        CitationMarkerMember(
                            text=text[marker_start:marker_end],
                            local_start=marker_start - clause_start,
                            local_end=marker_end - clause_start,
                            reference_ids=list(citation.reference_ids),
                            marker_type=citation.marker_type,
                        )
                    ],
                }
            )
        )
    return split


def _group_collective_narrative_markers(
    detections: list[tuple[int, InTextCitation]],
) -> list[InTextCitation] | None:
    """Group adjacent narrative sources that share one explicit predicate."""
    if any(citation.marker_type != "narrative" for _index, citation in detections):
        return None
    marker_spans = _located_marker_detections(detections)
    if marker_spans is None or len(marker_spans) < 2:
        return None
    text = detections[0][1].text
    if not _COLLECTIVE_PREFIX.fullmatch(text[:marker_spans[0][0]]):
        return None
    gaps = [
        text[left[1]:right[0]]
        for left, right in zip(marker_spans, marker_spans[1:])
    ]
    if (
        not all(_COORDINATED_MARKER_GAP.fullmatch(gap) for gap in gaps)
        or not any(re.search(r"\b(?:and|or)\b|&", gap, re.IGNORECASE) for gap in gaps)
    ):
        return None
    cluster_start = marker_spans[0][0]
    cluster_end = marker_spans[-1][1]
    if not _COLLECTIVE_REPORTING_SUFFIX.match(text[cluster_end:]):
        return None

    ordered_reference_ids: list[str] = []
    candidate_reference_ids: list[str] = []
    marker_members: list[CitationMarkerMember] = []
    for marker_start, marker_end, citation in marker_spans:
        for reference_id in citation.reference_ids:
            if reference_id not in ordered_reference_ids:
                ordered_reference_ids.append(reference_id)
        for reference_id in citation.candidate_reference_ids:
            if reference_id not in candidate_reference_ids:
                candidate_reference_ids.append(reference_id)
        marker_members.append(
            CitationMarkerMember(
                text=text[marker_start:marker_end],
                local_start=marker_start,
                local_end=marker_end,
                reference_ids=list(citation.reference_ids),
                marker_type=citation.marker_type,
            )
        )
    first = marker_spans[0][2]
    merged = first.model_copy(
        update={
            "reference_ids": ordered_reference_ids,
            "candidate_reference_ids": candidate_reference_ids,
            "citation_key": "|".join(
                citation.citation_key
                for _start, _end, citation in marker_spans
                if citation.citation_key
            ),
            "citation_marker": text[cluster_start:cluster_end],
            "citation_markers": marker_members,
            "marker_member": " | ".join(
                citation.marker_member or citation.citation_marker
                for _start, _end, citation in marker_spans
            ),
            "marker_start": min(citation.marker_start for _s, _e, citation in marker_spans),
            "marker_end": max(citation.marker_end for _s, _e, citation in marker_spans),
            "page_number": next(
                (
                    citation.page_number
                    for _s, _e, citation in marker_spans
                    if citation.page_number
                ),
                "",
            ),
        }
    )
    return [merged]


def _located_marker_detections(
    detections: list[tuple[int, InTextCitation]],
) -> list[tuple[int, int, InTextCitation]] | None:
    """Locate every distinct marker exactly once in sentence order."""
    text = detections[0][1].text
    ordered = sorted(detections, key=lambda item: (item[1].marker_start, item[0]))
    located: list[tuple[int, int, InTextCitation]] = []
    cursor = 0
    for _index, citation in ordered:
        marker = citation.citation_marker
        if not marker:
            return None
        start = text.find(marker, cursor)
        if start < 0:
            return None
        end = start + len(marker)
        located.append((start, end, citation))
        cursor = end
    return located


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def bounded_antecedent_context(
    body_text: str,
    passage_start: int,
    *,
    max_sentences: int = 2,
    max_characters: int = 2_000,
) -> list[ClaimContextSegment]:
    """Return up to two exact preceding sentences with paper coordinates."""
    if passage_start < 0 or passage_start > len(body_text):
        raise PaperExtractionError("Citation start is outside the paper body")
    bounded_count = max(0, min(max_sentences, 2))
    if bounded_count == 0:
        return []

    sentence_spans: list[tuple[int, int, str]] = []
    paragraph_pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for paragraph_match in paragraph_pattern.finditer(body_text):
        paragraph = paragraph_match.group(0)
        cursor = 0
        for sentence in split_sentences(paragraph):
            local_start = paragraph.find(sentence, cursor)
            if local_start < 0:
                continue
            local_end = local_start + len(sentence)
            cursor = local_end
            start = paragraph_match.start() + local_start
            end = paragraph_match.start() + local_end
            if end <= passage_start:
                sentence_spans.append((start, end, body_text[start:end]))

    selected: list[tuple[int, int, str]] = []
    characters = 0
    for item in reversed(sentence_spans):
        added = len(item[2])
        if selected and characters + added > max_characters:
            break
        if not selected and added > max_characters:
            item = (item[0], item[0] + max_characters, item[2][:max_characters])
            added = max_characters
        selected.append(item)
        characters += added
        if len(selected) >= bounded_count:
            break
    selected.reverse()
    total = len(selected)
    return [
        ClaimContextSegment(
            context_index=index,
            distance_before=total - index,
            text=text,
            paper_start=start,
            paper_end=end,
        )
        for index, (start, end, text) in enumerate(selected)
    ]


def _format_name(parser: type) -> str:
    name = parser.__name__.lower()
    if name.startswith("apa"):
        return "apa"
    if name.startswith("mla"):
        return "mla"
    raise PaperExtractionError(f"Unsupported detected parser: {parser.__name__}")


def _body_before_reference_section(
    paper_text: str,
    reference_section: str,
    parser: type,
) -> str:
    """Return the body preceding the exact extracted bibliography section.

    Heading-based extraction returns text after the heading. Remove that
    heading from the body as well; a content-based fallback has no heading and
    therefore simply cuts at the first bibliography entry.
    """

    section_start = paper_text.rfind(reference_section)
    if section_start < 0:
        raise PaperExtractionError("Extracted reference section lost source alignment")

    body = paper_text[:section_start].rstrip()
    headings = getattr(parser, "HEADINGS", [])
    if headings:
        tag = r"(?:<[^>]+>)*"
        heading_pattern = "|".join(headings)
        body = re.sub(
            rf"(?im)^\s*{tag}\s*(?:{heading_pattern})\s*{tag}\s*:?\s*$",
            "",
            body,
        ).rstrip()
    return body
