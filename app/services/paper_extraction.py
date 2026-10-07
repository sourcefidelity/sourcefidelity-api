"""Application-level orchestration for paper reference/citation extraction.

This module joins the already-validated extraction services without taking on
upload persistence, source retrieval, verification judgment, or reporting.
Keeping that boundary explicit lets the application preserve stable identities
and auditable rejected spans before later pipeline stages are implemented.
"""

from __future__ import annotations

import re
import hashlib
import unicodedata
from typing import Optional, Literal

from pydantic import BaseModel, Field, model_serializer
from app.services.body_title_formatting import BodyTitleAssessment, assess_body_title_italics

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
from app.services.reference_consistency import (
    ReferenceConsistencyAssessment,
    assess_reference_consistency,
)
from app.services.reference_layout import ReferenceLayoutArtifact
from app.services.submitted_locator_inventory import SubmittedLocatorInventory
from app.services.assessment_configuration import AssessmentConfiguration
from app.services.reference_formatting import ReferenceFormattingAssessment
from app.services.quotation_locator_requirement import QuotationLocatorRequirement, assess_quotation_locators
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


class CitationMarkerCensusEntry(BaseModel):
    """One structurally visible marker that must never disappear from a report."""

    marker_id: str
    text: str
    passage_start: int
    passage_end: int
    reference_ids: list[str] = Field(default_factory=list)
    candidate_reference_ids: list[str] = Field(default_factory=list)
    link_status: str
    marker_type: str
    paragraph_index: int = Field(ge=0)
    member_count: int = Field(ge=1)
    coverage_status: str = "not_assessed"
    reason_code: str = "citation_scope_or_member_unresolved"
    claim_id: str | None = None
    report_text: str | None = None
    report_passage_start: int | None = None
    report_passage_end: int | None = None
    # citation-reference-tolerance-v1: what differs when a tolerant rule linked it.
    link_differences: list[str] = Field(default_factory=list)


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
    citation_marker_census: list[CitationMarkerCensusEntry] = Field(
        default_factory=list
    )
    body_character_count: int = 0
    reference_section_character_count: int = 0
    total_word_count: int = 0
    body_word_count: int = 0
    reference_word_count: int = 0
    reference_consistency: ReferenceConsistencyAssessment | None = None
    reference_layout: ReferenceLayoutArtifact | None = None
    submitted_locator_inventory: SubmittedLocatorInventory | None = None
    assessment_configuration: AssessmentConfiguration = Field(default_factory=AssessmentConfiguration)
    # Missing historical policy retains its original evidence requirements.
    required_doi_policy_version: Literal['apa7_verified_doi_required_v1', 'apa7_bibliographic_doi_required_v2'] = 'apa7_verified_doi_required_v1'
    required_author_policy_version: Literal['apa7_verified_journal_author_required_v1'] | None = None
    reference_formatting: ReferenceFormattingAssessment | None = None
    body_title_formatting: BodyTitleAssessment | None = None
    quotation_locator_requirements: list[QuotationLocatorRequirement] = Field(default_factory=list)
    text_extraction: dict = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def preserve_body_title_snapshot_absence(self, handler):
        """Do not add a new null field to a previously hash-bound snapshot.

        Explicit nulls and observations remain serialized. Only the field that
        was absent before this extension is omitted, not other legacy defaults.
        """
        value = handler(self)
        if 'body_title_formatting' not in self.model_fields_set:
            value.pop('body_title_formatting', None)
        return value


_DATE_QUALIFIER = r"(?:(?:c\.|ca\.|circa|after|before|since|until|from|by|early|late|mid-?)\s*)*"
_DATE_ONLY_ASIDE = re.compile(
    rf"\(\s*{_DATE_QUALIFIER}(?:1[5-9]|20)\d{{2}}s?"
    rf"\s*(?:(?:[-–—/]|\bto\b|\buntil\b)\s*(?:{_DATE_QUALIFIER}(?:(?:1[5-9]|20)\d{{2}}|\d{{2}})s?|present|now|today))?\s*\)",
    re.IGNORECASE,
)

def extract_paper_evidence(
    paper_text: str,
    *,
    paper_version_id: str,
    reference_text: str | None = None,
    reference_layout_text: str | None = None,
    format_hint: Optional[str] = None,
    use_llm_boundaries: bool = True,
    use_llm_atomizer: bool = True,
    use_llm_reference_fallback: bool = True,
    signals: Optional[SignalConfig] = None,
    docx_content: bytes | None = None,
    link_targets: frozenset[str] = frozenset(),
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

    reference_source_text = reference_text or paper_text
    reference_section = extract_reference_section(reference_source_text, citation_format)
    if not reference_section:
        raise PaperExtractionError("No reference section found")

    body_reference_section = extract_reference_section(paper_text, citation_format)
    if not body_reference_section:
        raise PaperExtractionError("No reference section found in semantic paper text")
    body_text = _body_before_reference_section(
        paper_text,
        body_reference_section,
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

    if reference_layout_text:
        from app.services.reference_url_repair import repair_reference_urls

        references = repair_reference_urls(references, reference_layout_text, link_targets=link_targets)

    structural_detections = extract_citations(
        body_text,
        references,
        format_hint=citation_format,
        use_llm_boundaries=False,
        signals=signals or SignalConfig.all_off(),
        extractor="cite",
    )
    marker_census = _build_marker_census(structural_detections, body_text, references)
    if use_llm_boundaries:
        llm_detections = extract_citations(
            body_text,
            references,
            format_hint=citation_format,
            use_llm_boundaries=True,
            signals=signals or SignalConfig.all_off(),
            extractor="cite",
        )
        extracted = _merge_additive_llm_recovery(
            structural_detections,
            llm_detections,
            marker_census,
            body_text,
        )
    else:
        extracted = structural_detections
    extracted = _recover_linked_sentence_citations(
        extracted,
        marker_census,
        body_text,
    )
    extracted = source_headed_sections(extracted, marker_census, body_text, references)
    extracted = join_adjacent_continuations(extracted, body_text, references)
    extracted = join_author_naming_sentences(extracted, marker_census, body_text, references)
    extracted = yearless_author_mentions(extracted, marker_census, body_text, references)
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
        incomplete_markers = _unrepresented_markers(parent_claim, marker_census)
        if incomplete_markers:
            claim_boundary_rejections.append(
                ClaimBoundaryRejection(
                    passage_start=citation.passage_start,
                    passage_end=citation.passage_end,
                    citation_marker=citation.citation_marker,
                    reason="citation_marker_membership_incomplete",
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

    marker_census = _bind_marker_census(marker_census, citation_claims)
    reference_consistency = assess_reference_consistency(
        paper_version_id=version_id,
        citation_format=citation_format,
        references=references,
        citations=[*accepted, *rejected],
    )
    return PaperExtractionArtifact(
        body_title_formatting=(BodyTitleAssessment.model_validate(assess_body_title_italics(
            content=docx_content, body=body_text, references=references,
            citation_format=citation_format)) if docx_content is not None else None),
        required_doi_policy_version='apa7_bibliographic_doi_required_v2',
        required_author_policy_version='apa7_verified_journal_author_required_v1',
        paper_version_id=version_id,
        citation_format=citation_format,
        references=references,
        citations=accepted,
        rejected_citations=rejected,
        citation_claims=citation_claims,
        atomizations=atomizations,
        eligible_atomic_claims=eligible_atomic_claims,
        claim_boundary_rejections=claim_boundary_rejections,
        citation_marker_census=marker_census,
        body_character_count=len(body_text),
        reference_section_character_count=len(reference_section),
        total_word_count=_word_count(paper_text),
        body_word_count=_word_count(body_text),
        reference_word_count=_word_count(reference_section),
        reference_consistency=reference_consistency,
        quotation_locator_requirements=assess_quotation_locators(
            body_text=body_text,citation_format=citation_format,claims=citation_claims,references=references,
        ),
    )


def _word_count(value: str) -> int:
    """Count human-readable word tokens without changing retained text."""
    return len(re.findall(r"[^\W_]+(?:['’\u2011-][^\W_]+)*", value, re.UNICODE))


def _build_marker_census(
    detections: list[InTextCitation],
    body_text: str,
    references: list[ParsedReference],
) -> list[CitationMarkerCensusEntry]:
    grouped: dict[tuple[int, int, str], list[InTextCitation]] = {}
    for citation in detections:
        marker = citation.citation_marker
        if not marker or marker == "implicit_continuation" or citation.passage_start < 0:
            continue
        local_start = citation.text.find(marker)
        if local_start < 0:
            continue
        start = citation.passage_start + local_start
        end = start + len(marker)
        grouped.setdefault((start, end, marker), []).append(citation)

    census: list[CitationMarkerCensusEntry] = []
    for (start, end, marker), members in sorted(grouped.items()):
        reference_ids = sorted(
            {
                reference_id
                for citation in members
                for reference_id in citation.reference_ids
            }
        )
        candidate_ids = sorted(
            {
                reference_id
                for citation in members
                for reference_id in citation.candidate_reference_ids
            }
        )
        statuses = {citation.link_status for citation in members}
        link_status = (
            "linked"
            if statuses == {"linked"} and reference_ids
            else "ambiguous"
            if "ambiguous" in statuses or candidate_ids
            else "missing_reference"
        )
        member_labels = {
            (citation.marker_member or citation.citation_marker).strip()
            for citation in members
        }
        marker_id = hashlib.sha256(
            f"citation-marker-v1:{start}:{end}:{marker}".encode("utf-8")
        ).hexdigest()
        report_span = _sentence_span_containing(body_text, start, required_end=end)
        census.append(
            CitationMarkerCensusEntry(
                marker_id=marker_id,
                text=marker,
                passage_start=start,
                passage_end=end,
                reference_ids=reference_ids,
                candidate_reference_ids=candidate_ids,
                link_status=link_status,
                marker_type=members[0].marker_type,
                paragraph_index=members[0].paragraph_index,
                member_count=max(1, len(member_labels)),
                reason_code=(
                    "citation_scope_or_member_unresolved"
                    if link_status == "linked"
                    else "citation_reference_ambiguous"
                    if link_status == "ambiguous"
                    else "citation_reference_missing"
                ),
                report_passage_start=(report_span[0] if report_span else None),
                report_passage_end=(report_span[1] if report_span else None),
                report_text=(report_span[2] if report_span else None),
            )
        )
    broad_parenthetical = re.compile(
        # A single line break may fall inside a parenthesis (PDF wrapping).
        r"\((?:(?!\n\s*\n)[^()]){0,250}\b(?:19|20)\d{2}[a-z]?\b(?:(?!\n\s*\n)[^()]){0,250}\)",
        re.IGNORECASE,
    )
    for match in broad_parenthetical.finditer(body_text):
        if any(
            marker.passage_start <= match.start()
            and match.end() <= marker.passage_end
            for marker in census
        ):
            continue
        marker = match.group(0)
        # A complete calendar date is an aside, not an author/year citation.
        # Previously detected explicit source markers remain preserved above.
        if re.fullmatch(r'\(\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(?:19|20)\d{2}\s*\)', marker, re.I):
            continue
        # A date-only range defining an explicitly named historical period is
        # not another cited source. Recognized narrative markers were retained
        # above, so this does not discard a matched title/author-year citation.
        if (re.fullmatch(r'\(\s*(?:18|19|20)\d{2}\s*[-–—]\s*(?:18|19|20)\d{2}\s*\)', marker)
                and re.search(r'\b(?:period|era|years)\s*$', body_text[max(0,match.start()-100):match.start()], re.I)):
            continue
        # A date range, open range, decade or qualified year with no author text
        # ("(1916–1965)", "(1966–present)", "(c. 1916–1965)", "(after c. 1966)") is
        # an aside, not a citation (paper 5, 2026-10-01), unless a reference's year
        # is exactly that date text. A bare single year is left to other rules.
        if _DATE_ONLY_ASIDE.fullmatch(marker) and not re.fullmatch(r'\(\s*(?:18|19|20)\d{2}[a-rt-z]?\s*\)', marker):
            inner = marker.strip('() ')
            if not any((getattr(r, 'year', '') or '').strip() == inner for r in references):
                continue
        linked_ids, candidate_ids = _link_broad_parenthetical_marker(
            marker, references
        )
        link_differences: list[str] = []
        if not linked_ids and not candidate_ids:
            linked_ids, link_differences = _misspelt_narrative_author(body_text, match.start(), marker, references)
        sentence_span = (
            _sentence_span_containing(
                body_text,
                match.start(),
                required_end=match.end(),
            )
        )
        marker_id = hashlib.sha256(
            f"citation-marker-v1:{match.start()}:{match.end()}:{marker}".encode(
                "utf-8"
            )
        ).hexdigest()
        census.append(
            CitationMarkerCensusEntry(
                marker_id=marker_id,
                text=marker,
                passage_start=match.start(),
                passage_end=match.end(),
                reference_ids=linked_ids,
                candidate_reference_ids=candidate_ids,
                link_status=(
                    "linked"
                    if linked_ids
                    else "ambiguous"
                    if candidate_ids
                    else "missing_reference"
                ),
                marker_type="parenthetical",
                paragraph_index=_paragraph_index_at(body_text, match.start()),
                member_count=max(1, marker.count(";") + 1),
                reason_code=(
                    "citation_scope_or_member_unresolved"
                    if linked_ids
                    else "citation_reference_ambiguous"
                    if candidate_ids
                    else "citation_marker_not_parsed"
                ),
                report_passage_start=(sentence_span[0] if sentence_span else None),
                report_passage_end=(sentence_span[1] if sentence_span else None),
                report_text=(sentence_span[2] if sentence_span else None),
                link_differences=link_differences,
            )
        )
    census.sort(key=lambda item: (item.passage_start, item.passage_end, item.marker_id))
    return census


def _misspelt_narrative_author(body_text: str, start: int, marker: str, references) -> tuple[list[str], list[str]]:
    """Link "Hartmn (2016)" to Langford (2010): a bare year after a word that is a
    close misspelling of exactly one reference's lead surname (paper 5, 2026-10-01).
    The year may differ; both differences are kept for the report."""
    from app.services.citation_extractor import _reference_lead_surnames, close_surname
    bare = re.fullmatch(r"\(\s*((?:19|20)\d{2})[a-z]?\s*\)", marker)
    word = re.search(r"([^\W\d_][^\W\d_'’-]*(?:['’-][^\W\d_]+)*)\s*$", body_text[max(0, start - 60):start])
    if not bare or not word:
        return [], []
    close = [ref for ref in references
             if _reference_lead_surnames(ref) and close_surname(word.group(1), _reference_lead_surnames(ref)[0])]
    if len(close) != 1:
        return [], []
    ref = close[0]
    differences = [f"author_spelling:{_reference_lead_surnames(ref)[0]}"]
    if (ref.year or "").strip()[:4] != bare.group(1):
        differences.append(f"year:{ref.year or 'n.d.'}")
    return [ref.reference_id], differences


def _link_broad_parenthetical_marker(
    marker: str,
    references: list[ParsedReference],
) -> tuple[list[str], list[str]]:
    """Recover uniquely identifiable author-year members outside strict syntax."""
    normalized_marker = unicodedata.normalize("NFKD", marker.casefold())
    marker_author_keys = {
        re.sub(r"[^a-z0-9]+", "", token.casefold())
        for token in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", normalized_marker)
    }
    years = set(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", normalized_marker))
    if not years:
        return [], []
    matches = []
    for reference in references:
        if reference.year.casefold() not in years:
            continue
        author = reference.author.strip()
        primary = author.split(",", 1)[0].strip()
        if "," not in author:
            primary = author.split()[0] if author.split() else ""
        author_key = re.sub(
            r"[^a-z0-9]+",
            "",
            unicodedata.normalize("NFKD", primary.casefold()),
        )
        if len(author_key) < 3:
            citation_key = re.sub(
                r"(?:19|20)\d{2}[a-z]?$", "", reference.citation_key.casefold()
            )
            author_key = re.sub(r"[^a-z0-9]+", "", citation_key).strip()
        if author_key and author_key in marker_author_keys:
            matches.append(reference.reference_id)
    unique = sorted(set(matches))
    if not unique:
        return [], []
    expected_members = max(1, marker.count(";") + 1)
    if len(unique) == expected_members:
        return unique, []
    return [], unique


def _sentence_span_containing(
    text: str,
    offset: int,
    *,
    required_end: int | None = None,
) -> tuple[int, int, str] | None:
    """Return one exact sentence span around a marker without guessing scope."""
    paragraph_start = 0
    paragraph_end = len(text)
    for separator in re.finditer(r"\n\s*\n", text):
        if separator.end() <= offset:
            paragraph_start = separator.end()
            continue
        paragraph_end = separator.start()
        break
    paragraph = text[paragraph_start:paragraph_end]
    stripped = paragraph.strip()
    if not stripped:
        return None
    local_base = paragraph_start + paragraph.find(stripped)
    cursor = 0
    sentence_spans: list[tuple[int, int]] = []
    for sentence in split_sentences(stripped):
        local_start = stripped.find(sentence, cursor)
        if local_start < 0:
            continue
        local_end = local_start + len(sentence)
        cursor = local_end
        sentence_spans.append((local_base + local_start, local_base + local_end))
    for index, (start, end) in enumerate(sentence_spans):
        if start <= offset < end:
            target_end = max(offset + 1, required_end or offset + 1)
            extension_index = index + 1
            while end < target_end and extension_index < len(sentence_spans):
                end = sentence_spans[extension_index][1]
                extension_index += 1
            if end < target_end:
                return None
            start = _trim_report_prefix(text, start, end)
            return start, end, text[start:end]
    return None


def _recover_linked_sentence_citations(
    citations: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
) -> list[InTextCitation]:
    """Recover exact sentence units when source membership is deterministic.

    This is the ordinary fallback for a visible, uniquely linked marker that
    strict extraction or the bounded LLM pass did not turn into a citation
    unit. Recovery is allowed only when every census marker in the exact
    sentence is linked; an ambiguous or missing member blocks the sentence.
    """
    accepted = [item for item in citations if item.drop_reason is None]
    uncovered = [
        marker
        for marker in census
        if marker.link_status == "linked"
        and marker.reference_ids
        and not set(marker.reference_ids).issubset(
            {
                reference_id
                for citation in accepted
                if citation.passage_start <= marker.passage_start
                and marker.passage_end <= citation.passage_end
                for reference_id in citation.reference_ids
            }
        )
    ]
    sentence_groups: dict[tuple[int, int, str], list[CitationMarkerCensusEntry]] = {}
    for marker in uncovered:
        sentence = _sentence_span_containing(
            body_text,
            marker.passage_start,
            required_end=marker.passage_end,
        )
        if sentence is not None:
            sentence_groups.setdefault(sentence, []).append(marker)

    recovered: list[InTextCitation] = []
    for (start, end, text), _uncovered_markers in sorted(
        sentence_groups.items(), key=lambda item: item[0][:2]
    ):
        sentence_markers = [
            marker
            for marker in census
            if start <= marker.passage_start and marker.passage_end <= end
        ]
        if not sentence_markers or any(
            marker.link_status != "linked" or not marker.reference_ids
            for marker in sentence_markers
        ):
            continue
        reference_ids = sorted(
            {
                reference_id
                for marker in sentence_markers
                for reference_id in marker.reference_ids
            }
        )
        marker_members = [
            CitationMarkerMember(
                text=marker.text,
                local_start=marker.passage_start - start,
                local_end=marker.passage_end - start,
                reference_ids=list(marker.reference_ids),
                marker_type=marker.marker_type,
            )
            for marker in sentence_markers
        ]
        locator_matches = re.findall(
            r"\bp{1,2}\.?\s*(\d+(?:\s*[-–—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*)\s*(?=[);]|$)",
            " ".join(marker.text for marker in sentence_markers),
            flags=re.IGNORECASE,
        )
        recovered.append(
            InTextCitation(
                reference_ids=reference_ids,
                text=text,
                claim_type=(
                    "quotation"
                    if re.search(r'["“][^"”\n]+["”]', text)
                    else "paraphrase"
                ),
                citation_marker="; ".join(
                    marker.text for marker in sentence_markers
                ),
                citation_markers=marker_members,
                marker_type=(
                    sentence_markers[0].marker_type
                    if len({marker.marker_type for marker in sentence_markers}) == 1
                    else "parenthetical"
                ),
                page_number=(locator_matches[0] if len(locator_matches) == 1 else ""),
                paragraph_index=sentence_markers[0].paragraph_index,
                passage_start=start,
                passage_end=end,
                confidence="high",
            )
        )
    return [*citations, *recovered]


_REPORT_SECTION_HEADING = re.compile(
    r"^\s*(?:abstract|introduction|background|literature\s+review|"
    r"method(?:s|ology)?|results?|discussion|conclusion|references?|"
    r"works\s+cited)\s*(?:\n+|$)",
    re.IGNORECASE,
)
_REPORT_SPEAKER_PREFIX = re.compile(
    r"^\s*\((?![^)\n]*(?:19|20)\d{2})[^)\n]{1,80}\)\s*\n+",
    re.IGNORECASE,
)


def _trim_report_prefix(text: str, start: int, end: int) -> int:
    candidate = text[start:end]
    while candidate:
        match = _REPORT_SECTION_HEADING.match(candidate)
        if match is None:
            match = _REPORT_SPEAKER_PREFIX.match(candidate)
        if match is None:
            break
        start += match.end()
        candidate = text[start:end]
    while start < end and text[start].isspace():
        start += 1
    return start


def _bounded_narrative_extension(
    base: InTextCitation,
    proposals: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body: str,
) -> InTextCitation | None:
    """Accept only an exact, unopposed forward proposal with its original marker.

    This checks structural eligibility, not semantic truth. Attribution is still
    the ordinary source-blind extractor's proposal; no source text is consulted.
    """
    if (base.drop_reason or base.marker_type != "narrative"
            or base.link_status != "linked" or len(base.reference_ids) != 1
            or base.passage_start < 0
            or body[base.passage_start:base.passage_end] != base.text):
        return None
    matching = [p for p in proposals if not p.drop_reason
                and p.passage_start == base.passage_start
                and p.passage_end >= base.passage_end
                and p.link_status == "linked"
                and p.reference_ids == base.reference_ids]
    # Different proposed extents are an unresolved scope disagreement, even if
    # the shorter extent is simply the original deterministic sentence.
    if len({p.passage_end for p in matching}) != 1:
        return None
    proposal = matching[0]
    if (proposal.passage_end <= base.passage_end
            or proposal.passage_end - base.passage_end > 1000
            or body[proposal.passage_start:proposal.passage_end] != proposal.text
            or re.search(r"\n\s*\n", proposal.text)):
        return None
    if any(m.passage_start >= base.passage_end
           and m.passage_start < proposal.passage_end for m in census):
        return None
    tail = body[base.passage_end:proposal.passage_end].strip()
    sentences = split_sentences(tail)
    if not 1 <= len(sentences) <= 2:
        return None
    cursor = base.passage_end
    for sentence in sentences:
        start = body.find(sentence, cursor, proposal.passage_end)
        if start < 0 or body[cursor:start].strip():
            return None
        end = start + len(sentence)
        span = _sentence_span_containing(body, start)
        if span is None or span[:2] != (start, end):
            return None
        cursor = end
    if cursor != proposal.passage_end:
        return None
    from app.services.citation_extractor import _detect_claim_type
    return base.model_copy(update={
        "text": proposal.text, "passage_end": proposal.passage_end,
        "claim_type": _detect_claim_type(proposal.text), "confidence": "medium",
    })


def _merge_additive_llm_recovery(
    structural: list[InTextCitation],
    llm: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
) -> list[InTextCitation]:
    """Preserve deterministic detections and admit only bounded LLM additions."""
    merged = []
    for base in structural:
        extension = _bounded_narrative_extension(base, llm, census, body_text)
        if extension is None:
            merged.append(base)
        else:
            # Retain the exact deterministic detection in the rejected/audit
            # collection; never silently overwrite its original wording/span.
            merged.append(base.model_copy(update={
                "drop_reason": "deterministic_base_of_bounded_narrative_extension_v1",
            }))
            merged.append(extension)
    structural_marker_ids = {
        marker.marker_id
        for marker in census
        if any(
            citation.passage_start <= marker.passage_start
            and marker.passage_end <= citation.passage_end
            and set(marker.reference_ids).issubset(citation.reference_ids)
            for citation in structural
        )
    }
    accepted_explicit = [citation for citation in merged if citation.drop_reason is None]
    for citation in llm:
        if citation.drop_reason is not None:
            merged.append(citation)
            continue
        if citation.citation_marker == "implicit_continuation":
            if _valid_implicit_continuation(citation, accepted_explicit, census, body_text):
                merged.append(citation)
            else:
                merged.append(
                    citation.model_copy(
                        update={"drop_reason": "llm_continuation_not_structurally_anchored"}
                    )
                )
            continue

        represented = [
            marker
            for marker in census
            if citation.passage_start <= marker.passage_start
            and marker.passage_end <= citation.passage_end
            and marker.link_status == "linked"
            and set(marker.reference_ids).issubset(citation.reference_ids)
        ]
        new_markers = [
            marker for marker in represented if marker.marker_id not in structural_marker_ids
        ]
        if not new_markers:
            continue
        sentence_spans = {
            _sentence_span_containing(
                body_text,
                marker.passage_start,
                required_end=marker.passage_end,
            )
            for marker in new_markers
        }
        sentence_spans.discard(None)
        if len(sentence_spans) != 1:
            merged.append(
                citation.model_copy(
                    update={"drop_reason": "llm_recovery_not_one_deterministic_sentence"}
                )
            )
            continue
        start, end, text = next(iter(sentence_spans))
        sentence_markers = [
            marker
            for marker in census
            if start <= marker.passage_start and marker.passage_end <= end
            and marker.link_status == "linked"
        ]
        reference_ids = sorted(
            {reference_id for marker in sentence_markers for reference_id in marker.reference_ids}
        )
        marker_members = [
            CitationMarkerMember(
                text=marker.text,
                local_start=marker.passage_start - start,
                local_end=marker.passage_end - start,
                reference_ids=list(marker.reference_ids),
                marker_type=marker.marker_type,
            )
            for marker in sentence_markers
        ]
        recovered = citation.model_copy(
            update={
                "text": text,
                "passage_start": start,
                "passage_end": end,
                "reference_ids": reference_ids,
                "citation_markers": marker_members,
                "citation_marker": "; ".join(marker.text for marker in sentence_markers),
                "link_status": "linked",
                "confidence": "medium",
            }
        )
        merged.append(recovered)
        accepted_explicit.append(recovered)
        structural_marker_ids.update(marker.marker_id for marker in sentence_markers)
    return merged


SOURCE_SECTION_VERSION = "source-headed-section-v1"
SOURCE_HEADING_MARKER_TYPE = "source_heading"


def source_heading_statement_start(text: str, marker: str) -> int:
    """Where a source-heading citation's statement begins: after the heading line."""
    at = text.find(marker) if marker else -1
    line_end = text.find("\n", max(at, 0))
    if line_end < 0:
        return 0
    return len(text) - len(text[line_end:].lstrip())


def _fold_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).casefold()))
_HEADING_MAX_WORDS = 30
_SECTION_MIN_WORDS = 8


def source_headed_sections(
    citations: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
    references,
) -> list[InTextCitation]:
    """Paragraphs under a heading that cites one source are about that source.

    A source-by-source review ("Article 2: Johnson, L. (2021)." followed by
    paragraphs discussing it; owner decision 2026-10-04) attributes each
    following paragraph with no citation of its own to the heading's source,
    up to the next such heading or another short heading ("Conclusion: …").
    It applies only when the paper has at least two such headings, so an
    ordinary one-line cited paragraph never claims the paragraphs after it.
    A film or other media heading is not a source of statements here.

    The heading's own citation is extended over the section's paragraphs, so
    the statement carries the exact marker its Evidence Package needs; no
    marker is made up. The heading-only citation and any model continuation
    inside the section are kept for audit with a drop reason. Returns the
    citations with these changes.
    """
    from app.services.citation_extractor import _original_paragraphs_with_offsets
    kinds = {reference.reference_id: reference.source_kind for reference in references}
    titles = {reference.reference_id: _fold_words(getattr(reference, "title", "") or "") for reference in references}
    linked = [c for c in citations if c.drop_reason is None and c.link_status == "linked"]

    def markers_in(start, end):
        return [m for m in census if start <= m.passage_start < end]

    def heading_source(start, end):
        text = body_text[start:end]
        markers = markers_in(start, end)
        words = re.findall(r"\w+", text)
        if not words or len(words) > _HEADING_MAX_WORDS or len(markers) != 1:
            return None
        marker = markers[0]
        if marker.link_status != "linked" or len(marker.reference_ids) != 1:
            return None
        reference_id = marker.reference_ids[0]
        if kinds.get(reference_id) in {"traditional_media", "video", "podcast_episode"}:
            return None
        before = body_text[start:marker.passage_start].strip()
        # A label ("Article 2:") or nothing may precede the citation; the
        # citation, not a sentence, carries the line.
        if before and not re.fullmatch(r"[^.!?]{0,60}:", before):
            return None
        # After the citation, nothing, a title-cased line or the work's own
        # title: never the rest of a sentence ("Jones (2020) argues …").
        after = body_text[marker.passage_end:end].strip(" .:;,-–—\n\t")
        if after:
            folded = _fold_words(after)
            long_words = [w for w in re.findall(r"[A-Za-z][\w'’-]*", after) if len(w) >= 4]
            title_cased = long_words and sum(w[0].isupper() for w in long_words) >= 0.75 * len(long_words)
            own_title = titles.get(reference_id) and folded.startswith(titles[reference_id][:40])
            if not (title_cased or own_title):
                return None
        return reference_id

    # Units: blank-line paragraphs, split at any line that is a source heading
    # (paper 7 runs a heading, its discussion and the next heading together).
    units = []   # (start, end, heading reference id or None, whole paragraph)
    for text, offset in _original_paragraphs_with_offsets(body_text):
        lines = [(offset + m.start(), offset + m.end()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()]
        run = None
        pieces = []
        for line_start, line_end in lines:
            rid = heading_source(line_start, line_end)
            if rid:
                if run:
                    pieces.append((run[0], run[1], None))
                    run = None
                pieces.append((line_start, line_end, rid))
            else:
                run = (run[0], line_end) if run else (line_start, line_end)
        if run:
            pieces.append((run[0], run[1], None))
        units.extend((a, b, rid, len(pieces) == 1) for a, b, rid in pieces)

    def section_break(start, end, whole):
        text = body_text[start:end].strip()
        words = re.findall(r"\w+", text)
        return whole and len(words) <= 15 and (
            not re.search(r"[.!?][\"”’')]*$", text) or bool(re.match(r"[^.!?]{1,40}:", text)))

    if sum(1 for unit in units if unit[2]) < 2:
        return citations
    # Each heading's run of attributable paragraphs, up to the first that
    # cites something itself, another heading or a section break.
    sections: list[tuple[int, int, str, int]] = []   # heading start/end, reference, section end
    current = None
    for start, end, rid, whole in units:
        if rid:
            current = [start, end, rid, None]
            sections.append(current)
            continue
        if current is None:
            continue
        text = body_text[start:end]
        if section_break(start, end, whole) or markers_in(start, end):
            current = None
            continue
        if len(re.findall(r"\w+", text)) < _SECTION_MIN_WORDS:
            continue
        current[3] = end
    result = list(citations)
    for heading_start, heading_end, rid, section_end in sections:
        if section_end is None:
            continue
        heading = next((i for i, c in enumerate(result) if c.drop_reason is None and c.link_status == "linked"
                        and c.citation_marker != "implicit_continuation" and rid in c.reference_ids
                        and heading_start <= c.passage_start < heading_end), None)
        if heading is None:
            continue
        base = result[heading]
        result[heading] = base.model_copy(update={"drop_reason": "base_of_source_headed_section_v1"})
        for i, other in enumerate(result):
            if (other.drop_reason is None and other.citation_marker == "implicit_continuation"
                    and other.passage_start < section_end and base.passage_start < other.passage_end):
                result[i] = other.model_copy(update={"drop_reason": "within_source_headed_section_v1"})
        result.append(base.model_copy(update={
            "text": body_text[base.passage_start:section_end],
            "passage_end": section_end,
            "confidence": "medium",
            # The heading only names the source: the paragraphs are the
            # statement shown and judged (owner decision 2026-10-04).
            "marker_type": SOURCE_HEADING_MARKER_TYPE,
        }))
    return result


JOINED_CONTINUATION_VERSION = "joined-continuation-v1"


def join_adjacent_continuations(citations: list[InTextCitation], body_text: str,
                                references=()) -> list[InTextCitation]:
    """A follow-on sentence with no citation of its own joins the citation right
    before it (owner decision 2026-10-04, option A: paper 9's "…as Smith (2019)
    outlines. Mara's capacity…"), so the statement carries the exact marker
    its Evidence Package needs and is assessed with it.

    Joined only when the citation before it cites exactly the same sources and
    only spaces lie between them, or sentences that each name the cited
    author with no citation of their own ("Smith contends that …"); never a
    paragraph break. The original citation and the continuation are kept for
    audit with drop reasons.
    """
    from app.services.relevance import extract_surnames
    surnames = {reference.reference_id: {name.casefold() for name in extract_surnames(reference.author or "")}
                for reference in references}

    def attributing_gap(gap: str, reference_ids) -> bool:
        if re.search(r"\n\s*\n", gap) or re.search(r"\((?:[^()]*\d{4}|n\.d\.)[^()]*\)", gap):
            return False
        if not gap.strip():
            return True
        names = set().union(*(surnames.get(rid, set()) for rid in reference_ids)) if reference_ids else set()
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", gap.strip()) if part]
        return bool(names) and all(any(re.search(rf"\b{re.escape(name)}\b", part, re.IGNORECASE) for name in names)
                                   for part in sentences)
    active = sorted((c for c in citations if c.drop_reason is None and c.link_status == "linked"),
                    key=lambda c: (c.passage_start, c.passage_end))
    result = list(citations)
    current = None   # (index in result of the growing citation, its citation)
    for citation in active:
        if citation.citation_marker != "implicit_continuation":
            current = (result.index(citation), citation)
            continue
        if citation.citation_markers or current is None:
            continue
        index, base = current
        gap = body_text[base.passage_end:citation.passage_start]
        if (set(citation.reference_ids) != set(base.reference_ids) or base.passage_end > citation.passage_start
                or not attributing_gap(gap, base.reference_ids)):
            continue
        extended = base.model_copy(update={
            "text": body_text[base.passage_start:citation.passage_end],
            "passage_end": citation.passage_end,
            "confidence": "medium",
        })
        if base.drop_reason is None and result[index] is base:
            result[index] = base.model_copy(update={"drop_reason": "base_of_joined_continuation_v1"})
            result.append(extended)
            index = len(result) - 1
        else:
            result[index] = extended
        result[result.index(citation)] = citation.model_copy(update={"drop_reason": "joined_to_preceding_citation_v1"})
        current = (index, extended)
    return result


def join_author_naming_sentences(
    citations: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
    references,
) -> list[InTextCitation]:
    """Sentences right after a one-source citation that name its author with no
    citation of their own ("Smith contends that …") join it (owner decision
    2026-10-04, paper 9), whether or not the model proposed them.

    Only within the paragraph, only spaces between sentences, and never into
    a sentence that is already part of a citation or holds a marker.
    """
    from app.services.relevance import extract_surnames
    surnames = {reference.reference_id: [name for name in extract_surnames(reference.author or "") if len(name) >= 3]
                for reference in references}
    result = list(citations)
    for index in range(len(result)):
        base = result[index]
        if (base.drop_reason is not None or base.link_status != "linked" or len(base.reference_ids) != 1
                or base.citation_marker == "implicit_continuation"):
            continue
        names = surnames.get(base.reference_ids[0]) or []
        if not names:
            continue
        end = base.passage_end
        while True:
            gap = re.match(r"[ \t]*\n?[ \t]*", body_text[end:])
            start = end + gap.end()
            if start >= len(body_text) or "\n" in gap.group() and re.match(r"\s*\n", body_text[start:]):
                break
            sentence = re.match(r"[^.!?]*[.!?]+[\"”’')]*", body_text[start:])
            if sentence is None:
                break
            stop = start + sentence.end()
            if (not any(re.search(rf"\b{re.escape(name)}\b", sentence.group(), re.IGNORECASE) for name in names)
                    or any(start <= m.passage_start < stop for m in census)
                    or any(c is not base and c.drop_reason is None and c.passage_start < stop and start < c.passage_end
                           for c in result)
                    or re.search(r"\n\s*\n", body_text[start:stop])):
                break
            end = stop
        if end == base.passage_end:
            continue
        result[index] = base.model_copy(update={"drop_reason": "base_of_author_naming_extension_v1"})
        result.append(base.model_copy(update={
            "text": body_text[base.passage_start:end], "passage_end": end, "confidence": "medium"}))
    return result


_REPORTING_VERBS = (r"(?:also\s+|further\s+)?(?:argues|argued|states|stated|notes|noted|claims|claimed|writes|wrote|"
                    r"documents|documented|suggests|suggested|explains|explained|observes|observed|mentions|"
                    r"mentioned|points out|pointed out|describes|described|contends|contended|asserts|asserted|"
                    r"maintains|maintained|uncovers|uncovered|identifies|identified|shows|showed|emphasi[sz]es|"
                    r"emphasi[sz]ed|highlights|highlighted|recovers|recovered|reports|reported)")
_WORK_NOUNS = r"(?:argument|study|biography|view|analysis|work|book|article|research|account|findings|theory|concept)"
_FIRST_NAMES = r"(?:[A-Z][\w'’.-]*\s+){0,3}"


def yearless_author_mentions(
    citations: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
    references,
) -> list[InTextCitation]:
    """A sentence that names a cited author with no year ("Hodges documents …",
    "according to Helsby", "Richard Dyer's argument") is a citation of that
    author's one reference (owner decision 2026-10-06, from the owner's
    highlights). Only for an author whose reference the paper cites elsewhere
    with a marker, whose surname leads exactly one reference, and only for a
    sentence outside every citation and holding no marker; no difference is
    recorded, because APA lets a later narrative mention omit the year.
    """
    from app.services.citation_extractor import _reference_lead_surnames
    active = [c for c in citations if c.drop_reason is None]
    cited = {rid for c in active if c.link_status == "linked" and c.citation_marker != "implicit_continuation"
             for rid in c.reference_ids}
    by_surname: dict[str, list] = {}
    for reference in references:
        lead = _reference_lead_surnames(reference)[:1]
        if lead and len(lead[0]) >= 3 and "," in (reference.author or ""):
            by_surname.setdefault(lead[0], []).append(reference)
    # The surname the paper itself cites a reference by, for an author the
    # list writes first name first ("(Helsby, 2005)" for "Wendy Helsby …").
    for marker in census:
        if marker.link_status == "linked" and len(marker.reference_ids) == 1:
            word = re.match(r"\(?\s*([A-Z][\w'’-]{2,})", marker.text)
            reference = next((r for r in references if r.reference_id == marker.reference_ids[0]), None)
            if word and reference is not None and "," not in (reference.author or "") \
                    and re.search(rf"\b{re.escape(word[1])}\b", reference.author or ""):
                by_surname.setdefault(word[1], [])
                if reference not in by_surname[word[1]]:
                    by_surname[word[1]].append(reference)
    names = {name: refs[0] for name, refs in by_surname.items() if len(refs) == 1 and refs[0].reference_id in cited}
    if not names:
        return citations
    result = list(citations)
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", body_text):
        offset = paragraph.start()
        cursor = 0
        for sentence in split_sentences(paragraph.group()):
            local = paragraph.group().find(sentence, cursor)
            if local < 0:
                continue
            cursor = local + len(sentence)
            start, stop = offset + local, offset + local + len(sentence)
            if (any(start <= m.passage_start < stop for m in census)
                    or any(c.drop_reason is None and c.passage_start < stop and start < c.passage_end for c in result)
                    # The student describing their own essay is not the source's claim.
                    or re.search(r"\b(?:this|my|our|the present)\s+(?:essay|paper|article|study|analysis|research)\b"
                                 r"|\b(?:I|we)\b", sentence)):
                continue
            for name, reference in names.items():
                n = re.escape(name)
                found = (re.search(rf"\b[Aa]ccording to {_FIRST_NAMES}{n}\b(?!\s*\()", sentence)
                         or re.search(rf"\b{n}(?:['’]s?)?\s+{_REPORTING_VERBS}\b", sentence)
                         or re.search(rf"\b{_FIRST_NAMES}{n}['’]s?\s+(?:\w+\s+){{0,2}}?{_WORK_NOUNS}\b", sentence)
                         or re.search(rf"\b(?:introduced|proposed|argued|developed|coined|defined)\s+by\s+{_FIRST_NAMES}{n}\b",
                                      sentence))
                if not found or re.match(r"\s*\(", sentence[found.end():]):
                    continue
                result.append(InTextCitation(
                    reference_ids=[reference.reference_id], link_status="linked", text=sentence,
                    citation_key=reference.citation_key, citation_marker=found.group(0).strip(),
                    marker_member=found.group(0).strip(), marker_type="narrative",
                    passage_start=start, passage_end=stop, confidence="medium"))
                break
    return result


def _valid_implicit_continuation(
    citation: InTextCitation,
    explicit: list[InTextCitation],
    census: list[CitationMarkerCensusEntry],
    body_text: str,
) -> bool:
    if (
        citation.link_status != "linked"
        or not citation.reference_ids
        or citation.passage_start < 0
        or citation.passage_end <= citation.passage_start
        or body_text[citation.passage_start:citation.passage_end] != citation.text
    ):
        return False
    prior = [
        item
        for item in explicit
        if item.passage_end <= citation.passage_start
        and item.paragraph_index == citation.paragraph_index
        and set(citation.reference_ids).issubset(item.reference_ids)
    ]
    if not prior:
        return False
    nearest = max(prior, key=lambda item: item.passage_end)
    gap = body_text[nearest.passage_end:citation.passage_start]
    if len(gap) > 500 or re.search(r"\n\s*\n", gap):
        return False
    return not any(
        nearest.passage_end <= marker.passage_start < citation.passage_start
        and not set(marker.reference_ids).issubset(citation.reference_ids)
        for marker in census
    )


def _paragraph_index_at(text: str, offset: int) -> int:
    return len(re.findall(r"\n\s*\n", text[:offset]))


def _claim_marker_keys(claim: ClaimEvidence) -> set[tuple[int, int, str]]:
    return {
        (
            claim.passage_start + marker.local_start,
            claim.passage_start + marker.local_end,
            marker.text,
        )
        for marker in claim.citation_markers
    }


def _claim_covers_census_marker(
    claim: ClaimEvidence,
    marker: CitationMarkerCensusEntry,
) -> bool:
    exact = (marker.passage_start, marker.passage_end, marker.text)
    if exact in _claim_marker_keys(claim):
        return set(marker.reference_ids).issubset(claim.reference_ids)
    contained = [
        item
        for item in claim.citation_markers
        if marker.passage_start
        <= claim.passage_start + item.local_start
        < claim.passage_start + item.local_end
        <= marker.passage_end
    ]
    represented_ids = {
        reference_id for item in contained for reference_id in item.reference_ids
    }
    return (
        marker.member_count > 1
        and len(contained) >= marker.member_count
        and set(marker.reference_ids).issubset(represented_ids)
    )


def _unrepresented_markers(
    claim: ClaimEvidence,
    census: list[CitationMarkerCensusEntry],
) -> list[str]:
    missing: list[str] = []
    for marker in census:
        if not (
            claim.passage_start <= marker.passage_start
            and marker.passage_end <= claim.passage_end
        ):
            continue
        if (
            not _claim_covers_census_marker(claim, marker)
            or marker.link_status != "linked"
            or not set(marker.reference_ids).issubset(claim.reference_ids)
        ):
            missing.append(marker.marker_id)
    return missing


def _bind_marker_census(
    census: list[CitationMarkerCensusEntry],
    claims: list[ClaimEvidence],
) -> list[CitationMarkerCensusEntry]:
    result: list[CitationMarkerCensusEntry] = []
    for marker in census:
        matches = [
            claim
            for claim in claims
            if _claim_covers_census_marker(claim, marker)
        ]
        if len(matches) == 1 and marker.link_status == "linked":
            result.append(
                marker.model_copy(
                    update={
                        "coverage_status": "covered",
                        "reason_code": "citation_member_bound",
                        "claim_id": matches[0].claim_id,
                    }
                )
            )
        else:
            result.append(marker)
    return result


def report_anchor_citations(
    artifact: PaperExtractionArtifact,
) -> list[InTextCitation]:
    """Return only report-safe claim spans plus every unresolved marker span."""
    claim_keys = {
        (
            claim.passage_start,
            claim.passage_end,
            claim.text,
            tuple(sorted(claim.reference_ids)),
        )
        for claim in artifact.citation_claims
    }
    result = [
        citation
        for citation in artifact.citations
        if (
            citation.passage_start,
            citation.passage_end,
            citation.text,
            tuple(sorted(citation.reference_ids)),
        )
        in claim_keys
    ]
    for group in unresolved_marker_report_groups(artifact.citation_marker_census):
        result.append(
            InTextCitation(
                text=group["text"],
                reference_ids=group["reference_ids"],
                candidate_reference_ids=group["candidate_reference_ids"],
                link_status=group["link_status"],
                citation_marker=group["citation_marker"],
                marker_type=group["marker_type"],
                paragraph_index=group["paragraph_index"],
                passage_start=group["passage_start"],
                passage_end=group["passage_end"],
                confidence="low",
            )
        )
    return result


def unresolved_marker_report_groups(
    census: list[CitationMarkerCensusEntry],
) -> list[dict]:
    """Group exact structural sentences for neutral report navigation only.

    A strict verification boundary can reject a citation unit even when the
    deterministic parser knows the sentence containing its marker. Reusing
    that exact sentence in the report prevents marker-only display without
    admitting the sentence to source-fidelity verification.
    """
    groups: dict[tuple[int, int, str, int], list[CitationMarkerCensusEntry]] = {}
    for marker in census:
        if marker.coverage_status == "covered":
            continue
        start = marker.report_passage_start
        end = marker.report_passage_end
        text = marker.report_text
        if start is None or end is None or not text:
            start, end, text = marker.passage_start, marker.passage_end, marker.text
        groups.setdefault((start, end, text, marker.paragraph_index), []).append(marker)

    result = []
    for (start, end, text, paragraph_index), markers in groups.items():
        reference_ids = sorted({value for item in markers for value in item.reference_ids})
        candidate_ids = sorted(
            {value for item in markers for value in item.candidate_reference_ids}
        )
        statuses = {item.link_status for item in markers}
        marker_texts = list(dict.fromkeys(item.text for item in markers))
        marker_types = {item.marker_type for item in markers}
        result.append(
            {
                "group_id": hashlib.sha256(
                    f"citation-report-group-v1:{start}:{end}:{text}".encode("utf-8")
                ).hexdigest(),
                "text": text,
                "passage_start": start,
                "passage_end": end,
                "paragraph_index": paragraph_index,
                "reference_ids": reference_ids,
                "candidate_reference_ids": candidate_ids,
                "link_status": (
                    "linked"
                    if statuses == {"linked"} and reference_ids
                    else "ambiguous"
                    if "ambiguous" in statuses or candidate_ids
                    else "missing_reference"
                ),
                "citation_marker": "; ".join(marker_texts),
                "marker_type": next(iter(marker_types)) if len(marker_types) == 1 else "mixed",
                "reason_codes": list(dict.fromkeys(item.reason_code for item in markers)),
                "recovered_sentence": any(
                    item.report_passage_start is not None and item.report_text
                    for item in markers
                ),
            }
        )
    return sorted(
        result,
        key=lambda item: (item["passage_start"], item["passage_end"], item["group_id"]),
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
        combined_markers: list[CitationMarkerMember] = []
        seen_markers: set[tuple[int, int, str]] = set()
        for _index, citation in detections:
            marker_items = list(citation.citation_markers)
            if not marker_items and citation.citation_marker:
                local_start = citation.text.find(citation.citation_marker)
                if local_start >= 0:
                    marker_items = [
                        CitationMarkerMember(
                            text=citation.citation_marker,
                            local_start=local_start,
                            local_end=local_start + len(citation.citation_marker),
                            reference_ids=list(citation.reference_ids),
                            marker_type=citation.marker_type,
                        )
                    ]
            for marker in marker_items:
                key = (marker.local_start, marker.local_end, marker.text)
                if key not in seen_markers:
                    seen_markers.add(key)
                    combined_markers.append(marker)
        combined_markers.sort(key=lambda item: (item.local_start, item.local_end))
        winner = winner.model_copy(update={"citation_markers": combined_markers})
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
        if len(reference_ids) <= 1:
            continue
        if len(marker_identities) == 1:
            if len(detections) == 1:
                # The extractor already emitted one compound citation with
                # complete membership.
                continue
            resolved = _group_compound_parenthetical_marker(detections)
        else:
            prepared = _collapse_compound_parenthetical_blocks(detections)
            resolved = (
                _split_source_specific_clauses(prepared)
                if prepared is not None
                else None
            )
        if resolved is None:
            resolved = _group_collective_narrative_markers(detections)
        if resolved is None:
            resolved = _group_complete_sentence_markers(detections)
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


def _group_complete_sentence_markers(
    detections: list[tuple[int, InTextCitation]],
) -> list[InTextCitation] | None:
    """Keep one exact sentence unit while preserving source-specific markers.

    When explicit clause splitting is unavailable, rejecting the complete
    sentence hides real citations from the evidence-first report. This fallback
    makes no claim that every source governs every clause: it retains each
    marker's exact source membership and fans the unchanged citation sentence
    out for source-specific evidence retrieval.
    """
    if not detections or any(
        citation.link_status != "linked" or not citation.reference_ids
        for _index, citation in detections
    ):
        return None
    prepared = _collapse_compound_parenthetical_blocks(detections)
    if prepared is None:
        return None
    located = _located_marker_detections(prepared)
    if located is None or len(located) < 2:
        return None
    text = detections[0][1].text
    reference_ids: list[str] = []
    candidate_reference_ids: list[str] = []
    marker_members: list[CitationMarkerMember] = []
    for marker_start, marker_end, citation in located:
        for reference_id in citation.reference_ids:
            if reference_id not in reference_ids:
                reference_ids.append(reference_id)
        for reference_id in citation.candidate_reference_ids:
            if reference_id not in candidate_reference_ids:
                candidate_reference_ids.append(reference_id)
        if citation.citation_markers:
            marker_members.extend(citation.citation_markers)
        else:
            marker_members.append(
                CitationMarkerMember(
                    text=text[marker_start:marker_end],
                    local_start=marker_start,
                    local_end=marker_end,
                    reference_ids=list(citation.reference_ids),
                    marker_type=citation.marker_type,
                )
            )
    marker_members.sort(key=lambda item: (item.local_start, item.local_end))
    first = located[0][2]
    return [
        first.model_copy(
            update={
                "reference_ids": reference_ids,
                "candidate_reference_ids": candidate_reference_ids,
                "citation_key": "|".join(
                    citation.citation_key
                    for _start, _end, citation in located
                    if citation.citation_key
                ),
                "citation_marker": "; ".join(
                    text[start:end] for start, end, _citation in located
                ),
                "citation_markers": marker_members,
                "marker_member": " | ".join(
                    marker.text for marker in marker_members
                ),
                "marker_start": min(
                    citation.marker_start for _start, _end, citation in located
                ),
                "marker_end": max(
                    citation.marker_end for _start, _end, citation in located
                ),
                "confidence": "medium",
            }
        )
    ]


def _collapse_compound_parenthetical_blocks(
    detections: list[tuple[int, InTextCitation]],
) -> list[tuple[int, InTextCitation]] | None:
    """Represent each exact compound marker once before clause grouping.

    The citation extractor emits one source-specific detection per member of a
    compound parenthetical block.  Clause grouping previously treated the
    semicolons *inside* that block as sentence-level separators and rejected
    otherwise explicit multi-source sentences.
    """
    grouped: dict[tuple[str, int, int], list[tuple[int, InTextCitation]]] = {}
    for item in detections:
        citation = item[1]
        grouped.setdefault(
            (citation.citation_marker, citation.marker_start, citation.marker_end),
            [],
        ).append(item)
    prepared: list[tuple[int, InTextCitation]] = []
    for block in grouped.values():
        if len(block) == 1:
            prepared.append(block[0])
            continue
        merged = _group_compound_parenthetical_marker(block)
        if merged is None or len(merged) != 1:
            return None
        prepared.append((min(index for index, _citation in block), merged[0]))
    return sorted(prepared, key=lambda item: (item[1].marker_start, item[0]))


def _group_compound_parenthetical_marker(
    detections: list[tuple[int, InTextCitation]],
) -> list[InTextCitation] | None:
    """Merge members parsed from one exact parenthetical citation block."""
    if not detections or any(
        citation.marker_type != "parenthetical"
        for _index, citation in detections
    ):
        return None
    raw_marker = detections[0][1].citation_marker
    text = detections[0][1].text
    if not raw_marker or any(
        citation.citation_marker != raw_marker
        for _index, citation in detections
    ):
        return None
    marker_start = text.find(raw_marker)
    if marker_start < 0 or text.find(raw_marker, marker_start + 1) >= 0:
        return None

    ordered = []
    cursor = 0
    for _index, citation in detections:
        member_text = (citation.marker_member or "").strip()
        if not member_text:
            return None
        member_start = raw_marker.find(member_text, cursor)
        if member_start < 0:
            return None
        member_end = member_start + len(member_text)
        ordered.append((member_start, member_end, citation))
        cursor = member_end
    if len({citation.marker_member for _s, _e, citation in ordered}) != len(ordered):
        return None

    reference_ids: list[str] = []
    candidate_reference_ids: list[str] = []
    marker_members: list[CitationMarkerMember] = []
    for member_start, member_end, citation in ordered:
        for reference_id in citation.reference_ids:
            if reference_id not in reference_ids:
                reference_ids.append(reference_id)
        for reference_id in citation.candidate_reference_ids:
            if reference_id not in candidate_reference_ids:
                candidate_reference_ids.append(reference_id)
        marker_members.append(
            CitationMarkerMember(
                text=raw_marker[member_start:member_end],
                local_start=marker_start + member_start,
                local_end=marker_start + member_end,
                reference_ids=list(citation.reference_ids),
                marker_type="parenthetical",
            )
        )
    first = detections[0][1]
    return [
        first.model_copy(
            update={
                "reference_ids": reference_ids,
                "candidate_reference_ids": candidate_reference_ids,
                "citation_key": "|".join(
                    citation.citation_key
                    for _start, _end, citation in ordered
                    if citation.citation_key
                ),
                "citation_markers": marker_members,
                "marker_member": " | ".join(
                    citation.marker_member for _start, _end, citation in ordered
                ),
                "page_number": next(
                    (
                        citation.page_number
                        for _start, _end, citation in ordered
                        if citation.page_number
                    ),
                    "",
                ),
            }
        )
    ]


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
    marker_ranges = [(item[0], item[1]) for item in marker_spans]
    for match in re.finditer(r";", text):
        if any(left <= match.start() < right for left, right in marker_ranges):
            continue
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
