"""Deterministic, paper-local reference/citation consistency evidence."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.services.schemas import InTextCitation, MediaAnalysisCandidate, ParsedReference


REFERENCE_CONSISTENCY_VERSION = "reference-consistency-v3"


def inspect_media_type_context(*, body: str, body_sha256: str,
                               candidate: MediaAnalysisCandidate,
                               references: list[ParsedReference]) -> dict:
    """Compare explicit submitted designations, never resolve work identity.

    Exact title equality supplies diagnostic candidates only. Multiple entries
    remain ambiguous even if one happens to agree with the proposed type.
    """
    result = dict(version='media-type-context-v1', status='not_assessed',
                  reference_ids=[], explicit_types={}, reference_bindings={},
                  source_identity_assessed=False, automatic_findings_enabled=False)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    if (digest(body) != body_sha256
            or not 0 <= candidate.passage_start < candidate.passage_end <= len(body)
            or digest(body[candidate.passage_start:candidate.passage_end]) != candidate.passage_sha256
            or not candidate.passage_start <= candidate.title_start < candidate.title_end <= candidate.passage_end
            or body[candidate.title_start:candidate.title_end] != candidate.title):
        result['status'] = 'invalid_binding'
        return result
    if candidate.title_role != 'work_title':
        return result
    labels = {'play': 'play', 'film': 'film', 'documentary': 'film',
              'song': 'song', 'album': 'album', 'film review': 'article'}
    def observed_title(reference):
        # The parser may retain an explicit terminal medium descriptor in title.
        # Remove only a recognized descriptor for this diagnostic comparison.
        title = reference.title.strip()
        suffix = re.search(r'\s+\[([^\[\]]+)\]$', title)
        if suffix and suffix.group(1).strip().casefold() in labels:
            title = title[:suffix.start()]
        return title.casefold()
    matches = [r for r in references if observed_title(r) == candidate.title.strip().casefold()]
    result['reference_ids'] = [r.reference_id for r in matches]
    if len(matches) != 1:
        if matches:
            result['status'] = 'ambiguous_reference_context'
        return result
    ref = matches[0]
    if sum(r.reference_id == ref.reference_id for r in references) != 1 or ref.needs_review:
        result['status'] = 'ambiguous_reference_context'
        return result
    types = {labels[label.strip().casefold()] for label in re.findall(r'\[([^\[\]]+)\]', ref.raw_ref)
             if label.strip().casefold() in labels}
    result['reference_bindings'] = {ref.reference_id: digest(ref.raw_ref)}
    result['explicit_types'] = {ref.reference_id: sorted(types)}
    if len(types) > 1:
        result['status'] = 'ambiguous_reference_context'
    elif types and candidate.media_type not in {'unknown', 'other'}:
        result['status'] = ('submitted_type_agrees' if candidate.media_type in types
                            else 'submitted_type_disagrees')
    return result


def inspect_media_reference_context(*, body: str, body_sha256: str, candidate: MediaAnalysisCandidate,
                                    references: list[ParsedReference],
                                    citations: list[InTextCitation]) -> dict:
    """Expose existing citation links, not a second identity/omission verifier.

    A film link in a song-analysis passage can support later application of the
    approved performance policy, but does not itself prove which version was used.
    """
    result = dict(version='media-reference-context-v1', status='not_assessed',
                  film_reference_ids=[], reference_bindings={},
                  source_use_assessed=False, reference_absence_assessed=False,
                  automatic_findings_enabled=False)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    if (digest(body) != body_sha256
            or not 0 <= candidate.passage_start < candidate.passage_end <= len(body)
            or digest(body[candidate.passage_start:candidate.passage_end]) != candidate.passage_sha256
            or not candidate.passage_start <= candidate.title_start < candidate.title_end <= candidate.passage_end
            or body[candidate.title_start:candidate.title_end] != candidate.title):
        result['status'] = 'invalid_binding'
        return result
    if candidate.media_type != 'song':
        return result
    reference_id_counts: dict[str, int] = defaultdict(int)
    for reference in references:
        reference_id_counts[reference.reference_id] += 1
    films = {r.reference_id:r for r in references if reference_id_counts[r.reference_id] == 1
             and not r.needs_review
             and r.source_kind == 'traditional_media' and r.source_kind_confidence == 'high'
             and re.search(r'\[film\]', r.raw_ref, re.I)}
    found = set()
    for citation in citations:
        if (citation.link_status != 'linked' or citation.candidate_reference_ids
                or not 0 <= citation.passage_start < citation.passage_end <= len(body)
                or body[citation.passage_start:citation.passage_end] != citation.text):
            continue
        markers = [(m.text, m.local_start, m.local_end, m.reference_ids)
                   for m in citation.citation_markers]
        if not markers and citation.citation_marker and citation.text.count(citation.citation_marker) == 1:
            start = citation.text.index(citation.citation_marker)
            markers = [(citation.citation_marker, start, start+len(citation.citation_marker), citation.reference_ids)]
        for text, start, end, ids in markers:
            if (not text or not 0 <= start < end <= len(citation.text)
                    or citation.text[start:end] != text
                    or not candidate.passage_start <= citation.passage_start+start < citation.passage_start+end <= candidate.passage_end):
                continue
            found.update(rid for rid in ids if rid in films and rid in citation.reference_ids)
    result['film_reference_ids'] = sorted(found)
    result['reference_bindings'] = {rid:digest(films[rid].raw_ref) for rid in sorted(found)}
    if found:
        result['status'] = 'film_reference_context_available' if len(found)==1 else 'multiple_film_contexts'
    return result

FindingType = Literal[
    "missing_reference_entry",
    "ambiguous_reference_link",
    "reference_not_linked_in_extracted_citations",
    "duplicate_reference_entry",
    "likely_reference_repetition",
    "duplicate_citation_key",
    "reference_parse_review",
]
FindingLevel = Literal["attention", "review", "neutral"]


class ReferenceConsistencyFinding(BaseModel):
    """One exact structural finding without reproducing student prose."""

    finding_id: str = Field(pattern=r"^reference-consistency:[0-9a-f]{64}$")
    finding_type: FindingType
    level: FindingLevel
    reason_code: str = Field(min_length=1, max_length=100)
    reference_ids: list[str] = Field(default_factory=list)
    candidate_reference_ids: list[str] = Field(default_factory=list)
    passage_start: int | None = Field(default=None, ge=0)
    passage_end: int | None = Field(default=None, ge=0)
    marker_text_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    explanation: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_citation_span(self):
        supplied = self.passage_start is not None or self.passage_end is not None
        if supplied and (
            self.passage_start is None
            or self.passage_end is None
            or self.passage_end <= self.passage_start
            or self.marker_text_sha256 is None
        ):
            raise ValueError("Citation findings require a complete exact span binding")
        return self


class ReferenceConsistencyAssessment(BaseModel):
    """Bounded consistency result for one extracted paper version."""

    assessment_version: str = REFERENCE_CONSISTENCY_VERSION
    paper_version_id: str = Field(min_length=1)
    citation_format: Literal["apa", "mla"]
    status: Literal["complete", "not_assessed"]
    reference_count: int = Field(ge=0)
    citation_detection_count: int = Field(ge=0)
    findings: list[ReferenceConsistencyFinding] = Field(default_factory=list)
    finding_counts: dict[str, int] = Field(default_factory=dict)
    formatting_status: Literal["not_assessed"] = "not_assessed"
    formatting_reason_codes: list[str] = Field(
        default_factory=lambda: ["normalized_text_lacks_layout_and_typography"]
    )
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        expected: dict[str, int] = defaultdict(int)
        for finding in self.findings:
            expected[finding.finding_type] += 1
        if dict(sorted(expected.items())) != self.finding_counts:
            raise ValueError("Reference consistency finding counts do not match findings")
        return self


def assess_reference_consistency(
    *,
    paper_version_id: str,
    citation_format: str,
    references: list[ParsedReference],
    citations: list[InTextCitation],
) -> ReferenceConsistencyAssessment:
    """Assess only relationships established by the current extracted records."""
    if citation_format not in {"apa", "mla"}:
        raise ValueError("Reference consistency supports APA and MLA papers only")

    findings: list[ReferenceConsistencyFinding] = []
    observed_reference_ids: set[str] = set()
    # Recheck retained original citation text before turning a missed link into
    # an uncited-entry diagnostic. This does not mutate historical detections.
    from app.services.citation_extractor import extract_citations
    for text in dict.fromkeys(c.text for c in citations):
        for recovered in extract_citations(text, references, format_hint=citation_format):
            observed_reference_ids.update(recovered.reference_ids)
            observed_reference_ids.update(recovered.candidate_reference_ids)
    seen_citation_findings: set[tuple] = set()
    for citation in citations:
        observed_reference_ids.update(citation.reference_ids)
        observed_reference_ids.update(citation.candidate_reference_ids)
        if citation.link_status not in {"missing_reference", "ambiguous"}:
            continue
        finding_type: FindingType = (
            "missing_reference_entry"
            if citation.link_status == "missing_reference"
            else "ambiguous_reference_link"
        )
        key = (
            finding_type,
            citation.passage_start,
            citation.passage_end,
            citation.citation_marker,
            tuple(citation.reference_ids),
            tuple(citation.candidate_reference_ids),
        )
        if key in seen_citation_findings:
            continue
        seen_citation_findings.add(key)
        reason_code = (
            "no_reference_entry_matched_marker"
            if finding_type == "missing_reference_entry"
            else "multiple_reference_entries_match_marker"
        )
        explanation = (
            "The extracted in-text citation has no matching reference-list entry."
            if finding_type == "missing_reference_entry"
            else "The extracted in-text citation could match more than one reference-list entry."
        )
        has_exact_span = (
            citation.passage_start >= 0
            and citation.passage_end > citation.passage_start
        )
        findings.append(
            _finding(
                finding_type=finding_type,
                level="neutral",
                reason_code=reason_code,
                reference_ids=citation.reference_ids,
                candidate_reference_ids=citation.candidate_reference_ids,
                passage_start=citation.passage_start if has_exact_span else None,
                passage_end=citation.passage_end if has_exact_span else None,
                marker_text=citation.citation_marker if has_exact_span else None,
                explanation=explanation,
            )
        )

    by_normalized_entry: dict[str, list[str]] = defaultdict(list)
    by_doi: dict[str, list[str]] = defaultdict(list)
    by_citation_key: dict[str, list[str]] = defaultdict(list)
    by_corroborated_web_identity: dict[tuple, list[str]] = defaultdict(list)
    for reference in references:
        if reference.needs_review:
            findings.append(
                _finding(
                    finding_type="reference_parse_review",
                    level="review",
                    reason_code=f"field_extraction_{reference.extraction_method}",
                    reference_ids=[reference.reference_id],
                    explanation=(
                        "The reference entry could not be parsed with the ordinary deterministic field extractor."
                    ),
                )
            )
        normalized_entry = _normalize_reference_entry(reference.raw_ref)
        if normalized_entry:
            by_normalized_entry[normalized_entry].append(reference.reference_id)
        if reference.doi:
            by_doi[reference.doi.casefold()].append(reference.reference_id)
        if reference.citation_key:
            by_citation_key[reference.citation_key.casefold()].append(reference.reference_id)
        web_key = _corroborated_web_key(reference)
        if web_key:
            by_corroborated_web_identity[web_key].append(reference.reference_id)

    duplicate_groups: set[tuple[str, ...]] = set()
    definite_duplicate_pairs: set[frozenset[str]] = set()
    for reason_code, groups in (
        ("same_normalized_reference_entry", by_normalized_entry),
        ("same_normalized_doi", by_doi),
        ("same_author_title_and_specific_url", by_corroborated_web_identity),
    ):
        for reference_ids in groups.values():
            unique = tuple(dict.fromkeys(reference_ids))
            if len(unique) < 2 or unique in duplicate_groups:
                continue
            duplicate_groups.add(unique)
            definite_duplicate_pairs.update(
                frozenset((left, right))
                for index, left in enumerate(unique)
                for right in unique[index + 1 :]
            )
            findings.append(
                _finding(
                    finding_type="duplicate_reference_entry",
                    level="attention",
                    reason_code=reason_code,
                    reference_ids=list(unique),
                    explanation="Two or more reference-list entries identify the same work record.",
                )
            )

    # Overlapping corroboration routes describe one duplicate group, not
    # independent errors. Preserve all reasons in the group's explanation.
    duplicate_findings = [f for f in findings if f.finding_type == "duplicate_reference_entry"]
    findings = [f for f in findings if f.finding_type != "duplicate_reference_entry"]
    groups: list[set[str]] = []
    for finding in duplicate_findings:
        group = set(finding.reference_ids)
        overlaps = [g for g in groups if g & group]
        for prior in overlaps:
            group.update(prior)
            groups.remove(prior)
        groups.append(group)
    duplicate_ids = set().union(*groups) if groups else set()
    for group in groups:
        reasons = sorted({f.reason_code for f in duplicate_findings if group & set(f.reference_ids)})
        findings.append(_finding(
            finding_type="duplicate_reference_entry", level="attention",
            reason_code="corroborated_duplicate_group", reference_ids=sorted(group),
            explanation=("These entries repeat the same source. " +
                         ("The author, title and specific source URL agree despite different submitted dates."
                          if "same_author_title_and_specific_url" in reasons else
                          "The entries have identical reference text or the same DOI."))))
        definite_duplicate_pairs.update(frozenset((a, b)) for a in group for b in group if a != b)

    likely_groups = _likely_repetition_groups(
        references,
        excluded_pairs=definite_duplicate_pairs,
    )
    for reference_ids in likely_groups:
        findings.append(
            _finding(
                finding_type="likely_reference_repetition",
                level="attention",
                reason_code="same_author_year_near_identical_title",
                reference_ids=reference_ids,
                explanation=(
                    "These reference entries appear to repeat the same source."
                ),
            )
        )

    by_id = {reference.reference_id: reference for reference in references}
    for citation_key, reference_ids in by_citation_key.items():
        unique = list(dict.fromkeys(reference_ids))
        if len(unique) < 2 or any(set(unique) <= group for group in groups):
            continue
        if len({apa_in_text_form(by_id[rid]) for rid in unique if rid in by_id}) == len(unique):
            continue  # their in-text citations already differ
        findings.append(
            _finding(
                finding_type="duplicate_citation_key",
                level="review",
                reason_code="reference_entries_share_in_text_key",
                reference_ids=unique,
                explanation=(
                    "Multiple reference-list entries share the same extracted author-year citation key."
                ),
            )
        )

    for reference in references:
        if reference.reference_id in observed_reference_ids or reference.reference_id in duplicate_ids:
            continue
        findings.append(
            _finding(
                finding_type="reference_not_linked_in_extracted_citations",
                level="neutral",
                reason_code="no_extracted_marker_links_reference",
                reference_ids=[reference.reference_id],
                explanation=(
                    "No extracted in-text citation was linked to this reference-list entry."
                ),
            )
        )

    findings.sort(
        key=lambda item: (
            {"attention": 0, "review": 1, "neutral": 2}[item.level],
            item.finding_type,
            item.passage_start if item.passage_start is not None else -1,
            item.reference_ids,
        )
    )
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.finding_type] += 1
    return ReferenceConsistencyAssessment(
        paper_version_id=paper_version_id,
        citation_format=citation_format,
        status="complete",
        reference_count=len(references),
        citation_detection_count=len(citations),
        findings=findings,
        finding_counts=dict(sorted(counts.items())),
        limitations=[
            "An unlinked reference is bounded to current citation extraction and is a review signal, not proof that the paper never cites it.",
            "Missing and ambiguous links from low-confidence citation detections remain neutral diagnostics until citation/reference linking is separately accepted.",
            "Typography, italics, hanging indents, capitalization rules and entry-specific APA/MLA punctuation are not assessed from normalized plain text.",
            "Personal communications and other in-text-only source exceptions require a later explicit rule before a missing-entry finding can be suppressed.",
            "Likely repetition is a cautious high-similarity attention finding, not a definitive duplicate-work conclusion.",
        ],
    )


def _finding(
    *,
    finding_type: FindingType,
    level: FindingLevel,
    reason_code: str,
    reference_ids: list[str],
    explanation: str,
    candidate_reference_ids: list[str] | None = None,
    passage_start: int | None = None,
    passage_end: int | None = None,
    marker_text: str | None = None,
) -> ReferenceConsistencyFinding:
    payload = {
        "finding_type": finding_type,
        "reason_code": reason_code,
        "reference_ids": reference_ids,
        "candidate_reference_ids": candidate_reference_ids or [],
        "passage_start": passage_start,
        "passage_end": passage_end,
        "marker_text_sha256": (
            hashlib.sha256(marker_text.encode()).hexdigest() if marker_text is not None else None
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ReferenceConsistencyFinding(
        finding_id=f"reference-consistency:{digest}",
        finding_type=finding_type,
        level=level,
        reason_code=reason_code,
        reference_ids=reference_ids,
        candidate_reference_ids=candidate_reference_ids or [],
        passage_start=passage_start,
        passage_end=passage_end,
        marker_text_sha256=payload["marker_text_sha256"],
        explanation=explanation,
    )


def _normalize_reference_entry(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    return normalized.rstrip(". ")


def _corroborated_web_key(reference: ParsedReference) -> tuple | None:
    """Submitted multi-field duplication, not URL-only work equivalence.

    Restrict date-tolerant matching to specific webpage records. Book editions,
    components, different authors/titles and generic site roots do not qualify.
    """
    from urllib.parse import urlsplit
    import unicodedata
    from app.services.source_type import classify_reference_source_kind

    if (reference.needs_review or reference.doi or reference.source_kind != "webpage"
            or reference.container_title.strip() or reference.pages.strip()):
        return None
    # A retained webpage label cannot override explicit book/component evidence.
    # Exclude URLs from this textual screen: a slug is not an edition statement.
    bibliographic_text = re.sub(r"https?://\S+|www\.\S+", " ", reference.raw_ref, flags=re.I)
    if re.search(
        r"\bISBN(?:-1[03])?\b|\bedition\b|"
        r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|"
        r"eighth|ninth|tenth|revised|rev\.?|expanded|updated)\s+ed\b",
        bibliographic_text + " " + reference.title, re.I,
    ):
        return None
    observed_kind = classify_reference_source_kind(
        bibliographic_text, title=reference.title
    ).kind
    if observed_kind not in {"unknown", "webpage", "blog_post", "news_article"}:
        return None
    def web_identity_text(value: str) -> str:
        normalized = unicodedata.normalize("NFC", value).casefold()
        return " ".join("".join(
            char if unicodedata.category(char)[0] in {"L", "M", "N"} else " "
            for char in normalized
        ).split())

    title, author = web_identity_text(reference.title), web_identity_text(reference.author)
    if len(title) < 15 or len(title.split()) < 3 or not author or author in {"unknown", "anonymous"}:
        return None
    try:
        url = urlsplit(reference.url or "")
        if url.scheme not in {"http", "https"} or not url.hostname or url.path.strip("/") == "":
            return None
        # Fragments can select distinct routed resources, not just page positions.
        return author, title, url.hostname.casefold(), url.port, url.path, url.query, url.fragment
    except ValueError:
        return None


def _likely_repetition_groups(
    references: list[ParsedReference],
    *,
    excluded_pairs: set[frozenset[str]],
) -> list[list[str]]:
    """Group only extremely close non-identical work identities.

    This is intentionally stricter than ordinary fuzzy similarity.  It requires
    a stable year, compatible author identity, compatible source kind, and a
    substantive title that is exact after safe normalization or nearly
    identical by both character and token overlap.
    """
    parents = {reference.reference_id: reference.reference_id for reference in references}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for index, left in enumerate(references):
        for right in references[index + 1 :]:
            pair = frozenset((left.reference_id, right.reference_id))
            if pair in excluded_pairs or not _is_likely_repetition(left, right):
                continue
            union(left.reference_id, right.reference_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for reference in references:
        groups[find(reference.reference_id)].append(reference.reference_id)
    return sorted(
        (sorted(values) for values in groups.values() if len(values) > 1),
        key=lambda values: values,
    )


def _is_likely_repetition(left: ParsedReference, right: ParsedReference) -> bool:
    if left.year == "n.d." or left.year.casefold() != right.year.casefold():
        return False
    if not _authors_compatible(left.author, right.author):
        return False
    if (
        left.source_kind != "unknown"
        and right.source_kind != "unknown"
        and left.source_kind != right.source_kind
    ):
        return False
    left_title = _identity_text(left.title)
    right_title = _identity_text(right.title)
    if min(len(left_title), len(right_title)) < 12:
        return False
    left_tokens, right_tokens = set(left_title.split()), set(right_title.split())
    if min(len(left_tokens), len(right_tokens)) < 2:
        return False
    if left_title == right_title:
        return True
    token_union = left_tokens | right_tokens
    token_jaccard = len(left_tokens & right_tokens) / len(token_union)
    character_similarity = SequenceMatcher(None, left_title, right_title).ratio()
    return token_jaccard >= 0.90 and character_similarity >= 0.96


def _authors_compatible(left: str, right: str) -> bool:
    left_normalized = _identity_text(left)
    right_normalized = _identity_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    left_family, left_given = _author_parts(left)
    right_family, right_given = _author_parts(right)
    if not left_family or left_family != right_family:
        return False
    if not left_given or not right_given:
        return False
    return left_given[0] == right_given[0]


def _author_parts(value: str) -> tuple[str, str]:
    first_author = re.split(r"\s+(?:and|&)\s+|;", value, maxsplit=1, flags=re.I)[0]
    if "," in first_author:
        family, given = first_author.split(",", 1)
    else:
        tokens = _identity_text(first_author).split()
        if len(tokens) < 2:
            return "", ""
        family, given = tokens[-1], " ".join(tokens[:-1])
    return _identity_text(family), _identity_text(given)


def _identity_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

def apa_in_text_form(reference) -> str:
    """The APA in-text author form: one surname, two joined, or "et al." for three or more.

    "Yang, M., O'Sullivan, P.S., ... (2019)" and "Yang, Y., & Wang, X. (2019)"
    share a first surname and year, but their citations ("Yang et al., 2019",
    "Yang & Wang, 2019") already differ (Academic Article, 2026-09-30).
    """
    author = str(getattr(reference, "author", "") or "")
    if not author.strip() and str(getattr(reference, "title", "") or "").strip():
        # A title-first entry is cited by its title (Franchise 2's "Sherlock
        # season 1" and "Sherlock season 4", both n.d., 2026-10-07).
        title = " ".join(str(reference.title).casefold().split())
        return f"“{title}” {str(getattr(reference, 'year', '') or '').casefold()}"
    names = re.findall(r"([^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*)?)\s*,\s*(?:[A-Z](?:\.|\b)[\s\-]*)+", author)
    surnames = [n.strip().casefold() for n in names] or [author.split(",")[0].strip().casefold()]
    year = str(getattr(reference, "year", "") or "").casefold()
    if len(surnames) >= 3:
        return f"{surnames[0]} et al. {year}"
    return " & ".join(surnames) + f" {year}"
