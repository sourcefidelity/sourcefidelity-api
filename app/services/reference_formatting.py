"""Conservative, rule-specific reference-format evidence.

This layer never turns one observed rule into a whole-entry or whole-paper
style verdict. Each rule remains independently assessed or not assessed.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.services.reference_layout import ReferenceLayoutArtifact, reference_span_style_observed
from app.services.schemas import ParsedReference
from app.services.source_type import _BOOK_PUBLISHER_RE, _JOURNAL_STRUCTURE_RE


REFERENCE_FORMATTING_VERSION = "reference-formatting-v1"
HANGING_INDENT_POINTS = 36.0
HANGING_INDENT_TOLERANCE_POINTS = 4.5
APA_TITLE_RULE_SOURCE = 'https://www.apa.org/ed/precollege/psn/2020/09/apa-style-student-papers'
APA_ORDER_RULE_SOURCE = 'https://www.apa.org/pubs/journals/resources/general-manuscript-preparation-guidelines'


class ContributionReferenceExpectation(BaseModel):
    """Explicit reviewed input, not an automatic source-use or identity verdict."""
    model_config = {'extra': 'forbid', 'frozen': True}
    paper_body_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reference_id: str
    reference_text_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    review_record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    use_scope: Literal['specific_contribution', 'whole_collection', 'unknown']
    contribution_title: str = Field(min_length=1)
    contribution_author: str = Field(min_length=1)
    contribution_pages: str = Field(min_length=1)
    basis: Literal['owner_review', 'source_bound_review']


def assess_contribution_reference(*, citation_format: str, body: str,
                                 reference: ParsedReference,
                                 expectation: ContributionReferenceExpectation,
                                 source_sha256: str, review_record_sha256: str) -> dict:
    """Development-only rule projection from reviewed contribution use.

    No runtime producer/consumer is enabled. Hashes bind the caller's reviewed
    inputs, not independent source authentication or permission to admit a file.
    """
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    result = dict(rule_id='apa7_contribution_reference_v1', status='not_assessed',
        reason_code='unsupported_or_unresolved', reference_id=reference.reference_id,
        reference_text_sha256=digest(reference.raw_ref), automatic_findings_enabled=False,
        wrong_author_assessed=False)
    if (expectation.paper_body_sha256 != digest(body)
            or expectation.reference_id != reference.reference_id
            or expectation.reference_text_sha256 != result['reference_text_sha256']
            or expectation.source_sha256 != source_sha256
            or expectation.review_record_sha256 != review_record_sha256):
        result['reason_code'] = 'review_binding_changed'
        return result
    if citation_format != 'apa' or reference.needs_review:
        return result
    if expectation.use_scope != 'specific_contribution':
        result['reason_code'] = 'specific_contribution_use_not_established'
        return result
    if (reference.source_kind != 'edited_collection'
            or reference.source_kind_confidence != 'high'):
        return result
    if expectation.contribution_title.casefold() in reference.raw_ref.casefold():
        result['reason_code'] = 'contribution_details_may_already_be_present'
        return result
    result.update(status='difference', reason_code='collection_instead_of_contribution',
        contribution_title=expectation.contribution_title,
        contribution_author=expectation.contribution_author,
        contribution_pages=expectation.contribution_pages,
        source_sha256=source_sha256, review_record_sha256=review_record_sha256,
        verification_basis=expectation.basis,
        explanation='This reference identifies the edited book rather than the specific contribution used. '
            'Include the contribution title, author and page range within the edited-book reference. '
            'This also applies when the contribution author is the book editor.')
    return result


class ReferenceTitleStyleResult(BaseModel):
    rule_id: Literal['apa7_reference_title_italics_v1', 'apa7_reference_title_italics_v2', 'apa7_marked_periodical_italics_v1', 'literal_title_emphasis_v1'] = 'apa7_reference_title_italics_v2'
    reference_id: str
    reference_text_sha256: str
    source_kind: str
    status: Literal['matches_rule', 'difference', 'not_assessed']
    reason_code: str
    title_start: int | None = None
    title_end: int | None = None
    title_sha256: str | None = None
    expected_italic: bool | None = None
    observed_italic: bool | None = None

    @model_validator(mode='after')
    def validate_binding(self):
        if self.status != 'not_assessed' and (
            self.title_start is None or self.title_end is None
            or self.title_start < 0 or self.title_end <= self.title_start
            or not self.title_sha256 or self.expected_italic is None
            or self.observed_italic is None
        ):
            raise ValueError('Assessed title rules require exact observed title bindings')
        return self


class ReferenceOrderResult(BaseModel):
    rule_id: Literal['apa7_distinct_surname_order_v1'] = 'apa7_distinct_surname_order_v1'
    status: Literal['matches_rule', 'difference', 'not_assessed'] = 'not_assessed'
    reason_code: str
    section_sha256: str | None = None
    reference_bindings: dict[str, str] = Field(default_factory=dict)
    observed_order: list[str] = Field(default_factory=list)
    expected_order: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_order(self):
        if self.status != 'not_assessed':
            if (not self.section_sha256 or len(self.observed_order) < 2
                    or len(set(self.observed_order)) != len(self.observed_order)
                    or sorted(self.observed_order) != sorted(self.expected_order)
                    or set(self.reference_bindings) != set(self.observed_order)
                    or (self.status == 'difference') != (self.observed_order != self.expected_order)):
                raise ValueError('Assessed ordering needs complete distinct bound permutations')
        return self


def assess_title_styles(layout: ReferenceLayoutArtifact, references: list[ParsedReference]):
    """Bounded APA book/article/explicit-film rule with observed typography."""
    entries = {entry.reference_id: entry for entry in layout.entries}
    results = []
    for ref in references:
        digest = hashlib.sha256(ref.raw_ref.encode()).hexdigest()
        data = dict(reference_id=ref.reference_id, reference_text_sha256=digest,
                    source_kind=ref.source_kind, status='not_assessed')
        reason = 'style_or_work_kind_not_accepted'
        entry = entries.get(ref.reference_id)
        supported = (layout.citation_format == 'apa' and not ref.needs_review
                     and ref.source_kind_confidence == 'high'
                     and ref.source_kind in {'monograph', 'journal_article', 'traditional_media'})
        if supported:
            reason = 'title_span_not_uniquely_observed'
            title = ref.title
            film = (ref.source_kind == 'traditional_media'
                    and bool(re.search(r'\[Film\]', ref.raw_ref, re.I))
                    and bool(re.search(r'\(Director\)', ref.raw_ref, re.I)))
            if film:
                title = re.sub(r'\s*\[Film\]\s*$', '', title, flags=re.I).rstrip()
            if (entry and entry.reference_text_sha256 == digest and len(title) >= 8
                    and len(title.split()) >= (2 if film else 3) and ref.raw_ref.count(title) == 1):
                start = ref.raw_ref.index(title); end = start + len(title)
                tail = ref.raw_ref[end:].lstrip(' .,*')
                # Routing kind is not sufficient formatting evidence: a press
                # named inside a title or a chapter followed by a book
                # publisher must not establish an independent book title.
                independent_kind = (
                    bool(_BOOK_PUBLISHER_RE.search(tail)) and not re.match(r'In\b', tail, re.I)
                    if ref.source_kind == 'monograph' else
                    bool(_JOURNAL_STRUCTURE_RE.search(tail.replace('*', '')) or re.search(
                        r',\s*\d{1,4}\s*,\s*\d+\s*[-–—]\s*\d+', tail.replace('*','')))
                    if ref.source_kind == 'journal_article' else
                    film and bool(re.match(r'\[Film\]', tail, re.I))
                )
                if not independent_kind:
                    results.append(ReferenceTitleStyleResult(**data, reason_code='independent_container_or_publisher_not_established'))
                    continue
                if reference_span_style_observed(entry, ref.raw_ref, start, end):
                    flags = {any(s.italic and s.start <= i < s.end for s in entry.text_style_spans)
                             for i in range(start, end) if ref.raw_ref[i].isalnum()}
                    reason = 'mixed_or_embedded_title_typography_not_assessed'
                    # Mixed styling can be correct for embedded titles or terms.
                    # An entirely plain book title is missing italics even if
                    # it contains a quoted phrase. Mixed styling still abstains.
                    expected = ref.source_kind in {'monograph', 'traditional_media'}
                    # A plain title followed by an italic larger work is a
                    # part of that work, correctly styled (Curle, 2026-09-30).
                    later_italic = any(sp.italic and sp.start >= end and any(ch.isalpha() for ch in ref.raw_ref[sp.start:sp.end])
                                       for sp in entry.text_style_spans)
                    if flags == {False} and later_italic and expected:
                        reason = 'larger_work_italicized_after_part_title'
                    elif len(flags) == 1 and (flags == {False} and expected or not any(c in title for c in '“”"[]')):
                        observed = flags.pop()
                        data.update(title_start=start, title_end=end,
                                    title_sha256=hashlib.sha256(title.encode()).hexdigest(),
                                    observed_italic=observed, expected_italic=expected,
                                    status='matches_rule' if observed == expected else 'difference')
                        reason = 'title_italic_style_matches' if observed == expected else 'title_italic_style_differs'
        results.append(ReferenceTitleStyleResult(**data, reason_code=reason))
        if (supported and ref.source_kind == 'journal_article' and entry
                and entry.reference_text_sha256 == digest and ref.title
                and ref.raw_ref.count(ref.title) == 1):
            offset = ref.raw_ref.index(ref.title) + len(ref.title)
            # Only explicit marked journal/volume structure is added here;
            # stars are student characters, never evidence of actual italics.
            match = re.match(r'\s*\.\s*\*(?P<field>[^*\n]+(?:\*\s*,\s*|,\s*)\d{1,4})\*?\s*(?:\([^)]{1,20}\))?\s*,\s*\d+\s*[-–—]\s*\d+', ref.raw_ref[offset:])
            if not match:
                # An unmarked journal and volume right after the title (Hess,
                # Paper 2, 2026-09-30: a plain journal name was never checked).
                # Their actual type style, not student characters, is compared.
                from app.services.ref_field_extractor import _journal_parts
                journal, volume, _issue, _pages = _journal_parts(ref.raw_ref)
                if journal and volume and '*' not in ref.raw_ref[offset:]:
                    match = re.match(r'\s*[.?!]\s*(?P<field>' + re.escape(journal)
                                     + r'\s*,\s*' + re.escape(volume) + r')\b', ref.raw_ref[offset:])
            if match:
                start, end = offset + match.start('field'), offset + match.end('field')
                if reference_span_style_observed(entry, ref.raw_ref, start, end):
                    flags = {any(s.italic and s.start <= i < s.end for s in entry.text_style_spans)
                             for i in range(start, end) if ref.raw_ref[i].isalnum()}
                    if len(flags) == 1:
                        observed = flags.pop()
                        results.append(ReferenceTitleStyleResult(
                            rule_id='apa7_marked_periodical_italics_v1', reference_id=ref.reference_id,
                            reference_text_sha256=digest, source_kind=ref.source_kind,
                            status='matches_rule' if observed else 'difference',
                            reason_code='periodical_italic_style_matches' if observed else 'periodical_italic_style_differs',
                            title_start=start, title_end=end, title_sha256=hashlib.sha256(ref.raw_ref[start:end].encode()).hexdigest(),
                            expected_italic=True, observed_italic=observed))
    for ref in references:
        entry=entries.get(ref.reference_id)
        title=ref.title.strip('*').rstrip('.')
        if (layout.citation_format!='apa' or ref.needs_review or not title or not entry
                or entry.reference_text_sha256!=hashlib.sha256(ref.raw_ref.encode()).hexdigest()
                or ref.raw_ref.count(title)!=1
                or any(r.reference_id==ref.reference_id and r.status=='difference' for r in results)):
            continue
        start=ref.raw_ref.index(title);end=start+len(title)
        if (start==0 or ref.raw_ref[start-1]!='*' or not re.match(r'\.?\*',ref.raw_ref[end:])
                or not reference_span_style_observed(entry,ref.raw_ref,start,end)
                or any(s.italic and s.start<end and s.end>start for s in entry.text_style_spans)):
            continue
        results.append(ReferenceTitleStyleResult(rule_id='literal_title_emphasis_v1',
            reference_id=ref.reference_id,reference_text_sha256=entry.reference_text_sha256,
            source_kind=ref.source_kind,status='difference',reason_code='literal_emphasis_not_rendered',
            title_start=start,title_end=end,title_sha256=hashlib.sha256(title.encode()).hexdigest(),
            expected_italic=True,observed_italic=False))
    return results


def assess_reference_order(layout: ReferenceLayoutArtifact, references: list[ParsedReference],
                           section: str | None) -> ReferenceOrderResult:
    """Compare only complete, simple distinct-surname APA lists.

    Exact section accounting plus one structurally independent author/date
    start per entry are required. Same-author, authorless, group-author,
    transliteration and compound-name cases remain outside this narrow rule.
    """
    if layout.citation_format != 'apa' or not section or len(references) < 2:
        return ReferenceOrderResult(reason_code='section_or_style_unavailable')
    compact = lambda s: re.sub(r'\s+', '', s)
    if compact(section) != compact(''.join(r.raw_ref for r in references)):
        return ReferenceOrderResult(reason_code='complete_section_accounting_unavailable')
    keys = []
    entries = {e.reference_id: e for e in layout.entries}
    for ref in references:
        entry = entries.get(ref.reference_id)
        # Restrict the accepted sort domain; do not guess special-case keys.
        key = re.match(r'^([A-Za-z]+),\s+[A-Z]\.', ref.raw_ref)
        dates = re.findall(r'\((?:19|20)\d{2}[a-z]?\)', ref.raw_ref)
        if (not key or ref.needs_review or len(dates) != 1 or not ref.author
                or not ref.raw_ref.startswith(ref.author)
                or entry is None or entry.mapping_status != 'matched'
                or entry.reference_text_sha256 != hashlib.sha256(ref.raw_ref.encode()).hexdigest()):
            return ReferenceOrderResult(reason_code='sort_key_or_entry_boundary_uncertain')
        keys.append(key.group(1).lower())
    if len(set(keys)) != len(keys) or len({r.reference_id for r in references}) != len(references):
        return ReferenceOrderResult(reason_code='same_surname_or_duplicate_identity_not_assessed')
    # A separate line-start census prevents merged entries from satisfying a
    # mere concatenation test. Wrapped author lists remain conservative.
    starts = re.findall(r'(?m)^\s*[A-Za-z]+,\s+[A-Z]\.', section)
    if len(starts) != len(references):
        return ReferenceOrderResult(reason_code='independent_entry_census_unavailable')
    order = [r.reference_id for r in references]
    expected = [r.reference_id for _, r in sorted(zip(keys, references), key=lambda p:p[0])]
    return ReferenceOrderResult(status='matches_rule' if order == expected else 'difference',
        reason_code='distinct_surname_order_matches' if order == expected else 'distinct_surname_order_differs',
        section_sha256=hashlib.sha256(section.encode()).hexdigest(),
        reference_bindings={r.reference_id:hashlib.sha256(r.raw_ref.encode()).hexdigest() for r in references},
        observed_order=order, expected_order=expected)


def book_publication_year_discrepancy(reference: ParsedReference, discovery: dict) -> dict | None:
    """Located catalog-date discrepancy, never an exact-edition verdict.

    Only independently acquired Google Books metadata with exact work-level
    title/author agreement is eligible. Conflicting plausible years, a matching
    submitted year, explicit edition qualifiers or stale field hashes abstain.
    """
    from app.services.reference_discovery import ReferenceDiscoveryRecord, _value_hash, credible_field_candidates
    if (reference.needs_review or reference.source_kind != 'monograph'
            or not re.fullmatch(r'(?:18|19|20)\d{2}', reference.year or '')
            or re.search(r'\b(?:edition|reprint|reissue|\d+(?:st|nd|rd|th)\s+ed\.)', reference.raw_ref, re.I)):
        return None
    try:
        record = ReferenceDiscoveryRecord.model_validate(discovery)
    except ValueError:
        return None
    expected = record.expected
    if (record.reference_id != reference.reference_id or expected.reference_parse_review
            or (expected.title, expected.year, expected.authors) !=
               (reference.title, reference.year, [reference.author])):
        return None
    years = set()
    qualified = []
    # Catalogs may omit the subtitle or initial article on an older edition.
    # Such a same-year lead vetoes a year-error finding; it does not confirm
    # source identity or silently replace the submitted title.
    def main_title(value):
        main=str(value or '').split(':',1)[0].casefold()
        words=re.findall(r'\w+',main)
        if words and words[0] in {'the','a','an'}: words=words[1:]
        return words
    expected_main=main_title(reference.title)
    for candidate in record.candidates:
        metadata=candidate.edition_metadata
        if (candidate.provider!='google_books' or not metadata
                or candidate.disposition_reason_code!='edition_metadata_only'
                or candidate.observed.year!=reference.year
                or metadata.published_date[:4]!=reference.year
                or len(expected_main)<2 or main_title(candidate.observed.title)!=expected_main
                or not any(a.attempt_id==candidate.attempt_id and a.provider=='google_books'
                    and a.permitted and a.outcome=='candidate_found' and a.completed_at for a in record.attempts)):
            continue
        comparisons={c.field_name:c for c in candidate.comparisons}
        fields=[('title',reference.title,candidate.observed.title),
                ('author',reference.author,' | '.join(candidate.observed.authors)),
                ('year',reference.year,candidate.observed.year)]
        if (all(name in comparisons and comparisons[name].expected_sha256==_value_hash(old)
                and comparisons[name].observed_sha256==_value_hash(new) for name,old,new in fields)
                and all(comparisons[name].outcome=='agreement' for name in ('author','year'))
                and not any(c.outcome=='material_conflict' and c.field_name!='title' for c in candidate.comparisons)):
            return None
    for candidate in credible_field_candidates(record):
        comparisons = {c.field_name:c for c in candidate.comparisons}
        bindings = [('title', reference.title, candidate.observed.title),
                    ('author', reference.author, ' | '.join(candidate.observed.authors)),
                    ('year', reference.year, candidate.observed.year)]
        if any(name not in comparisons or comparisons[name].expected_sha256 != _value_hash(old)
               or comparisons[name].observed_sha256 != _value_hash(new) for name,old,new in bindings):
            continue
        if any(comparisons[name].outcome != 'agreement' for name in ('title','author')):
            continue
        if candidate.observed.year == reference.year:
            return None
        metadata = candidate.edition_metadata
        if (candidate.provider == 'google_books' and metadata
                and metadata.published_date[:4] == candidate.observed.year
                and not candidate.has_material_conflict
                and any(a.attempt_id == candidate.attempt_id and a.provider == 'google_books'
                        and a.permitted and a.outcome == 'candidate_found' and a.completed_at
                        for a in record.attempts)):
            qualified.append(candidate)
            years.add(candidate.observed.year)
    records = {(c.edition_metadata.volume_id, c.edition_metadata.record_sha256) for c in qualified}
    if len(records) < 2 or len(years) != 1 or reference.year in years:
        return None
    located_year = next(iter(years))
    return dict(
        finding_type='publication_year_discrepancy', reference_id=reference.reference_id,
        rule_id='book_catalog_publication_year_discrepancy_v3',
        reference_text_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
        finding=f'The reference gives {reference.year}; matching Google Books publication records give {located_year}.',
        field_difference=dict(field_name='year', submitted_value=reference.year, located_value=located_year),
        catalog_records=[dict(provider=c.provider, volume_id=c.edition_metadata.volume_id,
                             record_sha256=c.edition_metadata.record_sha256) for c in qualified],
        exact_edition_established=False, rectangles=[], localization_status='not_assessed',
    )


def reference_style_findings(assessment: dict, references: dict[str, ParsedReference]) -> list[dict]:
    """Project stored v2 results only; never upgrade historical assessments."""
    if assessment.get('assessment_version') != 'reference-formatting-v2':
        return []
    findings = []
    for payload in assessment.get('title_results') or []:
        result = ReferenceTitleStyleResult.model_validate(payload)
        ref = references.get(result.reference_id)
        if (result.status != 'difference' or ref is None
                or result.reference_text_sha256 != hashlib.sha256(ref.raw_ref.encode()).hexdigest()):
            continue
        title = ref.raw_ref[result.title_start:result.title_end]
        if (hashlib.sha256(title.encode()).hexdigest() != result.title_sha256
                or result.expected_italic == result.observed_italic):
            continue
        findings.append(dict(finding_type='reference_title_style', reference_id=ref.reference_id,
            finding=('Literal asterisks surround this title; the title itself is not italicized.'
                     if result.rule_id == 'literal_title_emphasis_v1' else
                     'The journal title and volume number are not italicized in this APA reference.'
                     if result.rule_id == 'apa7_marked_periodical_italics_v1' else
                     'The title is not italicized in this APA reference.' if result.expected_italic else
                     'Use regular type, rather than italics, for the article title in this APA reference.'),
            rule_id=result.rule_id, rule_source=APA_TITLE_RULE_SOURCE,
            field_difference={'field_name':'title','submitted_value':title},
            rectangles=[], localization_status='not_assessed'))
    payload = assessment.get('order_result')
    if payload:
        result = ReferenceOrderResult.model_validate(payload)
        if (result.status == 'difference' and all(
            rid in references and hashlib.sha256(references[rid].raw_ref.encode()).hexdigest() == digest
            for rid,digest in result.reference_bindings.items()
        )):
            for position,rid in enumerate(result.observed_order):
                expected = result.expected_order.index(rid)
                if position == expected:
                    continue
                findings.append(dict(finding_type='reference_order', reference_id=rid,
                    finding=f'This reference is {position+1} in the list; alphabetical order by first-author surname places it at {expected+1}.',
                    rule_id=result.rule_id, rule_source=APA_ORDER_RULE_SOURCE,
                    field_difference={'field_name':'author','submitted_value':references[rid].raw_ref.split(',',1)[0]},
                    expected_reference_order=result.expected_order,
                    observed_reference_order=result.observed_order,
                    rectangles=[], localization_status='not_assessed'))
    return findings


def _longest_ordered_subsequence(positions: list[int]) -> set[int]:
    """Indexes of one deterministic longest strictly increasing subsequence."""
    from bisect import bisect_left
    tails: list[int] = []
    tail_index: list[int] = []
    previous = [-1] * len(positions)
    for index, value in enumerate(positions):
        slot = bisect_left(tails, value)
        if slot == len(tails):
            tails.append(value)
            tail_index.append(index)
        else:
            tails[slot] = value
            tail_index[slot] = index
        previous[index] = tail_index[slot - 1] if slot else -1
    kept: set[int] = set()
    cursor = tail_index[-1] if tail_index else -1
    while cursor != -1:
        kept.add(cursor)
        cursor = previous[cursor]
    return kept


def reference_order_projection(findings: list[dict], ordinal=None) -> tuple[list[dict], bool]:
    """Keep only entries that must move; report a pervasively unsorted list once.

    A position that merely shifts because another entry is misplaced is not
    itself misplaced: moving one entry from the end to the front changes every
    position. The misplaced set is everything outside one longest correctly
    ordered subsequence -- the fewest moves that restore the order. When it
    reaches the hanging-indent threshold, ``max(3, n // 2)``, the list is
    reported once and no entry is flagged. Findings whose observed order can
    not be established are returned unchanged.
    """
    order_findings = [f for f in findings if f.get('finding_type') == 'reference_order']
    if not order_findings:
        return list(findings), False
    expected = list(order_findings[0].get('expected_reference_order') or [])
    observed = list(order_findings[0].get('observed_reference_order') or [])
    if not observed and ordinal is not None and expected:
        ordinals = [ordinal(rid) for rid in expected]
        if all(value is not None for value in ordinals) and len(set(ordinals)) == len(ordinals):
            observed = [rid for _, rid in sorted(zip(ordinals, expected))]
    if (not expected or sorted(observed) != sorted(expected)
            or any(list(f.get('expected_reference_order') or []) != expected for f in order_findings)):
        return list(findings), False
    rank = {rid: position for position, rid in enumerate(expected)}
    kept_positions = _longest_ordered_subsequence([rank[rid] for rid in observed])
    misplaced = {rid for position, rid in enumerate(observed) if position not in kept_positions}
    pervasive = len(misplaced) >= max(3, len(observed) // 2)
    projected = [f for f in findings if f.get('finding_type') != 'reference_order'
                 or (not pervasive and f.get('reference_id') in misplaced)]
    return projected, pervasive


class ReferenceFormattingRuleResult(BaseModel):
    reference_id: str = Field(min_length=1)
    rule_id: Literal["reference_list_hanging_indent_0_5_in"]
    status: Literal["matches_rule", "difference", "not_assessed"]
    observed_points: float | None = None
    expected_points: float = HANGING_INDENT_POINTS
    reason_code: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_observation(self):
        if self.status == "not_assessed" and self.observed_points is not None:
            raise ValueError("An unassessed rule cannot carry an observation")
        if self.status != "not_assessed" and self.observed_points is None:
            raise ValueError("An assessed rule requires an observation")
        return self


class ReferenceFormattingAssessment(BaseModel):
    assessment_version: str = REFERENCE_FORMATTING_VERSION
    citation_format: Literal["apa", "mla"]
    status: Literal["partial", "not_assessed"]
    assessed_rule_ids: list[str] = Field(default_factory=list)
    results: list[ReferenceFormattingRuleResult] = Field(default_factory=list)
    result_counts: dict[str, int] = Field(default_factory=dict)
    primary_rule_sources: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    title_results: list[ReferenceTitleStyleResult] = Field(default_factory=list)
    order_result: ReferenceOrderResult | None = None

    @model_validator(mode="after")
    def validate_counts(self):
        expected = dict(sorted(Counter(item.status for item in self.results).items()))
        if self.result_counts != expected:
            raise ValueError("Reference-formatting counts do not match results")
        assessed = (any(item.status != "not_assessed" for item in self.results + self.title_results)
                    or bool(self.order_result and self.order_result.status != 'not_assessed'))
        if (self.status == "partial") != assessed:
            raise ValueError("Partial formatting status requires assessed evidence")
        return self


def assess_reference_formatting(
    layout: ReferenceLayoutArtifact,
    *, references: list[ParsedReference] | None = None, reference_section: str | None = None,
) -> ReferenceFormattingAssessment:
    """Retain legacy indentation results and optional versioned narrow rules."""
    sources = (
        [
            "https://www.apa.org/ed/precollege/psn/2020/09/apa-style-student-papers"
        ]
        if layout.citation_format == "apa"
        else [
            "https://style.mla.org/hanging-indents/",
            "https://style.mla.org/app/uploads/sites/3/2020/12/Formatting-a-Research-Paper_v3_-The-MLA-Style-Center.pdf",
        ]
    )
    results: list[ReferenceFormattingRuleResult] = []
    for entry in layout.entries:
        observed = entry.observed_hanging_indent_points
        if entry.mapping_status != "matched":
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="not_assessed",
                    reason_code="reference_layout_not_uniquely_matched",
                )
            )
        elif observed is None:
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="not_assessed",
                    reason_code="continuation_indent_not_observable",
                )
            )
        else:
            matches = (
                abs(observed - HANGING_INDENT_POINTS)
                <= HANGING_INDENT_TOLERANCE_POINTS
            )
            results.append(
                ReferenceFormattingRuleResult(
                    reference_id=entry.reference_id,
                    rule_id="reference_list_hanging_indent_0_5_in",
                    status="matches_rule" if matches else "difference",
                    observed_points=observed,
                    reason_code=(
                        "observed_hanging_indent_within_tolerance"
                        if matches
                        else "observed_hanging_indent_outside_tolerance"
                    ),
                )
            )
    counts = dict(sorted(Counter(item.status for item in results).items()))
    assessed = any(item.status != "not_assessed" for item in results)
    title_results = assess_title_styles(layout, references) if references is not None else []
    order_result = assess_reference_order(layout, references, reference_section) if references is not None else None
    rule_ids = ['reference_list_hanging_indent_0_5_in'] if assessed else []
    rule_ids += list(dict.fromkeys(r.rule_id for r in title_results if r.status != 'not_assessed'))
    if order_result and order_result.status != 'not_assessed':
        rule_ids.append(order_result.rule_id)
    return ReferenceFormattingAssessment(
        assessment_version='reference-formatting-v2' if references is not None else REFERENCE_FORMATTING_VERSION,
        citation_format=layout.citation_format,
        status="partial" if rule_ids else "not_assessed",
        assessed_rule_ids=rule_ids,
        results=results,
        result_counts=counts,
        primary_rule_sources=list(dict.fromkeys(sources + ([APA_TITLE_RULE_SOURCE, APA_ORDER_RULE_SOURCE]
                                  if references is not None and layout.citation_format == 'apa' else []))),
        title_results=title_results,
        order_result=order_result,
        limitations=[
            "Indentation counts describe only the observable 0.5-inch hanging-indent rule; title and order results are separate, not whole-entry or whole-paper verdicts.",
            "A one-line PDF entry has no visible continuation indentation and remains not assessed.",
            "Title checks cover uniformly styled, independently identified APA book/article and explicit film titles; mixed styles and unsupported work types abstain. Ordering covers only exactly accounted, distinct simple surnames. Spacing, heading placement, capitalization, punctuation and field order remain not assessed.",
        ],
    )


APA_CONTRIBUTION_RULE_SOURCE = (
    'https://apastyle.apa.org/style-grammar-guidelines/references/examples/edited-book-chapter-references'
)

# "In J. Belton (Ed.)," — one editor, captured for comparison with the author.
_SINGLE_EDITOR_RE = re.compile(
    r'\bIn\s+(?P<editor>[^(),;]{2,60}?)\s*\(\s*Eds?\.?\s*\)\s*,', re.IGNORECASE)


def _surname_and_initial(name: str) -> tuple[str, str]:
    """Reduce "Belton, J." and "J. Belton" to the same (surname, initial)."""
    cleaned = re.sub(r'[^A-Za-z,\'\-\s]', ' ', name or '').strip()
    if not cleaned:
        return '', ''
    if ',' in cleaned:
        surname, _, rest = cleaned.partition(',')
    else:
        words = cleaned.split()
        surname, rest = words[-1], ' '.join(words[:-1])
    initials = [word[0] for word in rest.split() if word]
    return surname.strip().casefold(), (initials[0].casefold() if initials else '')


def contribution_editor_findings(references, container_records: dict | None = None) -> list[dict]:
    """Report a chapter cited from a monograph as though the book were edited.

    The pattern alone proves nothing: an editor very often writes the
    introduction or a chapter of the collection they edited, and

        Smith, A. (2019). Introduction. In A. Smith (Ed.), The handbook of
        things (pp. 1-10). Routledge.

    is correct APA. What distinguishes Belton's reference is the located work:
    *American Cinema/American Culture* is a single-authored book, so there is
    no edited collection for a chapter to sit in. The finding therefore
    requires an identified container that is a monograph, and stays silent
    when the container was not identified or is genuinely edited.
    """
    findings = []
    records = container_records or {}
    for reference in references:
        if getattr(reference, 'source_kind', '') not in {'book_section', 'edited_collection'}:
            continue
        raw = getattr(reference, 'raw_ref', '') or ''
        match = _SINGLE_EDITOR_RE.search(raw)
        author = getattr(reference, 'author', '') or ''
        if not match or not author:
            continue
        editor = match.group('editor').strip()
        if re.search(r'\b(and|&)\b', editor, re.IGNORECASE):
            continue
        if _surname_and_initial(editor) != _surname_and_initial(author):
            continue
        record = records.get(getattr(reference, 'reference_id', '')) or {}
        if not record.get('is_monograph'):
            continue  # unidentified, or a genuine edited collection
        located = str(record.get('title') or '').strip()
        findings.append(dict(
            finding_type='contribution_author_is_volume_editor',
            reference_id=reference.reference_id,
            finding=(
                f'The located work{" " + located if located else ""} is a single-authored '
                'book, not an edited collection, so this reference cites a part of it as '
                'though it were a chapter in an edited volume. Cite the book itself and '
                'give the page range of the part used.'),
            rule_id='apa7_contribution_author_is_editor_v1',
            rule_source=APA_CONTRIBUTION_RULE_SOURCE,
            located_record={**{k: record.get(k) for k in ('title', 'authors', 'provider')},
                            'source_kind': record.get('located_source_kind') or record.get('source_kind')},
            field_difference={'field_name': 'editor', 'submitted_value': editor},
            rectangles=[], localization_status='not_assessed'))
    return findings


# Fields whose disagreement is a citation detail, not an identity question.
# Title and author are excluded deliberately: when either materially conflicts
# the located record is a different work, and reporting its year or journal as
# the reader's "correct" value would assert that the cited work exists with
# those details. Measured 2026-09-22, every large year gap among articles in
# the development corpus was exactly that -- a different paper, sometimes a
# later one by the same authors -- and none was a wrong year.
# `year` is deliberately absent. Measured over 307 stored references, a
# year difference on a record already anchored by title and author was
# never a citation error: 15 of 18 differed by exactly one, which is the
# online-first and issue-year pattern, and the other three differed by 17
# to 28 years, which is a reprint or a provider record error. Monograph
# years are covered by `book_publication_year_discrepancy`, whose two
# independent edition records are what make that claim safe to publish.
# Kept in step with `_MATERIAL_CONFLICT_FIELDS` in reference_discovery: pages
# and publisher are excluded because their measured conflicts were input
# defects, not reference errors. See the note there.
_CONFLICT_REPORTABLE_FIELDS = ('volume', 'issue')
_CONFLICT_FIELD_LABELS = {
    'container_title': 'journal or book title', 'volume': 'volume',
    'issue': 'issue', 'pages': 'pages', 'publisher': 'publisher',
}
BIBLIOGRAPHIC_FIELD_CONFLICT_RULE = 'bibliographic-field-conflict-v1'


def bibliographic_field_conflicts(reference: ParsedReference, discovery: dict) -> dict | None:
    """Name the reference details that disagree with the located record.

    The work must first be recognisable: the title must agree and the author
    must agree or differ only minorly. That is what separates a citation
    detail a reader can check from a search result about another work. Values
    are reported only when the stored comparison still binds the reference as
    it reads now, and qualifying candidates that contradict each other abstain.

    This states a discrepancy. It is not a verdict on the reference, and it
    never asserts which value is correct.
    """
    from app.services.reference_discovery import ReferenceDiscoveryRecord, _value_hash
    if reference.needs_review:
        return None
    try:
        record = ReferenceDiscoveryRecord.model_validate(discovery)
    except ValueError:
        return None
    if record.reference_id != reference.reference_id or record.expected.reference_parse_review:
        return None

    located: dict[str, set[str]] = {}
    providers: set[str] = set()
    for candidate in record.candidates:
        if not candidate.is_credible:
            continue
        if not any(a.attempt_id == candidate.attempt_id and a.provider == candidate.provider
                   and a.permitted and a.completed_at
                   and a.outcome in {'candidate_found', 'candidates_processed'}
                   for a in record.attempts):
            continue
        comparisons = {item.field_name: item for item in candidate.comparisons}
        title = comparisons.get('title')
        author = comparisons.get('author')
        if (title is None or title.outcome != 'agreement'
                or author is None or author.outcome not in {'agreement', 'minor_difference'}):
            continue
        # The anchors must still bind the reference as it reads now.
        if (title.expected_sha256 != _value_hash(reference.title)
                or author.expected_sha256 != _value_hash(reference.author)):
            continue
        for field in _CONFLICT_REPORTABLE_FIELDS:
            comparison = comparisons.get(field)
            submitted = str(getattr(reference, field, '') or '')
            observed = str(getattr(candidate.observed, field, '') or '')
            if (comparison is None or comparison.outcome != 'material_conflict'
                    or not submitted or not observed
                    or comparison.expected_sha256 != _value_hash(submitted)
                    or comparison.observed_sha256 != _value_hash(observed)):
                continue
            located.setdefault(field, set()).add(observed)
            providers.add(candidate.provider)

    # A field whose located value is disputed between qualifying records is
    # not a discrepancy anyone can act on.
    differences = [
        dict(field_name=field,
             submitted_value=str(getattr(reference, field, '') or ''),
             located_value=next(iter(values)))
        for field, values in sorted(located.items())
        if len(values) == 1
    ]
    if not differences:
        return None
    named = ', '.join(_CONFLICT_FIELD_LABELS.get(d['field_name'], d['field_name'])
                      for d in differences)
    detail = '; '.join(f"{_CONFLICT_FIELD_LABELS.get(d['field_name'], d['field_name'])}: "
                       f"reference gives {d['submitted_value']}, located record gives "
                       f"{d['located_value']}" for d in differences)
    return dict(
        finding_type='bibliographic_field_conflict',
        reference_id=reference.reference_id,
        rule_id=BIBLIOGRAPHIC_FIELD_CONFLICT_RULE,
        reference_text_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
        finding=(f'The located record matches this work by title and author but differs on '
                 f'{named}. {detail}. Check these details against the source; a record that '
                 f'agrees on the work may still index a different version of it.'),
        field_differences=differences,
        located_record_providers=sorted(providers),
        exact_edition_established=False,
        rectangles=[], localization_status='not_assessed',
    )
