"""Paper-local APA/MLA locator omissions, not source/locator accuracy."""
from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.schemas import ParsedReference
from app.services.verification_evidence import ClaimEvidence


RULE_SOURCE = 'https://apastyle.apa.org/style-grammar-guidelines/citations/quotations'
_QUOTES = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
_ALTERNATIVE = re.compile(
    r'\b(?:pp?|pages?|paras?|paragraphs?|sections?|chapters?|chap|'
    r'headings?|lines?|verses?|acts?|scenes?|timestamps?|'
    r'locations?|loc|appendix|appendices)\b|\b(?:tables?|figures?)\s+\d|\b\d{1,2}:\d{2}\b|\(\s*\d', re.I,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class QuotationLocatorRequirement(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    rule_id: Literal['apa7_explicit_quotation_locator_v1', 'apa7_narrative_quotation_locator_v2', 'apa7_attributed_quotation_locator_v3', 'mla9_explicit_passage_locator_v1'] = 'apa7_explicit_quotation_locator_v1'
    passage_kind: Literal['quotation', 'paraphrase'] = 'quotation'
    pagination_text: str | None = None
    pagination_start: int | None = None
    pagination_end: int | None = None
    claim_id: str
    claim_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reference_id: str | None = None
    reference_sha256: str | None = None
    status: Literal['missing', 'locator_present', 'not_assessed']
    reason_code: str
    quote_start: int | None = Field(default=None, ge=0)
    quote_end: int | None = Field(default=None, gt=0)
    quote_sha256: str | None = None
    context_start: int | None = Field(default=None, ge=0)
    context_end: int | None = Field(default=None, gt=0)
    context_sha256: str | None = None
    locator_accuracy: Literal['not_assessed'] = 'not_assessed'

    @model_validator(mode='after')
    def validate_missing(self):
        if self.status == 'missing' and (
            not self.reference_id or not self.reference_sha256 or not self.quote_sha256
            or self.quote_start is None or self.quote_end is None or self.quote_end <= self.quote_start
            or self.context_start is None or self.context_end is None or self.context_end <= self.context_start
            or not self.context_sha256
        ):
            raise ValueError('Missing-locator findings require exact quotation/reference/context bindings')
        if self.status == 'missing' and self.rule_id == 'mla9_explicit_passage_locator_v1' and (
            not self.pagination_text or self.pagination_start is None or self.pagination_start < 0
            or self.pagination_end is None or self.pagination_end <= self.pagination_start
        ):
            raise ValueError('MLA omissions require the original reference pagination span')
        return self


def assess_quotation_locators(*, body_text: str, citation_format: str,
                              claims: list[ClaimEvidence], references: list[ParsedReference]) -> list[QuotationLocatorRequirement]:
    """A narrow explicit single-source quotation followed by an APA marker.

    Absence is not inferred from the parsed page field alone. Inspect the
    complete bounded paragraph for alternative/trailing location information.
    Supplied numbers are not validated here. A bounded explicit narrative
    reporting clause is supported; block quotations and uncertain scopes abstain.
    """
    refs = {r.reference_id:r for r in references}
    if citation_format == 'mla':
        return _assess_mla_locators(body_text, claims, refs)
    results = []
    for claim in claims:
        if claim.claim_type != 'quotation':
            continue
        data = dict(claim_id=claim.claim_id, claim_sha256=_hash(claim.text))
        narrative = claim.citation_marker_type == 'narrative'
        if narrative:
            data['rule_id'] = 'apa7_narrative_quotation_locator_v2'
        def record(status, reason, **extra):
            results.append(QuotationLocatorRequirement(**data,status=status,reason_code=reason,**extra))
        if citation_format != 'apa':
            record('not_assessed','style_requires_consulted_version_locator_evidence');continue
        if (claim.extraction_confidence != 'high' or len(claim.reference_ids) != 1
                or claim.citation_marker_type not in {'parenthetical','narrative'} or len(claim.citation_markers) != 1):
            record('not_assessed','quotation_attribution_not_in_accepted_scope');continue
        ref=refs.get(claim.reference_ids[0])
        member=claim.citation_markers[0]
        if (ref is None or ref.needs_review or ref.is_media_source
                or member.reference_ids != claim.reference_ids):
            record('not_assessed','source_membership_or_work_kind_uncertain');continue
        if (claim.passage_start < 0 or claim.passage_end <= claim.passage_start
                or body_text[claim.passage_start:claim.passage_end] != claim.text):
            record('not_assessed','original_paper_span_not_exact');continue
        quotes=list(_QUOTES.finditer(claim.text))
        if not quotes:
            record('not_assessed','single_complete_quotation_not_established');continue
        if not narrative and member.local_start is not None:
            quotes=[q for q in quotes if q.end() <= member.local_start]
        if not quotes:
            record('not_assessed','quotation_marker_not_immediately_attached');continue
        quoted=quotes[-1];quote=quoted.group(1) or quoted.group(2)
        if re.sub(r'\W+', '', quote.casefold()) == re.sub(r'\W+', '', ref.title.casefold()):
            record('not_assessed','short_quoted_term_or_work_title');continue
        # The complete marker must immediately follow the quote. A marker
        # attached to a later sentence cannot establish this quote's source.
        trailing=claim.text[quoted.end():].strip()
        marker=claim.citation_marker
        if narrative:
            lead = claim.text[:quotes[0].start()]
            according = bool(re.fullmatch(r'According\s+to\s+'+re.escape(marker)+r'\s*,[^.!?\n]{0,200}', lead, re.I))
            if (len(quote.split()) >= 40 or member.local_start is None or member.local_end is None
                    or claim.text[member.local_start:member.local_end] != marker
                    or not (according or re.fullmatch(r'[^.!?\n]{0,100}?'+re.escape(marker)
                        +r'\s+(?:writes|wrote|notes|noted|argues|argued|states|stated|reports|reported|explains|explained|says|said|points out|pointed out)\b[^.!?\n]{0,180}', lead, re.I)
                    ) or (not according and not re.fullmatch(r'[.!?\s]*', trailing))):
                record('not_assessed','narrative_quotation_scope_not_explicit');continue
        elif not re.match(r'(?:[A-Za-z’\x27-]+\s+){0,5}[.,;:!?\s]*'+re.escape(marker),trailing):
            record('not_assessed','quotation_marker_not_immediately_attached');continue
        if claim.page_locator.strip():
            record('locator_present','parsed_locator_present_accuracy_not_assessed');continue
        # Require a bare author/date marker. Additional text can be a valid
        # heading, locator, note or secondary attribution; never discard it.
        bare = (r'[^();\n]{1,180}\s+\((?:19|20)\d{2}[a-z]?\)' if narrative
                else r'\([^();\n]{1,180},\s*(?:19|20)\d{2}[a-z]?\)')
        if (member.text != marker or not re.fullmatch(bare,marker)):
            record('not_assessed','marker_contains_possible_locator_or_other_context');continue
        start=body_text.rfind('\n\n',0,claim.passage_start)+2
        if start==1:start=0
        end=body_text.find('\n\n',claim.passage_end)
        if end<0:end=len(body_text)
        context=body_text[start:end]
        if len(context)>8000:
            record('not_assessed','complete_local_context_exceeds_bound');continue
        # An unrelated date or the word "figure" elsewhere in the paragraph
        # is not a locator for this quotation. Inspect the exact attributed
        # claim plus immediately following parenthetical location information.
        after=body_text[claim.passage_end:end]
        adjacent=re.match(r'\s*(?:\([^()\n]{1,180}\)\s*)+',after)
        location_context=claim.text+(adjacent.group() if adjacent else '')
        unquoted=_QUOTES.sub('',location_context)
        # Exact bare author/year markers are not alternative locators.
        # Preserve any additional location text, including unknown formats.
        for other in claims:
            if (start <= other.passage_start and other.passage_end <= end
                    and body_text[other.passage_start:other.passage_end] == other.text):
                for m in other.citation_markers:
                    if (m.local_start is not None and m.local_end is not None
                            and other.text[m.local_start:m.local_end] == m.text
                            and re.fullmatch(r'[^();\n]{1,180}\s+\((?:19|20)\d{2}[a-z]?\)|\([^();\n]{1,180},\s*(?:19|20)\d{2}[a-z]?\)', m.text)):
                        unquoted = unquoted.replace(m.text, '')
        if (_ALTERNATIVE.search(unquoted)
                or re.search(r'\([^)]*\)',unquoted.replace(marker,''))
                or re.search(r'personal communication|as cited in|epigraph',unquoted,re.I)):
            record('not_assessed','possible_alternative_locator_or_exception_in_context');continue
        data['rule_id']='apa7_attributed_quotation_locator_v3'
        record('missing','explicit_apa_quotation_has_no_observed_locator',
               reference_id=ref.reference_id,reference_sha256=_hash(ref.raw_ref),
               quote_start=quoted.start()+1,quote_end=quoted.end()-1,quote_sha256=_hash(quote),
               context_start=start,context_end=end,context_sha256=_hash(context))
    return results


def _assess_mla_locators(body_text, claims, refs):
    """Bounded MLA author-attributed passages with explicit reference pagination.

    Bibliographic page ranges are premises, never the missing pinpoint location.
    No inference from book kind, DOI, article ID, PDF page count or absent fields.
    """
    results = []
    page_pattern = re.compile(r'\bpp?\.\s*(\d{1,5})(?:\s*[–—-]\s*(\d{1,5}))?(?!\w)', re.I)
    for claim in claims:
        data = dict(rule_id='mla9_explicit_passage_locator_v1', claim_id=claim.claim_id,
                    claim_sha256=_hash(claim.text), passage_kind=claim.claim_type)
        def record(status, reason, **extra):
            results.append(QuotationLocatorRequirement(**data, status=status, reason_code=reason, **extra))
        if (claim.extraction_confidence != 'high' or len(claim.reference_ids) != 1
                or claim.context_dependency_status not in {'not_required', 'resolved'}):
            record('not_assessed', 'mla_attribution_uncertain'); continue
        ref = refs.get(claim.reference_ids[0])
        if ref is None or ref.needs_review or ref.is_media_source or not ref.title.strip():
            record('not_assessed', 'mla_reference_uncertain'); continue
        if body_text[claim.passage_start:claim.passage_end] != claim.text or claim.passage_start < 0:
            record('not_assessed', 'original_paper_span_not_exact'); continue
        if (not claim.citation_markers or any(m.reference_ids != claim.reference_ids or
                claim.text[m.local_start:m.local_end] != m.text for m in claim.citation_markers)):
            record('not_assessed', 'mla_marker_binding_uncertain'); continue
        if claim.page_locator.strip():
            record('locator_present', 'parsed_locator_present_accuracy_not_assessed'); continue
        pages = list(page_pattern.finditer(ref.raw_ref))
        if len(pages) != 1:
            record('not_assessed', 'mla_reference_pagination_unknown'); continue
        page = pages[0]
        # An unrelated page-like string in a title or URL is not pagination.
        prefix = ref.raw_ref[:page.start()]
        if (not re.search(r'\b(?:18|19|20)\d{2}\b', prefix)
                or (ref.title and ref.title in ref.raw_ref and page.start() < ref.raw_ref.index(ref.title)+len(ref.title))):
            record('not_assessed', 'mla_pagination_field_unbound'); continue
        links = re.findall(r'(?:https?://|www\.)\S+', ref.raw_ref, re.I)
        if (re.search(r'\b(?:unpaginated|HTML|EPUB|Kindle|e-book|ebook)\b', ref.raw_ref, re.I)
                or any('doi.org/' not in link.lower() and not re.search(r'\.pdf(?:[?#]|$)', link.rstrip('.,;'), re.I) for link in links)):
            record('not_assessed', 'mla_online_representation_uncertain'); continue
        first, last = page.group(1), page.group(2)
        if last is None or first == last:
            record('not_assessed', 'mla_one_page_work_exception'); continue
        # MLA elided endings: 149–66 means 149–166; reject malformed ranges.
        end_page = int(first[:len(first)-len(last)] + last) if len(last) < len(first) else int(last)
        if end_page <= int(first):
            record('not_assessed', 'mla_pagination_range_uncertain'); continue
        surname = ref.author.split(',')[0].strip()
        if not re.fullmatch(r'[A-Z][A-Za-z\x27’-]+', surname):
            record('not_assessed', 'mla_author_marker_scope_unsupported'); continue
        parenthetical = [m for m in claim.citation_markers if m.marker_type == 'parenthetical']
        narrative = [m for m in claim.citation_markers if m.marker_type == 'narrative']
        if (len(parenthetical) > 1 or len(narrative) > 1 or
                len(parenthetical)+len(narrative) != len(claim.citation_markers) or
                any(not m.text.startswith(surname+' ') for m in narrative) or
                any(m.text != '('+surname+')' for m in parenthetical)):
            record('not_assessed', 'mla_possible_alternative_locator'); continue
        text = claim.text
        end = parenthetical[0].local_start if parenthetical else len(text.rstrip('.!? '))
        if parenthetical and text[parenthetical[0].local_end:].strip(' .!?'):
            record('not_assessed', 'mla_marker_not_final'); continue
        quoted = list(_QUOTES.finditer(text))
        if claim.claim_type == 'quotation':
            if len(quoted) != 1 or len((quoted[0].group(1) or quoted[0].group(2)).split()) < 8:
                record('not_assessed', 'single_complete_quotation_not_established'); continue
            q = quoted[0]
            start_span, end_span = q.start()+1, q.end()-1
            if text[q.end():end].strip(' .,:;!?'):
                record('not_assessed', 'quotation_marker_not_immediately_attached'); continue
            if re.sub(r'\W+', '', text[start_span:end_span].casefold()) == re.sub(r'\W+', '', ref.title.casefold()):
                record('not_assessed', 'short_quoted_term_or_work_title'); continue
        else:
            # Explicit attribution of a proposition, not merely a work mention.
            # Wider source-blind paraphrase-scope classification is not assumed.
            proposition = re.search(r'\b'+re.escape(surname)+r'\s+(?:argues|states|reports|notes|observes|claims|suggests|explains|concludes|finds)\s+that\s+\S', text)
            if quoted or not proposition:
                record('not_assessed', 'mla_specific_paraphrase_scope_unestablished'); continue
            if re.search(r'\b(?:as a whole|overall|throughout|central thesis|main argument|entire (?:book|work|article))\b', text, re.I):
                record('not_assessed', 'mla_whole_work_reference_possible'); continue
            start_span, end_span = proposition.start(), end
            if len(text[start_span:end].split()) < 10:
                record('not_assessed', 'mla_specific_paraphrase_scope_unestablished'); continue
        start_context = body_text.rfind('\n\n', 0, claim.passage_start)+2
        if start_context == 1: start_context = 0
        end_context = body_text.find('\n\n', claim.passage_end)
        if end_context < 0: end_context = len(body_text)
        context = body_text[start_context:end_context]
        outside_quotes = _QUOTES.sub('', context)
        for marker in parenthetical: outside_quotes = outside_quotes.replace(marker.text, '')
        if (len(context) > 8000 or _ALTERNATIVE.search(outside_quotes)
                or re.search(r'\([^)]*\)|\b(?:epigraph|qtd\.|as cited in)\b', outside_quotes, re.I)):
            record('not_assessed', 'possible_alternative_locator_or_exception_in_context'); continue
        record('missing', 'mla_specific_passage_missing_locator_with_reference_page_range',
               reference_id=ref.reference_id, reference_sha256=_hash(ref.raw_ref),
               quote_start=start_span, quote_end=end_span, quote_sha256=_hash(text[start_span:end_span]),
               context_start=start_context, context_end=end_context, context_sha256=_hash(context),
               pagination_text=page.group(), pagination_start=page.start(), pagination_end=page.end())
    return results


def quotation_locator_findings(results: list[QuotationLocatorRequirement],
                               claims: list[ClaimEvidence], references: dict[str,ParsedReference]) -> list[dict]:
    """Project only retained, hash-matching omissions; historical absence is empty."""
    indexed={c.claim_id:c for c in claims};findings=[]
    for result in results:
        claim=indexed.get(result.claim_id);ref=references.get(result.reference_id)
        if (result.status!='missing' or claim is None or ref is None
                or _hash(claim.text)!=result.claim_sha256 or _hash(ref.raw_ref)!=result.reference_sha256):continue
        quote=claim.text[result.quote_start:result.quote_end]
        if _hash(quote)!=result.quote_sha256:continue
        mla = result.rule_id == 'mla9_explicit_passage_locator_v1'
        if mla and (not result.pagination_text or ref.raw_ref[result.pagination_start:result.pagination_end] != result.pagination_text):
            continue
        findings.append(dict(finding_type='required_quotation_locator_missing',reference_id=ref.reference_id,
            claim_id=claim.claim_id,citation_text=claim.text,quote_text=quote,citation_marker=claim.citation_marker,
            passage_kind=result.passage_kind, citation_style='mla' if mla else 'apa',
            pagination_text=result.pagination_text,
            finding=(f'Add a page number or another appropriate numbered division for this {result.passage_kind}. Your reference gives {result.pagination_text}; that is the work’s page range, not the specific location of the borrowed passage.' if mla else 'Add a page number or another suitable location for this quotation. For an unpaginated source, use a paragraph number or a heading/section that helps the reader find it.'),
            rule_id=result.rule_id,rule_source='https://style.mla.org/in-text-citations-overview/' if mla else RULE_SOURCE,
            field_difference={'field_name':'quotation','submitted_value':quote},
            rectangles=[],localization_status='not_assessed'))
    return findings
