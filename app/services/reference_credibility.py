"""Evidence-bound reference findings; not an AI or source-existence detector.

Consumes the existing identity comparisons and completion assessor. Never
changes discovery outcomes, acquisition/admission or required search routes.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from html import escape
from urllib.parse import quote

from app.services.reference_discovery import (
    ReferenceDiscoveryRecord, ReferenceDiscoveryTrace, assess_reference_discovery_trace,
    _normalize_doi, _normalize_text, _value_hash, _search_is_incomplete,
    is_edition_sensitive_reference, build_reference_discovery_candidate,
)
from app.services.retrieval.base import RetrievalResult
from app.services.metadata_identity import POLICY as METADATA_IDENTITY_POLICY, observation_is_bound

POLICY = 'reference-credibility-v3'
FINDING_TEXT = ('Potentially fabricated reference. Searches using the supplied '
                'bibliographic details did not establish a matching work.')


def credibility_finding_html(finding):
    kind = finding.get('finding_type')
    if kind in {'unverified_reference', 'potentially_fabricated_reference'}:
        # The former review flag is presented as the current one.
        from app.services.reference_verification import FINDING_TEXT, LABEL
        return ('<mark class="issue-heading unverified">' + LABEL + '.</mark>'
                + escape(FINDING_TEXT[len(LABEL) + 1:]))
    if kind == 'reference_identifier_conflict':
        # The record itself follows; restating its title and authors here
        # only repeated it (owner decision 2026-09-24).
        return 'The submitted DOI identifies:'
    return escape(str(finding.get('finding') or ''))


def _placeholder_finding(reference):
    # Only unfinished identifier syntax, not common names, plausible titles,
    # Anonymous, n.d., forthcoming or "to appear" publication statuses.
    match = re.search(r'\barXiv\s*:\s*\d{4}\.X{4,5}\b|\b10\.X{4,9}/X{3,}\b', reference.raw_ref, re.I)
    if not match or reference.needs_review:
        return None
    return dict(finding_type='reference_identifier_placeholder', reference_id=reference.reference_id,
        policy_version=POLICY, finding='The reference contains an unfinished identifier placeholder: '+match.group()+'.',
        reference_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
        field_difference=dict(field_name='identifier', submitted_value=match.group()),
        original_span=dict(start=match.start(), end=match.end()), rectangles=[], localization_status='not_assessed')


def credibility_records_html(finding):
    """Shared live/portable/PDF metadata explanation; no source-text evidence."""
    if finding.get('finding_type') in {'unverified_reference', 'potentially_fabricated_reference'}:
        explanation = str(finding.get('evidence_explanation') or '')
        limits = ''.join('<p class="muted">' + escape(str(item)) + '</p>'
                         for item in finding.get('limitations') or [])
        return ('<details><summary>Search Details</summary><p>' + escape(explanation) +
                '</p>' + limits + '</details>') if explanation else ''
    rows = []
    if finding.get('evidence_explanation'):
        rows.append('<p>' + escape(str(finding['evidence_explanation'])) + '</p>')
    for record in finding.get('records') or []:
        observed = record.get('observed') or {}
        text = '. '.join(str(v) for v in (', '.join(observed.get('authors') or []),
            observed.get('year'), observed.get('title'), observed.get('container_title')) if v)
        doi = _normalize_doi(observed.get('doi') or '')
        link = (' <a href="https://doi.org/' + escape(quote(doi, safe='/'), quote=True)
                + '" target="_blank" rel="noopener noreferrer">' + escape(doi) + '</a>') if doi else ''
        rows.append('<p class="full-reference">' + escape(text) + link + '</p>')
    return ''.join(rows)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _bound_candidate(record, candidate, *, fields=('title', 'author', 'doi'), metadata_bound=False):
    """Check observation, field and successful-attempt bindings, not similarity alone."""
    if not any(a.attempt_id == candidate.attempt_id and a.provider == candidate.provider
               and a.permitted and a.completed_at and a.outcome == 'candidate_found'
               for a in record.attempts):
        return False
    if candidate.acquisition_outcome in {'not_attempted', 'unknown'}:
        return False
    registered = (candidate.provider == 'crossref' and candidate.registration_record_sha256
                  and candidate.registration_observation_sha256 == hashlib.sha256(
                      candidate.observed.model_dump_json().encode()).hexdigest())
    independent = (candidate.location_provenance == 'independently_acquired_content'
                   and candidate.validated_identity_content_sha256 and candidate.location_sha256)
    if not (registered or independent or metadata_bound):
        return False
    comparisons = {c.field_name: c for c in candidate.comparisons}
    if len(comparisons) != len(candidate.comparisons):
        return False
    recomputed = build_reference_discovery_candidate(attempt_id=candidate.attempt_id,
        provider=candidate.provider, expected=record.expected,
        result=RetrievalResult(source_name='comparison_only', success=True,
            title=candidate.observed.title, authors=candidate.observed.authors,
            year=candidate.observed.year, doi=candidate.observed.doi))
    for name in fields:
        old = ' | '.join(record.expected.authors) if name == 'author' else getattr(record.expected, name)
        new = ' | '.join(candidate.observed.authors) if name == 'author' else getattr(candidate.observed, name)
        comparison = comparisons.get(name)
        if not old or not new or comparison is None:
            return False
        if comparison.expected_sha256 != _value_hash(old) or comparison.observed_sha256 != _value_hash(new):
            return False
        if comparison.outcome != _outcome(recomputed, name):
            return False
    return True


def _eligible_metadata_observation(record, candidate):
    return observation_is_bound(candidate) and any(
        a.attempt_id == candidate.attempt_id and a.route_category == 'academic_adapter'
        and any(q.query_id in a.query_ids and q.execution_provider == candidate.provider
                and q.execution_outcome == 'results' for q in record.queries)
        for a in record.attempts)


def _corroborated_metadata(record, candidate):
    """Qualify title/author comparison, not DOI error, source admission or absence.

    Separate service observations can share underlying deposits. This is a
    reproducible agreement check, not independent votes or a probability.
    """
    if not _eligible_metadata_observation(record, candidate) or not _bound_candidate(
            record, candidate, fields=('title', 'author'), metadata_bound=True):
        return None
    old = candidate.observed
    if not re.fullmatch(r'[12]\d{3}', old.year or ''):
        return None
    def author_key(authors):
        # Exact normalized name tokens; no surname-only/semantic corroboration.
        return sorted(tuple(sorted(_normalize_text(a).split())) for a in authors)
    peers = []
    for peer in record.candidates:
        if peer.candidate_id == candidate.candidate_id or peer.provider == candidate.provider:
            continue
        receipt_bound = _eligible_metadata_observation(record, peer)
        if not _bound_candidate(record, peer, fields=('title', 'author'), metadata_bound=receipt_bound):
            continue
        new = peer.observed
        same_doi = bool(old.doi and new.doi and _normalize_doi(old.doi) == _normalize_doi(new.doi))
        same_title = _normalize_text(old.title) == _normalize_text(new.title)
        same_authors = author_key(old.authors) == author_key(new.authors)
        same_year = bool(new.year and old.year == new.year)
        doi_conflict = bool(old.doi and new.doi and not same_doi)
        # A conflicting observation of the same ID/work defeats corroboration,
        # even when a third record would otherwise agree.
        if same_doi and not (same_title and same_authors and same_year):
            return None
        if same_title and same_authors and (doi_conflict or not same_year):
            return None
        if same_title and same_authors and same_year and not doi_conflict:
            peers.append(peer.candidate_id)
    if not peers:
        return None
    outcomes = [_outcome(candidate, f) for f in ('title', 'author')]
    disposition = ('compatible_work' if all(o in {'agreement', 'minor_difference'} for o in outcomes)
                   else 'different_work' if all(o == 'material_conflict' for o in outcomes)
                   else 'bibliographic_difference')
    return dict(candidate_id=candidate.candidate_id, corroborating_candidate_ids=sorted(peers),
                policy_version=METADATA_IDENTITY_POLICY,
                metadata_record_sha256=candidate.metadata_identity.record_sha256,
                compared_fields=['title', 'author'],
                disposition=disposition)


def _outcome(candidate, name):
    return next((c.outcome for c in candidate.comparisons if c.field_name == name), 'unknown')


def _possible_title_extraction_damage(expected, observed):
    """Suppress a negative for joined words/title-container spill, never confirm identity."""
    compact = lambda text: ''.join(c for c in _normalize_text(text) if c.isalnum())
    old, new = compact(expected), compact(observed)
    return len(new) >= 16 and (old == new or old.startswith(new))


def _search_title_boundary_unresolved(reference):
    """Do not turn visibly damaged search inputs into cumulative negatives.

    This is an abstention, not a corrected title or a source-identity claim.
    It also works when a catalog returned nothing to compare against.
    """
    title = reference.title.rstrip('. ')
    if any(len(re.findall(r'[a-z][A-Z]', token)) >= 3
           for token in re.findall(r'[A-Za-z]+', title)):
        return True
    if (reference.source_kind == 'journal_article'
            and re.search(r'\.\s+\*[^*]+\*$', title)):
        return True  # Article title still includes a marked container element.
    if reference.source_kind in {'monograph', 'edited_collection'} and '.' in title:
        from app.services.source_type import _BOOK_PUBLISHER_RE
        tail = title.rsplit('.', 1)[1].strip()
        if _BOOK_PUBLISHER_RE.fullmatch(tail):
            return True
    return False


def _record_view(candidate):
    fields = candidate.observed.model_dump(mode='json')
    doi = _normalize_doi(fields.get('doi') or '')
    return dict(candidate_id=candidate.candidate_id, provider=candidate.provider,
                record_sha256=candidate.registration_record_sha256 or candidate.validated_identity_content_sha256
                    or (candidate.metadata_identity.record_sha256 if candidate.metadata_identity else None),
                observed=fields, url='https://doi.org/' + quote(doi, safe='/') if doi else '')


def _input_matches(reference, record):
    expected = record.expected
    return (record.reference_id == reference.reference_id and not reference.needs_review
            and not expected.reference_parse_review
            and expected.title == reference.title and expected.authors == [reference.author]
            and expected.year == reference.year and expected.doi == reference.doi
            and bool(reference.title and reference.author)
            and _normalize_text(reference.title) in _normalize_text(reference.raw_ref)
            and _normalize_text(reference.author) in _normalize_text(reference.raw_ref)
            and not re.search(r'https?://|\bdoi\b', reference.title, re.I)
            and len(re.findall(r'\((?:19|20)\d{2}[a-z]?(?:[,)]|\s)', reference.raw_ref)) <= 1)


def _assess_compound_credibility(reference, discovery=None, trace=None):
    """Return auditable assessment even when no finding can safely be shown.

The first enabled strong rule is a compound cross-record inconsistency: the
submitted DOI identifies different title/authors, and the submitted title is
registered to a second work with different authors. A copied wrong DOI alone
is only a factual identifier error. Books/editions remain outside this rule.
"""
    result = dict(policy_version=POLICY, reference_id=reference.reference_id,
                  reference_sha256=hashlib.sha256(reference.raw_ref.encode()).hexdigest(),
                  status='not_assessed', reason_code='discovery_unavailable', findings=[])
    placeholder = _placeholder_finding(reference)
    if placeholder:
        result.update(status='finding', reason_code='unfinished_identifier', findings=[placeholder])
        return result
    try:
        record = ReferenceDiscoveryRecord.model_validate(discovery)
    except (ValueError, TypeError):
        return result
    result['discovery_sha256'] = _digest(discovery)
    if not _input_matches(reference, record):
        result['reason_code'] = 'original_reference_or_parse_unresolved'
        return result
    result.update(status='no_finding', reason_code='no_qualified_compound_discrepancy')
    bound = [c for c in record.candidates if _bound_candidate(record, c)]
    compatible = [c for c in bound if all(_outcome(c, field) in {'agreement', 'minor_difference'}
                                         for field in ('title', 'author'))]
    identifiers = [c for c in bound if _outcome(c, 'doi') == 'agreement'
                   and _outcome(c, 'title') == 'material_conflict'
                   and _outcome(c, 'author') == 'material_conflict'
                   and not _possible_title_extraction_damage(record.expected.title, c.observed.title)]
    if not identifiers:
        return result
    if any(_outcome(c, 'doi') == 'agreement' for c in compatible):
        result['reason_code'] = 'conflicting_qualified_identifier_observations'
        return result
    # Repeated observations of one registration are one discrepancy, not votes.
    first = sorted(identifiers, key=lambda c: c.candidate_id)[0]
    text = (f'The submitted DOI identifies “{first.observed.title}” by '
            f'{", ".join(first.observed.authors)}, not the title and authors in this reference.')
    finding = dict(finding_type='reference_identifier_conflict', reference_id=reference.reference_id,
        policy_version=POLICY, finding=text, records=[_record_view(first)],
        field_difference=dict(field_name='doi', submitted_value=reference.doi),
        rectangles=[], localization_status='not_assessed',
        reference_sha256=result['reference_sha256'], discovery_sha256=result['discovery_sha256'])
    result.update(status='finding', reason_code='identifier_points_to_different_work', findings=[finding])
    if compatible or is_edition_sensitive_reference(record.expected, reference.raw_ref):
        result['reason_code'] = 'credible_correction_or_edition_requires_only_identifier_finding'
        return result
    if not first.registration_record_sha256:
        return result
    titles = [c for c in bound if _outcome(c, 'title') == 'agreement'
              and _outcome(c, 'author') == 'material_conflict'
              and _outcome(c, 'doi') == 'material_conflict'
              and c.registration_record_sha256
              and c.registration_record_sha256 != first.registration_record_sha256]
    if not titles:
        return result
    if record.search_policy_version != 'api-first-search-v2' or set(record.required_web_providers) != {'brave', 'exa'}:
        result['reason_code'] = 'current_required_route_policy_missing'
        return result
    # The strong label must not bypass missing routes or unresolved candidates.
    try:
        parsed = ReferenceDiscoveryTrace.model_validate(trace)
        completion = assess_reference_discovery_trace(parsed)
        same = (parsed.reference_id == record.reference_id and parsed.expected == record.expected
                and parsed.candidates == record.candidates and parsed.attempts == record.attempts
                and parsed.queries == record.queries and parsed.search_policy_version == record.search_policy_version
                and parsed.search_retention_policy == record.search_retention_policy
                and parsed.required_web_providers == record.required_web_providers
                and parsed.required_route_categories == record.required_route_categories)
    except (ValueError, TypeError):
        same, completion = False, None
    if not same or not completion.ready or completion.record.outcome != record.outcome or _search_is_incomplete(
            record.required_route_categories, record.attempts, record.queries, record.search_policy_version):
        result['reason_code'] = 'compound_discrepancy_search_incomplete'
        return result
    if any(not _bound_candidate(record, c) or any(
            _outcome(c, field) == 'unknown' for field in ('title', 'author')) for c in record.candidates):
        result['reason_code'] = 'compound_discrepancy_candidate_unresolved'
        return result
    second = sorted(titles, key=lambda c: c.candidate_id)[0]
    finding.update(finding_type='potentially_fabricated_reference',
        finding=FINDING_TEXT,
        evidence_explanation=(text + f' The submitted title is registered separately to {", ".join(second.observed.authors)} '
                              f'under DOI {second.observed.doi}.'),
        records=[_record_view(first), _record_view(second)],
        field_difference=dict(field_name='entry', submitted_value=reference.raw_ref),
        trace_sha256=_digest(trace))
    result.update(reason_code='corroborated_cross_record_identity_conflicts', trace_sha256=_digest(trace))
    return result


def assess_reference_credibility(reference, discovery=None, trace=None):
    """Separate review-worthy cumulative evidence from source-existence outcomes.

    Archive access is corroboration, not a prerequisite. Failed required searches
    remain incomplete; they do not count as completed negative observations.
    """
    if isinstance(trace, dict) and trace.get('credibility_policy_version') == 'reference-credibility-v4':
        from app.services.bounded_reference_review import assess
        try:
            return assess(reference, discovery, trace)
        except (ValueError, TypeError, KeyError, AttributeError):
            return dict(policy_version='reference-credibility-v4', reference_id=reference.reference_id,
                        status='not_assessed', reason_code='invalid_bounded_review_evidence',findings=[])
    result = _assess_compound_credibility(reference, discovery, trace)
    if any(f['finding_type'] in {'potentially_fabricated_reference', 'reference_identifier_placeholder'}
           for f in result['findings']):
        return result
    is_book = reference.source_kind in {'monograph', 'edited_collection'}
    if reference.source_kind not in {'journal_article', 'monograph', 'edited_collection'}:
        return result
    try:
        parsed = ReferenceDiscoveryTrace.model_validate(trace)
        if parsed.credibility_policy_version != POLICY or parsed.credibility_reference_sha256 != result['reference_sha256']:
            return result  # Never reinterpret historical no-match traces.
        completion = assess_reference_discovery_trace(parsed)
        if not completion.ready:
            return result
        record = completion.record
        if discovery is not None:
            supplied = ReferenceDiscoveryRecord.model_validate(discovery)
            if any(getattr(supplied, key) != getattr(record, key) for key in (
                'reference_id','expected','queries','attempts','candidates','outcome',
                'required_route_categories','required_web_providers','search_policy_version','search_retention_policy')):
                return result
    except (ValueError, TypeError):
        return result
    if not _input_matches(reference, record) or record.expected.source_kind != reference.source_kind:
        return result
    if _search_title_boundary_unresolved(reference):
        result['cumulative_reason_code'] = 'search_title_boundary_unresolved'
        return result
    if not is_book and is_edition_sensitive_reference(record.expected, reference.raw_ref):
        return result
    if (record.search_policy_version != 'api-first-search-v2'
            or set(parsed.required_web_providers) != {'brave','exa'}
            or not {'academic_adapter','bounded_web'} <= set(record.required_route_categories)):
        return result
    if _search_is_incomplete(record.required_route_categories, record.attempts, record.queries, record.search_policy_version):
        result['cumulative_reason_code'] = 'required_identity_search_incomplete'
        return result
    # No guess about missing fields, access failures or source completeness.
    # Every retained candidate must have independently inspectable identity data.
    metadata_adjudications = []
    for c in record.candidates:
        metadata = (_corroborated_metadata(record, c)
                    if parsed.metadata_identity_policy_version == METADATA_IDENTITY_POLICY else None)
        if metadata:
            metadata_adjudications.append(metadata)
            result['metadata_adjudications'] = metadata_adjudications
            if metadata['disposition'] == 'bibliographic_difference':
                result['cumulative_reason_code'] = 'metadata_correction_unresolved'
                return result
        if not _bound_candidate(record,c,fields=('title','author'),metadata_bound=bool(metadata)) or any(_outcome(c,f) == 'unknown' for f in ('title','author')):
            result['cumulative_reason_code'] = 'candidate_identity_unresolved'
            return result
        if all(_outcome(c,f) in {'agreement','minor_difference'} for f in ('title','author')):
            return result
        if _possible_title_extraction_damage(record.expected.title,c.observed.title):
            return result
    normalize = lambda s: re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
    title_query = normalize(f'title:{reference.title} author:{reference.author}')
    catalog_query = normalize(f'intitle:{reference.title} inauthor:{reference.author.split(",", 1)[0]}')
    doi_query = normalize(f'doi:{reference.doi}')
    queries = {q.query_id:q for q in record.queries}
    scholarly = {}
    missing_identifier = []
    completed_ids = []
    for attempt in record.attempts:
        if not attempt.permitted or not attempt.completed_at or attempt.outcome not in {'no_match','candidate_found','candidates_processed'}:
            continue
        for qid in attempt.query_ids:
            q = queries[qid]
            if q.execution_outcome not in {'results','no_results'}:
                continue
            if attempt.route_category != 'academic_adapter':
                continue
            if q.provider == 'crossref' and q.normalized_query == doi_query and q.execution_outcome == 'no_results' and q.result_count == 0 and q.reason_code == 'identifier_not_registered':
                missing_identifier.append(qid)
            catalog = is_book and q.provider == 'google_books' and q.normalized_query == catalog_query
            if not catalog and (q.provider not in {'crossref','openalex','semantic_scholar','core'} or q.normalized_query != title_query):
                continue
            # Empty means an explicitly observed empty result list, not a
            # relevance filter discarding candidates or a malformed response.
            empty = q.execution_outcome == 'no_results' and q.result_count == 0
            rejected = q.execution_outcome == 'results' and any(
                c.attempt_id == attempt.attempt_id for c in record.candidates)
            if empty or rejected:
                scholarly.setdefault(q.provider,[]).append(qid)
                completed_ids.append(attempt.attempt_id)
    identifier_findings = [f for f in result['findings'] if f['finding_type']=='reference_identifier_conflict']
    if len(scholarly) < 2 or (is_book and 'google_books' not in scholarly):
        result['cumulative_reason_code'] = 'insufficient_independent_identity_checks'
        return result
    # Require actual title-led web searches, not merely DOI lookups. Existing
    # completion verifies required APIs, transient dispositions and failures.
    web = set()
    for attempt in record.attempts:
        if attempt.route_category != 'bounded_web' or not attempt.permitted or not attempt.completed_at:
            continue
        for qid in attempt.query_ids:
            q = queries[qid]
            if (q.execution_outcome in {'results','no_results'} and q.execution_provider in {'brave','exa'}
                    and _normalize_text(reference.title) in _normalize_text(q.normalized_query)):
                web.add(q.execution_provider)
    if web != {'brave','exa'}:
        result['cumulative_reason_code'] = 'title_led_web_checks_missing'
        return result
    records = [r for f in identifier_findings for r in f.get('records',[])]
    identifier_text = ('The supplied DOI was not found in Crossref.' if missing_identifier and not records
                       else identifier_findings[0]['finding'] if identifier_findings else '')
    providers = ', '.join({'crossref':'Crossref','openalex':'OpenAlex','semantic_scholar':'Semantic Scholar','core':'CORE',
                          'google_books':'Google Books'}[p]
                          for p in sorted(scholarly))
    limitations = ['Book-catalog coverage is not exhaustive.' if is_book else 'The journal archive has not been verified.']
    finding = dict(finding_type='potentially_fabricated_reference', reference_id=reference.reference_id,
        policy_version=POLICY, evidence_basis='cumulative_identity_nonverification',
        finding=FINDING_TEXT,
        evidence_explanation=' '.join(part for part in (identifier_text, f'Title/author checks: {providers}. '
            'Title-led web searches: Brave and Exa.') if part),
        records=records, limitations=limitations,
        field_difference=dict(field_name='entry',submitted_value=reference.raw_ref), rectangles=[],localization_status='not_assessed',
        reference_sha256=result['reference_sha256'],trace_sha256=_digest(trace),
        search_evidence=dict(scholarly_query_ids=scholarly,identifier_query_ids=missing_identifier,
                             web_providers=sorted(web),attempt_ids=sorted(set(completed_ids))))
    if metadata_adjudications:
        finding['metadata_adjudication_policy_version'] = METADATA_IDENTITY_POLICY
        finding['search_evidence']['metadata_adjudications'] = metadata_adjudications
        used = {item['candidate_id'] for item in metadata_adjudications}
        used.update(peer for item in metadata_adjudications for peer in item['corroborating_candidate_ids'])
        already = {item['candidate_id'] for item in finding['records']}
        finding['records'].extend(_record_view(c) for c in record.candidates
                                  if c.candidate_id in used - already)
    result.update(status='finding',reason_code='cumulative_identity_nonverification',findings=[finding],
                  trace_sha256=_digest(trace))
    result.pop('cumulative_reason_code',None)
    return result
