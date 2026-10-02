"""Prospective, bounded human-review threshold, separate from source existence."""
import hashlib

from app.services.reference_review_scope import scope_for, input_hash, text_key
from app.services.source_type import (
    BOOK_CATALOGUE_KINDS,
    is_bibliographically_searchable,
)

POLICY = 'reference-credibility-v4'
SCREEN_RESOLUTION_POLICY = 'screened-metadata-work-binding-v1'


POSITIVE_ONLY_PROVIDERS = frozenset({'openaire', 'eric', 'doaj', 'europepmc'})
POSITIVE_ONLY_POLICY = 'positive-only-corroboration-v1'


def same_screened_work(candidate, observation):
    """Bind a screened lead to an already qualified different-work record.

    This does not qualify either record or initiate a lookup. Initials, missing
    names and fuzzy title matches are deliberately not expanded here.
    """
    from app.services.reference_discovery import _normalize_doi, _normalize_text
    old = candidate.observed
    left_doi = _normalize_doi(old.doi or '')
    right_doi = _normalize_doi(observation.get('doi') or '')
    if left_doi and right_doi and left_doi != right_doi:
        return False
    if (text_key(old.title) == text_key(observation.get('title'))
            and old.authors == observation.get('authors')):
        return True
    # A common surname or shared topic cannot bridge two records. This extra
    # mechanical equivalence requires the SAME observed DOI and name tokens.
    if not left_doi or left_doi != right_doi:
        return False
    def names(values):
        return sorted(tuple(sorted(_normalize_text(text_key(v)).split())) for v in values)
    other_authors = observation.get('authors') or []
    return bool(old.authors and other_authors
        and names(old.authors) == names(other_authors)
        and _normalize_text(text_key(old.title))
        == _normalize_text(text_key(observation.get('title'))))


def assess(reference, discovery, trace):
    from app.services.reference_discovery import (
        ReferenceDiscoveryTrace, assess_reference_discovery_trace, _normalize_doi,
        OBSERVED_TITLE_ABSENCE,
    )
    from app.services.reference_credibility import (
        _assess_compound_credibility, _input_matches, _search_title_boundary_unresolved,
        _bound_candidate, _corroborated_metadata, _outcome, _possible_title_extraction_damage,
        _digest, _record_view, FINDING_TEXT,
    )
    result = _assess_compound_credibility(reference, discovery, trace)
    result['policy_version'] = POLICY
    # Keep factual DOI/placeholder findings when the bounded review abstains.
    result['findings'] = [dict(f, policy_version=POLICY) for f in result['findings']
                          if f['finding_type'] != 'potentially_fabricated_reference']
    def stop(reason):
        result['bounded_review_reason'] = reason
        return result
    # An absent kind is a classifier gap, not an unsearchable work: the review
    # below turns on author/title/year coverage, which is kind-independent.
    # Only kinds the indexes genuinely cannot settle are excluded here.
    if not is_bibliographically_searchable(reference.source_kind):
        return stop('unsupported_source_kind')
    try:
        parsed = ReferenceDiscoveryTrace.model_validate(trace)
        completion = assess_reference_discovery_trace(parsed)
        if (parsed.credibility_policy_version != POLICY or not completion.ready
                or parsed.credibility_reference_sha256 != hashlib.sha256(reference.raw_ref.encode()).hexdigest()):
            return stop('input_or_trace_unbound')
        record = completion.record
        if not _input_matches(reference, record) or record.expected.source_kind != reference.source_kind:
            return stop('reference_not_searchable')
        if _search_title_boundary_unresolved(reference):
            return stop('search_title_boundary_unresolved')
        if (parsed.search_policy_version != 'api-first-search-v2'
                or set(parsed.required_web_providers) != {'brave','exa'}
                or not {'academic_adapter','bounded_web'} <= set(parsed.required_route_categories)):
            return stop('required_review_policy_missing')
        if discovery is not None:
            from app.services.reference_discovery import ReferenceDiscoveryRecord
            supplied = ReferenceDiscoveryRecord.model_validate(discovery)
            if any(getattr(supplied,k) != getattr(record,k) for k in (
                    'reference_id','expected','queries','attempts','candidates','outcome',
                    'search_policy_version','required_route_categories','required_web_providers','search_retention_policy')):
                return stop('discovery_trace_mismatch')
    except (ValueError, TypeError):
        return stop('invalid_trace')

    queries = {q.query_id:q for q in record.queries}
    scope_version = parsed.bounded_review_policy_version or 'bounded-reference-review-v1'
    scope = scope_for(scope_version)
    dispositions = []
    # Decide materiality before looking at whether a webpage happened to open.
    for candidate in record.candidates:
        title = candidate.observed.title
        metadata = _corroborated_metadata(record, candidate)
        bound = _bound_candidate(record, candidate, fields=('title','author'), metadata_bound=bool(metadata))
        title_out, author_out, doi_out = (_outcome(candidate,f) for f in ('title','author','doi'))
        title_scope = scope(reference.title, title)
        if (scope_version in {'bounded-reference-review-v2','bounded-reference-review-v3','bounded-reference-review-v4','bounded-reference-review-v5','bounded-reference-review-v6','bounded-reference-review-v7','bounded-reference-review-v8','bounded-reference-review-v9'} and not title
                and not candidate.observed.authors and not candidate.observed.doi
                and candidate.acquisition_outcome == 'not_attempted'
                and candidate.disposition_reason_code == 'bounded_location_limit'
                and candidate.location_rank is not None and candidate.location_rank > 5
                and (candidate.submitted_identifier_location_match is False
                     or not (record.expected.doi or record.expected.isbn))):
            decision = 'outside_bound'  # Uninformative tail lead, not a plausible match.
        elif bound and title_out == 'material_conflict' and author_out == 'material_conflict':
            decision = 'resolved_different'
        elif (candidate.submitted_identifier_location_match or doi_out == 'agreement'
                or (scope_version in {'bounded-reference-review-v5','bounded-reference-review-v6','bounded-reference-review-v7','bounded-reference-review-v8','bounded-reference-review-v9'} and _outcome(candidate,'isbn') == 'agreement')
                or title_out in {'agreement','minor_difference'}
                or _possible_title_extraction_damage(reference.title, title)):
            decision = 'material'
        elif bound and title_out == 'material_conflict' and title_scope == 'outside_bound':
            decision = 'resolved_different'  # Same author can write different works.
        elif scope_version in {'bounded-reference-review-v7', 'bounded-reference-review-v8','bounded-reference-review-v9'} and (
                candidate.acquisition_outcome == 'identity_rejected'
                or (not candidate.plausible_identity_match
                    and not candidate.access_restricted
                    and candidate.acquisition_outcome != 'access_restricted'
                    and candidate.title_absence_reason in OBSERVED_TITLE_ABSENCE
                    and not (
                        candidate.observed.title or candidate.observed.authors
                        or candidate.observed.doi or candidate.observed.isbn))):
            # Discovery already answered this, and the review was re-deriving
            # 'unknown' from the missing fields instead of reading the answer.
            # A lead whose identity was rejected is not a plausible match, and
            # neither is one that returned no title, author or identifier at
            # all. Leads that did observe a title keep the ordinary comparison
            # path below.
            #
            # A blocked lead is excluded from that reasoning. Failing to reach
            # a page is not the same as reading it and finding no work there,
            # and the distinction decides whether a student is told their
            # reference may be fabricated. The candidate now records which of
            # the two happened, and only an observed absence counts: an
            # unexplained empty record leaves the reference open, because an
            # extraction failure looks exactly like a page with no work on it.
            decision = 'outside_bound'
        else:
            decision = scope(reference.title, title, reference.author, candidate.observed.authors)
        if (scope_version in {'bounded-reference-review-v8', 'bounded-reference-review-v9'}
                and candidate.provider in POSITIVE_ONLY_PROVIDERS
                and decision == 'unknown'):
            # A positive-only source may confirm a work, never impugn one.
            #
            # Two measured hazards. A loose matcher answers a fabricated
            # reference with a near-title — OpenAIRE returns "Fair Competition:
            # The Law and Economics of Antitrust" for "Competition policy and
            # the limits of antitrust" — and an unresolved lead like that would
            # block the finding. A domain-limited source returns nothing for an
            # out-of-scope work: ERIC holds no law, DOAJ no paywalled articles,
            # and that silence is not absence.
            #
            # So only `unknown` is overridden. A genuine title or identifier
            # agreement still reaches the material branch above and still
            # protects the reference, and a resolved different work still
            # resolves. Their silence never reaches quorum either, because the
            # coverage check names its providers explicitly and these are not
            # among them.
            decision = 'outside_bound'
        dispositions.append(dict(candidate_id=candidate.candidate_id, disposition=decision,
                                 basis='qualified_metadata' if bound else 'discovery_scope_only'))
    result['bounded_review_candidates'] = dispositions
    if any(d['disposition'] in {'material','unknown'} for d in dispositions):
        return stop('material_candidate_unresolved')
    resolved_ids = {d['candidate_id'] for d in dispositions if d['disposition']=='resolved_different'}

    scholarly = {}; web = set(); ignored_failures = []; evidence_queries = []
    title_query = text_key(f'title:{reference.title} author:{reference.author}')
    catalog_query = text_key(f'intitle:{reference.title} inauthor:{reference.author.split(",",1)[0]}')
    for attempt in record.attempts:
        succeeded = attempt.permitted and attempt.completed_at and attempt.outcome in {
            'no_match','candidate_found','candidates_processed'}
        bound_queries = [queries[qid] for qid in attempt.query_ids]
        if attempt.route_category == 'library_metadata' and attempt.required and (
                not succeeded or not bound_queries or any(
                    q.execution_outcome not in {'results','no_results'}
                    or (q.execution_outcome == 'no_results' and q.result_count != 0)
                    for q in bound_queries)):
            return stop('required_library_route_incomplete')
        for audit in attempt.transient_search_audits:
            screen = audit.bounded_review_screen
            if screen is None:
                if audit.unresolved or audit.not_attempted or audit.identity_established:
                    return stop('transient_materiality_unavailable')
            elif not screen.resolved_for(reference.title, reference.author):
                return stop('transient_material_candidate_unresolved')
            elif screen.policy_version != scope_version:
                return stop('transient_scope_policy_mismatch')
        for q in bound_queries:
            success = succeeded and q.execution_outcome in {'results','no_results'}
            if attempt.route_category == 'bounded_web' and q.execution_provider in {'brave','exa'}:
                if q.execution_outcome == 'no_results' and q.result_count != 0:
                    return stop('web_empty_response_uncertified')
                if not success and q.required is not False:
                    # One provider being unavailable is not an incomplete
                    # search of the category. Measured 2026-09-21: an Exa
                    # outage failed 10 of 20 queries and removed six
                    # fabrication findings from paper 11, five of which Brave
                    # had already answered with a certified empty result. A
                    # university cannot stop marking because a vendor is
                    # having a bad hour. The failure is recorded and never
                    # counted as negative evidence; the category quorum below
                    # still requires that at least one title-led provider
                    # completed, so a total web-search failure still blocks
                    # the finding.
                    ignored_failures.append(q.query_id)
                    continue
                if (success and q.execution_outcome == 'results'
                        and not attempt.transient_search_audits
                        and not any(c.attempt_id == attempt.attempt_id for c in record.candidates)):
                    return stop('web_candidate_dispositions_missing')
                if success and text_key(reference.title) in text_key(q.normalized_query):
                    web.add(q.execution_provider); evidence_queries.append(q.query_id)
            if attempt.route_category != 'academic_adapter':
                continue
            provider = q.execution_provider or q.provider
            if provider not in {'crossref','openalex','semantic_scholar','core','google_books'}:
                continue
            if text_key(q.normalized_query) not in {title_query, catalog_query}:
                continue
            if not success:
                ignored_failures.append(q.query_id)
                continue  # Never a negative vote; quorum is checked below.
            screen = q.bounded_review_screen
            acceptable = q.execution_outcome == 'no_results' and q.result_count == 0
            if screen:
                if screen.policy_version != scope_version:
                    return stop('metadata_scope_policy_mismatch')
                if screen.input_sha256 != input_hash(reference.title, reference.author):
                    return stop('metadata_screen_input_mismatch')
                if screen.observations is None:
                    return stop('metadata_screen_observations_missing')
                if (q.result_count != screen.candidate_count or
                        (not screen.candidate_count and q.execution_outcome != 'no_results')):
                    return stop('metadata_screen_count_unbound')
                for observation in screen.observations:
                    if (record.expected.doi and observation.get('doi') and
                            _normalize_doi(record.expected.doi) == _normalize_doi(observation['doi'])):
                        return stop('material_metadata_identifier_match')
                    decision = scope(reference.title, observation.get('title'), reference.author,
                                     observation.get('authors') or [])
                    if decision != observation.get('disposition'):
                        return stop('metadata_screen_binding_invalid')
                    if decision != 'outside_bound' and not any(
                            c.candidate_id in resolved_ids
                            and (same_screened_work(c, observation)
                                 if parsed.metadata_screen_resolution_policy_version == SCREEN_RESOLUTION_POLICY
                                 else (text_key(c.observed.title) == text_key(observation.get('title'))
                                       and c.observed.authors == observation.get('authors')))
                            for c in record.candidates):
                        return stop('material_metadata_candidate_unresolved')
                acceptable = True
            elif q.execution_outcome == 'results':
                # Existing catalog route retains every bounded volume record.
                acceptable = provider == 'google_books' and any(
                    c.attempt_id == attempt.attempt_id and c.candidate_id in resolved_ids
                    for c in record.candidates)
                if provider == 'google_books' and scope_version in {'bounded-reference-review-v5','bounded-reference-review-v6','bounded-reference-review-v7','bounded-reference-review-v8','bounded-reference-review-v9'}:
                    catalog = [c for c in record.candidates if c.attempt_id == attempt.attempt_id]
                    reviewed_ids = {d['candidate_id'] for d in dispositions
                                    if d['disposition'] in {'resolved_different', 'outside_bound'}}
                    # A completed catalog search can return unrelated books.
                    # Count coverage, not independently adjudicated identities;
                    # missing records/counts cannot become successful emptiness.
                    acceptable = (bool(catalog) and q.result_count == len(catalog)
                        and all(c.provider == 'google_books' and c.edition_metadata is not None
                                and c.candidate_id in reviewed_ids for c in catalog))
            if acceptable:
                scholarly.setdefault(provider, []).append(q.query_id)
                evidence_queries.append(q.query_id)
    if len(scholarly) < 2:
        return stop('metadata_coverage_quorum_incomplete')
    if reference.source_kind in BOOK_CATALOGUE_KINDS and 'google_books' not in scholarly:
        return stop('book_catalog_coverage_missing')
    if not web:
        return stop('title_led_web_coverage_missing')
    if 'library_metadata' in record.required_route_categories and not any(
            a.required and a.route_category=='library_metadata' for a in record.attempts):
        return stop('required_library_route_missing')
    finding = dict(finding_type='potentially_fabricated_reference', reference_id=reference.reference_id,
        policy_version=POLICY, evidence_basis='bounded_identity_nonverification', finding=FINDING_TEXT,
        evidence_explanation='Title/author searches: ' + ', '.join(sorted(scholarly)) +
            '. Title-led web searches: ' + ', '.join(sorted(web)) +
            '. No credible matching work was established within these search bounds.',
        records=[_record_view(c) for c in record.candidates if c.candidate_id in resolved_ids],
        limitations=['Search coverage is bounded, not an exhaustive catalog of published works.'] +
            (['Additional metadata lookups failed; they were not counted as negative evidence.'] if ignored_failures else []) +
            ([f"Only one title-led web provider completed this search ({sorted(web)[0]}); "
              "the other was unavailable, so web coverage was narrower than usual."]
             if len(web) < 2 else []),
        field_difference=dict(field_name='entry',submitted_value=reference.raw_ref),
        rectangles=[],localization_status='not_assessed',reference_sha256=result['reference_sha256'],
        trace_sha256=_digest(trace),search_evidence=dict(query_ids=evidence_queries,
            scholarly_query_ids=scholarly,web_providers=sorted(web),candidate_dispositions=dispositions,
            uncounted_failure_query_ids=ignored_failures))
    result.update(status='finding',reason_code='bounded_identity_nonverification',findings=[finding])
    return result
