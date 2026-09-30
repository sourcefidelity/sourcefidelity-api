"""Bounded APA journal author omissions; no search or source admission."""
import hashlib

from app.services.ref_field_extractor import extract_authorless_apa_journal
from app.services.reference_discovery import ReferenceDiscoveryRecord, _value_hash

AUTHOR_POLICY = 'apa7_verified_journal_author_required_v1'
RULE_SOURCE = 'https://apastyle.apa.org/style-grammar-guidelines/references/elements-list-entry'


def eligible_author_lookup(reference, policy_version):
    """Missing attribution must not prevent bibliographic field inspection."""
    parsed = extract_authorless_apa_journal(reference.raw_ref)
    return bool(policy_version == AUTHOR_POLICY and not reference.author
                and not reference.needs_review and parsed
                and (parsed.title, parsed.year, parsed.doi) ==
                    (reference.title, reference.year, reference.doi))


def discover_author_metadata(reference, sources, *, permitted=True):
    """One configured registration lookup, never acquisition or completed search."""
    import time
    import uuid
    from datetime import datetime, timezone
    from app.services.reference_discovery import (
        ExpectedBibliographicFields, ReferenceRouteAttempt, ReferenceSearchQuery,
        build_reference_discovery_candidate, derive_reference_discovery_record,
    )
    expected = ExpectedBibliographicFields(title=reference.title, year=reference.year,
        doi=reference.doi or '', source_kind=reference.source_kind)
    adapter = next((s for s in sources if s.name == 'crossref'), None)
    started = datetime.now(timezone.utc)
    clock_start = time.monotonic()
    attempt_id = str(uuid.uuid4())
    candidates = []
    calls = 0
    outcome, reason = 'unavailable', 'registration_adapter_unavailable'
    if not permitted:
        outcome = 'not_permitted'
        reason = 'author_metadata_budget_exhausted'
    elif adapter is not None:
        calls = 1
        try:
            result = adapter.search_by_title_author(reference.title, None)
            if result.success:
                candidates.append(build_reference_discovery_candidate(
                    attempt_id=attempt_id, provider='crossref', expected=expected, result=result))
                outcome, reason = 'candidate_found', 'author_registration_metadata_only'
            else:
                outcome, reason = 'operational_failure', 'registration_lookup_unresolved'
        except Exception:
            outcome, reason = 'operational_failure', 'registration_lookup_failed'
    elapsed = time.monotonic()-clock_start
    query = ReferenceSearchQuery(query_id=attempt_id, route_category='academic_adapter',
        provider='crossref', normalized_query=reference.title,
        query_sha256=hashlib.sha256(reference.title.encode()).hexdigest(),
        execution_outcome=('results' if candidates else 'operational_failure') if calls else 'budget_skipped' if not permitted else 'unknown',
        result_count=len(candidates), provider_calls=calls, latency_seconds=elapsed,
        reason_code=reason, required=True)
    attempt = ReferenceRouteAttempt(attempt_id=attempt_id, provider='crossref',
        query_ids=[query.query_id],
        route_category='academic_adapter', required=True, permitted=permitted,
        outcome=outcome, reason_code=reason, started_at=started,
        completed_at=datetime.now(timezone.utc))
    record = derive_reference_discovery_record(reference_id=reference.reference_id,
        expected=expected, required_route_categories=['academic_adapter', 'bounded_web'],
        queries=[query], attempts=[attempt], candidates=candidates)
    return {'reference_id':reference.reference_id, 'status':'unavailable',
        'reason_code':'bibliographic_metadata_only',
        'reference_discovery':record.model_dump(mode='json'),
        'author_metadata_lookup':{'policy_version':AUTHOR_POLICY,
            'provider':'crossref', 'provider_calls':calls,
            'latency_seconds':elapsed,
            'full_text_attempted':False, 'paid_provider_calls':0}}


def required_author_omissions(*, citation_format, references, discoveries, inventory,
                             policy_version=None):
    if citation_format.lower() != 'apa' or policy_version != AUTHOR_POLICY or not inventory:
        return []
    findings = []
    for entry in inventory.get('entries', []):
        ref = references.get(entry.get('reference_id'))
        if ref is None or ref.author or ref.needs_review:
            continue
        raw_hash = hashlib.sha256(ref.raw_ref.encode()).hexdigest()
        if entry.get('reference_text_sha256') != raw_hash:
            continue
        parsed = extract_authorless_apa_journal(ref.raw_ref)
        if not parsed or (parsed.title, parsed.year) != (ref.title, ref.year):
            continue
        try:
            record = ReferenceDiscoveryRecord.model_validate(discoveries.get(ref.reference_id))
        except ValueError:
            continue
        if (record.reference_id != ref.reference_id or record.outcome != 'confirmed'
                or record.expected.reference_parse_review or record.expected.authors
                or record.expected.title != ref.title or record.expected.year != ref.year
                or record.expected.doi != (ref.doi or '') or record.expected.isbn
                or record.expected.edition_sensitive):
            continue
        author_sets = {}
        for candidate in record.candidates:
            # Same bibliographic registration boundary as the accepted DOI rule.
            registered = candidate.provider == 'crossref' and any(
                a.attempt_id == candidate.attempt_id and a.provider == 'crossref'
                and a.route_category == 'academic_adapter' and a.permitted
                and a.outcome == 'candidate_found' and a.completed_at is not None
                for a in record.attempts)
            if (not registered or candidate.has_material_conflict or candidate.has_minor_difference
                    or candidate.edition_metadata or candidate.observed.edition_sensitive
                    or candidate.acquisition_outcome == 'identity_rejected'):
                continue
            comparisons = {c.field_name: c for c in candidate.comparisons}
            if any(c.expected_sha256 and c.outcome != 'agreement'
                   and not (c.field_name == 'source_kind' and c.outcome == 'unknown')
                   for c in candidate.comparisons):
                continue
            if any(k not in comparisons or comparisons[k].outcome != 'agreement'
                   or comparisons[k].expected_sha256 != _value_hash(v)
                   or comparisons[k].observed_sha256 != _value_hash(getattr(candidate.observed, k))
                   for k, v in [('title', ref.title), ('year', ref.year)]):
                continue
            authors = candidate.observed.authors
            author_comparison = comparisons.get('author')
            if (not authors or any(not a.strip() or a.casefold().strip() == 'anonymous' for a in authors)
                    or author_comparison is None or author_comparison.expected_sha256
                    or author_comparison.outcome != 'unknown'
                    or author_comparison.observed_sha256 != _value_hash(' | '.join(authors))):
                continue
            author_sets[tuple(a.casefold().strip() for a in authors)] = (authors, candidate)
        if len(author_sets) != 1:
            continue
        authors, candidate = next(iter(author_sets.values()))
        findings.append({
            'finding_type': 'required_author_missing', 'reference_id': ref.reference_id,
            'reference_text_sha256': raw_hash, 'rule_id': AUTHOR_POLICY, 'rule_source': RULE_SOURCE,
            'conflicting_fields': ['required author'], 'verified_authors': authors,
            'finding': 'This reference starts with the date but omits the article author. '
                       + 'The verified bibliographic record lists: ' + '; '.join(authors) + '.',
            'candidate_id': candidate.candidate_id, 'provider': candidate.provider,
            'verification_basis': 'crossref_registration_metadata',
            'bibliographic_record_sha256': hashlib.sha256(candidate.observed.model_dump_json().encode()).hexdigest(),
            'rectangles': entry.get('rectangles', []),
            'localization_status': 'bound_entry' if entry.get('rectangles') else 'not_assessed',
        })
    return findings
