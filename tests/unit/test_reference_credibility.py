"""Synthetic identities only; no student/source content in public fixtures."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib

import pytest

from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryCandidate, ReferenceDiscoveryTrace,
    ReferenceRouteAttempt, ReferenceSearchQuery, assess_reference_discovery_trace,
    build_reference_discovery_candidate,
)
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.schemas import ParsedReference


def case(*, alternative=True, completed=True):
    ref = ParsedReference(reference_id='ref-test', author='Alvarez, A.', year='2020',
        title='Archival Methods in Coastal Communities', doi='10.1234/coastal',
        source_kind='journal_article', raw_ref='Alvarez, A. (2020). Archival Methods in Coastal Communities. doi:10.1234/coastal')
    expected = ExpectedBibliographicFields(title=ref.title, authors=[ref.author], year=ref.year,
        doi=ref.doi, source_kind=ref.source_kind)
    trace = ReferenceDiscoveryTrace(reference_id=ref.reference_id, expected=expected,
        search_policy_version='api-first-search-v2', required_web_providers=['brave','exa'],
        required_route_categories=['academic_adapter', 'bounded_web'])
    messages = [dict(DOI=ref.doi, title=['Particle Measurements in Stellar Plasmas'],
        author=[dict(given='Bea',family='Bishop')], issued={'date-parts':[[2020]]})]
    if alternative:
        messages.append(dict(DOI='10.1234/archives', title=[ref.title],
            author=[dict(given='Cora', family='Chan')], issued={'date-parts':[[2020]]}))
    now = datetime.now(timezone.utc)
    for i,message in enumerate(messages):
        attempt_id = f'metadata-{i}'
        query = message['DOI'] if i == 0 else ref.title
        trace.queries.append(ReferenceSearchQuery(query_id=attempt_id, route_category='academic_adapter',
            provider='crossref', normalized_query=query, query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            execution_outcome='results', provider_calls=1, result_count=1))
        trace.attempts.append(ReferenceRouteAttempt(attempt_id=attempt_id, route_category='academic_adapter',
            provider='crossref', required=True, permitted=True, query_ids=[attempt_id],
            outcome='candidate_found', started_at=now, completed_at=now))
        trace.candidates.append(build_reference_discovery_candidate(attempt_id=attempt_id,
            provider='crossref', expected=expected, result=CrossrefRetriever()._parse_message(message)))
    if completed:
        for provider in ('brave', 'exa'):
            trace.queries.append(ReferenceSearchQuery(query_id=provider, provider=provider,
                execution_provider=provider, route_category='bounded_web', normalized_query=ref.title,
                query_sha256=hashlib.sha256(ref.title.encode()).hexdigest(), required=True,
                execution_outcome='no_results', provider_calls=1, result_count=0))
            trace.attempts.append(ReferenceRouteAttempt(attempt_id=provider, provider=provider,
                route_category='bounded_web', required=True, permitted=True, query_ids=[provider],
                outcome='no_match', started_at=now, completed_at=now))
    record = assess_reference_discovery_trace(trace).record
    assert record is not None
    return ref, record.model_dump(mode='json'), trace.model_dump(mode='json')


def kinds(result):
    return [f['finding_type'] for f in result['findings']]


def test_corroborated_different_records_completed_alternatives_yield_one_flag():
    ref, record, trace = case()
    before = deepcopy((record, trace))
    result = assess_reference_credibility(ref, record, trace)
    assert kinds(result) == ['potentially_fabricated_reference']
    assert len(result['findings'][0]['records']) == 2
    assert result['findings'][0]['field_difference']['submitted_value'] == ref.raw_ref
    assert (record, trace) == before


@pytest.mark.parametrize('alternative,completed', [(False,True), (True,False), (False,False)])
def test_wrong_doi_alone_or_incomplete_search_is_only_identifier_error(alternative, completed):
    ref, record, trace = case(alternative=alternative, completed=completed)
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


@pytest.mark.parametrize('failure', ['timeout','rate_limited','captcha','budget_skipped','cooldown_skipped'])
def test_required_provider_failure_never_becomes_fabrication(failure):
    ref, record, trace = case()
    trace['queries'][-1]['execution_outcome'] = failure
    record['queries'][-1]['execution_outcome'] = failure
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


@pytest.mark.parametrize('field', ['title','author','year','doi','reference_id'])
def test_changed_submitted_fields_reject_old_findings(field):
    ref, record, trace = case()
    ref = ref.model_copy(update={field:'changed'})
    assert not assess_reference_credibility(ref, record, trace)['findings']


@pytest.mark.parametrize('mutation', ['snippet','field_hash','observation_hash','outcome','failed_attempt','review'])
def test_stale_untrusted_or_incomplete_premises_cannot_supply_flags(mutation):
    ref, record, trace = case()
    if mutation == 'review':
        ref.needs_review = True
    elif mutation == 'failed_attempt':
        record['attempts'][0]['outcome'] = 'operational_failure'
    else:
        for c in record['candidates']:
            if mutation == 'snippet':
                c.pop('registration_record_sha256'); c.pop('registration_observation_sha256')
            elif mutation == 'observation_hash':
                c['observed']['title'] = 'Changed record'
            elif mutation == 'field_hash':
                next(x for x in c['comparisons'] if x['field_name']=='title')['expected_sha256'] = 'f'*64
            elif mutation == 'outcome':
                next(x for x in c['comparisons'] if x['field_name']=='author')['outcome'] = 'agreement'
    assert not assess_reference_credibility(ref, record, trace)['findings']


def test_mirrored_observation_is_not_independent_corroboration():
    ref, record, trace = case()
    record['candidates'][1]['registration_record_sha256'] = record['candidates'][0]['registration_record_sha256']
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


def test_edition_and_parse_uncertainty_are_not_fabrication():
    ref, record, trace = case()
    record['expected']['edition_sensitive'] = True
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']
    ref.needs_review = True
    assert not assess_reference_credibility(ref, record, trace)['findings']


def test_complete_no_match_and_no_doi_sources_are_not_fabrication():
    ref, record, trace = case()
    for payload in (record,trace):
        payload['candidates'] = []
        for a in payload['attempts']:
            a['outcome'] = 'no_match'
    record['outcome'] = 'unlocated_after_search'
    assert not assess_reference_credibility(ref, record, trace)['findings']
    ref.doi = ''
    assert not assess_reference_credibility(ref, None, None)['findings']


def test_old_candidate_serialization_does_not_gain_new_hash_fields():
    _, record, _ = case()
    old = record['candidates'][0]
    old.pop('registration_record_sha256'); old.pop('registration_observation_sha256')
    assert ReferenceDiscoveryCandidate.model_validate(old).model_dump(mode='json') == old


def test_title_author_correction_suppresses_suspected_fabrication():
    ref, record, trace = case()
    raw = dict(DOI='10.1234/correction', title=[ref.title],
        author=[dict(given='A.',family='Alvarez')], issued={'date-parts':[[2020]]})
    c = build_reference_discovery_candidate(attempt_id='metadata-1',provider='crossref',
        expected=ExpectedBibliographicFields.model_validate(record['expected']),
        result=CrossrefRetriever()._parse_message(raw)).model_dump(mode='json')
    record['candidates'][1] = c
    trace['candidates'][1] = c
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


def test_disagreeing_qualified_records_for_same_identifier_abstain():
    ref, record, trace = case()
    raw = dict(DOI=ref.doi, title=[ref.title],
        author=[dict(given='A.', family='Alvarez')], issued={'date-parts': [[2020]]})
    record['candidates'][1] = build_reference_discovery_candidate(attempt_id='metadata-1', provider='crossref',
        expected=ExpectedBibliographicFields.model_validate(record['expected']),
        result=CrossrefRetriever()._parse_message(raw)).model_dump(mode='json')
    result = assess_reference_credibility(ref, record, trace)
    assert not result['findings']
    assert result['reason_code'] == 'conflicting_qualified_identifier_observations'


@pytest.mark.parametrize('index', [0, 1])
def test_independent_content_is_not_a_second_registered_work(index):
    ref, record, trace = case()
    for payload in (record, trace):
        c = payload['candidates'][index]
        c['registration_record_sha256'] = None
        c['registration_observation_sha256'] = None
        c['location_provenance'] = 'independently_acquired_content'
        c['validated_identity_content_sha256'] = 'a'*64
        c['location_sha256'] = 'b'*64
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


@pytest.mark.parametrize('wording', ['To appear', 'forthcoming', 'Anonymous', 'John Smith', 'n.d.', 'Firstname Lastname'])
def test_plausible_names_titles_and_publication_status_are_not_signals(wording):
    ref, _, _ = case()
    ref.raw_ref += ' '+wording
    assert not assess_reference_credibility(ref)['findings']


def test_explicit_identifier_placeholder_is_formatting_not_fabrication():
    ref, _, _ = case()
    ref.raw_ref += ' arXiv:2305.XXXX'
    result = assess_reference_credibility(ref)
    assert kinds(result) == ['reference_identifier_placeholder']
    f = result['findings'][0]
    span=f['original_span']
    assert ref.raw_ref[span['start']:span['end']] == f['field_difference']['submitted_value']


def test_unknown_candidate_identity_keeps_only_factual_identifier_error():
    ref, record, trace = case()
    for payload in (record,trace):
        payload['candidates'][1]['acquisition_outcome'] = 'not_attempted'
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


def test_legacy_route_policy_does_not_authorize_stronger_flag():
    ref, record, trace = case()
    record['search_policy_version'] = 'configured-search-v1'
    assert kinds(assess_reference_credibility(ref, record, trace)) == ['reference_identifier_conflict']


def test_false_candidate_outcomes_cannot_turn_matching_metadata_into_conflict():
    ref, record, trace = case()
    c = record['candidates'][0]
    c['comparisons'] = [{**x,'outcome':'material_conflict'} if x['field_name']=='doi' else x for x in c['comparisons']]
    assert not assess_reference_credibility(ref, record, trace)['findings']


def test_joined_words_and_journal_spill_cannot_make_matching_title_a_wrong_doi():
    ref, record, trace = case()
    ref.title='ArchivalMethodsinCoastalCommunities.JOURNALREVIEW'
    ref.raw_ref=f'{ref.author} ({ref.year}). {ref.title}. doi:{ref.doi}'
    record['expected']['title']=ref.title
    candidate=build_reference_discovery_candidate(attempt_id='metadata-0',provider='crossref',
        expected=ExpectedBibliographicFields.model_validate(record['expected']),
        result=CrossrefRetriever()._parse_message(dict(DOI=ref.doi,title=['Archival Methods in Coastal Communities'],
            author=[dict(given='Anna',family='Al Varez')],issued={'date-parts':[[2020]]})))
    record['candidates']=[candidate.model_dump(mode='json')]
    assert not assess_reference_credibility(ref, record, trace)['findings']


def test_malformed_registration_receipt_cannot_break_discovery_or_certify_candidate():
    from app.services.retrieval.base import RetrievalResult
    ref, record, _ = case()
    candidate = build_reference_discovery_candidate(attempt_id='metadata-0', provider='crossref',
        expected=ExpectedBibliographicFields.model_validate(record['expected']),
        result=RetrievalResult(source_name='crossref', success=True, title='Different work',
            authors=['Other author'], doi=ref.doi, metadata={'message': {'author': [None]}}))
    assert candidate.registration_record_sha256 is None
    assert candidate.registration_observation_sha256 is None
