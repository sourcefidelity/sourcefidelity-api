"""Prospective metadata-only identity adjudication, not source admission."""
from copy import deepcopy
import hashlib

import pytest

from app.services.metadata_identity import POLICY, observation_is_bound
from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryTrace,
    assess_reference_discovery_trace, build_reference_discovery_candidate,
)
from app.services.reference_credibility import assess_reference_credibility
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.core import CoreRetriever
from app.services.retrieval.crossref import CrossrefRetriever
from test_reference_credibility_cumulative import fixture


def candidate(provider, expected, *, title='Volcanic sediment dynamics', author='Stone',
              year=2002, doi='10.1234/other', acquisition='metadata_only'):
    if provider == 'openalex':
        raw = dict(id='https://openalex.org/W123', title=title, publication_year=year,
                   doi='https://doi.org/' + doi if doi else None,
                   authorships=[dict(author=dict(display_name=author, id='not-retained'))],
                   abstract_inverted_index={'not-retained': [0]}, cited_by_count=99)
        result = OpenAlexRetriever.__new__(OpenAlexRetriever)._parse_work(raw)
    elif provider == 'core':
        raw = dict(id=123, title=title, yearPublished=year, doi=doi,
                   authors=[dict(name=author, id='not-retained')], abstract='not-retained')
        result = CoreRetriever.__new__(CoreRetriever)._parse_output(raw)
    else:
        raw = dict(DOI=doi, title=[title], author=[dict(family=author)],
                   issued={'date-parts': [[year]]})
        result = CrossrefRetriever()._parse_message(raw)
    return build_reference_discovery_candidate(attempt_id=provider, provider=provider,
        expected=ExpectedBibliographicFields(**expected), result=result,
        acquisition_outcome=acquisition)


def scenario(**kwargs):
    ref, trace = fixture()
    trace['metadata_identity_policy_version'] = POLICY
    rows = [candidate(p, trace['expected'], **kwargs) for p in ('crossref', 'openalex')]
    trace['candidates'] = [c.model_dump(mode='json') for c in rows]
    for i in (0, 1):
        trace['attempts'][i]['outcome'] = 'candidate_found'
        trace['queries'][i].update(execution_outcome='results', result_count=1)
    return ref, trace


@pytest.mark.parametrize('provider', ['openalex', 'core'])
def test_receipt_replays_existing_parser_and_excludes_incidental_payload(provider):
    _, trace = fixture()
    c = candidate(provider, trace['expected'])
    assert observation_is_bound(c)
    text = c.metadata_identity.model_dump_json()
    assert 'not-retained' not in text and 'abstract' not in text
    c.observed.title = 'Changed observation'
    assert not observation_is_bound(c)


@pytest.mark.parametrize('acquisition', ['metadata_only', 'transport_failure', 'access_restricted', 'completeness_rejected'])
def test_two_bound_records_can_adjudicate_without_source_download(acquisition):
    ref, trace = scenario(acquisition=acquisition)
    before = deepcopy(trace)
    assessment = assess_reference_credibility(ref, None, trace)
    assert [f['finding_type'] for f in assessment['findings']] == ['potentially_fabricated_reference']
    audit = assessment['findings'][0]['search_evidence']['metadata_adjudications'][0]
    assert audit['disposition'] == 'different_work'
    assert audit['corroborating_candidate_ids'] == [trace['candidates'][0]['candidate_id']]
    assert trace == before  # Neither acquisition outcomes nor source admission change.
    if acquisition != 'metadata_only':
        assert assess_reference_discovery_trace(trace).record.outcome == 'search_incomplete'


def test_two_distinct_metadata_services_work_without_student_or_candidate_doi():
    ref, trace = scenario(doi='')
    ref.doi = ''; ref.raw_ref = ref.raw_ref.split(' https://doi.org/')[0]
    trace['expected']['doi'] = ''
    trace['credibility_reference_sha256'] = hashlib.sha256(ref.raw_ref.encode()).hexdigest()
    trace['candidates'] = [candidate(p, trace['expected'], doi='').model_dump(mode='json')
                           for p in ('core', 'openalex')]
    trace['attempts'][0].update(attempt_id='core', provider='core', query_ids=['core'])
    trace['queries'] = trace['queries'][:4]
    trace['queries'][0].update(query_id='core', provider='core', execution_provider='core')
    assert assess_reference_credibility(ref, None, trace)['findings']


@pytest.mark.parametrize('mutation', ['no_policy', 'no_receipt', 'changed_observation',
    'changed_native', 'wrong_hash', 'wrong_id', 'missing_author', 'missing_year',
    'conflicting_doi', 'conflicting_year', 'different_title', 'different_author',
    'not_attempted', 'failed_route', 'filtered_route', 'one_provider'])
def test_unqualified_and_incomplete_observations_never_authorize_flag(mutation):
    ref, trace = scenario()
    c = trace['candidates'][1]
    if mutation == 'no_policy': trace.pop('metadata_identity_policy_version')
    elif mutation == 'no_receipt': c.pop('metadata_identity')
    elif mutation == 'changed_observation': c['observed']['title'] = 'Tampered title'
    elif mutation == 'changed_native': c['metadata_identity']['native_fields']['title'] = 'Tampered title'
    elif mutation == 'wrong_hash': c['metadata_identity']['observation_sha256'] = '0' * 64
    elif mutation == 'wrong_id': c['metadata_identity']['record_id'] = 'https://openalex.org/W999'
    elif mutation == 'not_attempted': c['acquisition_outcome'] = 'not_attempted'
    elif mutation == 'failed_route': trace['queries'][1]['execution_outcome'] = 'timeout'
    elif mutation == 'filtered_route': trace['queries'][1]['reason_code'] = 'metadata_candidates_filtered'
    elif mutation == 'one_provider':
        trace['candidates'] = [c]; trace['attempts'][0]['outcome'] = 'no_match'
        trace['queries'][0].update(execution_outcome='no_results', result_count=0)
    else:
        changes = {'missing_author': dict(author=''), 'missing_year': dict(year=None),
                   'conflicting_doi': dict(doi='10.1234/different'), 'conflicting_year': dict(year=2003),
                   'different_title': dict(title='Volcanic sediment patterns'),
                   'different_author': dict(author='Someone Else')}
        trace['candidates'][1] = candidate('openalex', trace['expected'], **changes[mutation]).model_dump(mode='json')
    assert not assess_reference_credibility(ref, None, trace)['findings']


def test_same_provider_duplicate_cannot_corroborate_itself():
    ref, trace = scenario()
    c = trace['candidates'][1]
    trace['candidates'][0] = deepcopy(c)
    trace['candidates'][0].update(candidate_id='duplicate', attempt_id='crossref')
    trace['attempts'][0]['provider'] = 'openalex'
    trace['queries'][0]['provider'] = 'openalex'
    trace['queries'][-1]['provider'] = 'openalex'
    assert not assess_reference_credibility(ref, None, trace)['findings']


def test_genuine_work_and_edition_compatibility_prevent_flag():
    ref, trace = fixture()
    ref, trace = scenario(title=ref.title, author=ref.author)
    assert not assess_reference_credibility(ref, None, trace)['findings']
    ref.source_kind = 'monograph'; trace['expected']['source_kind'] = 'monograph'
    # Even a conflicting date on an otherwise compatible book is not fabrication.
    trace['candidates'] = [candidate(p, trace['expected'], title=ref.title,
        author=ref.author, year=2003).model_dump(mode='json') for p in ('crossref', 'openalex')]
    assert not assess_reference_credibility(ref, None, trace)['findings']


def test_old_trace_serialization_does_not_gain_observations_or_policy():
    _, trace = fixture()
    saved = ReferenceDiscoveryTrace.model_validate(trace).model_dump(mode='json')
    assert 'metadata_identity_policy_version' not in saved
    _, prospective = scenario()
    prospective['candidates'][1].pop('metadata_identity')
    parsed = ReferenceDiscoveryTrace.model_validate(prospective)
    assert 'metadata_identity' not in parsed.candidates[1].model_dump(mode='json')


def test_conflicting_third_record_defeats_two_record_agreement():
    ref, trace = scenario()
    c = candidate('core', trace['expected'], year=2003)
    trace['candidates'].append(c.model_dump(mode='json'))
    query = deepcopy(trace['queries'][1])
    query.update(query_id='core', provider='core', execution_provider='core')
    attempt = deepcopy(trace['attempts'][1])
    attempt.update(attempt_id='core', provider='core', query_ids=['core'])
    trace['queries'].append(query); trace['attempts'].append(attempt)
    assert not assess_reference_credibility(ref, None, trace)['findings']


@pytest.mark.parametrize('mutation', ['budget', 'invalid_record_id', 'missing_record_id', 'overwritten_result'])
def test_raw_metadata_cannot_certify_changed_or_unattempted_candidate(mutation):
    _, trace = fixture()
    c = candidate('openalex', trace['expected'])
    raw = deepcopy(c.metadata_identity.native_fields)
    if mutation == 'invalid_record_id': raw['id'] = 'https://example.org/W123'
    if mutation == 'missing_record_id': raw.pop('id')
    result = OpenAlexRetriever.__new__(OpenAlexRetriever)._parse_work(raw)
    if mutation == 'budget':
        result.success = False; result.metadata['candidate_budget_skipped'] = True
    if mutation == 'overwritten_result': result.title = 'Student-supplied replacement'
    built = build_reference_discovery_candidate(attempt_id='openalex', provider='openalex',
        expected=ExpectedBibliographicFields(**trace['expected']), result=result)
    assert built.metadata_identity is None


def test_finding_contains_inspectable_metadata_and_corroboration_links():
    ref, trace = scenario()
    finding = assess_reference_credibility(ref, None, trace)['findings'][0]
    assert {r['provider'] for r in finding['records']} == {'crossref', 'openalex'}
    assert all(r['record_sha256'] and r['observed']['title'] and r['observed']['authors']
               for r in finding['records'])


@pytest.mark.parametrize('authors', [[{}], [None], {'author': 'not a list'}])
def test_malformed_nested_receipt_abstains_without_crashing(authors):
    from app.services.metadata_identity import digest
    ref, trace = scenario()
    receipt = trace['candidates'][1]['metadata_identity']
    receipt['native_fields']['authorships'] = authors
    receipt['record_sha256'] = digest(receipt['native_fields'])
    assert not assess_reference_credibility(ref, None, trace)['findings']


@pytest.mark.parametrize('matching_field', ['title', 'author'])
def test_single_field_difference_is_not_promoted_to_fabrication(matching_field):
    ref, _ = fixture()
    changes = {'title': ref.title} if matching_field == 'title' else {'author': ref.author}
    ref, trace = scenario(**changes)
    result = assess_reference_credibility(ref, None, trace)
    assert not result['findings']
    if matching_field == 'author':
        assert result['cumulative_reason_code'] == 'metadata_correction_unresolved'
    # Exact title agreement already triggers the earlier correction safeguard.


def test_optional_failed_metadata_query_cannot_be_corroboration():
    from app.services.reference_credibility import _corroborated_metadata
    ref, trace = scenario()
    trace['attempts'][1]['required'] = False
    trace['queries'][1]['execution_outcome'] = 'timeout'
    record = assess_reference_discovery_trace(trace).record
    assert _corroborated_metadata(record, record.candidates[1]) is None
