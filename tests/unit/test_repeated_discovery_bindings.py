"""Repeated metadata observations retain each successful lookup's provenance."""
from copy import deepcopy

from app.services.reference_discovery import ExpectedBibliographicFields, assess_reference_discovery_trace
from app.services.retrieval.base import AcquisitionLocation, RetrievalResult
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE


def test_repeated_provider_location_binds_both_attempts_without_extra_acquisition():
    trace = {'reference_id': 'ref', 'expected': ExpectedBibliographicFields(
        title='A bounded study', authors=['Writer, W.'], year='2020'),
        'required': {'academic_adapter'}, 'queries': [], 'attempts': [],
        'candidates': [], 'limitations': []}
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    url = 'https://publisher.example/article'
    try:
        result = RetrievalResult(source_name='crossref', success=True,
            title='A bounded study', authors=['Writer, W.'], year='2020',
            doi='10.1234/example', locations=[
                AcquisitionLocation(url=url, provider='crossref'),
                AcquisitionLocation(url=url, provider='crossref')])
        for _ in range(2):
            SourceResolver._record_discovery_attempt(category='academic_adapter',
                provider='crossref', result=result, required=True)
        assert len(trace['attempts']) == len(trace['candidates']) == 2
        assert {c.attempt_id for c in trace['candidates']} == {a.attempt_id for a in trace['attempts']}
        assert len({c.location_sha256 for c in trace['candidates']}) == 1
        assert len({c.candidate_id for c in trace['candidates']}) == 2
        SourceResolver._record_canonical_location_outcomes(RetrievalResult(
            source_name='canonical', success=False, metadata={'location_attempts':[
                {'url':url, 'outcome':'access_restricted', 'reason_code':'access_restricted'}]}))
        assert all(c.acquisition_outcome == 'access_restricted' for c in trace['candidates'])
        assert all(not c.validated_identity_content_sha256 for c in trace['candidates'])
        payload, _ = SourceResolver._discovery_artifacts()
        before = deepcopy(payload)
        completion = assess_reference_discovery_trace(payload)
        assert 'candidate_binding_incomplete' not in completion.blocker_codes
        assert payload == before
        # Removing one attempt's observation must still fail closed.
        payload['candidates'] = payload['candidates'][:1]
        assert 'candidate_binding_incomplete' in assess_reference_discovery_trace(payload).blocker_codes
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
