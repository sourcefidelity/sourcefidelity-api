"""Same-author comparisons reuse qualified records, not name/topic guesses."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services.bounded_reference_review import same_screened_work, SCREEN_RESOLUTION_POLICY
from app.services.reference_discovery import ReferenceDiscoveryTrace
from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_review_scope import screen_metadata
from app.services.retrieval.base import RetrievalResult
from test_bounded_reference_review import review_fixture, kinds


def candidate():
    return SimpleNamespace(observed=SimpleNamespace(title='Volcanic sediment dynamics',
        authors=['Andrew River'],doi='10.1234/other'))


def test_same_doi_full_name_order_and_case_are_mechanical():
    assert same_screened_work(candidate(),dict(title='VOLCANIC sediment dynamics',
        authors=['River, Andrew'],doi='https://doi.org/10.1234/other'))


@pytest.mark.parametrize('change',[
    {'doi':'10.1234/different'}, {'doi':''}, {'authors':['River, A.']},
    {'authors':[]}, {'authors':['Alex River']}, {'title':'Volcanic sediment chemistry'},
])
def test_missing_conflicting_or_fuzzy_binding_does_not_resolve(change):
    observation=dict(title='Volcanic sediment dynamics',authors=['River, Andrew'],doi='10.1234/other')
    observation.update(change)
    assert not same_screened_work(candidate(),observation)


def test_exact_names_do_not_override_conflicting_doi():
    assert not same_screened_work(candidate(),dict(title='Volcanic sediment dynamics',
        authors=['Andrew River'],doi='10.1234/different'))


def test_prospective_trace_reuses_already_qualified_record_without_rewriting_history():
    from app.services.retrieval.crossref import CrossrefRetriever
    from app.services.reference_discovery import build_reference_discovery_candidate, ExpectedBibliographicFields
    ref,trace=review_fixture()
    # This comparison specifically exercises the historical author-only screen;
    # scope v5 excludes clearly different titles earlier, without adjudication.
    trace['bounded_review_policy_version']='bounded-reference-review-v4'
    result=CrossrefRetriever()._parse_message(dict(DOI='10.1234/other',title=['Volcanic sediment dynamics'],
        author=[dict(family='River',given='A.')],issued={'date-parts':[[2002]]}))
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),result=result)
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][0]['outcome']='candidate_found'
    trace['queries'][0].update(execution_outcome='results',result_count=1,
        bounded_review_screen=screen_metadata(ref.title,ref.author,[result]))
    trace['queries'][1].update(execution_outcome='results',result_count=1,
        bounded_review_screen=screen_metadata(ref.title,ref.author,[RetrievalResult(
            source_name='openalex',success=True,title=result.title,authors=['River, A.'],doi=result.doi)]))
    from app.services.reference_review_scope import scope_for
    for q in trace['queries'][:2]:
        screen=q['bounded_review_screen'];screen['policy_version']='bounded-reference-review-v4'
        for o in screen['observations']:
            o['disposition']=scope_for('bounded-reference-review-v4')(ref.title,o['title'],ref.author,o['authors'])
        screen.update(outside_bound=0,material=1,unknown=0)
    before=deepcopy(trace)
    assert not assess_reference_credibility(ref,None,trace)['findings']
    assert 'metadata_screen_resolution_policy_version' not in ReferenceDiscoveryTrace(**trace).model_dump()
    fresh=deepcopy(trace);fresh['metadata_screen_resolution_policy_version']=SCREEN_RESOLUTION_POLICY
    assert kinds(assess_reference_credibility(ref,None,fresh))==['potentially_fabricated_reference']
    assert trace==before
    fresh['candidates'][0]['registration_observation_sha256']='0'*64
    assert not assess_reference_credibility(ref,None,fresh)['findings']
