"""Multiple references share provider counters, not independent fresh allowances."""
from test_api_first_search_policy import cascade
from test_bibliography_identity import resolver, reference
from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_discovery import assess_reference_discovery_trace


def test_identity_references_share_exhausted_required_search_allowance(cascade, monkeypatch):
    search,providers=cascade
    search._escalation_limits.update(brave=2,exa=1)  # Exa runs no filetype variant
    # Without a per-reference floor (WEB_SEARCH_PER_REFERENCE_FLOOR) the job-wide
    # allowance is shared; the floor itself is covered in test_web_search_reference_floor.
    search._per_reference_floor={}
    shared=resolver(monkeypatch,[search])
    traces=[]
    for index in range(4):
        ref=reference(reference_id=f'control-{index}',title=f'Competition policy in region {index}')
        result=shared.resolve_reference(ref,identity_only=True)
        trace=result.metadata['reference_discovery_trace'];traces.append(trace)
        assert not result.full_text and not result.abstract and not result.representation
        assessed=assess_reference_discovery_trace(trace)
        assert assessed.ready and assessed.record.outcome=='search_incomplete'
        review=assess_reference_credibility(ref,result.metadata.get('reference_discovery'),trace)
        assert not any(f['finding_type']=='potentially_fabricated_reference' for f in review['findings'])
    assert providers['brave'].search.call_count==2
    assert providers['exa'].search.call_count==1
    for trace in traces[1:]:
        queries=[q for q in trace['queries'] if q.get('execution_provider') in {'brave','exa'}]
        assert {q['execution_provider'] for q in queries}=={'brave','exa'}
        assert all(q['execution_outcome']=='budget_skipped' and q['provider_calls']==0 for q in queries)
    assert search.search_metrics['provider_calls:brave']==2
    assert search.search_metrics['provider_calls:exa']==1
    providers['searxng'].search.assert_not_called()
    providers['tavily'].search.assert_not_called()


def test_one_exhausted_provider_is_not_hidden_by_other_provider_success(cascade, monkeypatch):
    search,providers=cascade
    search._escalation_limits.update(brave=0,exa=2)
    shared=resolver(monkeypatch,[search]);ref=reference()
    result=shared.resolve_reference(ref,identity_only=True)
    trace=result.metadata['reference_discovery_trace']
    assert assess_reference_discovery_trace(trace).record.outcome=='search_incomplete'
    queries=trace['queries']
    assert any(q.get('execution_provider')=='exa' and q['execution_outcome']=='no_results' for q in queries)
    assert any(q.get('execution_provider')=='brave' and q['execution_outcome']=='budget_skipped' for q in queries)
    assert not assess_reference_credibility(ref,result.metadata.get('reference_discovery'),trace)['findings']
    providers['brave'].search.assert_not_called()
