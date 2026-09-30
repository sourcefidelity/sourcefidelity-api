import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from app.services.evidence_report import (_restore_retained_continuations, _responsive_display_excerpt,
    _sentence_start_context, _identity_view)
from app.services.usage_cost import llm_price_context, llm_cost, search_trace_cost


def test_continuation_reassembly_requires_parent_hash_and_overlap():
    text='General discussion. Another sentence about actor Jordan.'
    parent=dict(passage_id='p',character_start=10,character_end=10+len(text),excerpt=text[:30],
                excerpt_truncated=True,passage_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                representation_id='r',content_sha256='h',page_index=1)
    child={**parent,'passage_id':'child','parent_passage_id':'p','character_start':30,'excerpt':text[20:]}
    assert _restore_retained_continuations([parent,child])[0]['excerpt']==text
    assert parent['excerpt']==text[:30]
    child['excerpt']='Wrong text'
    assert _restore_retained_continuations([parent,child])[0]['excerpt']==text[:30]


def test_excerpt_prefers_named_actor_over_generic_vocabulary():
    text='Publicity and studio marketing manufactured images through magazines. Jordan was presented as a foreign star.'
    assert 'Jordan' in _responsive_display_excerpt(text,'Publicity and studio marketing created Jordan’s image.')


def test_clear_leading_fragment_is_not_presented_as_sentence_start():
    assert _sentence_start_context('continued from before. A complete sentence begins here.')=='A complete sentence begins here.'
    assert _sentence_start_context('only a surviving fragment')=='only a surviving fragment'


def test_unbound_matching_year_does_not_neutralize_edition_uncertainty():
    match=dict(comparisons=[dict(field_name=f,outcome='agreement') for f in ('title','author','year')])
    other=dict(comparisons=[dict(field_name='year',outcome='unknown',reason_code='book_edition_year_unresolved')])
    assert _identity_view(dict(candidates=[match,other]))['edition_year_unresolved']
    assert _identity_view(dict(candidates=[other]))['edition_year_unresolved']


def test_costs_bind_model_endpoint_time_and_actual_token_categories():
    price=llm_price_context('deepseek-v4-flash','https://api.deepseek.com',datetime(2026,9,16,2,tzinfo=timezone.utc))
    usage=dict(prompt_tokens=1000,prompt_cache_hit_tokens=900,prompt_cache_miss_tokens=100,completion_tokens=100,**price)
    assert abs(llm_cost(usage)-.0001554)<1e-10
    assert not llm_price_context('deepseek-v4-flash','https://example.org')
    assert llm_cost(dict(total_tokens=1200)) is None
    usage['prompt_tokens']=1001
    assert llm_cost(usage) is None


def test_search_receipts_do_not_count_cached_queries():
    q=[dict(execution_provider='brave',provider_calls=2),
       dict(execution_provider='exa',provider_calls=1,cost_usd=.007),
       dict(execution_provider='brave',provider_calls=2,cache_hit=True)]
    result=search_trace_cost([dict(reference_discovery_trace=dict(queries=q))])
    assert result['search_api_calls']==3 and result['search_cost_usd']==.017
