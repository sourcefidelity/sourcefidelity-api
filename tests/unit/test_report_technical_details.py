from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bs4 import BeautifulSoup

from app.services import processing_metrics as pm
from app.services.evidence_report import _render_technical_details, render_evidence_report_html
from app.services.processing_metrics import _breakdowns
from app.services.safe_fetch import UnsafeUrlError, safe_request


@pytest.fixture
def stage():
    current = {'llm_calls': 0, 'total_tokens': 0, 'missing_usage_calls': 0, 'metrics_version': 2,
               'adapter_requests': {}, 'direct_fetches': {}}
    token = pm._current.set(current)
    yield current
    pm._current.reset(token)


def test_worker_threads_are_counted_and_concurrent_increments_are_not_lost(stage):
    usage = SimpleNamespace(total_tokens=10, prompt_tokens=6, completion_tokens=4)

    def call(_):
        pm.record_llm_attempt(model='m', endpoint_host='h')
        pm.record_llm_usage(usage, model='m', endpoint_host='h')
        for _ in range(200):
            pm.record_provider_request('openalex')

    with ThreadPoolExecutor(8) as pool:
        for future in [pool.submit(copy_context().run, call, i) for i in range(8)]:
            future.result()
    assert stage['llm_calls'] == 8 and stage['total_tokens'] == 80 and stage['missing_usage_calls'] == 0
    assert stage['adapter_requests'] == {'openalex': 1600}


def test_reference_parser_pools_carry_the_metrics_context():
    source = Path('app/services/reference_parser.py').read_text()
    assert 'pool.submit(copy_context().run, _fallback_one, i)' in source
    assert 'pool.submit(copy_context().run, _process_batch, indices, refs_list)' in source


@pytest.mark.parametrize('module', ['crossref', 'openalex', 'core', 'semantic_scholar', 'datacite', 'eric',
                                    'open_library', 'elsevier', 'google_books', 'gutenberg', 'wikisource',
                                    'internet_archive_catalog'])
def test_every_academic_adapter_records_its_requests(module):
    source = Path(f'app/services/retrieval/{module}.py').read_text()
    assert source.count('record_provider_request(') >= 1
    assert source.count('httpx.get(') + source.count('httpx.request(') + source.count('_core_request(') <= \
        source.count('record_provider_request(') + source.count('_core_request(') + 1


def test_internet_archive_counts_as_an_adapter_through_safe_fetch():
    assert 'usage_label="adapter:internet_archive"' in Path('app/services/retrieval/internet_archive.py').read_text()


def test_safe_request_records_one_fetch_after_the_safety_check(stage, monkeypatch):
    with pytest.raises(UnsafeUrlError):
        safe_request('http://127.0.0.1/private', usage_label='landing page')
    assert stage['direct_fetches'] == {}                     # rejected before any request
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={'location': '/final'}, request=request)
        return httpx.Response(200, content=b'ok', headers={'content-type': 'text/html'}, request=request)

    real_client = httpx.Client                                # patched below; keep the class
    monkeypatch.setattr('app.services.safe_fetch.httpx.Client',
                        lambda **_kwargs: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr('app.services.safe_fetch._validate_url', lambda _url: None)
    safe_request('https://publisher.example/a', usage_label='landing page')
    safe_request('https://archive.example/a', usage_label='adapter:internet_archive')
    assert len(requests) == 3                                 # one redirect hop
    assert stage['direct_fetches'] == {'landing page': 1} and stage['adapter_requests'] == {'internet_archive': 1}


def test_breakdown_groups_by_model_and_provider_and_keeps_unknowns_unknown():
    priced = {'model': 'deepseek-v4-flash', 'endpoint_host': 'api.deepseek.com', 'usd_per_million': [.003, .15, .6],
              'prompt_tokens': 100, 'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60,
              'completion_tokens': 10, 'total_tokens': 110}
    local = {'model': 'llama3.1', 'endpoint_host': 'localhost', 'prompt_tokens': 50, 'completion_tokens': 5,
             'total_tokens': 55}
    stages = [{'metrics_version': 2, 'adapter_requests': {'crossref': 3, 'openalex': 2},
               'direct_fetches': {'landing page': 4},
               'llm_attempt_records': [{'model': 'deepseek-v4-flash', 'endpoint_host': 'api.deepseek.com'}] * 2
               + [{'model': 'llama3.1', 'endpoint_host': 'localhost'}],
               'llm_usage_records': [priced, local]}]
    searches = [{'execution_provider': 'brave', 'provider_calls': 2, 'cost_usd': None},
                {'execution_provider': 'exa', 'provider_calls': 1, 'cost_usd': .006},
                {'execution_provider': 'tavily', 'provider_calls': 1, 'cost_usd': None},
                {'execution_provider': 'searxng', 'provider_calls': 3, 'cost_usd': 0.}]
    result = _breakdowns(stages, searches, None)
    assert result['metrics_version'] == 'processing-metrics-v2' and result['breakdown_complete']
    assert result['adapter_requests'] == [{'provider': 'crossref', 'requests': 3}, {'provider': 'openalex', 'requests': 2}]
    rows = {row['model']: row for row in result['llm_by_model']}
    assert rows['deepseek-v4-flash']['calls'] == 2 and rows['deepseek-v4-flash']['missing_usage_calls'] == 1
    assert rows['deepseek-v4-flash']['cost_usd'] > 0 and not rows['deepseek-v4-flash']['cost_complete']
    assert rows['llama3.1']['cost_usd'] is None and rows['llama3.1']['endpoint_host'] == 'localhost'
    search = {row['provider']: row for row in result['search_by_provider']}
    assert search['brave'] == {'provider': 'brave', 'calls': 2, 'cost_usd': .01, 'cost_basis': 'tariff', 'unpriced_calls': 0}
    assert search['exa']['cost_basis'] == 'observed' and search['tavily']['cost_usd'] is None
    assert search['searxng']['cost_basis'] == 'free'
    html = _render_technical_details({**result, 'estimated_cost_usd': .02, 'cost_estimate_partial': True})
    text = BeautifulSoup(html, 'html.parser').get_text(' ')
    for expected in ('Crossref', 'OpenAlex', 'Landing page', 'Brave Search', 'no price on record', 'no charge',
                     'llama3.1', '(partial)'):
        assert expected in text
    # Owner request 2026-09-28: none of the explanatory notes.
    for removed in ('Partial: usage with no price on record', 'reported by provider', 'Totals cover all processing',
                    'Active worker time only', 'Requests actually sent', 'Search and model API charges only'):
        assert removed not in text


def test_older_reports_say_not_recorded_rather_than_zero():
    html = _render_technical_details({'wall_seconds': 12.5, 'llm_calls': 3, 'total_tokens': None,
                                      'search_calls_by_provider': {'brave': 2}})
    text = BeautifulSoup(html, 'html.parser').get_text(' ')
    assert 'Per-source requests: not recorded for this report.' in text
    assert 'Per-model detail: not recorded for this report.' in text and 'Tokens: not recorded' in text
    assert 'Estimated Cost (Before Credits) not recorded' in ' '.join(text.split())
    assert 'Direct Web Fetches' not in text


def test_section_is_in_the_one_report_collapsed_and_after_how_to_read():
    # Shown to everyone for now (owner decision 2026-09-25); an old audience
    # parameter changes nothing.
    view = {'title': 'Report', 'citation_format': 'APA', 'citations': [], 'paper_surface': {},
            'processing_metrics': {'wall_seconds': 1.0, 'cpu_seconds': .5}}
    html = render_evidence_report_html(view, csp_nonce='technical-detail-nonce')
    soup = BeautifulSoup(html, 'html.parser')
    details = soup.select_one('details.technical-details')
    assert details and not details.has_attr('open')
    assert details.find_previous_sibling('details')['class'] == ['read-guide']
    assert not soup.select('.technical-export')
    assert render_evidence_report_html({**view, 'audience': 'student'}, csp_nonce='technical-detail-nonce') == html
    assert 'processing-footer' not in html
