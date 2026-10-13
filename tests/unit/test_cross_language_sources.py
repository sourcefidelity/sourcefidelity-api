"""Cross-language finding and checking of sources (owner decision 2026-10-12).

native-script-query-v1, native-script-route-v1, cross-language-identity-v1 and
cross-language-evidence-v1. Invented fixtures only.
"""
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services import cross_language_identity as cli
from app.services.cross_language_evidence import (
    claim_inputs, evidence_record, language_profile, queries, translate_statement,
)
from app.services.native_script_query import propose_native_query, validate_proposal
from app.services.other_language import (
    dominant_script, other_language_signals, pinyin_like, romanised_title, split_cited_title,
)
from app.services.reference_discovery import (
    ExpectedBibliographicFields, build_reference_discovery_candidate, derive_reference_discovery_record,
)
from app.services.retrieval.base import RetrievalResult

ROMANISED = 'Shuju quanli yu falv baohu yanjiu (Research on data rights and legal protection)'
NATIVE = '数据权利与法律保护研究'
ENTRY = f'Pan, Y., & Lin, Q. (2016). {ROMANISED}. Example Law Review, (11), 6-9.'


def reference(title=ROMANISED, raw=ENTRY, kind='journal_article', author='Pan, Y., & Lin, Q', year='2016'):
    return SimpleNamespace(reference_id='ref-0001-000000000001', title=title, raw_ref=raw, source_kind=kind,
                           author=author, year=year, url=None, doi=None, container_title='Example Law Review',
                           source_kind_confidence='high', source_kind_evidence=())


# ---- other-language detection --------------------------------------------

def test_the_cited_romanised_title_and_translation_are_separate_fields():
    parts = split_cited_title(ROMANISED)
    assert parts == {'raw': ROMANISED, 'romanised': 'Shuju quanli yu falv baohu yanjiu',
                     'translation': 'Research on data rights and legal protection'}
    assert split_cited_title('Learning to teach (A practical guide)') is None
    assert romanised_title(ROMANISED) == 'Shuju quanli yu falv baohu yanjiu'
    assert romanised_title('Shu ju quan li') == 'Shu ju quan li'
    assert romanised_title('Learning to teach') is None
    assert pinyin_like('2016 nian zhong guo') and not pinyin_like('The data rights')


def test_signals_are_shared_with_discovery_and_accept_dicts():
    assert other_language_signals(reference()) == ['romanised_title_with_translation']
    assert other_language_signals({'title': ROMANISED, 'raw_ref': ENTRY}) == ['romanised_title_with_translation']
    assert other_language_signals(reference(title='Rivers', raw='Lin, Q. (2021). Rivers. Example Press.')) == []


def test_dominant_script_reads_letters_only():
    assert dominant_script('数据权利 2016 研究') == 'han'
    assert dominant_script('Data rights, 2016.') == 'latin'
    assert dominant_script('データの権利と法律') == 'kana'
    assert dominant_script('2016 — 11') is None


# ---- deterministic romanised identity -------------------------------------

def test_a_han_title_read_as_the_cited_romanisation_agrees():
    result = cli.compare_titles(ROMANISED, NATIVE)
    assert result == {'outcome': 'agreement', 'reason_code': 'romanised_title_match',
                      'method': 'cross-language-identity-v1', 'similarity': 1.0}
    # Spacing and joined syllables do not matter; a site suffix is ignored.
    assert cli.compare_titles('Shu ju quan li yu fa lv bao hu yan jiu', NATIVE + ' - 示例网')['outcome'] == 'agreement'


def test_every_reading_of_a_character_is_allowed():
    # 行 reads xing or hang; either cited reading aligns.
    assert cli.compare_titles('Yin hang ye yan jiu bao gao', '银行业研究报告')['outcome'] == 'agreement'
    assert cli.compare_titles('Yin xing ye yan jiu bao gao', '银行业研究报告')['outcome'] == 'agreement'


def test_a_misspelled_syllable_is_a_minor_difference_and_a_main_title_match_too():
    near = cli.compare_titles('Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)',
                              '数据权利与法律保护制度研究')
    assert near['outcome'] == 'minor_difference' and near['reason_code'] == 'romanised_title_near_match'
    main = cli.compare_titles('Shuju quanli yanjiu (Research on data rights)', '数据权利研究——以示例为中心')
    assert main['outcome'] == 'minor_difference' and main['reason_code'] == 'romanised_main_title_match'


def test_a_different_han_title_conflicts_and_the_middle_band_is_inconclusive():
    assert cli.compare_titles(ROMANISED, '城市交通规划方法')['outcome'] == 'material_conflict'
    assert cli.compare_titles(ROMANISED, '数据权利研究')['outcome'] == 'material_conflict'
    # One different word of a similar sound: the romanisation cannot decide.
    assert cli.compare_titles(ROMANISED, '数据权利与法律保障研究') is None


def test_a_record_title_starting_with_the_cited_title_is_a_minor_difference():
    prefix = cli.compare_titles(ROMANISED, NATIVE + ' 潘越 林青 法学 【简介】 示例书院')
    assert prefix['outcome'] == 'minor_difference' and prefix['reason_code'] == 'romanised_title_prefix_match'
    # Further title words without a separator are not that.
    assert (cli.compare_titles(ROMANISED, NATIVE + '方法论') or {}).get('reason_code') != 'romanised_title_prefix_match'
    assert cli.compare_titles(ROMANISED, NATIVE + '-示例政府网')['outcome'] == 'agreement'


def test_a_different_number_in_the_title_is_another_work_of_the_series():
    cited = '2017 nian quan guo shu zi chu ban chan ye fa zhan bao gao (2017 digital publishing industry report)'
    assert cli.compare_titles(cited, '2017年全国数字出版产业发展报告')['outcome'] == 'agreement'
    assert cli.compare_titles(cited, '2016年全国数字出版产业发展报告')['outcome'] == 'material_conflict'
    assert cli.compare_titles(cited, '2017年全国数字出版产业发展报告_2016年回顾')['outcome'] == 'agreement'


@pytest.mark.parametrize('observed', ['データの権利と法律', '데이터 권리', 'Data rights and legal protection'])
def test_kana_hangul_and_latin_titles_are_not_read_in_pinyin(observed):
    assert cli.compare_titles(ROMANISED, observed) is None


def test_an_english_cited_title_is_never_romanised():
    assert cli.compare_titles('Research on data rights and legal protection', NATIVE) is None


def test_without_the_library_every_comparison_is_inconclusive(monkeypatch):
    monkeypatch.setattr(cli, '_pypinyin', lambda: None)
    assert cli.compare_titles(ROMANISED, NATIVE) is None
    assert cli.compare_authors(['Pan, Y.'], ['潘越']) is None
    assert cli.romanise(NATIVE) == NATIVE and not cli.applies_to_text(ROMANISED, NATIVE)


def test_romanised_names_match_surname_and_initial():
    assert cli.compare_authors(['Pan, Y., & Lin, Q'], ['潘越', '林青'])['outcome'] == 'agreement'
    assert cli.compare_authors(['Pan, Y., & Lin, Q'], ['潘越', '王明'])['outcome'] == 'minor_difference'
    assert cli.compare_authors(['Pan, Y., & Lin, Q'], ['王明', '赵强'])['outcome'] == 'material_conflict'
    # The initial must agree: 潘明 is not "Pan, Y.".
    assert cli.compare_authors(['Pan, Y.'], ['潘明'])['outcome'] == 'material_conflict'
    # A compound surname is read as one.
    assert cli.compare_authors(['Ouyang, M.'], ['欧阳明'])['outcome'] == 'agreement'
    assert cli.compare_authors(['Pan, Y.'], ['Yue Pan']) is None


def test_a_latin_record_carrying_the_romanised_title_or_the_translation():
    assert cli.latin_title_part_match(ROMANISED, 'Shuju quanli yu falv baohu yanjiu')['outcome'] == 'agreement'
    translated = cli.latin_title_part_match(ROMANISED, 'Research on Data Rights and Legal Protection')
    assert translated['reason_code'] == 'cited_translation_title_match' and translated['outcome'] == 'minor_difference'
    assert cli.latin_title_part_match(ROMANISED, 'Data rights') is None


def test_text_views_for_admission():
    opening = ['示例法学', NATIVE, '潘越  林青', '摘要：本文讨论数据权利。']
    assert cli.romanised_title_in_text(ROMANISED, '\n'.join(opening))
    assert cli.romanised_title_in_text(ROMANISED, '数据权利与法律\n保护研究')  # broken across lines
    assert not cli.romanised_title_in_text(ROMANISED, '城市交通规划方法')
    credited, names = cli.credited_people('Pan, Y., & Lin, Q', opening)
    assert credited == [('pan', 'y'), ('lin', 'q')] and '潘越' in names


# ---- discovery candidates --------------------------------------------------

def _expected(**change):
    values = dict(title=ROMANISED, authors=['Pan, Y., & Lin, Q'], year='2016', source_kind='journal_article')
    values.update(change)
    return ExpectedBibliographicFields(**values)


def _candidate(title=NATIVE, authors=('潘越', '林青'), year='2016', metadata=None):
    return build_reference_discovery_candidate(
        attempt_id='attempt-1', provider='openalex', expected=_expected(),
        result=RetrievalResult(source_name='openalex', success=True, title=title, authors=list(authors),
                               year=year, metadata=metadata or {}))


def test_a_han_record_romanising_to_the_citation_is_confirmed_with_its_method_recorded():
    candidate = _candidate()
    outcomes = {c.field_name: (c.outcome, c.reason_code, c.method) for c in candidate.comparisons}
    assert outcomes['title'] == ('agreement', 'romanised_title_match', 'cross-language-identity-v1')
    assert outcomes['author'] == ('agreement', 'romanised_author_match', 'cross-language-identity-v1')
    assert outcomes['year'][2] is None and 'method' not in candidate.comparisons[2].model_dump()
    from app.services.reference_discovery import ReferenceRouteAttempt, ReferenceSearchQuery
    text = f'title:{NATIVE}'
    query = ReferenceSearchQuery(query_id='query-1', route_category='academic_adapter', provider='openalex',
                                 execution_provider='openalex', execution_outcome='results',
                                 normalized_query=text, query_sha256=hashlib.sha256(text.encode()).hexdigest())
    now = datetime.now(timezone.utc)
    attempt = ReferenceRouteAttempt(attempt_id='attempt-1', route_category='academic_adapter', provider='openalex',
                                    required=False, permitted=True, query_ids=['query-1'],
                                    outcome='candidate_found', started_at=now, completed_at=now)
    record = derive_reference_discovery_record(
        reference_id='ref-0001-000000000001', expected=_expected(), required_route_categories=[],
        queries=[query], attempts=[attempt], candidates=[candidate])
    assert record.outcome == 'confirmed'


def test_a_different_han_record_conflicts_and_is_not_credible():
    candidate = _candidate(title='城市交通规划方法', authors=('王明',))
    assert candidate.has_material_conflict and not candidate.plausible_identity_match


def test_an_undecided_han_title_falls_back_to_the_model_answer():
    judged = {'same_work': 'unsure', 'author': 'not_shown', 'page_authors': []}
    candidate = _candidate(title='数据权利与法律保障研究', metadata={'cross_script_judgment': judged})
    title = next(c for c in candidate.comparisons if c.field_name == 'title')
    assert (title.outcome, title.reason_code) == ('unknown', 'cross_script_model_unsure')
    yes = _candidate(title='データの権利と法律', authors=('Pan Yue',),
                     metadata={'cross_script_judgment': {'same_work': 'yes', 'author': 'agrees',
                                                         'page_authors': ['Pan Yue']}})
    assert next(c for c in yes.comparisons if c.field_name == 'title').reason_code == 'cross_script_model_same_work'


# ---- native-script query proposal -----------------------------------------

def test_the_proposal_is_validated_against_its_schema():
    good = validate_proposal({'language': 'zh', 'native_title': NATIVE, 'native_authors': ['潘越', 3],
                              'native_container': '示例法学', 'confidence': 'certain'})
    assert good == {'language': 'zh', 'native_title': NATIVE, 'native_authors': ['潘越'],
                    'native_container': '示例法学', 'confidence': 'low'}
    assert validate_proposal({'language': 'zh', 'native_title': 'Data rights'}) is None
    assert validate_proposal({'language': 'the original language', 'native_title': NATIVE}) is None
    assert validate_proposal(['not', 'a', 'dict']) is None


def test_the_proposal_sends_only_the_reference_entry_and_records_provenance():
    seen = {}

    def completion(system, payload, **kwargs):
        seen['payload'] = payload
        kwargs['receipt'].update(returned_model='model-x', total_tokens=90, reported_cost_usd=0.0001)
        return {'language': 'zh', 'native_title': NATIVE, 'native_authors': ['潘越'], 'confidence': 'high'}

    proposal = propose_native_query(reference(), completion=completion)
    assert proposal['call'] == 'answered' and proposal['native_title'] == NATIVE
    assert proposal['payload_sha256'] == hashlib.sha256(seen['payload'].encode()).hexdigest()
    assert '"reference"' in seen['payload'] and ENTRY[:30] in seen['payload']
    assert proposal['returned_model'] == 'model-x' and proposal['total_tokens'] == 90
    assert proposal['cited_title']['translation'] == 'Research on data rights and legal protection'


def test_a_failed_or_invalid_proposal_proposes_nothing():
    def failing(*args, **kwargs):
        raise RuntimeError('boom')
    assert propose_native_query(reference(), completion=failing)['call'] == 'failed'
    invalid = propose_native_query(reference(), completion=lambda *a, **k: {'native_title': 'x'})
    assert invalid['call'] == 'invalid_answer' and 'native_title' not in invalid


def test_a_title_already_in_its_own_script_needs_no_model_call():
    def never(*args, **kwargs):
        raise AssertionError('no call expected')
    proposal = propose_native_query(reference(title=f'{NATIVE} [Data rights]'), completion=never)
    assert proposal['call'] == 'not_needed' and proposal['native_title'] == NATIVE


# ---- the native-script route in the resolver ------------------------------

class _Adapter:
    def __init__(self, name, result):
        self.name, self._result, self.titles = name, result, []
        self.capabilities = frozenset({'title_author'})

    def handles_source_kind(self, kind):
        return True

    def search_by_title_author(self, title, author=None):
        self.titles.append(title)
        return self._result(title)


class _Web:
    name = 'web_search'
    capabilities = frozenset({'web_discovery'})

    def __init__(self, outcome='no_results'):
        self.outcome, self.calls = outcome, []

    def search_by_title_author(self, title, author=None, year=None):
        self.calls.append((title, author))
        return RetrievalResult(source_name='web_search', success=False, error='No results', metadata={
            'search_attempts': [{'provider': 'brave', 'query': f'"{title}"', 'outcome': self.outcome,
                                 'result_count': 0}]})


def _resolver(monkeypatch, adapters, web):
    from app.services import native_script_query
    from app.services.source_resolver import SourceResolver
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._acquisition_capabilities = None
    resolver._retrieval_sources = [*adapters, web]
    monkeypatch.setattr(native_script_query, 'propose_native_query', lambda ref: {
        'version': 'native-script-query-v1', 'call': 'answered', 'native_title': NATIVE,
        'native_authors': ['潘越'], 'language': 'zh'})
    return resolver


def _trace(expected):
    return {'reference_id': 'ref-0001-000000000001', 'submitted_reference': ENTRY,
            'search_policy_version': 'api-first-search-v2', 'required_web_providers': ['brave', 'exa'],
            'expected': expected, 'required': {'academic_adapter', 'bounded_web'},
            'queries': [], 'attempts': [], 'candidates': [], 'limitations': []}


def _han_record(title):
    if title == NATIVE:
        return RetrievalResult(source_name='openalex', success=True, title=NATIVE, authors=['潘越', '林青'],
                               year='2016', abstract='本文讨论数据权利与法律保护。',
                               metadata={'structured_search_attempts': []})
    return RetrievalResult(source_name='openalex', success=False, error='No results',
                           metadata={'identity_search_result_count': 0})


def test_the_route_searches_in_the_works_own_script_and_confirms_by_romanisation(monkeypatch):
    from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
    openalex = _Adapter('openalex', _han_record)
    crossref = _Adapter('crossref', lambda t: RetrievalResult(source_name='crossref', success=False,
                                                              error='No results',
                                                              metadata={'identity_search_result_count': 0}))
    web = _Web()
    resolver = _resolver(monkeypatch, [openalex, crossref], web)
    token = _ACTIVE_DISCOVERY_TRACE.set(_trace(_expected()))
    try:
        admitted = resolver._native_script_route(reference(), identity_only=False)
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        route = trace['native_script_route']
        _trace_payload, discovery = SourceResolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    # Native title and then the cited translation, on the free adapters.
    assert openalex.titles == [NATIVE, 'Research on data rights and legal protection']
    # The web search asked for the native title and native author; identity
    # checks still compare with the cited fields.
    assert web.calls == [(NATIVE, '潘越')]
    assert route['status'] == 'completed' and route['policy_version'] == 'native-script-route-v1'
    assert route['adapters_completed'] == ['crossref', 'openalex'] and route['web_completed'] == ['brave']
    assert set(route['attempt_ids']) <= {a.attempt_id for a in trace['attempts']}
    assert route['cited_title']['romanised'] == 'Shuju quanli yu falv baohu yanjiu'
    assert discovery['outcome'] == 'confirmed'
    # The located record's abstract in its own language is admitted as abstract evidence.
    assert admitted is not None and admitted.abstract.startswith('本文') and admitted.metadata['native_script_route']


def test_a_web_provider_skipped_by_a_budget_leaves_the_route_incomplete(monkeypatch):
    from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE
    resolver = _resolver(monkeypatch, [_Adapter('openalex', _han_record)], _Web(outcome='budget_skipped'))
    token = _ACTIVE_DISCOVERY_TRACE.set(_trace(_expected()))
    try:
        resolver._native_script_route(reference(), identity_only=True)
        route = _ACTIVE_DISCOVERY_TRACE.get()['native_script_route']
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    assert route['status'] == 'incomplete' and route['web_skipped'] == ['brave']


def test_no_proposal_and_no_signal(monkeypatch):
    from app.services import native_script_query
    from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE
    web = _Web()
    resolver = _resolver(monkeypatch, [], web)
    monkeypatch.setattr(native_script_query, 'propose_native_query', lambda ref: {'call': 'failed'})
    token = _ACTIVE_DISCOVERY_TRACE.set(_trace(_expected()))
    try:
        assert resolver._native_script_route(reference(), identity_only=True) is None
        assert _ACTIVE_DISCOVERY_TRACE.get()['native_script_route']['status'] == 'proposal_unavailable'
        english = reference(title='Rivers of the plain', raw='Lin, Q. (2021). Rivers of the plain. Example Press.')
        _ACTIVE_DISCOVERY_TRACE.get().pop('native_script_route')
        assert resolver._native_script_route(english, identity_only=True) is None
        assert 'native_script_route' not in _ACTIVE_DISCOVERY_TRACE.get()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    assert web.calls == []


def test_the_route_has_its_own_elapsed_allowance():
    from app.services.retrieval_deadline import deadline_scope, expired, separate_deadline_scope
    with deadline_scope(0):
        assert expired()
        with separate_deadline_scope(60):
            assert not expired()
        assert expired()


def test_a_han_page_title_read_as_the_cited_title_is_read_not_set_aside():
    from app.services.source_resolver import _romanised_page_title_agrees
    assert _romanised_page_title_agrees(ROMANISED, [NATIVE + ' - 示例网'])
    assert not _romanised_page_title_agrees(ROMANISED, ['城市交通规划方法'])


# ---- verification and re-run reuse ----------------------------------------

def _verified_discovery(route=None, native_attempt=False):
    query = {'query_id': 'q1', 'provider': 'crossref', 'execution_provider': 'crossref',
             'execution_outcome': 'no_results', 'normalized_query': f'title:{ROMANISED.lower()}',
             'query_sha256': 'x'}
    queries = [query, {**query, 'query_id': 'q2', 'provider': 'openalex', 'execution_provider': 'openalex'},
               {**query, 'query_id': 'w1', 'provider': 'web_search', 'execution_provider': 'brave',
                'normalized_query': f'"{ROMANISED.lower()}"'}]
    def attempt(aid, category, provider, qid):
        return {'attempt_id': aid, 'route_category': category, 'provider': provider, 'required': True,
                'permitted': True, 'query_ids': [qid], 'outcome': 'no_match',
                'started_at': '2026-10-12T00:00:00Z', 'completed_at': '2026-10-12T00:00:01Z'}
    attempts = [attempt('a1', 'academic_adapter', 'crossref', 'q1'),
                attempt('a2', 'academic_adapter', 'openalex', 'q2'),
                attempt('b1', 'bounded_web', 'web_search', 'w1')]
    record = {'reference_id': 'ref-0001-000000000001', 'outcome': 'search_incomplete',
              'expected': {'title': ROMANISED, 'source_kind': 'journal_article'},
              'queries': queries, 'attempts': attempts, 'candidates': []}
    if route is not None:
        record['native_script_route'] = route
    return record


@pytest.mark.parametrize('route,status,reason', [
    (None, 'not_assessed', 'other_language_route_not_searched'),
    ({'policy_version': 'native-script-route-v1', 'status': 'incomplete'}, 'not_assessed',
     'other_language_route_incomplete'),
    ({'policy_version': 'native-script-route-v1', 'status': 'proposal_unavailable'}, 'not_assessed',
     'other_language_route_incomplete'),
    ({'policy_version': 'native-script-route-v1', 'status': 'completed'}, 'cannot_be_verified',
     'suited_searches_completed_without_match'),
])
def test_cannot_be_verified_only_after_the_native_script_route_completed(route, status, reason):
    from app.services.reference_verification import assess_reference_verification
    result = assess_reference_verification(reference(), _verified_discovery(route))
    assert (result['status'], result['reason_code']) == (status, reason)


def test_native_route_searches_do_not_complete_the_ordinary_required_routes():
    from app.services.reference_verification import assess_reference_verification
    record = _verified_discovery({'policy_version': 'native-script-route-v1', 'status': 'completed',
                                  'attempt_ids': ['a2']})
    result = assess_reference_verification(reference(), record)
    assert result['status'] == 'search_incomplete' and 'openalex' in result['incomplete_routes']


def test_a_rerun_searches_again_until_the_native_route_completed():
    from app.services.search.rerun_reuse import POLICY, native_script_route_due
    ref = {'title': ROMANISED, 'raw_ref': ENTRY, 'source_kind': 'journal_article'}
    assert POLICY == 'paper-search-reuse-v5'
    assert native_script_route_due({'reference_discovery': _verified_discovery()}, ref)
    incomplete = {'policy_version': 'native-script-route-v1', 'status': 'incomplete'}
    assert native_script_route_due({'reference_discovery': _verified_discovery(incomplete)}, ref)
    done = {'policy_version': 'native-script-route-v1', 'status': 'completed'}
    assert not native_script_route_due({'reference_discovery': _verified_discovery(done)}, ref)
    assert not native_script_route_due({'reference_discovery': {**_verified_discovery(), 'outcome': 'confirmed'}}, ref)
    assert not native_script_route_due({'reference_discovery': _verified_discovery()},
                                       {**ref, 'title': 'Rivers', 'raw_ref': 'Lin, Q. (2021). Rivers.'})


# ---- admission of other-language sources ----------------------------------

def test_a_han_document_with_the_cited_title_and_romanised_byline_is_high_identity():
    from app.services.source_resolver import _verify_text_content_identity
    from app.services.source_validator import _identity_observation
    front = '\n'.join(['示例法学 2016年第11期', NATIVE, '潘越  林青', '摘要：本文讨论数据权利。', '正文……'])
    seen = _identity_observation(front, '', None, ROMANISED, 'Pan, Y., & Lin, Q', '2016',
                                 prominent_title=True, prominent_years={'2016'})
    assert seen.confidence == 'high' and seen.observed['method'] == 'cross-language-identity-v1'
    assert seen.observed['authors'] == ['Pan, Y., & Lin, Q'] and seen.observed['title'] == ROMANISED
    other_byline = front.replace('潘越  林青', '王明  赵强')
    conflicted = _identity_observation(other_byline, '', None, ROMANISED, 'Pan, Y., & Lin, Q', '2016',
                                       prominent_title=True, prominent_years={'2016'})
    assert conflicted.confidence == 'medium' and conflicted.author_conflict['basis'] == 'han_name_line'
    page = ('示例网 首页 ' + NATIVE + ' 潘越 林青 示例法学 2016 摘要 本文讨论数据权利。').encode()
    assert _verify_text_content_identity(page, None, ROMANISED, 'Pan, Y., & Lin, Q', '2016')[0] == 'high'
    wrong = ('示例网 首页 城市交通规划方法 王明 2016').encode()
    assert _verify_text_content_identity(wrong, None, ROMANISED, 'Pan, Y., & Lin, Q', '2016')[0] == 'low'


def test_a_han_title_zone_is_prominent_for_the_romanised_citation():
    from app.services.source_validator import _title_in_layout_spans
    spans = [{'text': NATIVE, 'bbox': (50, 40, 500, 60), 'size': 18.0},
             {'text': '正文内容' * 20, 'bbox': (50, 400, 500, 420), 'size': 10.0}]
    assert _title_in_layout_spans(spans, 800.0, ROMANISED)
    assert not _title_in_layout_spans(spans, 800.0, 'Lu xian dai hua yan jiu (Road modernisation)')


# ---- checking how the student used the source -----------------------------

def test_han_runs_become_character_pairs_and_latin_tokens_are_unchanged():
    from app.services.verification_evidence import _meaningful_tokens
    assert _meaningful_tokens('Data rights protection') == ['data', 'rights', 'protection']
    assert _meaningful_tokens('数据权利') == ['数据', '据权', '权利']
    assert _meaningful_tokens('GDP数据') == ['gdp', '数据']


def test_han_sentences_split_at_their_own_full_stops():
    from app.services.facet_evidence_judgment import _passage_sentences
    passage = SimpleNamespace(passage_id='p1', text='数据权利日益重要。他说：“需要保护。”法律尚不完善！')
    texts = [s.text for s in _passage_sentences(passage)]
    assert texts == ['数据权利日益重要。', '他说：“需要保护。”', '法律尚不完善！']


def _source(text):
    content = text.encode('utf-8')
    now = datetime.now(timezone.utc)
    from app.services.verification_evidence import AuthorizedRepresentation
    return AuthorizedRepresentation(
        representation_id='han-source', canonical_work_id='han-work', content_object_id='han-object',
        content_sha256=hashlib.sha256(content).hexdigest(), content=content, representation_kind='plain_text',
        media_type='text/plain', provenance='authorized_upload', scope_type='personal_owner', scope_id='owner-1',
        identity_verdict='verified', identity_confidence=0.99, completeness_verdict='complete',
        text_quality='digital', edition_or_version=None, created_at=now, admitted_at=now)


HAN_SOURCE = '\n\n'.join([
    '城市交通规划需要长期投入，示例城市的经验表明公共交通优先。' * 2,
    '数据权利保护需要明确的法律规则，个人数据的收集应当取得同意。' * 2,
    '示例研究的样本来自三个地区，方法为问卷调查和访谈。' * 2,
])


def _claim(text, claim_type='paraphrase'):
    from app.services.verification_evidence import ClaimEvidence
    return ClaimEvidence(claim_id='claim-1', paper_version_id='paper-v1', text=text, claim_type=claim_type,
                         reference_ids=['ref-1'], citation_marker='(Pan & Lin, 2016)',
                         citation_marker_type='parenthetical', passage_start=0, passage_end=len(text))


def test_translated_queries_find_the_other_language_passage_and_are_recorded_as_such():
    from app.services.verification_evidence import build_passage_evidence
    source = _source(HAN_SOURCE)
    statement = 'Collecting personal data requires consent (Pan & Lin, 2016).'

    def completion(system, payload, **kwargs):
        kwargs['receipt'].update(returned_model='model-x', total_tokens=50)
        assert '"zh"' in payload and statement in payload and '城市' not in payload
        return {'translation': '收集个人数据需要取得同意', 'key_phrases': ['个人数据', '取得同意', 'consent']}

    record, translated = claim_inputs(source, statement, llm_enabled=True, completion=completion)
    assert record['differs'] and record['source_script'] == 'han' and record['statement_script'] == 'latin'
    assert translated == ['收集个人数据需要取得同意', '个人数据', '取得同意']
    assert record['query_translation']['translation_derived'] is True
    assert record['query_translation']['query_sha256s'][0] == hashlib.sha256(translated[0].encode()).hexdigest()
    plain = build_passage_evidence(source, claim=_claim(statement))
    assert plain.passages == [] and 'cross_language' not in plain.model_dump()
    artifact = build_passage_evidence(source, claim=_claim(statement), cross_language=record,
                                      translated_queries=translated)
    assert artifact.passages and '个人数据' in artifact.passages[0].text
    assert artifact.passages[0].retrieval_method == 'translated_query_lexical'
    assert artifact.model_dump()['cross_language']['source_script'] == 'han'
    assert artifact._translated_queries == translated


def test_an_english_quotation_against_an_other_language_source_is_not_assessable():
    from app.services.verification_evidence import build_passage_evidence
    source = _source(HAN_SOURCE)
    statement = '"Personal data collection requires consent" (Pan & Lin, 2016, p. 7).'
    record, _ = claim_inputs(source, statement, llm_enabled=False)
    claim = _claim(statement, 'quotation').model_copy(update={'page_locator': '7'})
    artifact = build_passage_evidence(source, claim=claim, cross_language=record)
    assert artifact.quotation_check.status == 'not_assessable'
    assert artifact.quotation_check.outcome == 'source_language_differs'
    assert artifact.locator_check.status == 'not_assessable'
    # Without the record the ordinary check would report the wording as not located.
    ordinary = build_passage_evidence(source, claim=claim)
    assert ordinary.quotation_check.outcome != 'source_language_differs'


def test_a_quotation_only_citation_of_an_other_language_source_is_judged_as_a_paraphrase():
    # Owner decision 2026-10-13: a quotation across languages is a
    # translation, so a paraphrase; it is judged and counted as one.
    from app.services.evidence_report import _effective_claim_type
    from app.services.judgment_input import judgment_eligibility
    from app.services.verification_evidence import build_passage_evidence
    source = _source(HAN_SOURCE)
    statement = 'Pan and Lin wrote that "personal data collection requires consent" (2016, p. 7).'
    record, _ = claim_inputs(source, statement, llm_enabled=False)
    claim = _claim(statement, 'quotation').model_copy(update={'quotation_form': 'quotation_only'})
    artifact = build_passage_evidence(source, claim=claim, cross_language=record)
    assert artifact.claim.quotation_form == 'embedded'
    assert judgment_eligibility(artifact.model_dump(mode='json')).reason_code != 'quotation_only'
    assert build_passage_evidence(source, claim=claim).claim.quotation_form == 'quotation_only'
    read = {'status': 'evidence_available', 'source_language_differs': True}
    assert _effective_claim_type('quotation', [read]) == 'paraphrase'
    assert _effective_claim_type('quotation', [read, {'status': 'evidence_available'}]) == 'quotation'
    assert _effective_claim_type('quotation', [{'status': 'unavailable'}]) == 'quotation'


def test_translated_queries_add_labelled_candidate_channels():
    from app.services.verification_evidence import _candidate_union_candidates, _extract_pages
    pages, _ = _extract_pages(_source(HAN_SOURCE))
    ranked, _rescue, facets = _candidate_union_candidates(
        pages, query_text='Collecting personal data requires consent.', page_locator='',
        broad_passages=[], top_k=3, translated_queries=['收集个人数据需要取得同意'])
    channels = {channel for _candidate, item_channels in ranked for channel in item_channels}
    assert any('translated' in channel for channel in channels)
    assert '收集个人数据需要取得同意' in facets


def test_the_language_profile_and_translation_guards():
    assert language_profile('Data rights.', 'Data rights matter.')['differs'] is False
    assert claim_inputs(_source('Data rights need law.'), 'Data rights.', llm_enabled=True,
                        completion=lambda *a, **k: pytest.fail('no call')) == (None, [])
    wrong = translate_statement('Data need consent.', 'han',
                                completion=lambda *a, **k: {'translation': 'Data need consent.'})
    assert wrong['call'] == 'invalid_answer' and queries(wrong) == []
    assert translate_statement('Data.', 'other')['call'] == 'not_supported'
    failed = translate_statement('Data.', 'han', completion=lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert failed['call'] == 'failed'
    assert 'query_translation' not in evidence_record({'source_script': 'han', 'statement_script': 'latin',
                                                       'differs': True}, None)


def test_claim_terms_are_not_counted_in_a_source_in_another_writing_system():
    from app.services.verification_evidence import claim_terms_present_in_source
    assert claim_terms_present_in_source(_source(HAN_SOURCE), 'Personal information collection requires consent') == (0, 0)
    english = _source('Personal information collection requires consent in every jurisdiction studied.')
    assert claim_terms_present_in_source(english, 'Personal information collection requires consent')[1] > 0


# ---- near readings, name forms and the student's own link ------------------

def test_a_near_reading_the_model_also_judged_the_same_work_agrees():
    yes = {'same_work': 'yes', 'author': 'not_shown', 'page_authors': []}
    near_title = '数据权利与法律保护制度研究'
    candidate = build_reference_discovery_candidate(
        attempt_id='attempt-1', provider='web_fetch',
        expected=_expected(title='Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)'),
        result=RetrievalResult(source_name='web_fetch', success=True, title=near_title, year='2016',
                               metadata={'cross_script_judgment': yes}))
    title = next(c for c in candidate.comparisons if c.field_name == 'title')
    assert (title.outcome, title.reason_code, title.method) == (
        'agreement', 'romanised_near_match_model_same_work', 'cross-language-identity-v1')
    # A model "yes" never overrides a romanised conflict.
    other = build_reference_discovery_candidate(
        attempt_id='attempt-1', provider='web_fetch', expected=_expected(),
        result=RetrievalResult(source_name='web_fetch', success=True, title='城市交通规划方法',
                               metadata={'cross_script_judgment': yes}))
    assert next(c for c in other.comparisons if c.field_name == 'title').outcome == 'material_conflict'


def test_a_same_work_page_in_another_script_dated_later_is_not_a_year_conflict():
    yes = {'same_work': 'yes', 'author': 'not_shown', 'page_authors': []}
    expected = ExpectedBibliographicFields(title='Measures for the governance of online services', year='2000',
                                           authors=['Northfield Ministry of Media'])
    def year_of(page_year, judgment):
        candidate = build_reference_discovery_candidate(
            attempt_id='attempt-1', provider='web_fetch', expected=expected,
            result=RetrievalResult(source_name='web_fetch', success=True, title='网络服务管理办法', year=page_year,
                                   metadata={'cross_script_judgment': judgment}))
        return next((c.outcome, c.reason_code) for c in candidate.comparisons if c.field_name == 'year')
    assert year_of('2024', yes) == ('unknown', 'cross_script_page_dated_later')
    # An earlier page date, or no same-work judgment, stays a conflict.
    assert year_of('1998', yes)[0] == 'material_conflict'
    assert year_of('2024', {})[0] == 'material_conflict'


def test_an_organisation_named_in_another_language_is_not_an_author_conflict():
    # Owner decision 2026-10-13: translations of an organisation's name vary.
    def author_of(cited, judged):
        candidate = build_reference_discovery_candidate(
            attempt_id='attempt-1', provider='web_fetch',
            expected=ExpectedBibliographicFields(title='Measures for the governance of online services',
                                                 year='2000', authors=[cited]),
            result=RetrievalResult(source_name='web_fetch', success=True, title='网络服务管理办法', year='2000',
                                   authors=['北原媒体部'], metadata={'cross_script_judgment': {
                                       'same_work': 'yes', 'author': judged, 'page_authors': ['北原媒体部']}}))
        return next((c.outcome, c.reason_code) for c in candidate.comparisons if c.field_name == 'author')
    assert author_of('Northfield Ministry of Media', 'differs') == ('unknown', 'cross_script_organisation_author_unresolved')
    assert author_of('Northfield Ministry of Media', 'agrees')[0] == 'agreement'


def test_han_name_entries_are_split_and_read_in_both_orders():
    assert cli.han_name_forms(['潘越 林青'])[:2] == ['潘越', '林青']
    assert set(cli.han_name_forms(['越 潘'])) == {'潘越', '越潘'}
    assert cli.compare_authors(['Pan, Y.'], ['越 潘'])['outcome'] == 'agreement'
    assert cli.compare_authors(['Pan, Y., & Lin, Q'], ['潘越 林青'])['outcome'] == 'agreement'


def _link_transport(monkeypatch, body):
    import httpx
    from app.services import safe_fetch
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=body, headers={'content-type': 'text/html'}, request=req))
    from httpx._client import Client as real_client  # unaffected by an earlier patch in the same test
    monkeypatch.setattr(safe_fetch.httpx, 'Client', lambda **kwargs: real_client(transport=transport))
    monkeypatch.setattr(safe_fetch, '_validate_url', lambda url: None)
    monkeypatch.setattr(safe_fetch, '_validate_connected_peer', lambda *a, **k: None)


def _link_page(title):
    paragraph = '本文讨论个人数据的收集、使用与保护，认为需要明确的法律规则和监督机制。' * 6
    return (f'<html><head><title>{title}</title><meta name="citation_title" content="{title}">'
            '<meta name="citation_author" content="潘越"><meta name="citation_author" content="林青">'
            '<meta name="citation_date" content="2016"></head><body><article>'
            f'<h1>{title}</h1><p>潘越 林青</p>' + ''.join(f'<p>{paragraph}</p>' for _ in range(4))
            + '</article></body></html>')


def test_a_student_link_titled_as_the_romanised_citation_is_read_without_a_model_call(monkeypatch):
    from app.services.source_resolver import SourceResolver
    _link_transport(monkeypatch, _link_page(NATIVE))

    def never(*args, **kwargs):
        raise AssertionError('no model call expected for an exact reading')
    monkeypatch.setattr('app.services.cross_script_identity.judge_cross_script_identity', never)
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/1', ROMANISED, expected_author='Pan, Y., & Lin, Q', expected_year='2016')
    # The page in its own language is read, identified and admitted as the source's text.
    assert result.success and result.representation is not None and '个人数据' in result.representation.content.decode()
    assert result.metadata['web_fetch_diagnostic']['reason'] == 'bibliographic_identity_confirmed'
    assert result.metadata['identity_confidence'] == 'high'


def test_a_student_link_with_a_near_reading_asks_the_model_once(monkeypatch):
    from app.services.source_resolver import SourceResolver
    near = '数据权利与法律保护制度研究'
    _link_transport(monkeypatch, _link_page(near))
    calls = []

    def judged(reference, page, **kwargs):
        calls.append(page['titles'])
        return {'version': 'cross-script-identity-v1', 'same_work': 'yes', 'author': 'agrees',
                'page_authors': ['潘越'], 'call': 'answered'}
    monkeypatch.setattr('app.services.cross_script_identity.judge_cross_script_identity', judged)
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/2', 'Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)',
        expected_author='Pan, Y., & Lin, Q', expected_year='2016')
    assert len(calls) == 1
    assert (result.metadata or {}).get('cross_script_judgment', {}).get('same_work') == 'yes'


def test_a_journal_abstract_page_in_its_own_language_stays_a_candidate(monkeypatch):
    # 2026-10-12 rerun: a journal article's link reached its abstract page in
    # Han script; the page type cannot be confirmed, but its title still names
    # the work and must stay a candidate rather than vanish.
    from app.services.reference_discovery import build_reference_discovery_candidate
    from app.services.source_resolver import SourceResolver
    from app.services.source_type import SourceKindAssessment
    near = '数据权利与法律保护制度研究'
    _link_transport(monkeypatch, _link_page(near))
    calls = []

    def judged(reference, page, **kwargs):
        calls.append(page['titles'])
        return {'version': 'cross-script-identity-v1', 'same_work': 'yes', 'author': 'agrees',
                'page_authors': ['潘越'], 'call': 'answered'}
    monkeypatch.setattr('app.services.cross_script_identity.judge_cross_script_identity', judged)
    cited = 'Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)'
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/3', cited,
        expected_source_kind=SourceKindAssessment(kind='journal_article', confidence='high'),
        expected_author='Pan, Y., & Lin, Q', expected_year='2016')
    assert not result.success and result.representation is None
    assert result.title == near and len(calls) == 1
    assert result.metadata['work_identity_basis'] == 'cross-language-identity-v1'
    candidate = build_reference_discovery_candidate(
        attempt_id='attempt-1', provider='web_fetch', expected=_expected(title=cited), result=result)
    assert next(c for c in candidate.comparisons if c.field_name == 'title').outcome == 'agreement'


def _page_with_byline_outside_metadata(title, byline):
    paragraph = '本文讨论个人数据的收集、使用与保护，认为需要明确的法律规则和监督机制。' * 6
    return (f'<html><head><title>{title}</title></head><body><nav><a>首页</a><a>新闻</a></nav>'
            f'<div class="title">{title}</div><div>发布时间：2016-03-01</div><span>来源：</span><span>{byline}</span>'
            '<article>' + ''.join(f'<p>{paragraph}</p>' for _ in range(4)) + '</article></body></html>')


def test_a_han_byline_beside_the_page_title_credits_the_cited_author(monkeypatch):
    # 2026-10-12 rerun: a ministry page named its writer only in a span
    # beside the title ("潘越" for "Pan, Y.").
    from app.services.source_resolver import SourceResolver
    near = '数据权利与法律保护制度研究'
    calls = []

    def judged(reference, page, **kwargs):
        calls.append(1)
        return {'version': 'cross-script-identity-v1', 'same_work': 'yes', 'author': 'not_shown',
                'page_authors': [], 'call': 'answered'}
    monkeypatch.setattr('app.services.cross_script_identity.judge_cross_script_identity', judged)
    cited = 'Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)'
    _link_transport(monkeypatch, _page_with_byline_outside_metadata(near, '潘越'))
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/4', cited, expected_author='Pan, Y.', expected_year='2016')
    assert result.success and result.metadata.get('work_identity_basis') == 'cross-language-identity-v1'
    assert result.authors == ['潘越']
    # A name given with its label beside the title counts the same way.
    _link_transport(monkeypatch, _page_with_byline_outside_metadata(near, '编辑：潘越'))
    labelled = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/6', cited, expected_author='Pan, Y.', expected_year='2016')
    assert labelled.success and labelled.authors == ['潘越']
    # Another person beside the title gives no credit.
    _link_transport(monkeypatch, _page_with_byline_outside_metadata(near, '林青'))
    other = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rights/5', cited, expected_author='Pan, Y.', expected_year='2016')
    assert not other.success and other.error != 'web_fetch_failed:runtime_error'
    assert other.metadata['web_fetch_diagnostic']['reason'] == 'bibliographic_identity_unconfirmed'


@pytest.mark.parametrize('author, cited, admitted', [
    ('not_shown', 'Northfield Ministry of Media', True),
    ('differs', 'Northfield Ministry of Media', True),   # an organisation's name is a translation
    ('differs', 'Pan, Y.', False),                        # a cited person the page does not show
])
def test_a_page_the_model_judged_the_same_work_is_admitted(monkeypatch, author, cited, admitted):
    # Owner decision 2026-10-13: the model's same-work answer is enough to
    # admit a page in another writing system cited by its English title.
    from app.services.source_resolver import SourceResolver
    page = _link_page('网络服务管理办法')
    if cited != 'Pan, Y.':
        # A ministry page names no people.
        page = page.replace('<meta name="citation_author" content="潘越"><meta name="citation_author" content="林青">', '')
        page = page.replace('<p>潘越 林青</p>', '')
    _link_transport(monkeypatch, page)
    monkeypatch.setattr('app.services.cross_script_identity.judge_cross_script_identity',
                        lambda *a, **k: {'version': 'cross-script-identity-v1', 'same_work': 'yes',
                                         'author': author, 'page_authors': [], 'call': 'answered'})
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        'https://pages.example/rules/7', 'Measures for the governance of online services',
        expected_author=cited, expected_year='2016')
    assert result.success is admitted
    if admitted:
        assert result.metadata['work_identity_basis'] == 'cross-script-identity-v1'
        assert '个人数据' in result.representation.content.decode()


def test_the_densest_text_block_is_read_when_the_extractor_finds_only_navigation():
    from app.services.source_resolver import _densest_text_block
    body = '第一条 为了规范网络服务活动，保护公民合法权益，制定本规定。' * 20
    html = (f'<html><body><nav>首页 简 繁 登录</nav><div class="wrap"><div class="menu">新闻 服务 互动</div>'
            f'<div class="pages_content">{body}</div></div><footer>版权所有</footer></body></html>')
    assert _densest_text_block(html).startswith('第一条 为了规范网络服务活动')
    assert _densest_text_block('<html><body><div>首页 新闻</div></body></html>') == ''


def test_an_organisation_is_not_contradicted_by_a_person_named_in_another_script():
    # A ministry page names its web editor; the citation names the ministry.
    candidate = build_reference_discovery_candidate(
        attempt_id='attempt-1', provider='web_fetch',
        expected=ExpectedBibliographicFields(title='Measures for the governance of online services', year='2016',
                                             authors=['Northfield Ministry of Media']),
        result=RetrievalResult(source_name='web_fetch', success=True, title='网络服务管理办法', year='2016',
                               authors=['周宁'], metadata={'cross_script_judgment': {
                                   'same_work': 'yes', 'author': 'agrees', 'page_authors': []}}))
    author = next(c for c in candidate.comparisons if c.field_name == 'author')
    assert (author.outcome, author.reason_code) == ('unknown', 'cross_script_organisation_author_unresolved')


def test_the_route_record_is_attached_to_both_payloads():
    from app.services.source_resolver import SourceResolver
    trace, record = {}, {}
    route = {'policy_version': 'native-script-route-v1', 'status': 'completed'}
    SourceResolver._attach_native_route(trace, record, route)
    assert trace['native_script_route'] == record['native_script_route'] == route
    assert trace['native_script_route'] is not route
    SourceResolver._attach_native_route(None, None, route)  # nothing to attach to


def test_translated_queries_survive_later_evidence_steps():
    from app.services.candidate_relationship_judgment import attach_verification_candidates
    from app.services.citation_use_router import attach_citation_use_routes
    from app.services.verification_evidence import build_passage_evidence
    source = _source(HAN_SOURCE)
    statement = 'Collecting personal data requires consent (Pan & Lin, 2016).'
    record = evidence_record(language_profile(HAN_SOURCE, statement), None)
    artifact = build_passage_evidence(source, claim=_claim(statement), cross_language=record,
                                      translated_queries=['收集个人数据需要取得同意'])
    artifact = attach_citation_use_routes(attach_verification_candidates(artifact))
    assert artifact._translated_queries == ['收集个人数据需要取得同意']
    assert '收集个人数据需要取得同意' not in artifact.report_payload().__repr__()
