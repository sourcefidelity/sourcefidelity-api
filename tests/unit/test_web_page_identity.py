"""Web page identity: one decision whichever route reached the page (web-page-identity-v1, 2026-10-21).

One case for each situation where the student-link, search-page, own-script
and uploaded-HTML routes used to differ. The model is the fixed-answer judge.
"""
import pytest

from app.config import settings
from app.services import source_resolver
from app.services.supplied_html_source import SuppliedHtmlRejected, qualify_supplied_html
from app.services.web_page_identity import ask_model, model_admits
from tests.unit.discovery_support import recorder_from

ROMANISED = 'Shuju quanli yu falv baohu yanjiu (Research on data rights and legal protection)'
NATIVE = '数据权利与法律保护研究'
NEAR = '数据权利与法律保护制度研究'
PARAGRAPH = '本文讨论个人数据的收集、使用与保护，认为需要明确的法律规则和监督机制。' * 6


def _han_page(title, *, people=('潘越', '林青'), date='2016', byline=None):
    meta = ''.join(f'<meta name="citation_author" content="{name}">' for name in people)
    return (f'<html><head><title>{title}</title><meta name="citation_title" content="{title}">{meta}'
            f'<meta name="citation_date" content="{date}"><meta property="og:type" content="article">'
            f'</head><body><article><h1>{title}</h1>' + (f'<span>{byline}</span>' if byline else '')
            + '<div class="article-body">' + ''.join(f'<p>{PARAGRAPH}</p>' for _ in range(14))
            + '</div></article></body></html>').encode()


def _supplied(content, **expected):
    return qualify_supplied_html(content, **expected)


def test_a_han_title_read_as_the_romanised_citation_needs_no_model(page_judge):
    page = _supplied(_han_page(NATIVE), expected_title=ROMANISED, expected_author='Pan, Y., & Lin, Q',
                     expected_year='2016')
    assert page.identity_confidence == 'high' and page_judge.calls == []


def test_a_page_the_model_judged_the_same_work_is_accepted_on_upload(page_judge):
    page_judge.answer = {'same_work': 'yes', 'author': 'not_shown'}
    page = _supplied(_han_page('网络服务管理办法', people=()), expected_title='Measures for the governance of online services',
                     expected_author='Northfield Ministry of Media', expected_year='2016')
    assert page.identity_confidence == 'high' and len(page_judge.calls) == 1


def test_a_page_the_model_judged_another_work_is_refused(page_judge):
    page_judge.answer = {'same_work': 'no'}
    with pytest.raises(SuppliedHtmlRejected) as refused:
        _supplied(_han_page('网络服务管理办法', people=()), expected_title='Measures for the governance of online services',
                  expected_author='Northfield Ministry of Media', expected_year='2016')
    assert refused.value.reason_code == 'bibliographic_fields_conflict'


def test_a_same_work_page_dated_later_is_not_a_year_conflict(page_judge):
    page_judge.answer = {'same_work': 'yes', 'author': 'not_shown'}
    page = _supplied(_han_page('网络服务管理办法', people=(), date='2019'),
                     expected_title='Measures for the governance of online services',
                     expected_author='Northfield Ministry of Media', expected_year='2016')
    assert page.identity_confidence == 'high'
    year = next(c for c in page.identity_comparisons if c['field_name'] == 'year')
    assert year['outcome'] == 'unknown' and year['reason_code'] == 'cross_script_page_dated_later'


def test_an_organisation_author_named_only_in_the_page_text_is_accepted_on_upload(page_judge):
    body = '<div class="article-body"><p>Presented by the Northfield Film Archive</p>' + ''.join(
        '<p>The archive restored the film from the original camera negative and records each repair.</p>'
        for _ in range(14)) + '</div>'
    html = ('<html><head><title>Restoring the northern films</title>'
            '<meta name="citation_title" content="Restoring the northern films">'
            '<meta name="citation_date" content="2020"><meta property="og:type" content="article"></head>'
            '<body><article><h1>Restoring the northern films</h1>' + body + '</article></body></html>').encode()
    page = _supplied(html, expected_title='Restoring the northern films', expected_author='Northfield Film Archive',
                     expected_year='2020')
    assert page.identity_confidence == 'high' and page_judge.calls == []


def test_a_han_byline_beside_the_title_credits_the_cited_author_on_upload(page_judge):
    page_judge.answer = {'same_work': 'yes', 'author': 'not_shown'}
    page = _supplied(_han_page(NEAR, people=(), byline='潘越'),
                     expected_title='Shuju quanli yu falv baohu zhidu yanjiou (Research on data rights)',
                     expected_author='Pan, Y.', expected_year='2016')
    assert page.identity_confidence == 'high' and len(page_judge.calls) == 1


def test_one_allowance_per_reference_across_routes_and_the_off_setting(page_judge, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_SCRIPT_PAGE_JUDGMENTS_PER_REFERENCE", 3)
    ask = lambda n: ask_model(title='Cited title', author='Writer, A.', year='2020', page_titles=[f'页面{n}'],
                              page_date='2021', page_text='text')
    token = source_resolver._ACTIVE_DISCOVERY_TRACE.set(recorder_from({}))
    try:
        answers = [ask(n) for n in range(4)]
        assert ask(0) is answers[0]          # the same page again: its first answer, no new call
    finally:
        source_resolver._ACTIVE_DISCOVERY_TRACE.reset(token)
    assert [a is not None for a in answers] == [True, True, True, False] and len(page_judge.calls) == 3
    monkeypatch.setattr(settings, "CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED", False)
    assert ask(9) is None and len(page_judge.calls) == 3


def test_every_route_asks_the_same_fields(page_judge):
    ask_model(title='Cited title', author='Writer, A.', year='2020', container_title='Northfield Review',
              page_titles=['页面'], page_date='2021', page_text='x' * 5000)
    reference, page = page_judge.calls[0]
    assert reference == {'title': 'Cited title', 'author': 'Writer, A.', 'year': '2020',
                         'container_title': 'Northfield Review'}
    assert page['titles'] == ['页面'] and page['date'] == '2021'


def test_a_cited_person_missing_from_the_page_stops_admission():
    assert model_admits({'same_work': 'yes', 'author': 'not_shown'}, 'Pan, Y.')
    assert not model_admits({'same_work': 'yes', 'author': 'differs'}, 'Pan, Y.')
    assert model_admits({'same_work': 'yes', 'author': 'differs'}, 'Northfield Ministry of Media')
    assert not model_admits({'same_work': 'unsure'}, 'Pan, Y.')
