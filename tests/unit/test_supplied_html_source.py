"""Qualification of an operator-supplied saved publisher page.

The supplied bytes are parsed offline and only the extracted article text may
become a representation. These checks fix that boundary: the shared identity
standard still rejects wrong works, and nothing is fetched or executed.
"""

import hashlib

import pytest

from app.services.source_type import SourceKindAssessment
from app.services.supplied_html_source import (
    SuppliedHtmlRejected,
    looks_like_html_document,
    qualify_supplied_html,
)

BODY = (
    '<p>The reviewed study compares documented archival practice across three '
    'national collections and records each difference in its own appendix.</p>'
)


def page(
    *,
    title='A Reviewed Work. Ada Perez. Oxford: Oxford University Press, 2016. '
          'Pp. xii+204. | Journal of Records: Vol 94, No 2',
    author='Morgan Chen',
    doi='10.1086/695968',
    date='2018-05-01',
    site='Journal of Records',
    body=BODY * 14,
    extra_head='',
):
    return (
        '<html><head>'
        f'<meta property="og:title" content="{title}">'
        f'<meta property="og:site_name" content="{site}">'
        '<meta property="og:type" content="article">'
        f'<meta name="citation_journal_title" content="{site}">'
        f'<meta name="dc.creator" content="{author}">'
        f'<meta name="dc.identifier" content="{doi}">'
        f'<meta name="dc.date" content="{date}">'
        f'{extra_head}'
        '</head><body><article><div class="article-body">'
        f'{body}'
        '</div></article></body></html>'
    ).encode('utf-8')


EXPECTED = dict(
    expected_title='A Reviewed Work. Ada Perez. Oxford: Oxford University Press, '
                   '2016. Pp. xii+204',
    expected_author='Chen, M',
    expected_year='2018',
    expected_doi='10.1086/695968',
    expected_source_kind=SourceKindAssessment(kind='journal_article', confidence='high'),
)


def test_supplied_page_qualifies_and_binds_its_hashes():
    content = page()
    qualified = qualify_supplied_html(content, **EXPECTED)
    assert qualified.html_sha256 == hashlib.sha256(content).hexdigest()
    assert qualified.extracted_sha256 == hashlib.sha256(
        qualified.article_text.encode('utf-8')
    ).hexdigest()
    assert qualified.completeness['verdict'] == 'complete'
    assert qualified.identity_confidence == 'high'
    assert {c['field_name'] for c in qualified.identity_comparisons} == {
        'title', 'author', 'year', 'doi'
    }
    assert all(c['outcome'] == 'agreement' for c in qualified.identity_comparisons)


def test_evidence_records_the_supplied_provenance():
    evidence = qualify_supplied_html(page(), **EXPECTED).as_evidence()
    assert evidence['supplied_document'] is True
    assert evidence['network_access'] == 'none'
    assert evidence['retained_representation'] == 'extracted_plain_text'


def test_active_content_is_recorded_and_never_retained():
    content = page(extra_head='<script src="https://cdn.example/app.js"></script>')
    qualified = qualify_supplied_html(content, **EXPECTED)
    assert 'script element' in qualified.active_content_markers
    # Only extracted text becomes the representation.
    assert '<script' not in qualified.article_text
    assert 'cdn.example' not in qualified.article_text


@pytest.mark.parametrize('override,reason', [
    (dict(expected_doi='10.1086/000000'), 'bibliographic_fields_conflict'),
    (dict(expected_author='Nguyen, T'), 'bibliographic_fields_conflict'),
    (dict(expected_title='An unrelated study of rainfall'), 'bibliographic_fields_conflict'),
    (dict(expected_year='1999'), 'bibliographic_fields_conflict'),
    (dict(expected_source_kind=SourceKindAssessment(kind='dataset', confidence='high')),
     'source_kind_unconfirmed'),
])
def test_wrong_work_is_rejected(override, reason):
    with pytest.raises(SuppliedHtmlRejected) as exc:
        qualify_supplied_html(page(), **{**EXPECTED, **override})
    assert exc.value.reason_code == reason


def test_truncated_review_title_still_conflicts():
    """The repair produces the right title; it does not weaken the comparison."""
    with pytest.raises(SuppliedHtmlRejected) as exc:
        qualify_supplied_html(page(), **{**EXPECTED, 'expected_title': 'A Reviewed Work'})
    assert exc.value.reason_code == 'bibliographic_fields_conflict'


def test_unreadable_page_is_rejected():
    with pytest.raises(SuppliedHtmlRejected) as exc:
        qualify_supplied_html(page(body='<p>Short.</p>'), **EXPECTED)
    assert exc.value.reason_code == 'readable_text_unavailable'


def test_partial_article_body_is_rejected():
    content = page(extra_head='<link rel="next" href="page-2.html">')
    with pytest.raises(SuppliedHtmlRejected) as exc:
        qualify_supplied_html(content, **EXPECTED)
    assert exc.value.reason_code == 'article_body_incomplete'


def test_non_html_upload_is_rejected():
    with pytest.raises(SuppliedHtmlRejected) as exc:
        qualify_supplied_html(b'%PDF-1.4\ntrailer', **EXPECTED)
    assert exc.value.reason_code == 'not_an_html_document'


def test_qualification_never_opens_the_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('supplied-page qualification must not fetch')

    import socket

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    assert qualify_supplied_html(page(), **EXPECTED).completeness['verdict'] == 'complete'


@pytest.mark.parametrize('content,media_type,expected', [
    (b'<!doctype html><html></html>', None, True),
    (b'<html><body>x</body></html>', None, True),
    (b'%PDF-1.7\n%\xe2\xe3', 'text/html', False),
    (b'plain words only', 'text/html', True),
    (b'plain words only', 'application/octet-stream', False),
])
def test_html_upload_detection(content, media_type, expected):
    assert looks_like_html_document(content, media_type) is expected
