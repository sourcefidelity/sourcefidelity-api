import hashlib
from datetime import datetime, timezone

import pytest

from app.services.schemas import ParsedReference
from app.services.submitted_links import (
    ACTIVE_LINKS, LinkRequest, initial_observations, identity_observed,
    page_observed, bind_authorized_admission, request_url, safe_origin,
    link_not_visited,
    source_validation_observed,
)
from app.services.submitted_link_display import render_link_checks, render_link_summary, saved_link_view


def observed():
    ref = ParsedReference(reference_id='r', url='https://example.org/paper', title='A source')
    rows = initial_observations(ref)
    digest = hashlib.sha256(b'response').hexdigest()
    now = datetime.now(timezone.utc)
    rows[0].requests = [LinkRequest(started_at=now, completed_at=now,
        request_sha256=rows[0].request_sha256, response_sha256=digest,
        http_status=200, outcome='response', destination_origin='https://example.org')]
    rows[0].state = 'observed'
    return ref, rows, digest


def test_homepage_requires_bound_structural_observation_not_just_root_url():
    from app.services.submitted_links import is_site_homepage
    from app.services.submitted_link_display import reference_link_findings
    html='<title>Organization</title><meta property="og:type" content="website"><nav>Publications</nav>'
    assert is_site_homepage(html, 'https://example.org/')
    assert is_site_homepage(html, 'https://example.org/jp/ja/index.html')
    assert not is_site_homepage(html+'<article>A report</article>', 'https://example.org/')
    assert not is_site_homepage(html, 'https://example.org/report')
    assert not is_site_homepage('<title>Organization</title>', 'https://example.org/')
    ref, rows, digest = observed()
    source={ref.reference_id:dict(url=ref.url,raw_reference=ref.raw_ref)}
    assert not reference_link_findings([r.model_dump(mode='json') for r in rows], source)
    token=ACTIVE_LINKS.set(rows)
    try: page_observed(ref.url, digest, 'site_homepage')
    finally: ACTIVE_LINKS.reset(token)
    findings=reference_link_findings([r.model_dump(mode='json') for r in rows], source)
    assert len(findings)==1 and findings[0]['link_outcome']=='site_homepage'
    assert 'home page' in findings[0]['finding']


def test_original_home_address_specificity_is_separate_from_http_failure():
    from app.services.submitted_link_display import reference_link_findings
    source=dict(url='https://example.org',raw_reference='Agency (2020). Communications Monitoring Report. https://example.org',
                title='Communications Monitoring Report',author='Agency',source_kind='webpage')
    findings=reference_link_findings([],{'r':source})
    assert len(findings)==1 and findings[0]['link_outcome']=='website_level_address'
    assert 'dead' not in findings[0]['finding']
    assert not reference_link_findings([],{'r':{**source,'title':'Agency official website'}})
    assert not reference_link_findings([],{'r':{**source,'raw_reference':'No submitted URL'}})
    assert not reference_link_findings([],{'r':{**source,'url':'https://example.org/report'}})


@pytest.mark.parametrize('reason,wording', [
    ('safety_rejected', 'did not pass file-safety checks'),
    ('safety_unavailable', 'File-safety checking was unavailable'),
    ('completeness_uncertain', 'its completeness is uncertain'),
    ('completeness_rejected', 'copy was judged incomplete'),
])
def test_bound_validation_limitations_survive_reload(reason, wording):
    ref, rows, digest = observed()
    token = ACTIVE_LINKS.set(rows)
    try:
        source_validation_observed(ref.url, digest, reason)
    finally:
        ACTIVE_LINKS.reset(token)
    saved = [r.model_dump(mode='json') for r in rows]
    projected = saved_link_view([ref], {'submitted_link_observations': saved})
    assert wording in render_link_checks(projected)
    assert projected[0]['requests'][0]['admitted_content'] == 'not_assessed'
    saved[0]['requests'][0]['source_validation_sha256'] = 'a' * 64
    assert wording not in render_link_checks(saved)


@pytest.mark.parametrize('change', ['url', 'hash', 'status', 'outcome', 'reason'])
def test_validation_limitation_rejects_unbound_or_invalid_input(change):
    ref, rows, digest = observed()
    if change == 'status':
        rows[0].requests[0].http_status = 404
    if change == 'outcome':
        rows[0].requests[0].outcome = 'safety_refused'
    token = ACTIVE_LINKS.set(rows)
    try:
        source_validation_observed('https://other.example/' if change == 'url' else ref.url,
            'a' * 64 if change == 'hash' else digest,
            'arbitrary' if change == 'reason' else 'safety_rejected')
    finally:
        ACTIVE_LINKS.reset(token)
    assert rows[0].requests[0].source_validation == 'not_assessed'


@pytest.mark.parametrize('url,expected', [
    ('https://example.org/private?token=SECRET#private', 'https://example.org'),
    ('https://user:SECRET@example.org/a', None), ('http://127.0.0.1/a', None),
    ('http://host.internal/a', None), ('javascript:alert(1)', None),
])
def test_origin_redaction(url, expected):
    assert safe_origin(url) == expected


@pytest.mark.parametrize('value,kind,expected', [
    (' www.example.org/a ', 'url', 'https://www.example.org/a'),
    ('https://doi.org/10.1234/a', 'doi', 'https://doi.org/10.1234/a'),
    ('doi:10.1234/a', 'doi', 'https://doi.org/10.1234/a'),
])
def test_request_normalization(value, kind, expected):
    assert request_url(value, kind) == expected


@pytest.mark.parametrize('wrong_url,wrong_content', [(False, False), (True, False), (False, True)])
def test_identity_and_admission_require_same_response(wrong_url, wrong_content):
    ref, rows, digest = observed()
    token = ACTIVE_LINKS.set(rows)
    try:
        identity_observed('https://other.example/' if wrong_url else ref.url,
            'a'*64 if wrong_content else digest, 'bibliographic_identity_confirmed', ['title', 'author'])
    finally:
        ACTIVE_LINKS.reset(token)
    metadata = {'durable_admission': {'state': 'accepted', 'representation_id': 'rep'},
        'web_fetch_diagnostic': {'reason': 'bibliographic_identity_confirmed', 'observed_content_sha256': digest},
        'requested_url_sha256': rows[0].request_sha256, 'accepted_representation_sha256': 'b'*64}
    projected = bind_authorized_admission([r.model_dump(mode='json') for r in rows], metadata, 'rep')
    request = projected[0]['requests'][0]
    assert (request['admitted_content'] == 'admitted') == (not wrong_url and not wrong_content)
    assert 'SECRET' not in render_link_checks(projected)


def test_http_success_and_soft_page_do_not_imply_identity():
    ref, rows, digest = observed()
    token = ACTIVE_LINKS.set(rows)
    try:
        page_observed(ref.url, digest, 'page_title_mismatch_unconfirmed')
    finally:
        ACTIVE_LINKS.reset(token)
    html = render_link_checks([r.model_dump(mode='json') for r in rows])
    assert 'HTTP 200' in html and 'identity was not assessed' in html
    assert 'this alone does not establish a different work' in html
    assert 'confirmed.' not in html


def test_historical_reports_unchanged_and_snapshot_mismatch_unknown():
    ref, rows, _ = observed()
    assert saved_link_view([ref], {}) == []
    aggregate = {'submitted_link_observations': [r.model_dump(mode='json') for r in rows]}
    assert saved_link_view([ref], aggregate)[0]['state'] == 'observed'
    changed = ref.model_copy(update={'title': 'Different reference'})
    assert saved_link_view([changed], aggregate)[0]['state'] == 'historical_unknown'
    assert 'reference-error counts' in render_link_summary(aggregate['submitted_link_observations'])


def test_malicious_saved_origin_never_rendered():
    _, rows, _ = observed()
    value = rows[0].model_dump(mode='json')
    value['requests'][0]['destination_origin'] = 'https://example.org/<script>SECRET</script>'
    assert 'SECRET' not in render_link_checks([value])


def test_legacy_request_binding_remains_legacy():
    ref = ParsedReference(reference_id='r', url='www.example.org/a')
    rows = initial_observations(ref, legacy=True)
    aggregate = {'submitted_link_observations': [r.model_dump(mode='json') for r in rows]}
    projected = saved_link_view([ref], aggregate)
    assert projected[0]['version'] == 'submitted-link-v1'
    assert projected[0]['request_sha256'] == rows[0].request_sha256


@pytest.mark.parametrize('reason', ['authorized_reuse', 'capability_disabled',
    'library_locator_only', 'candidate_budget_exhausted', 'unsupported_source_type'])
def test_explicit_no_visit_reason_cannot_replace_observed_request(reason):
    ref, rows, _ = observed()
    unvisited = initial_observations(ref)
    token = ACTIVE_LINKS.set(rows + unvisited)
    try:
        link_not_visited(ref.url, reason)
    finally:
        ACTIVE_LINKS.reset(token)
    assert rows[0].state == 'observed' and rows[0].not_checked_reason == 'not_recorded'
    assert unvisited[0].not_checked_reason == reason
    assert 'Not checked' in render_link_checks([unvisited[0].model_dump(mode='json')])


@pytest.mark.parametrize('tail,uncertain', [(' y-shirky-2027393.html', True),
    ('\narticle/part-two', True), (' Retrieved yesterday.', False), ('', False)])
def test_possible_wrapped_url_warns_without_repair_or_request(tail, uncertain):
    url = 'https://example.org/part'
    ref = ParsedReference(reference_id='r', url=url, raw_ref='Writer. ' + url + tail)
    row = initial_observations(ref)[0]
    assert (row.address_extraction == 'possible_truncation') is uncertain
    assert row.state == 'not_checked' and not row.requests
    assert ref.url == url
    assert ('extracted address may be incomplete' in render_link_checks([row.model_dump(mode='json')])) is uncertain
