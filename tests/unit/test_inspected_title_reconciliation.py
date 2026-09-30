import hashlib
from dataclasses import replace
import pytest
from app.services import source_validator as sv


@pytest.fixture
def case(monkeypatch):
    data = b'bounded PDF fixture'
    value = sv.ValidationResult(False, 'medium', 'uncertain', 'digital', 'Needs review')
    value.source_inspection = dict(version='source-inspection-v3', status='complete',
        identity='same_work', representation_role='source_text', completeness='not_established',
        content_sha256=hashlib.sha256(data).hexdigest(), decision_applied=False,
        observations=[dict(field='title', quote='Storytelling in the New Hollywood')],
        differences=[dict(field='title', kind='typographic')])
    expected = dict(expected_title='Storytelling in the New Holywood',
                    expected_author='Thompson', expected_year='1999')
    monkeypatch.setattr(sv, '_has_prominent_front_title_support', lambda *_: True)
    monkeypatch.setattr(sv, '_validate_retrieved_pdf_deterministic',
                        lambda *a, **k: replace(value, identity_confidence='high', source_inspection=None))
    return data, value, expected


def test_reconciliation_reuses_validator_without_granting_completeness(case):
    data, value, expected = case
    out = sv._reconcile_inspected_title(data, value, expected)
    assert out.identity_confidence == 'high'
    assert not out.accept and out.completeness == 'uncertain'
    assert out.source_inspection['decision_applied']
    assert not value.source_inspection['decision_applied']
    assert expected['expected_title'].endswith('Holywood')


@pytest.mark.parametrize('block', ['hash','role','warning','year','doi','isbn','edition','title','prominence','quality'])
def test_reconciliation_abstains_on_unaccepted_conditions(case, monkeypatch, block):
    data, value, expected = case
    finding = value.source_inspection
    if block == 'hash': finding['content_sha256'] = '0'*64
    if block == 'role': finding['representation_role'] = 'review'
    if block == 'warning': finding['completeness'] = 'warning_found'
    if block == 'year': expected['expected_year'] = None
    if block in {'doi','isbn'}: expected['expected_'+block] = 'identifier'
    if block == 'edition': finding['differences'][0]['field'] = 'edition'
    if block == 'title': finding['observations'][0]['quote'] = 'A Completely Different Title'
    if block == 'prominence': monkeypatch.setattr(sv, '_has_prominent_front_title_support', lambda *_: False)
    if block == 'quality': monkeypatch.setattr(sv, '_validate_retrieved_pdf_deterministic',
        lambda *a, **k: replace(value, identity_confidence='high', completeness='complete'))
    assert sv._reconcile_inspected_title(data, value, expected) is value


def test_insufficient_observations_allow_inspection_but_not_automatic_promotion(monkeypatch):
    value = sv.ValidationResult(False, 'rejected', 'skipped', 'digital', 'Insufficient',
                               reason_code='identity_insufficient_observations')
    monkeypatch.setattr(sv, '_validate_retrieved_pdf_deterministic', lambda *a, **k: value)
    calls = []
    out = sv.validate_retrieved_pdf(b'pdf', inspection_provider=lambda *_: calls.append(True) or
                                  {'status':'complete','identity':'same_work','decision_applied':False})
    assert calls == [True] and not out.accept and out.identity_confidence == 'rejected'


def test_hard_rejection_never_calls_inspector(monkeypatch):
    value = sv.ValidationResult(False, 'rejected', 'skipped', 'digital', 'Type conflict')
    monkeypatch.setattr(sv, '_validate_retrieved_pdf_deterministic', lambda *a, **k: value)
    def forbidden(*_): raise AssertionError('Must not inspect')
    out = sv.validate_retrieved_pdf(b'pdf', inspection_provider=forbidden)
    assert out.source_inspection is None and not out.accept


def test_insufficient_front_matter_rechecks_completeness(case, monkeypatch):
    data, value, expected = case
    value.identity_confidence = 'rejected'
    value.reason_code = 'identity_insufficient_observations'
    value.completeness = 'skipped'
    monkeypatch.setattr(sv, '_validate_retrieved_pdf_deterministic', lambda *a, **k:
        replace(value, identity_confidence='high', completeness='complete', accept=True))
    out = sv._reconcile_inspected_title(data, value, expected)
    assert out.accept and out.completeness == 'complete'


def test_late_book_title_uses_existing_fallback(case, monkeypatch):
    data, value, expected = case
    expected['expected_source_kind'] = 'monograph'
    monkeypatch.setattr(sv, '_has_prominent_front_title_support', lambda *_: False)
    calls = []
    monkeypatch.setattr(sv, '_late_book_title_identity', lambda *a: calls.append(a) or True)
    assert sv._reconcile_inspected_title(data, value, expected).identity_confidence == 'high'
    assert len(calls) == 1


@pytest.mark.parametrize('title', ['Storytelling in the New Hollywood 2', 'Storytelling in the Old Hollywood'])
def test_numbers_and_substantive_title_changes_abstain(case, title):
    data, value, expected = case
    value.source_inspection['observations'][0]['quote'] = title
    assert sv._reconcile_inspected_title(data, value, expected) is value


@pytest.mark.parametrize('author,allowed', [('Thompson',True),('Another Author',False)])
def test_redundant_model_difference_requires_independent_exact_agreement(case, author, allowed):
    data, value, expected = case
    value.source_inspection['differences'].append(dict(field='author',kind='typographic'))
    value.source_inspection['observations'].append(dict(field='author',quote=author))
    out=sv._reconcile_inspected_title(data,value,expected)
    assert bool(out.source_inspection.get('decision_applied')) is allowed
