"""A long abstract must be assessable, and one malformed reply must not end it.

Measured over 136 stored abstract/claim pairs on 2026-09-21: the model saw
text[:1_400] while admission required equality with the untruncated abstract,
so 35 assessments (26%) were paid for and discarded, and a further 6 (4.4%)
were lost to a single malformed response.
"""
from types import SimpleNamespace

import pytest

from app.services import passage_relevance as pr
from app.services.verification_evidence import ClaimEvidence

CLAIM = 'The study reports falling participation (Smith, 2020).'
SPAN = 'participation fell in every region studied'


def claim_evidence():
    return ClaimEvidence(claim_id='c', paper_version_id='p', text=CLAIM, reference_ids=['r1'],
                         citation_marker='(Smith, 2020)', citation_marker_type='parenthetical',
                         passage_start=0, passage_end=len(CLAIM), granularity='citation_unit')


def reply(**scope_overrides):
    scope = {'relevance': 'generally_relevant', 'confidence': 'high', 'discrepancy': None,
             'abstract_span': SPAN, 'claim_span': 'falling participation',
             'rationale': 'Same subject.', 'topic_relation': 'overlapping',
             'broad_subject_relation': 'compatible', 'plausible_connection': 'present',
             'subject_comparison': 'Both concern participation.',
             'stated_scope_conflict': 'absent', 'scope_dimension': None,
             'source_scope': None, 'claim_scope': None}
    scope.update(scope_overrides)
    return {'assessments': [{'passage_id': 'abstract', 'relevance': 'relevant',
                             'confidence': 'high', 'evidence_role': 'source_own_claim_or_finding',
                             'rationale': 'Directly addresses participation.'}],
            'scope': scope, 'related_excerpt': ''}


def long_abstract(length):
    filler = 'This article examines participation across several regions. '
    body = (filler * (length // len(filler) + 2))[:length - len(SPAN) - 2]
    return f'{body}{SPAN}.'


def test_an_abstract_over_the_old_passage_cap_is_still_assessed(monkeypatch):
    abstract = long_abstract(3_000)
    assert len(abstract) > pr.MAX_RELEVANCE_PASSAGE_CHARACTERS
    assert len(abstract) <= pr.MAX_ABSTRACT_SCOPE_CHARACTERS
    monkeypatch.setattr(pr, 'chat_completion_json', lambda *a, **k: reply())
    result = pr.assess_abstract_relevance(claim_evidence(), abstract)
    assert result['status'] == 'complete'
    scope = result['scope_assessment']
    assert scope['status'] == 'complete'
    assert scope['abstract_truncated'] is False


def test_an_abstract_beyond_the_abstract_cap_is_recorded_as_truncated(monkeypatch):
    abstract = long_abstract(pr.MAX_ABSTRACT_SCOPE_CHARACTERS + 400)
    monkeypatch.setattr(pr, 'chat_completion_json', lambda *a, **k: reply(
        relevance='apparent_mismatch', topic_relation='disjoint',
        broad_subject_relation='incompatible', plausible_connection='absent',
        discrepancy='different_subject', abstract_span='This article examines participation'))
    result = pr.assess_abstract_relevance(claim_evidence(), abstract)
    scope = result['scope_assessment']
    assert scope['abstract_truncated'] is True
    # Truncated input still cannot authorize a reader-visible warning.
    assert scope['attention'] is False
    assert scope['relevance'] == 'uncertain'


def test_one_malformed_response_is_retried_rather_than_discarded(monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {'assessments': [], 'scope': None, 'unexpected_field': True}
        return reply()

    monkeypatch.setattr(pr, 'chat_completion_json', flaky)
    result = pr.assess_abstract_relevance(claim_evidence(), 'Participation fell. ' + SPAN + '.')
    assert result['status'] == 'complete'
    assert result['response_attempts'] == 2
    assert len(calls) == 2


def test_the_retry_is_bounded_and_reports_the_last_failure(monkeypatch):
    calls = []

    def always_bad(*args, **kwargs):
        calls.append(1)
        return {'assessments': [], 'scope': None, 'unexpected_field': True}

    monkeypatch.setattr(pr, 'chat_completion_json', always_bad)
    result = pr.assess_abstract_relevance(claim_evidence(), 'Participation fell.')
    assert result['status'] == 'not_assessed'
    assert result['failure_category'] == 'schema_validation_failed'
    assert result['response_attempts'] == pr._ABSTRACT_RESPONSE_ATTEMPTS
    assert len(calls) == pr._ABSTRACT_RESPONSE_ATTEMPTS


def test_a_dropped_passage_id_is_also_retried_and_keeps_its_category(monkeypatch):
    calls = []

    def wrong_id(*args, **kwargs):
        calls.append(1)
        payload = reply()
        payload['assessments'][0]['passage_id'] = 'p001'
        return payload

    monkeypatch.setattr(pr, 'chat_completion_json', wrong_id)
    result = pr.assess_abstract_relevance(claim_evidence(), 'Participation fell.')
    assert result['failure_category'] == 'invalid_passage_ids'
    assert len(calls) == pr._ABSTRACT_RESPONSE_ATTEMPTS


def test_an_unfamiliar_dimension_label_does_not_destroy_the_assessment(monkeypatch):
    """Measured: the model answered "geography" and the whole judgment was lost."""
    monkeypatch.setattr(pr, 'chat_completion_json', lambda *a, **k: reply(
        stated_scope_conflict='present', scope_dimension='geography',
        source_scope='Australia', claim_scope='Hollywood'))
    result = pr.assess_abstract_relevance(claim_evidence(), 'Participation fell. ' + SPAN + '.')
    assert result['status'] == 'complete'
    assert result['scope_assessment']['scope_dimension'] == 'geography'


def test_an_unknown_scope_key_is_ignored_rather_than_fatal(monkeypatch):
    payload = reply()
    payload['scope']['scope_conflict_explanation'] = 'Surplus commentary.'
    payload['top_level_surprise'] = True
    monkeypatch.setattr(pr, 'chat_completion_json', lambda *a, **k: payload)
    result = pr.assess_abstract_relevance(claim_evidence(), 'Participation fell. ' + SPAN + '.')
    assert result['status'] == 'complete'
    assert result['scope_assessment']['status'] == 'complete'
    assert 'scope_conflict_explanation' not in result['scope_assessment']
