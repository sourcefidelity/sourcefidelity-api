from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.services.alternate_edition import AlternateEditionRecord
from app.services.evidence_report import _render_alternate_edition


def reviewed(decision='confirmed'):
    record=AlternateEditionRecord(submitted_reference_sha256='a'*64,
        retrieved_representation_sha256='b'*64,work_identity='verified',
        relationship='verified_alternate_edition',evidence=tuple(
            dict(evidence_id=p,purpose=p,representation_sha256='b'*64,passage_sha256='c'*64)
            for p in ('work_identity','edition_relationship')))
    value=record.model_dump(mode='json')
    value['human_review']=dict(reviewer_id='reviewer-1',reviewed_at=datetime.now(timezone.utc).isoformat(),
        decision=decision,record_sha256=record.review_input_sha256(),
        reviewed_evidence_ids=['work_identity','edition_relationship'],notes='Private review note.')
    return value


def test_confirmed_review_roundtrip_and_presentation():
    record=AlternateEditionRecord.model_validate(reviewed())
    assert record.human_verified
    assert AlternateEditionRecord.model_validate_json(record.model_dump_json()).human_verified
    html=_render_alternate_edition(record.model_dump(mode='json'))
    assert 'Human review confirmed' in html
    assert 'Private review note' not in html and 'reviewer-1' not in html
    assert not record.quotation_usable and not record.paraphrase_usable


@pytest.mark.parametrize('decision',['uncertain','rejected'])
def test_unconfirmed_reviews_remain_neutral(decision):
    record=AlternateEditionRecord.model_validate(reviewed(decision))
    assert not record.human_verified
    assert 'Human review confirmed' not in _render_alternate_edition(record.model_dump(mode='json'))


@pytest.mark.parametrize('field',['claim_sha256','submitted_reference_sha256','retrieved_representation_sha256','limitations'])
def test_changed_record_invalidates_review(field):
    value=reviewed()
    value[field]=['New limitation'] if field=='limitations' else 'd'*64
    with pytest.raises(ValidationError):AlternateEditionRecord.model_validate(value)


def test_partial_evidence_review_is_rejected():
    value=reviewed();value['human_review']['reviewed_evidence_ids'].pop()
    with pytest.raises(ValidationError):AlternateEditionRecord.model_validate(value)


def test_naive_timestamp_is_rejected():
    value=reviewed();value['human_review']['reviewed_at']='2026-09-09T12:00:00'
    with pytest.raises(ValidationError):AlternateEditionRecord.model_validate(value)
