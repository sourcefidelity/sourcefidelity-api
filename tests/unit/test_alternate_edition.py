import pytest
from pydantic import ValidationError
from app.services.alternate_edition import AlternateEditionRecord


def record(**changes):
    data=dict(submitted_reference_sha256='a'*64,retrieved_representation_sha256='b'*64)
    data.update(changes)
    return AlternateEditionRecord(**data)


def evidence(*purposes):
    return tuple(dict(evidence_id=p,purpose=p,representation_sha256='b'*64,
                      passage_sha256='c'*64) for p in purposes)


def test_defaults_do_not_authorize_any_task():
    r=record()
    assert not r.quotation_usable and not r.paraphrase_usable
    assert r.cited_edition_locator_correspondence=='unverified'


def test_quote_and_retrieved_page_do_not_prove_cross_edition_mapping():
    r=record(work_identity='verified',relationship='verified_alternate_edition',
        claim_sha256='d'*64,quotation_usable=True,retrieved_locator_match='verified',
        evidence=evidence('work_identity','edition_relationship','quotation','retrieved_locator'))
    assert not r.paraphrase_usable
    assert r.cited_edition_locator_correspondence=='unverified'
    assert AlternateEditionRecord.model_validate_json(r.model_dump_json())==r


@pytest.mark.parametrize('changes',[
    dict(relationship='verified_alternate_edition'),
    dict(quotation_usable=True), dict(paraphrase_usable=True),
    dict(cited_edition_locator_correspondence='verified'),
    dict(exact_edition_match=True),
])
def test_unsubstantiated_promotions_rejected(changes):
    with pytest.raises(ValidationError):record(**changes)


def test_unknown_fields_cannot_add_global_equivalence():
    with pytest.raises(ValidationError):record(verified_equivalent=True)
