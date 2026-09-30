import pytest
from app.services import edition_review_entry as entry
from app.services.personal_edition_answer import PersonalEditionAnswer, save_personal_answer, separate_work_signal
from app.services.edition_review_html import review_html
from tests.unit.test_edition_review_entry import review_env


@pytest.mark.parametrize('answer', ['yes', 'no'])
def test_simple_answer_roundtrip_no_grants(review_env, answer):
    session, storage, principal, source = review_env
    snapshot = entry.prepare_review(session, storage, principal, source.id, 'Synthetic Work (2004)')
    html = review_html(snapshot)
    assert 'value="yes"' in html and 'value="no"' in html
    assert 'textarea' not in html.split('<body>')[1] and 'type="checkbox"' not in html
    result = save_personal_answer(session, storage, principal, snapshot.id,
        PersonalEditionAnswer(snapshot_sha256=snapshot.snapshot_sha256, answer=answer))
    session.commit(); session.expire_all()
    assert result.payload['outcome'] == ('owner_confirmed_cited_edition' if answer == 'yes' else 'unverified')
    assert not result.payload['task_usability_granted']
    assert source.admission_state == 'needs_review'
    assert 'Saved answer:' in review_html(snapshot, result)
    with pytest.raises(entry.EditionReviewError, match='already'):
        save_personal_answer(session, storage, principal, snapshot.id,
            PersonalEditionAnswer(snapshot_sha256=snapshot.snapshot_sha256, answer=answer))


def test_translation_cannot_be_overridden(review_env):
    session, storage, principal, source = review_env
    snapshot = entry.prepare_review(session, storage, principal, source.id, 'Translation of Synthetic Work')
    assert snapshot.payload['separate_retrieval_required']
    assert 'data-answer' not in review_html(snapshot).split('<body>')[1]
    with pytest.raises(entry.EditionReviewError, match='separate retrieval'):
        save_personal_answer(session, storage, principal, snapshot.id,
            PersonalEditionAnswer(snapshot_sha256=snapshot.snapshot_sha256, answer='yes'))


def test_separate_work_signal():
    assert separate_work_signal('Translated by A. Person')
    assert separate_work_signal('Abridged edition')
    assert not separate_work_signal('Unabridged reprint')
