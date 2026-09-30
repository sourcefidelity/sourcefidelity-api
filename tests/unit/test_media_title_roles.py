import pytest
from app.services.subject_identifier import bind_media_analysis_candidates
from app.services.media_analysis_windows import plan_media_analysis_windows, summarize_media_analysis_windows


def proposal(**updates):
    return dict(title='Example Magazine', passage='Example Magazine frames the performer as independent.',
        media_type='periodical', title_role='publication_title', proposed_role='substantive_analysis') | updates


def test_publication_analysis_survives_without_an_article_identity():
    raw = proposal()
    result = bind_media_analysis_candidates([raw], raw['passage'], [], require_title_role=True)
    assert result.version == 'media-analysis-candidates-v3'
    assert result.status == 'bound_candidates'
    assert result.candidates[0].proposed_role == 'substantive_analysis'
    assert result.candidates[0].title_role == 'publication_title'
    body = raw['passage']
    plan = plan_media_analysis_windows(body, max_input_tokens=1500, max_windows=1)
    summary = summarize_media_analysis_windows(body, plan, {0: result.model_dump(mode='json')})
    assert summary['coverage_status'] == 'processed_all_windows'
    assert summary['candidates'][0]['title_role'] == 'publication_title'
    assert not summary['reference_absence_assessed']
    assert not summary['automatic_findings_enabled']


@pytest.mark.parametrize('changes', [dict(media_type='article'), dict(title_role='invented'),
    dict(title='Invented Article')])
def test_inconsistent_or_invented_roles_fail_closed(changes):
    raw = proposal(**changes)
    result = bind_media_analysis_candidates([raw], raw['passage'], [], require_title_role=True)
    assert result.status == 'invalid'


def test_live_role_required_but_legacy_role_not_inferred():
    raw = proposal(media_type='article')
    del raw['title_role']
    live = bind_media_analysis_candidates([raw], raw['passage'], [], require_title_role=True)
    assert live.status == 'invalid'
    legacy = bind_media_analysis_candidates([raw], raw['passage'], [])
    assert legacy.version == 'media-analysis-candidates-v2'
    assert legacy.candidates[0].title_role == 'uncertain'
