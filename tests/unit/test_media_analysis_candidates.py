import pytest
from app.services.subject_identifier import bind_media_analysis_candidates, identify_subject
from app.services.schemas import SubjectIdentification


@pytest.mark.parametrize('kind', ['film','tv_series','tv_episode','book','album','song',
    'radio_program','radio_episode','podcast','podcast_episode','play','poem','video',
    'video_game','article','other','unknown'])
def test_media_types_retain_exact_work_and_unassessed_semantics(kind):
    text='Example Work uses repetition to complicate the ending.'
    result=bind_media_analysis_candidates([dict(title='Example Work',passage=text,
        proposed_role='substantive_analysis',media_type=kind)],text,[])
    assert result.status=='bound_candidates'
    assert result.candidates[0].media_type==kind
    assert result.version=='media-analysis-candidates-v2'
    assert result.automatic_findings_enabled is False


def test_type_is_required_and_invalid_type_is_not_silently_coerced():
    base=dict(title='Work',passage='Work is mentioned.',proposed_role='uncertain')
    for item in [base,dict(base,media_type='invented')]:
        result=bind_media_analysis_candidates([item],base['passage'],[])
        assert result.status=='invalid' and not result.candidates
    assert SubjectIdentification().media_analysis_preflight.status=='not_requested'
    assert identify_subject('',[],collect_media_candidates=True).media_analysis_preflight.status=='unavailable'
    with pytest.raises(ValueError):
        identify_subject('',[],collect_media_candidates=True,collect_film_candidates=True)


def test_repeated_title_retains_complete_unique_context_and_all_occurrences():
    text='Work begins quietly. Work ends in silence.'
    raw=[dict(title='Work',passage=text,media_type='song',proposed_role='substantive_analysis')]
    result=bind_media_analysis_candidates(raw,text,[])
    assert result.status=='bound_candidates'
    c=result.candidates[0]
    assert c.title_occurrences==[(0,4),(21,25)]
    assert text[c.passage_start:c.passage_end]==text
    assert bind_media_analysis_candidates(raw,text+' '+text,[]).status=='invalid'


def test_generalized_candidates_use_single_existing_call(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    text='Example Song repeats the refrain to undermine the narrator.'
    def reply(**kwargs):
        calls.append(kwargs)
        return {'media_analysis_candidates':[dict(title='Example Song',passage=text,
            media_type='song',title_role='work_title',proposed_role='substantive_analysis')]}
    monkeypatch.setattr('app.services.subject_identifier.chat_completion_json',reply)
    monkeypatch.setattr('app.services.providers.get_provider_config',
                        lambda:SimpleNamespace(input_batch_tokens=10000))
    result=identify_subject(text,[],collect_media_candidates=True)
    assert len(calls)==1
    assert 'secondary scholarship' in calls[0]['system_prompt']
    assert result.media_analysis_preflight.status=='bound_candidates'
    assert result.film_analysis_preflight.status=='not_requested'
