from app.services.reference_consistency import inspect_media_type_context
from app.services.subject_identifier import bind_media_analysis_candidates
from app.services.schemas import ParsedReference
import pytest


@pytest.mark.parametrize('variant,expected', [
    ('play','submitted_type_disagrees'),('film','submitted_type_agrees'),
    ('multiple','ambiguous_reference_context'),('duplicate_id','ambiguous_reference_context'),
    ('missing','not_assessed'),('unknown','not_assessed'),('publication','not_assessed'),
    ('changed_body','invalid_binding'),('descriptor','submitted_type_disagrees'),
    ('unrecognized_suffix','not_assessed')])
def test_submitted_designation_context_never_resolves_identity(variant,expected):
    body='Example shapes the narrative.'
    bound=bind_media_analysis_candidates([dict(title='Example',passage=body,media_type='film',
        title_role='work_title',proposed_role='substantive_analysis')],body,[],require_title_role=True)
    c=bound.candidates[0]
    refs=[ParsedReference(reference_id='one',title='Example',raw_ref='Author. Example [Play].')]
    if variant=='film':refs[0].raw_ref='Author. Example [Film].'
    if variant=='descriptor':refs[0].title='Example [Play]'
    if variant=='unrecognized_suffix':refs[0].title='Example [Unspecified]'
    if variant=='multiple':refs.append(ParsedReference(reference_id='two',title='Example',raw_ref='Example [Film].'))
    if variant=='duplicate_id':refs.append(ParsedReference(reference_id='one',title='Different',raw_ref='Different [Film].'))
    if variant=='missing':refs=[]
    if variant=='unknown':c=c.model_copy(update={'media_type':'unknown'})
    if variant=='publication':c=c.model_copy(update={'title_role':'publication_title','media_type':'periodical'})
    if variant=='changed_body':body+=' changed'
    result=inspect_media_type_context(body=body,body_sha256=bound.body_sha256,candidate=c,references=refs)
    assert result['status']==expected
    assert not result['source_identity_assessed']
    assert not result['automatic_findings_enabled']
