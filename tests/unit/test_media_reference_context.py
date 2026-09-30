import hashlib
import pytest
from app.services.subject_identifier import bind_media_analysis_candidates
from app.services.schemas import InTextCitation, ParsedReference
from app.services.reference_consistency import inspect_media_reference_context


def test_song_context_reuses_bound_film_citation_without_omission_conclusion():
    text='The Song performance frames the character (Director, 1946).'
    candidate=bind_media_analysis_candidates([dict(title='Song',passage=text,
        proposed_role='substantive_analysis',media_type='song')],text,[]).candidates[0]
    reference=ParsedReference(reference_id='film',author='Director',year='1946',title='Example',
        raw_ref='Director. (1946). Example [Film]. Studio.',source_kind='traditional_media',
        source_kind_confidence='high')
    citation=InTextCitation(text=text,reference_ids=['film'],citation_marker='(Director, 1946)',
        passage_start=0,passage_end=len(text))
    args=dict(body=text,body_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate=candidate,
              references=[reference],citations=[citation])
    result=inspect_media_reference_context(**args)
    assert result['film_reference_ids']==['film']
    assert result['status']=='film_reference_context_available'
    assert not result['source_use_assessed'] and not result['reference_absence_assessed']
    assert not result['automatic_findings_enabled']
    assert inspect_media_reference_context(**(args|{'body_sha256':'0'*64}))['status']=='invalid_binding'
    for changes in [{'link_status':'ambiguous'},{'candidate_reference_ids':['other']},
                    {'text':'changed text'},{'citation_marker':'(Other, 1946)'}]:
        got=inspect_media_reference_context(**(args|{'citations':[citation.model_copy(update=changes)]}))
        assert not got['film_reference_ids']
    assert inspect_media_reference_context(**(args|{'references':[]}))['status']=='not_assessed'


@pytest.mark.parametrize('case', ['duplicate', 'review', 'low_confidence', 'not_film',
    'unlinked_member', 'outside_passage', 'two_films'])
def test_context_requires_unique_eligible_reference_and_contained_marker(case):
    from app.services.schemas import CitationMarkerMember

    passage = 'Song frames the performance (Director, 1946).'
    body = passage + ' Another film is cited (Other, 1950).'
    candidate = bind_media_analysis_candidates([dict(title='Song', passage=passage,
        proposed_role='substantive_analysis', media_type='song')], body, []).candidates[0]
    ref = ParsedReference(reference_id='film', raw_ref='Director. (1946). Work [Film].',
        source_kind='traditional_media', source_kind_confidence='high')
    refs = [ref]
    marker = '(Director, 1946)'
    start = body.index(marker)
    ids = ['film']
    linked = ['film']
    if case == 'duplicate':
        refs.append(ref.model_copy(update={'raw_ref': 'Other. (1950). Other [Film].'}))
    elif case == 'review':
        refs = [ref.model_copy(update={'needs_review': True})]
    elif case == 'low_confidence':
        refs = [ref.model_copy(update={'source_kind_confidence': 'medium'})]
    elif case == 'not_film':
        refs = [ref.model_copy(update={'raw_ref': 'Director. Work [Album].'})]
    elif case == 'unlinked_member':
        linked = ['other']
    elif case == 'outside_passage':
        marker = '(Other, 1950)'
        start = body.index(marker)
    elif case == 'two_films':
        refs.append(ref.model_copy(update={'reference_id': 'second'}))
        ids = linked = ['film', 'second']
    citation = InTextCitation(text=body, passage_start=0, passage_end=len(body),
        reference_ids=linked, citation_markers=[CitationMarkerMember(text=marker,
            local_start=start, local_end=start+len(marker), reference_ids=ids)])
    result = inspect_media_reference_context(body=body,
        body_sha256=hashlib.sha256(body.encode()).hexdigest(), candidate=candidate,
        references=refs, citations=[citation])
    assert result['status'] == ('multiple_film_contexts' if case == 'two_films' else 'not_assessed')
    assert result['film_reference_ids'] == (['film', 'second'] if case == 'two_films' else [])
    assert not result['source_use_assessed']
    assert not result['reference_absence_assessed']
    assert not result['automatic_findings_enabled']
