import pytest
from app.services.verification_evidence import (
    _SourcePage, _SourceStructuralSpan, _PassageCandidate, _following_body_context,
)


def fixture(next_text='However, the cobalt dates remain uncertain because the archive was incomplete.', gap='\n', role='body_prose', index=0):
    first='The archive establishes cobalt dates.'
    text=first+gap+next_text
    page=_SourcePage(index=index,label='1',text=text,structural_spans=(
        _SourceStructuralSpan(start=0,end=len(first),role='body_prose'),
        _SourceStructuralSpan(start=len(first)+len(gap),end=len(text),role=role)))
    anchor=_PassageCandidate(page_index=0,page_label='1',start=0,end=len(first),
        text=first,method='lexical_overlap',score=.8,passage_role='body_prose')
    return page,[(anchor,{'candidate_lexical'})]


def test_following_window_preserves_exact_text_and_anchor():
    page,ranked=fixture()
    candidate,key=_following_body_context([page],ranked,query_text='The archive establishes cobalt dates.')
    assert candidate.text==page.text[candidate.start:candidate.end]
    assert key==(0,0,ranked[0][0].end)
    assert candidate.method=='following_body_context'


@pytest.mark.parametrize('kwargs',[
    {'gap':'\n\n\n'}, {'gap':'x'}, {'role':'citation_notes'},
    {'role':'reference_list'}, {'index':1},
    {'next_text':'2. Results\nThe archive establishes cobalt dates.'},
    {'next_text':'Unrelated flowering species grow in distant woodland.'},
    {'next_text':'However, the archive dates remain uncertain because the'},
])
def test_unsafe_or_unconnected_continuation_abstains(kwargs):
    page,ranked=fixture(**kwargs)
    assert _following_body_context([page],ranked,query_text='The archive establishes cobalt dates.') is None
