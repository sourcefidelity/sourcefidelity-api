from dataclasses import replace

from app.services.verification_evidence import _PdfLayoutSpan, _byline_first_opening_roles
from app.services.verification_evidence import _pdf_structural_spans


def opening():
    def s(text,x,y,h):
        return _PdfLayoutSpan(0,0,len(text),text,x,y,x+180,y+h,600,800)
    return [s('doi: 10.1234/example',100,40,12),
            s('Jane Example\nExample University',100,180,25),
            s('A long research title\nin two lines',100,240,55),
            s('Abstract',100,325,12),s('Keywords',420,325,12),
            s('This is the abstract prose.',100,342,100),
            s('Topic\nContext',420,342,50),
            s('1. Introduction',100,500,14)]


def test_byline_first_roles_keep_columns_separate():
    assert _byline_first_opening_roles(0,opening()) == {
        1:'publication_metadata',2:'document_metadata',3:'abstract',
        4:'publication_metadata',5:'abstract',6:'publication_metadata'}


def test_no_doi_or_late_page_cannot_hide_body():
    assert not _byline_first_opening_roles(3,opening())
    assert not _byline_first_opening_roles(0,opening()[1:])


def test_missing_abstract_and_wrong_alignment_abstain():
    spans=opening();spans[3]=replace(spans[3],text='Methods')
    assert not _byline_first_opening_roles(0,spans)
    spans=opening();spans[2]=replace(spans[2],x0=300)
    assert not _byline_first_opening_roles(0,spans)


def test_ordinary_heading_size_is_not_article_title():
    spans=opening();spans[2]=replace(spans[2],y1=260)
    assert not _byline_first_opening_roles(0,spans)


def test_ambiguous_abstract_follower_not_classified():
    spans=opening();spans.append(replace(spans[5],text='Competing block'))
    result=_byline_first_opening_roles(0,spans)
    assert 5 not in result and 8 not in result


def test_top_margin_note_requires_substantive_text_and_adjacent_main_column():
    note=_PdfLayoutSpan(1,0,100,'1. This is a substantive note beside the main article text.',40,25,120,150,600,800)
    body=_PdfLayoutSpan(1,100,250,'An ordinary body paragraph discussing the research.',150,25,530,300,600,800)
    roles=_pdf_structural_spans({1:[note,body]})[1]
    assert any(r.start==0 and r.role=='citation_notes' for r in roles)
    assert not any(r.start==100 and r.role=='citation_notes' for r in roles)
    assert not any(r.role=='citation_notes' for r in _pdf_structural_spans({1:[note]})[1])
    short=replace(note,text='1. 2019')
    assert not any(r.role=='citation_notes' for r in _pdf_structural_spans({1:[short,body]})[1])
