from app.services.verification_evidence import _SourcePage,_source_blocks

def test_unparenthesized_year_bibliography_and_continuation_are_excluded():
    pages=[_SourcePage(index=0,label='1',text='References\nSmith, J. 2020. A Work.\n'),
           _SourcePage(index=1,label='2',text='Jones, A.B. 2021. Another Work.\n'),
           _SourcePage(index=2,label='3',text='Chapter Three\nOrdinary source discussion resumes here.')]
    blocks=_source_blocks(pages)
    assert all(role=='reference_list' for p,_,_,_,role in blocks if p.index<2)
    assert all(role!='reference_list' for p,_,_,_,role in blocks if p.index==2)

def test_toc_references_heading_without_entries_does_not_swallow_body():
    pages=[_SourcePage(index=0,label='1',text='References\nChapter One begins on page 14.'),
           _SourcePage(index=1,label='2',text='The source explains the substantive findings. Repeated measurements establish the observed pattern, with several independent observations and clear methodological limitations.')]
    assert any(p.index==1 and role!='reference_list' for p,_,_,_,role in _source_blocks(pages))


def test_contributor_heading_covers_long_bio_but_stops_at_references():
    from app.services.verification_evidence import _PdfLayoutSpan,_pdf_structural_spans
    lines=['Notes on contributor']+['Researcher biography describes teaching and publications.']*7
    lines+=['References','Smith, J. 2020. A Work on Research Methods.']
    text='';layout=[]
    for i,line in enumerate(lines):
        start=len(text);text+=line+'\n'
        layout.append(_PdfLayoutSpan(page_index=0,start=start,end=len(text),text=line,
            x0=72,x1=480,y0=150+i*14,y1=161+i*14,page_width=600,page_height=800))
    spans=_pdf_structural_spans({0:layout})[0]
    bio=''.join(text[s.start:s.end] for s in spans if s.role=='author_biography')
    assert bio.count('Researcher biography')==7 and 'References' not in bio
