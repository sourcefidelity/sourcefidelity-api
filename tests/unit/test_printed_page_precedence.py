from app.services.verification_evidence import (
    _SourcePage, _SourceStructuralSpan, _resolved_pdf_page_labels,
)


def resolve(printed, embedded):
    pages=[];spans={}
    for index,(number,label) in enumerate(zip(printed,embedded)):
        text=number+'\nBody text'
        pages.append(_SourcePage(index=index,label=label,text=text))
        spans[index]=(_SourceStructuralSpan(start=0,end=len(number),role='page_furniture'),)
    return _resolved_pdf_page_labels(pages,spans)


def test_adjacent_printed_sequence_overrides_numeric_pdf_labels():
    assert resolve(['209','210','211'],['5','6','7'])=={0:'209',1:'210',2:'211'}
    assert resolve(['84','85'],['1','2'])=={0:'84',1:'85'}


def test_isolated_or_nonsequential_printed_numbers_do_not_override():
    assert resolve(['209'],['5'])=={0:'5'}
    assert resolve(['209','2024'],['5','6'])=={0:'5',1:'6'}


def test_explicit_prefixes_and_roman_labels_survive():
    assert resolve(['209','210'],['App-1','ii'])=={0:'App-1',1:'ii'}


def test_missing_and_ambiguous_printed_labels_are_not_inferred():
    assert resolve(['209','210\n2024','211'],['5','6','7'])=={0:'5',1:'6',2:'7'}
    assert resolve(['209',''],[None,None])=={0:'209',1:None}


def test_real_pdf_extraction_propagates_labels_and_records_conflict():
    import fitz
    from types import SimpleNamespace
    from app.services.verification_evidence import _extract_pages

    with fitz.open() as doc:
        for number in (84,85,86):
            page=doc.new_page()
            page.insert_text((280,30),str(number))
            page.insert_text((72,120),'Independent article body with checkable research details.')
        doc.set_page_labels([{'startpage':0,'prefix':'','style':'D','firstpagenum':1}])
        content=doc.tobytes()
    pages,limits=_extract_pages(SimpleNamespace(representation_kind='pdf',content=content))
    assert [p.label for p in pages]==['84','85','86']
    assert any('corroborated on adjacent pages' in s for s in limits)
