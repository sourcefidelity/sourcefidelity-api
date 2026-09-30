import fitz
from app.services.reference_layout import extract_reference_layout_from_bytes
from app.services.schemas import ParsedReference


def layout(duplicate=False):
    text = '(1985). A complete article title with a long identifying subtitle and several distinctive words.'
    doc=fitz.open(); page=doc.new_page(width=1000,height=800)
    page.insert_text((40,80),'References')
    page.insert_text((40,110),'9')  # Numeric content is retained, not globally filtered.
    page.insert_text((40,140),text)
    if duplicate: page.insert_text((40,190),text)
    data=doc.tobytes();doc.close()
    ref=ParsedReference(reference_id='r',raw_ref=text,title='A complete article title',year='1985')
    return extract_reference_layout_from_bytes(data,'paper.pdf',references=[ref],citation_format='apa').entries[0]


def test_exact_span_is_not_ambiguous_with_its_own_larger_window():
    result=layout()
    assert result.mapping_status=='matched' and result.match_confidence==1
    assert min(r.y0 for r in result.rectangles)>120


def test_distinct_identical_reference_occurrences_still_abstain():
    assert layout(duplicate=True).mapping_status=='ambiguous'
