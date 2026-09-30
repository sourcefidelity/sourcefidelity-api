import fitz
from app.services.verification_evidence import _pdf_reading_order_text_and_spans

def test_top_column_fragments_are_not_combined_as_headers():
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((350,35),'RUNNING HEADER',fontsize=9)
        for x,label in [(330,'RIGHT'),(40,'LEFT')]:
            page.insert_text((x,75),label+' TOP FRAGMENT',fontsize=9)
            page.insert_textbox(fitz.Rect(x,130,x+230,500),label+' BODY '+('evidence words '*70),fontsize=9)
        text,spans,reordered=_pdf_reading_order_text_and_spans(page,0)
        assert reordered
        assert text.index('RUNNING HEADER')<text.index('LEFT TOP')<text.index('LEFT BODY')<text.index('RIGHT TOP')<text.index('RIGHT BODY')
        assert ''.join(s.text for s in spans)==text
