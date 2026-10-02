"""A passage the paper repeats word for word binds to the copy matching its
place in the text (paper 4 pasted one paragraph twice; 2026-10-02)."""
import fitz

from app.services.presentation_anchors import bind_citations_to_pdf
from app.services.schemas import InTextCitation

SENTENCE = "The first animated feature film pioneered a new visual language for its audience."


def pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), SENTENCE, fontsize=9)
    page.insert_text((72, 300), SENTENCE, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


def test_the_second_copy_binds_to_the_second_place_on_the_page():
    body = f"{SENTENCE}\n\nOther words come between the two copies.\n\n{SENTENCE}"
    second = body.rindex(SENTENCE)
    citation = InTextCitation(text=SENTENCE, citation_key="", passage_start=second,
                              passage_end=second + len(SENTENCE), paragraph_index=2)
    without = bind_citations_to_pdf(pdf(), citations=[citation])
    assert without.anchors[0].mapping_status == "ambiguous"
    anchor = bind_citations_to_pdf(pdf(), citations=[citation], body_text=body).anchors[0]
    assert anchor.mapping_method == "body_order_disambiguation"
    assert min(r.y0 for r in anchor.rectangles) > 250
