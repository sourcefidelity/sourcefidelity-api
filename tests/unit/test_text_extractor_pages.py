"""Owner decision 2026-10-04 (paper 9): a sentence a page break cuts is not two paragraphs."""
from app.services.text_extractor import join_pdf_pages


def test_a_cut_sentence_joins_and_real_paragraphs_stay():
    assert join_pdf_pages(["First page ends. This part", "of her experience relates."]) == (
        "First page ends.\n\nThis part of her experience relates.")
    assert join_pdf_pages(["no boundary at all", "continues here."]) == "no boundary at all continues here."
    assert join_pdf_pages(["A finished sentence.", "Next page starts."]) == "A finished sentence.\n\nNext page starts."
    assert join_pdf_pages(["Heading", "Body text begins here."]) == "Heading\n\nBody text begins here."


def test_a_heading_without_a_blank_line_becomes_its_own_paragraph():
    from app.services.text_extractor import isolate_pdf_headings
    text = ("Her developing identity.\nConceptualisation of Identity and Representation\nIdentity is a dynamic construct."
            "\n\nReferences\nSmith, J. (2019). The Complexity of Identity\nFormation. Press.")
    out = isolate_pdf_headings(text)
    assert "identity.\n\nConceptualisation of Identity and Representation\n\nIdentity is" in out
    assert out.endswith("References\nSmith, J. (2019). The Complexity of Identity\nFormation. Press.")
    wrapped = "It was studied.\nThe research shows that people\nadapt quickly."
    assert isolate_pdf_headings(wrapped) == wrapped

