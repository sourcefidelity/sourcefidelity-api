from app.services.citation_extractor import extract_citations
from app.services.schemas import ParsedReference


def test_compound_lead_surname_et_al_is_preserved_and_linked_exactly():
    ref=ParsedReference(reference_id='collection',author='Jeffers McDonald, T., Lanckman, L., & Polley, S.',
        year='2023',title='Example collection',raw_ref='Example',citation_key='Jeffers McDonald 2023')
    body='The design encourages desire (Jeffers McDonald et al., 2023).'
    citations=extract_citations(body,[ref],use_llm_boundaries=False)
    assert any(c.reference_ids==['collection'] and c.citation_marker=='(Jeffers McDonald et al., 2023)' for c in citations)
    other=ref.model_copy(update={'reference_id':'different','author':'Jeffers, T., Lanckman, L., & Polley, S.'})
    citations=extract_citations(body,[other],use_llm_boundaries=False)
    assert not any(c.reference_ids for c in citations)
