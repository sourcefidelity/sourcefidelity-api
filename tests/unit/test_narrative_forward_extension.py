import pytest

from app.services.citation_extractor import extract_citations
from app.services.paper_extraction import _build_marker_census, _merge_additive_llm_recovery
from app.services.schemas import ParsedReference


def inputs(tail='These cues establish character traits. Based on this, the model adds actions.'):
    text = 'According to Morgan (2015), characters emerge through stories. ' + tail
    refs = [ParsedReference(reference_id='r', author='Morgan, D.', year='2015', title='Characters')]
    rows = extract_citations(text, refs, format_hint='apa', use_llm_boundaries=False)
    census = _build_marker_census(rows, text, refs)
    base = rows[0]
    proposed = base.model_copy(update={'text': text, 'passage_end': len(text), 'marker_type': 'cite_tag'})
    return text, rows, census, proposed


def test_exact_forward_extension_keeps_base_audit_and_exact_marker():
    text, rows, census, proposed = inputs()
    result = _merge_additive_llm_recovery(rows, [proposed], census, text)
    accepted = [r for r in result if not r.drop_reason]
    assert len(accepted) == 1
    assert accepted[0].text == text
    assert accepted[0].citation_markers == rows[0].citation_markers
    assert result[0].text == rows[0].text and result[0].passage_end == rows[0].passage_end
    assert result[0].drop_reason == 'deterministic_base_of_bounded_narrative_extension_v1'


@pytest.mark.parametrize('change', ['backward', 'partial', 'altered', 'binding', 'disagreement', 'failed'])
def test_invalid_or_competing_extensions_leave_deterministic_unit(change):
    text, rows, census, proposed = inputs()
    proposals = [proposed]
    if change == 'backward':
        proposals = [proposed.model_copy(update={'passage_start': -1})]
    elif change == 'partial':
        proposals = [proposed.model_copy(update={'passage_end': len(text)-8, 'text': text[:-8]})]
    elif change == 'altered':
        proposals = [proposed.model_copy(update={'text': text.replace('traits', 'roles')})]
    elif change == 'binding':
        proposals = [proposed.model_copy(update={'reference_ids': ['other']})]
    elif change == 'disagreement':
        proposals.append(rows[0])
    else:
        proposals = [proposed.model_copy(update={'drop_reason': 'invalid_model_output'})]
    result = _merge_additive_llm_recovery(rows, proposals, census, text)
    assert any(r == rows[0] for r in result)
    assert not any(r.text == text and not r.drop_reason for r in result)


@pytest.mark.parametrize('tail', [
    'These cues matter. They shape characters. They also imply motives.',
    '\n\nThese cues matter.',
    'Another claim follows (Unknown, 2020).',
])
def test_scope_boundaries_prevent_extension(tail):
    text, rows, census, proposed = inputs(tail)
    result = _merge_additive_llm_recovery(rows, [proposed], census, text)
    assert any(r == rows[0] for r in result)


def test_parenthetical_scope_is_not_forward_extended():
    text, rows, census, proposed = inputs()
    base = rows[0].model_copy(update={'marker_type': 'parenthetical'})
    result = _merge_additive_llm_recovery([base], [proposed], census, text)
    assert base in result


def test_orchestrator_persists_extension_binding_and_original_audit(monkeypatch):
    from app.services import paper_extraction
    text, rows, census, proposed = inputs()
    ref = ParsedReference(reference_id='r', author='Morgan, D.', year='2015', title='Characters',
                          raw_ref='Morgan, D. (2015). Characters. Press.')
    monkeypatch.setattr(paper_extraction, 'extract_and_parse_references', lambda *a, **k: [ref])
    monkeypatch.setattr(paper_extraction, 'extract_citations',
                        lambda *a, **k: [proposed] if k.get('use_llm_boundaries') else rows)
    artifact = paper_extraction.extract_paper_evidence(
        text+'\n\nReferences\n'+ref.raw_ref, paper_version_id='forward-unit',
        format_hint='apa', use_llm_atomizer=False, use_llm_reference_fallback=False)
    assert len(artifact.citation_claims) == 1
    assert artifact.citation_claims[0].text == text
    assert artifact.citation_claims[0].reference_ids == ['r']
    assert len(artifact.rejected_citations) == 1
    assert artifact.rejected_citations[0].text == rows[0].text
    assert not artifact.claim_boundary_rejections
