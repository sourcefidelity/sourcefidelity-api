import pytest

from app.services.paper_extraction import extract_paper_evidence, PaperExtractionArtifact
from app.services.quotation_locator_requirement import quotation_locator_findings

REF = 'Smith, John. “Experimental Outcomes.” Journal of Testing, vol. 2, no. 1, 2020, pp. 12–24.'
QUOTE = 'The measured rate increased by twenty percent during the controlled experiment'
PARAPHRASE = 'Smith reports that the measured rate increased by twenty percent during the controlled experiment.'


def extract(body, ref=REF):
    return extract_paper_evidence(body+'\n\nWorks Cited\n'+ref, paper_version_id='mla-test', format_hint='mla',
        use_llm_boundaries=False, use_llm_atomizer=False, use_llm_reference_fallback=False)


@pytest.mark.parametrize('body', [f'“{QUOTE}” (Smith).', PARAPHRASE, PARAPHRASE[:-1]+' (Smith).'])
def test_mla_quote_and_explicit_paraphrase_without_locator(body):
    a = extract(body)
    missing = [r for r in a.quotation_locator_requirements if r.status == 'missing']
    assert len(missing) == 1
    r = missing[0]
    assert r.rule_id == 'mla9_explicit_passage_locator_v1'
    assert r.pagination_text == 'pp. 12–24'
    assert r.locator_accuracy == 'not_assessed'
    assert PaperExtractionArtifact.model_validate_json(a.model_dump_json()) == a
    findings = quotation_locator_findings(missing, a.citation_claims, {r.reference_id:r for r in a.references})
    assert findings[0]['citation_style'] == 'mla'
    assert 'paragraph number' not in findings[0]['finding']
    a.references[0].raw_ref += ' changed'
    assert not quotation_locator_findings(missing, a.citation_claims, {r.reference_id:r for r in a.references})


@pytest.mark.parametrize('pages', ['p. 7', 'pp. 7–7', 'pp. 24–12', 'article 1234', '', 'pp. xi–xv', 'pp. A1–A7'])
def test_one_page_and_unaccepted_pagination_never_flag(pages):
    a = extract(PARAPHRASE, REF.replace('pp. 12–24', pages))
    assert not any(r.status == 'missing' for r in a.quotation_locator_requirements)


@pytest.mark.parametrize('body', [
    f'“{QUOTE}” (Smith 14).',
    PARAPHRASE[:-1]+' (14).',
    PARAPHRASE+' (ch. 2)',
    PARAPHRASE+' (lines 10–12)',
    'Smith broke new ground with this work (Smith).',
    'Smith reports that the entire book explores the cultural history of scientific experimentation.',
    'The experiment used a controlled procedure with a measured outcome (Smith).',
    f'“{QUOTE}”. A different claim appears here (Smith).',
])
def test_locator_or_uncertain_specific_scope_abstains(body):
    a = extract(body)
    assert not any(r.status == 'missing' for r in a.quotation_locator_requirements)


def test_online_representation_and_elided_page_range():
    a = extract(PARAPHRASE, REF+' https://example.org/article')
    assert not any(r.status == 'missing' for r in a.quotation_locator_requirements)
    a = extract(PARAPHRASE, REF.replace('12–24', '149–66'))
    assert any(r.status == 'missing' for r in a.quotation_locator_requirements)


@pytest.mark.parametrize('suffix', [' Kindle edition.', ' EPUB.', ' Unpaginated.', ' HTML edition.'])
def test_reflowable_or_unpaginated_version_overrides_bibliographic_range(suffix):
    a = extract(PARAPHRASE, REF+suffix)
    assert not any(r.status == 'missing' for r in a.quotation_locator_requirements)


def test_mla_panel_and_body_geometry():
    import fitz
    from app.services.evidence_report import _attach_reference_field_geometry, _render_reference_panel_template, _reference_view
    a = extract(PARAPHRASE)
    findings = quotation_locator_findings(a.quotation_locator_requirements, a.citation_claims, {r.reference_id:r for r in a.references})
    assert len(findings) == 1
    findings[0]['source'] = _reference_view(a.references[0])
    doc = fitz.open(); page = doc.new_page()
    page.insert_text((40, 100), PARAPHRASE, fontsize=8)
    page.insert_text((40, 300), REF, fontsize=8)
    _attach_reference_field_geometry({'reference_practice':findings}, doc, 'a'*64)
    assert findings[0]['rectangles'] and all(r['y1'] < 150 for r in findings[0]['rectangles'])
    panel = _render_reference_panel_template(findings[0], 1)
    assert 'Citation and reference formatting' in panel and 'MLA citation guidance' not in panel
    assert 'APA quotation guidance' not in panel


def test_paraphrase_flag_excludes_merged_heading():
    a = extract('Controlled MLA locator test '+PARAPHRASE)
    findings = quotation_locator_findings(a.quotation_locator_requirements, a.citation_claims, {r.reference_id:r for r in a.references})
    assert len(findings) == 1
    assert findings[0]['quote_text'].startswith('Smith reports that')
    assert 'Controlled MLA' not in findings[0]['quote_text']
