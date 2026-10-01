import hashlib
import pytest
from app.services.paper_extraction import extract_paper_evidence, PaperExtractionArtifact
from app.services.quotation_locator_requirement import assess_quotation_locators, quotation_locator_findings

QUOTE='The ordinary activity of the audience changes how this story is understood'
REF='Smith, J. (2020). A complete book title. University Press.'


def extract(body, style='apa'):
    return extract_paper_evidence(body+'\n\nReferences\n'+REF,paper_version_id='test-paper',format_hint=style,
        use_llm_boundaries=False,use_llm_atomizer=False,use_llm_reference_fallback=False)


def test_explicit_quotation_omission_is_separate_from_accuracy_and_persists():
    body=f'“{QUOTE}” (Smith, 2020).'
    a=extract(body)
    findings=[r for r in a.quotation_locator_requirements if r.status=='missing']
    assert len(findings)==1
    r=findings[0]
    assert r.locator_accuracy=='not_assessed'
    assert r.context_sha256==hashlib.sha256(body.encode()).hexdigest()
    assert PaperExtractionArtifact.model_validate_json(a.model_dump_json())==a
    old=a.model_dump();old.pop('quotation_locator_requirements')
    assert PaperExtractionArtifact.model_validate(old).quotation_locator_requirements==[]
    projected=quotation_locator_findings(findings,a.citation_claims,{r.reference_id:r for r in a.references})
    assert len(projected)==1 and projected[0]['quote_text']==QUOTE
    a.references[0].raw_ref+=' changed'
    assert not quotation_locator_findings(findings,a.citation_claims,{r.reference_id:r for r in a.references})


@pytest.mark.parametrize('suffix',[
    '(p. 4)', '(pp. 4–5)', '(para. 2)', '(paragraph 2)', '(“Results” section)',
    '(“Overview”)', '(Chapter 3)', '(00:42)', '(lines 3–5)', '(4)',
])
def test_trailing_alternative_locators_are_not_discarded(suffix):
    body=f'“{QUOTE}” (Smith, 2020). {suffix}'
    a=extract(body)
    assert not any(r.status=='missing' for r in a.quotation_locator_requirements)


@pytest.mark.parametrize('body',[
    f'“{QUOTE}” (Smith, 2020, p. 4).',
    f'“{QUOTE}” (Smith, 2020, “Results” section).',
    f'“{QUOTE}”. A different statement comes next (Smith, 2020).',
    f'In the Results section, “{QUOTE}” (Smith, 2020).',
    f'The epigraph reads “{QUOTE}” (Smith, 2020).',
])
def test_unaccepted_boundaries_do_not_produce_omissions(body):
    a=extract(body)
    assert not any(r.status=='missing' for r in a.quotation_locator_requirements)


def test_mla_unknown_work_multiple_markers_and_changed_body_abstain():
    body=f'“{QUOTE}” (Smith, 2020).';a=extract(body)
    for style,claims,refs,text in [
        ('mla',a.citation_claims,a.references,body),
        ('apa',a.citation_claims,a.references,'changed '+body),
        ('apa',[c.model_copy(update={'reference_ids':['a','b']}) for c in a.citation_claims],a.references,body),
    ]:
        results=assess_quotation_locators(body_text=text,citation_format=style,claims=claims,references=refs)
        assert results and all(r.status=='not_assessed' for r in results)


def test_work_title_case_and_punctuation_do_not_create_quotation_omission():
    body=f'“{QUOTE}” (Smith, 2020).'
    a=extract(body)
    refs=[r.model_copy(update={'title': QUOTE.upper()+'.'}) for r in a.references]
    results=assess_quotation_locators(body_text=body,citation_format='apa',claims=a.citation_claims,references=refs)
    assert results and all(r.status=='not_assessed' for r in results)


def test_omission_is_localized_to_quote_not_reference_list():
    import fitz
    from app.services.evidence_report import _reference_view,_attach_reference_field_geometry,_render_reference_panel_template
    body=f'"{QUOTE}" (Smith, 2020).';a=extract(body)
    findings=quotation_locator_findings(a.quotation_locator_requirements,a.citation_claims,{r.reference_id:r for r in a.references})
    assert len(findings)==1
    findings[0]['source']=_reference_view(a.references[0])
    doc=fitz.open();page=doc.new_page()
    page.insert_text((50,100),body,fontsize=8);page.insert_text((50,300),REF,fontsize=8)
    _attach_reference_field_geometry({'reference_practice':findings},doc,'a'*64)
    assert findings[0]['localization_status']=='exact_field'
    assert all(r['y1']<150 for r in findings[0]['rectangles'])
    assert 'Citation and Reference Formatting' in _render_reference_panel_template(findings[0],1)


def test_explicit_narrative_quotation_has_versioned_omission():
    a=extract(f'Smith (2020) writes “{QUOTE}”.')
    finding, = [r for r in a.quotation_locator_requirements if r.status=='missing']
    assert finding.rule_id=='apa7_attributed_quotation_locator_v3'
    assert PaperExtractionArtifact.model_validate_json(a.model_dump_json())==a


@pytest.mark.parametrize('suffix',['(p. 4)','(para. 2)','(“Background” section)','(“Background”)','(00:42)','(4)'])
def test_narrative_alternatives_are_never_removed(suffix):
    a=extract(f'Smith (2020) writes “{QUOTE}” {suffix}.')
    assert not any(r.status=='missing' for r in a.quotation_locator_requirements)


@pytest.mark.parametrize('body',[
    f'Smith (2020) discusses a different argument. “{QUOTE}”.',
    f'Smith (2020) writes “{QUOTE}” and Jones provides another account.',
    f'In the Results section, Smith (2020) writes “{QUOTE}”.',
])
def test_narrative_uncertain_scope_abstains(body):
    assert not any(r.status=='missing' for r in extract(body).quotation_locator_requirements)


@pytest.mark.parametrize('body',[
    'The author calls it “audience activity” (Smith, 2020).',
    f'Smith (2020) writes “{QUOTE}” and “another lengthy quotation to assess here”.',
    'Smith (2020) writes “a short term”.',
    'Scholar James Smith (2020) states: “This is a complete sentence. This is its continuation.”',
    'The “selected” narrative (Smith, 2020).',
])
def test_explicit_short_multiple_and_multisentence_quotes_require_locators(body):
    a=extract(body)
    assert any(r.status=='missing' and r.rule_id=='apa7_attributed_quotation_locator_v3'
               for r in a.quotation_locator_requirements)


def test_exact_attribution_does_not_require_source_kind_or_acquisition():
    body=f'“{QUOTE}” (Smith, 2020).';a=extract(body)
    refs=[r.model_copy(update={'source_kind':'unknown','source_kind_confidence':'unknown'}) for r in a.references]
    assert any(r.status=='missing' for r in assess_quotation_locators(body_text=body,citation_format='apa',claims=a.citation_claims,references=refs))


def test_other_sentence_date_and_ordinary_figure_are_not_locators():
    a=extract('The historical period (1927-1954) is separate. Smith (2020) writes “a figure of speech”.')
    assert any(r.status=='missing' for r in a.quotation_locator_requirements)


def test_according_to_is_explicit_source_attribution():
    a=extract('According to Smith (2020), the figure was “unpredictable”, but also “careful with words”.')
    assert any(r.status=='missing' for r in a.quotation_locator_requirements)
