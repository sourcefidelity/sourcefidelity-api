from app.services.evidence_report import summary_text
import hashlib
import io

import fitz
import pytest
from docx import Document
from pydantic import ValidationError

from app.services.body_title_formatting import BodyTitleAssessment, assess_body_title_italics, body_title_findings
from app.services.paper_extraction import PaperExtractionArtifact, extract_paper_evidence
from app.services.schemas import ParsedReference
from app.services.evidence_report import (_attach_reference_field_geometry, _build_report_summary,
                                         _reference_view, _render_reference_panel_template)


def case():
    doc = Document()
    text = 'In Example Film (1991) the staging changes.'
    doc.add_paragraph(text)
    raw = io.BytesIO(); doc.save(raw)
    ref = ParsedReference(reference_id='film', title='Example Film', year='1991',
        raw_ref='Director (1991). Example Film [Film]. Studio.',
        source_kind='traditional_media', source_kind_confidence='high')
    result = BodyTitleAssessment.model_validate(assess_body_title_italics(
        content=raw.getvalue(), body=text, references=[ref]))
    return raw.getvalue(), text, ref, result


def test_saved_observations_and_historical_absence():
    raw, text, ref, result = case()
    artifact = PaperExtractionArtifact(paper_version_id='paper', citation_format='apa',
                                      references=[ref], body_title_formatting=result)
    assert PaperExtractionArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    assert PaperExtractionArtifact(paper_version_id='old', citation_format='apa').body_title_formatting is None
    fresh = extract_paper_evidence(text+'\nReferences\n'+ref.raw_ref, paper_version_id='fresh', format_hint='apa',
        use_llm_boundaries=False, use_llm_atomizer=False, use_llm_reference_fallback=False, docx_content=raw)
    assert fresh.body_title_formatting.paper_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('mutation', ['title', 'paragraph', 'span', 'status'])
def test_invalid_saved_observations_fail_closed(mutation):
    _, _, _, result = case()
    data = result.model_dump(); item = data['observations'][0]
    if mutation == 'title': item['title_sha256'] = '0'*64
    elif mutation == 'paragraph': item['paragraph_text'] += ' changed'
    elif mutation == 'span': item['title_end'] = 9999
    else: item['status'] = 'matches_rule'
    with pytest.raises(ValidationError): BodyTitleAssessment.model_validate(data)


@pytest.mark.parametrize('mutation', ['paper', 'reference', 'review', 'absent'])
def test_unbound_observations_never_become_findings(mutation):
    _, _, ref, result = case(); digest = result.paper_sha256
    if mutation == 'paper': digest = '0'*64
    elif mutation == 'reference': ref.raw_ref += ' changed'
    elif mutation == 'review': ref.needs_review = True
    else: result = None
    assert body_title_findings(result, {'film': ref}, digest) == []


@pytest.mark.parametrize('duplicate', [False, True])
def test_exact_body_highlight_panel_and_localized_summary(duplicate):
    _, text, ref, result = case()
    findings = body_title_findings(result, {'film': ref}, result.paper_sha256)
    for f in findings: f['source'] = _reference_view(ref)
    def summaries():
        return _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
            reference_practice=findings, require_paper_flags=True)
    assert not any('lack required italics' in s for s in map(summary_text, summaries()['reference_formatting']))
    doc = fitz.open(); page = doc.new_page(); page.insert_text((72, 100), text)
    page.insert_text((72, 300), ref.raw_ref)
    if duplicate: page.insert_text((72, 200), text)
    _attach_reference_field_geometry({'reference_practice': findings}, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    assert bool(findings[0]['rectangles']) is not duplicate
    if not duplicate:
        box = fitz.Rect(*[findings[0]['rectangles'][0][k] for k in ('x0','y0','x1','y1')])
        assert page.get_textbox(box) == 'Example Film'
        assert any('lacks required italics' in s for s in map(summary_text, summaries()['reference_formatting']))
    assert 'Citation and Reference Formatting' in _render_reference_panel_template(findings[0], 1)


def test_export_body_title_destinations_and_repeatability():
    import unicodedata
    from app.services.report_export import _render_pdf
    _, text, ref, result = case()
    findings = body_title_findings(result, {'film': ref}, result.paper_sha256)
    findings[0]['source'] = _reference_view(ref)
    doc = fitz.open(); page = doc.new_page(); page.insert_text((72,100), text)
    content = doc.tobytes()
    view = {'citations': [], 'reference_practice': findings, 'overview': {}}
    def render():
        return _render_pdf(content, citations=[], reference_practice=findings,
                           export_binding='body-title-test', view=view)
    output, counts = render()
    assert output == render()[0]
    assert counts['reference_practice_findings'] == 1
    with fitz.open(stream=output, filetype='pdf') as pdf:
        text = ' '.join(unicodedata.normalize('NFKC', ''.join(p.get_text() for p in pdf)).split())
        assert 'Citation and Reference Formatting 1' in text
        assert 'Back to marked passage' in text
        assert 'Italicize this film or book title' in text
        assert all(0 <= link['page'] < len(pdf) for p in pdf for link in p.get_links() if link['kind'] == fitz.LINK_GOTO)
        assert pdf[0].get_links()


@pytest.mark.parametrize('tail,expected', [('University Press.',1), ('In A larger book. University Press.',0), ('Unknown.',0)])
def test_book_routing_kind_needs_independent_publisher(tail, expected):
    raw, text, ref, _ = case()
    ref.source_kind = 'monograph'
    ref.raw_ref = 'Writer (1991). Example Film. '+tail
    result = assess_body_title_italics(content=raw, body=text, references=[ref])
    assert len(result['observations']) == expected
