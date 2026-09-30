"""Patchwriting passages in the report, in the owner's wording (2026-09-29)."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace
import uuid

import fitz
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.evidence_report import (
    _build_report_summary, _how_to_read_sections, _render_report_summary,
    render_evidence_report_html, summary_text,
)
from app.services.patchwriting_report import attach_passage_geometry, build_passages, stored_block

CLOSE = 'This passage closely follows the wording and structure of the source without quotation marks.'
EXACT = "This passage uses the source's exact wording without quotation marks."
QUOTED = 'This passage uses wording that the source quotes from another author, without quotation marks.'
STUDENT = 'Remote learning reduced the engagement of first year students in large lectures'
INTERNALS = ('close_paraphrase', 'unquoted_verbatim', 'source_quoted', 'patchwriting-v2', 'source_quoted_share',
             'sentence_key', 'alignment_score', 'student_density', 'thresholds', '0:120:180')


def reference(n):
    return {'author': 'Writer, A.', 'year': '2020', 'title': f'Title {n}',
            'raw_reference': f'Writer, A. (2020). Title {n}. Journal of Tests, 1, 1-9.'}


def finding(kind='close_paraphrase', start=10, text=STUDENT, *, share=0.0, page_index=3, page_label=None,
            source_text='Remote learning reduced the engagement of first-year students in large lectures.'):
    return {
        'kind': kind, 'label': 'body_sentence', 'claim_ids': [],
        'student_sentence': {'paper_start': start, 'paper_end': start + len(text)},
        'student_region': {'paper_start': start, 'paper_end': start + len(text), 'text': text,
                           'text_truncated': False},
        'student_matched_spans': [],
        'source_sentences': [{'sentence_key': '0:120:180', 'page_index': page_index, 'page_label': page_label,
                              'absolute_start': 120, 'absolute_end': 180, 'role': 'body',
                              'text': source_text, 'text_truncated': False, 'matched_spans': []}],
        'measures': {'candidate_score': 0.9, 'student_density': 0.8, 'alignment_score': 12.0,
                     'source_quoted_share': share},
    }


def block(**sources):
    return {'policy_version': 'patchwriting-v2', 'decision_applied': False,
            'sources': {rid: {'policy_version': 'patchwriting-v2', 'status': 'compared',
                              'thresholds': {'paraphrase_min_matched': 6}, 'findings': rows}
                        for rid, rows in sources.items()}}


def passages_for(value):
    return build_passages(value, lambda rid: reference(rid[-1]) if rid.startswith('ref-') else None)


def window(soup, template_id):
    template = soup.select_one(f'template#{template_id}')
    return template and BeautifulSoup(template.decode_contents(), 'html.parser')


SURFACE = {'page_dimensions': [{'page_index': 0, 'width': 612, 'height': 792}],
           'page_href_template': 'p-{page_index}',
           'selectable_words': {0: [(72, 90, 540, 100, 'Body')]}}
BOX = {'page_index': 0, 'x0': 72.0, 'y0': 100.0, 'x1': 400.0, 'y1': 112.0}


def render(passages, *, members=None, citations=None):
    for passage in passages:
        passage['paper_location'] = {'localization_level': 'exact_rectangle', 'rectangles': [dict(BOX)]}
    view = {'title': 'Report', 'citation_format': 'APA', 'reference_practice': [], 'paper_surface': SURFACE,
            'citations': citations if citations is not None else [
                {'student_text': 'Other words (Writer, 2020).', 'members': members or [],
                 'paper_location': {'localization_level': 'exact_rectangle',
                                    'rectangles': [dict(BOX, y0=300, y1=312)]}}],
            'patchwriting_passages': passages}
    view['summary'] = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
                                            patchwriting_passages=passages)
    html = render_evidence_report_html(view, csp_nonce='patchwriting-report-nonce')
    return html, BeautifulSoup(html, 'html.parser')


def test_overlapping_findings_merge_into_numbered_passages_listing_each_source():
    value = block(**{
        'ref-1': [finding(start=10), finding(start=200, text='A later passage with other words entirely')],
        'ref-2': [finding('unquoted_verbatim', start=30, text=STUDENT[20:] + ' today')],
        'unknown': [finding(start=500)],
    })
    value['sources']['ref-3'] = {'status': 'not_assessed', 'reason': 'body_offsets_unverified'}
    passages = passages_for(value)
    assert [p['number'] for p in passages] == [1, 2]
    first = passages[0]
    assert (first['paper_character_start'], first['paper_character_end']) == (10, 30 + len(STUDENT) - 20 + 6)
    assert first['student_text'] == STUDENT + ' today'
    assert [(s['reference_id'], s['wording']) for s in first['sources']] == [
        ('ref-1', 'close_paraphrase'), ('ref-2', 'unquoted_verbatim')]
    assert first['kind'] == 'unquoted_verbatim' and passages[1]['kind'] == 'close_paraphrase'
    [comparison] = first['sources'][0]['comparisons']
    assert comparison['excerpts'] == [
        {'page': '4', 'text': 'Remote learning reduced the engagement of first-year students in large lectures.'}]
    assert ''.join(g['text'] for g in comparison['student']) == STUDENT
    stored = str(passages)
    assert not any(term in stored for term in ('source_quoted_share', 'sentence_key', 'alignment_score',
                                               'thresholds', 'patchwriting-v2'))
    assert passages_for(None) == [] and passages_for(block()) == []


SOURCE_SENTENCE = ('4.1 Transcending expectations Female rebellion against rigid patriarchal structures is '
                   'rich fuel that powers the engine of female-led Disney narratives.')
REWRITE = 'rebellion against rigid patriarchal structures fueled the female-dominated Disney narrative'


def rewrite_finding():
    """A student clause inside a longer sentence, following part of a source sentence."""
    item = finding(start=108, text=REWRITE, source_text=SOURCE_SENTENCE, page_index=None)
    item['student_sentence'] = {'paper_start': 100, 'paper_end': 108 + len(REWRITE) + 40}
    item['student_matched_spans'] = [
        {'paper_start': 108 + REWRITE.index(word), 'paper_end': 108 + REWRITE.index(word) + len(word)}
        for word in ('rebellion', 'rigid patriarchal structures fueled', 'female', 'Disney narrative')]
    origin = 5000
    sentence = item['source_sentences'][0]
    sentence.update(absolute_start=origin, absolute_end=origin + len(SOURCE_SENTENCE), matched_spans=[
        {'absolute_start': origin + SOURCE_SENTENCE.index(word),
         'absolute_end': origin + SOURCE_SENTENCE.index(word) + len(word)}
        for word in ('rebellion', 'rigid patriarchal structures is rich fuel', 'female', 'Disney narratives')])
    return item


def test_comparison_marks_copied_words_and_cuts_the_source_to_the_followed_clause():
    [passage] = passages_for(block(**{'ref-1': [rewrite_finding()]}))
    [comparison] = passage['sources'][0]['comparisons']
    shown = ''.join(f"**{g['text']}**" if g['copied'] else g['text'] for g in comparison['student'])
    # Matched words and the identical runs containing them; a hyphenated word
    # is marked only where it matches.
    assert shown == ('… **rebellion against rigid patriarchal structures fueled** the '
                     '**female**-dominated **Disney narrative** …')
    # The heading and the words before the first followed word are cut.
    assert comparison['excerpts'] == [{'page': None, 'text': (
        '… rebellion against rigid patriarchal structures is rich fuel that powers the engine of '
        'female-led Disney narratives')}]


def test_a_source_sentence_that_cannot_be_bound_is_shown_whole():
    item = rewrite_finding()
    item['source_sentences'][0]['text_truncated'] = True
    [passage] = passages_for(block(**{'ref-1': [item]}))
    assert passage['sources'][0]['comparisons'][0]['excerpts'][0]['text'] == SOURCE_SENTENCE


def test_source_quoted_line_needs_half_of_the_matched_words_quoted_in_the_source():
    quoted = passages_for(block(**{'ref-1': [finding('unquoted_verbatim', share=0.5)]}))[0]
    below = passages_for(block(**{'ref-1': [finding('unquoted_verbatim', share=0.49)]}))[0]
    assert quoted['sources'][0]['wording'] == 'source_quoted' and quoted['kind'] == 'unquoted_verbatim'
    assert below['sources'][0]['wording'] == 'unquoted_verbatim'


MEMBER = {'reference_id': 'ref-1', 'coverage_level': 'full_text', 'source': reference('1'),
          'source_action': {'enabled': True, 'status': 'authenticated_source_view_available',
                            'label': 'Open source', 'href': '/report/r/source/v'}}


def citing(members, start=0, end=200):
    return {'student_text': STUDENT + ' (Writer, 2020).', 'members': members,
            'paper_character_start': start, 'paper_character_end': end,
            'paper_location': {'localization_level': 'exact_rectangle', 'rectangles': [dict(BOX, x1=500)]}}


def test_citation_window_shows_patchwriting_under_academic_practice_with_the_approved_text():
    for kind, share, line in (('close_paraphrase', 0.0, CLOSE), ('unquoted_verbatim', 0.0, EXACT),
                              ('unquoted_verbatim', 0.8, QUOTED), ('close_paraphrase', 0.7, QUOTED)):
        passages = passages_for(block(**{'ref-1': [finding(kind, share=share, page_label='iv')]}))
        other = {**MEMBER, 'reference_id': 'ref-9', 'source': reference('9')}
        html, soup = render(passages, citations=[citing([other, MEMBER])])
        assert not soup.select('template[id^="passage-panel-"]')
        win = window(soup, 'citation-panel-1')
        first, second = win.select('[data-source-member]')
        assert 'Academic Practice' not in first.get_text()
        finding_html = second.select_one('.patchwriting-finding')
        heading = finding_html.find_previous_sibling('p')
        assert heading.select_one('.issue-heading.academic').get_text() == 'Academic Practice'
        assert [p.get_text() for p in finding_html.select('p')] == [
            line, STUDENT, 'Source (p. iv):',
            'Remote learning reduced the engagement of first-year students in large lectures.']
        assert not win.select('blockquote.source-excerpt') and not finding_html.select('blockquote')
        assert 'coach' not in win.get_text().casefold()
        assert not any(term in html for term in INTERNALS)


def test_words_no_citation_covers_are_shown_in_the_reference_window():
    passages = passages_for(block(**{'ref-1': [finding(start=400)]}))
    html, soup = render(passages, citations=[citing([MEMBER])])
    assert not window(soup, 'citation-panel-1').select('.patchwriting-finding')
    win = window(soup, 'reference-entry-panel-1')
    section = win.select_one('.reference-finding[data-category="academic"]')
    assert section.select_one('.issue-heading.academic').get_text() == 'Academic Practice'
    assert CLOSE in section.select_one('.patchwriting-finding').get_text()
    overlay = soup.select_one('a.patchwriting-overlay')
    assert overlay['data-panel-template'] == 'reference-entry-panel-1' and not overlay.get('data-member-index')


def test_summary_lines_singular_and_plural_link_to_the_windows_and_highlights():
    cited = [citing([MEMBER], 0, 180), citing([MEMBER], 190, 380)]
    one = passages_for(block(**{'ref-1': [finding()]}))
    summary = _build_report_summary(citations=cited, overview={}, pervasive_hanging_indent=False,
                                    patchwriting_passages=one)
    assert [summary_text(i) for i in summary['academic_practice']] == [
        "1 passage closely follows a source's wording (citation 1)."]
    many = passages_for(block(**{'ref-1': [finding('unquoted_verbatim', start=s) for s in (0, 200, 400, 600)]
                                 + [finding(start=800)]}))
    many[0]['kind'] = 'close_paraphrase'
    summary = _build_report_summary(citations=cited, overview={}, pervasive_hanging_indent=False,
                                    patchwriting_passages=many)
    assert [summary_text(i) for i in summary['academic_practice']] == [
        "2 passages closely follow sources' wording (citation 1; reference 1).",
        "3 passages use sources' exact wording without quotation marks (citation 2; reference 1)."]
    soup = BeautifulSoup(_render_report_summary(summary, placed_citations=frozenset({1})), 'html.parser')
    items = soup.select('.summary-column li')
    assert [li.get_text() for li in items] == [
        "2 passages closely follow sources' wording (citation 1; reference 1).",
        "3 passages use sources' exact wording without quotation marks (citation 2; reference 1)."]
    links = items[0].select('a.summary-instance')
    assert [(a['data-go-to'], a['data-go-to-mark'], a['href']) for a in links] == [
        ('citation-panel-1', 'patchwriting-mark-1', '#citation-location-1'),
        ('reference-entry-panel-1', 'patchwriting-mark-5', '#evidence-panel')]
    seven = [{'number': n, 'kind': 'close_paraphrase', 'paper_character_start': 10 * n,
              'paper_character_end': 10 * n + 5,
              'sources': [{'reference_id': 'ref-1'}]} for n in range(1, 8)]
    cited = [citing([MEMBER], 10 * n, 10 * n + 9) for n in range(1, 8)]
    summary = _build_report_summary(citations=cited, overview={}, pervasive_hanging_indent=False,
                                    patchwriting_passages=seven)
    li = BeautifulSoup(_render_report_summary(summary), 'html.parser').select_one('li')
    assert li.get_text() == "7 passages closely follow sources' wording (citations 1, 2, 3, 4, 5, …)."


def test_how_to_read_adds_the_full_text_sentence_after_the_yellow_highlights():
    text = BeautifulSoup(_how_to_read_sections(), 'html.parser').select('p')[1].get_text()
    assert text.startswith('Yellow highlights mark attribution issues')
    assert text.endswith('quoted wording that differs from the source. Patchwriting is checked only against sources '
                         'whose full text was retrieved.')


def test_paper_overlay_uses_the_academic_highlight_without_badge_and_opens_the_citation_at_its_source():
    passages = passages_for(block(**{'ref-1': [finding()]}))
    other = {**MEMBER, 'reference_id': 'ref-9', 'source': reference('9')}
    html, soup = render(passages, citations=[citing([other, MEMBER])])
    overlay = soup.select_one('a.patchwriting-overlay')
    assert overlay['data-panel-template'] == 'citation-panel-1' and overlay['data-member-index'] == '1'
    assert overlay['id'] == 'patchwriting-mark-1' and overlay['data-passage'] == '1'
    hit = overlay.select_one('rect.patchwriting-hit.academic-highlight')
    assert hit and overlay.select_one('rect.patchwriting-selection')
    assert '.patchwriting-overlay .patchwriting-hit{fill:#ffe45c;fill-opacity:.4;stroke:none}' in html
    assert ('.patchwriting-overlay.hovered .patchwriting-selection,.patchwriting-overlay.selected '
            '.patchwriting-selection,.patchwriting-overlay:focus .patchwriting-selection'
            '{fill:#b9dcff;fill-opacity:.22}') in html
    assert '.patchwriting-overlay:focus{outline:none}' in html
    assert not soup.select('.paper-badge[data-panel-template^="passage-panel-"], [id^="passage-location-"]')
    # Drawn after the citation, so a click on the highlight reaches it first.
    svg = str(soup.select_one('svg.page-surface'))
    assert svg.index('citation-overlay') < svg.index('patchwriting-overlay')


def test_navigation_hover_and_click_precedence_follow_the_window():
    script = Path('app/services/report_interactions.js').read_text()
    ordered = script[script.index('function orderedTargets'):script.index('// Scroll only the paper')]
    assert ('.page-container .patchwriting-overlay[data-panel-template]:not([data-panel-template^="citation-panel-"])'
            in ordered)
    assert "'passage-' + el.dataset.passage" in ordered
    at = script[script.index('function citationAt'):script.index('// Selecting a paper mark')]
    assert at.index(".closest('.patchwriting-overlay')") < at.index('const flag')
    assert "'.patchwriting-overlay'" in script[script.index('function paperTargetFor'):]
    hover = script[script.index('function hoverCitation'):script.index('function showMember')]
    assert '.patchwriting-overlay[data-panel-template="${id}"]' in hover
    assert "mark && document.getElementById(mark)" in script and 'link.dataset.goToMark' in script
    judgment = Path('app/services/report_judgment.js').read_text()
    draw = judgment[judgment.index('function draw()'):judgment.index('// One source part')]
    assert ".patchwriting-overlay').forEach(el => el.parentNode.append(el))" in draw


def test_nothing_is_rendered_without_a_patchwriting_block():
    assert stored_block(SimpleNamespace(verification_summary={'report_ids': []}), {}) is None
    html, soup = render([])
    assert not soup.select('.patchwriting-finding, .patchwriting-overlay, [id^="patchwriting-mark-"]')
    assert not soup.select('[data-summary-kind^="passage_"]')


def _paper(lines):
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        for i, line in enumerate(lines):
            page.insert_text((72, 100 + 20 * i), line, fontsize=11)
        return document.tobytes(no_new_id=True)


def test_passage_words_bind_to_the_page_and_unbound_passages_stay_reachable():
    paper = _paper(['Some opening words. Remote learning reduced the engagement',
                    'of first year students in large lectures, as shown.'])
    passages = passages_for(block(**{'ref-1': [finding(start=20),
                                               finding(start=400, text='Words that are not on the page at all')]}))
    attach_passage_geometry(passages, paper, [])
    placed, unplaced = passages
    assert placed['paper_location']['localization_level'] == 'exact_rectangle'
    assert {r['page_index'] for r in placed['paper_location']['rectangles']} == {0}
    assert len(placed['paper_location']['rectangles']) == 2
    assert unplaced['paper_location']['rectangles'] == []
    citations = [citing([MEMBER], 0, 200), citing([MEMBER], 390, 500)]
    view = {'title': 'Report', 'citation_format': 'APA', 'reference_practice': [], 'paper_surface': SURFACE,
            'citations': citations, 'patchwriting_passages': passages}
    view['summary'] = _build_report_summary(citations=citations, overview={}, pervasive_hanging_indent=False,
                                            patchwriting_passages=passages)
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce='patchwriting-report-nonce'), 'html.parser')
    assert window(soup, 'citation-panel-2').select_one('.patchwriting-finding')
    assert [a['id'] for a in soup.select('a.patchwriting-overlay[id]')] == ['patchwriting-mark-1']
    link = soup.select('a.summary-instance')[-1]
    assert (link['data-go-to'], link['data-go-to-mark']) == ('citation-panel-2', 'patchwriting-mark-2')


def test_pdf_export_carries_patchwriting_in_the_citation_and_reference_sections_without_internals():
    from app.services.report_export import _render_pdf
    paper = _paper(['Some opening words. Remote learning reduced the engagement',
                    'of first year students in large lectures, as shown.',
                    'Later the student wrote these other words about lectures.'])
    first = finding(start=20, share=0.9)
    second = finding(start=500, text='the student wrote these other words about lectures')
    passages = passages_for(block(**{'ref-1': [first, second]}))
    attach_passage_geometry(passages, paper, [])
    citations = [citing([MEMBER], 0, 200)]
    citations[0]['paper_location'] = {'localization_level': 'semantic_only', 'rectangles': []}
    view = {'citations': citations, 'reference_practice': [], 'patchwriting_passages': passages,
            'paper_surface': {}, 'overview': {}}
    content, counts = _render_pdf(paper, citations=citations, reference_practice=[], export_binding='x', view=view)
    assert counts['patchwriting_passages'] == 2
    with fitz.open(stream=content, filetype='pdf') as document:
        import unicodedata
        appendix = unicodedata.normalize('NFKC', ' '.join(' '.join(page.get_text().split())
                                                          for page in list(document)[1:]))
        toc = [entry[1] for entry in document.get_toc()]
        links = [link for link in document[0].get_links() if link.get('kind') == fitz.LINK_GOTO]
    assert 'Citation 1, source 1' in toc and 'Passage 1' not in appendix
    assert links
    section = appendix[appendix.index('Citation 1, source 1'):]
    for text in ('Academic Practice ' + QUOTED, '“' + STUDENT + '”', 'Source (p. 4):',
                 '“Remote learning reduced the engagement of first-year students in large lectures.”'):
        assert text in section
    assert ('Reference 1 ' + reference('1')['raw_reference'] + ' Academic Practice ' + CLOSE
            + ' “the student wrote these other words about lectures”') in appendix
    assert not any(term in appendix for term in INTERNALS)


def test_interactive_export_keeps_patchwriting_in_the_citation_window_without_internals(export_store):
    from app.services.interactive_report_export import build_interactive_report_html
    _, _, _, _, view, paper = export_store
    view = deepcopy(view)
    view['reference_practice'] = []
    view['citations'][0]['members'][0]['reference_id'] = 'ref-1'
    view['citations'][0]['paper_character_start'] = 0
    view['citations'][0]['paper_character_end'] = 200
    view['paper_surface']['presentation_sha256'] = hashlib.sha256(paper).hexdigest()
    passages = passages_for(block(**{'ref-1': [finding()]}))
    passages[0]['paper_location'] = {'localization_level': 'exact_rectangle',
                                     'rectangles': [dict(BOX, y0=60.0, y1=78.0)]}
    view['patchwriting_passages'] = passages
    view['summary'] = _build_report_summary(citations=view['citations'], overview={},
                                            pervasive_hanging_indent=False, patchwriting_passages=passages)
    html = build_interactive_report_html(view, paper).decode()
    soup = BeautifulSoup(html, 'html.parser')
    win = window(soup, 'citation-panel-1')
    assert CLOSE in win.select_one('.patchwriting-finding').get_text()
    assert soup.select_one('a.patchwriting-overlay[data-panel-template="citation-panel-1"]')
    assert "1 passage closely follows a source's wording (citation 1)." in soup.get_text()
    assert not win.select('form') and not any(term in html for term in INTERNALS)


def test_projection_refresh_picks_passages_up_from_the_job_summary():
    from app.models import Base
    from app.models.job import Job
    from app.models.report import Report, ReportPaperArtifactRecord
    from app.services.evidence_report_refresh import refresh_evidence_report_projection
    from app.services.paper_extraction import extract_paper_evidence
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 90), 'Remote learning reduced the engagement of students (Vidor, 1946).')
        page.insert_text((72, 200), 'References')
        page.insert_text((72, 230), 'Vidor, C. (1946). Remote learning. Columbia Press.')
        text = page.get_text()
        paper = doc.tobytes()
    extracted = extract_paper_evidence(text, paper_version_id='paper-v1', format_hint='apa',
                                       use_llm_boundaries=False, use_llm_atomizer=False,
                                       use_llm_reference_fallback=False)
    reference_id = extracted.references[0].reference_id
    region = 'Remote learning reduced the engagement of students'
    summary = {'patchwriting': block(**{reference_id: [finding('unquoted_verbatim', start=0, text=region)]})}
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    backend = SimpleNamespace(download=lambda key: paper)
    with Session(engine) as session:
        job = Job(filename='paper.pdf', title=' Remake ', paper_version_id='paper-v1', status='completed', stage='completed',
                  scope_id='owner', scope_type='personal_owner', input_sha256=hashlib.sha256(paper).hexdigest(),
                  input_media_type='application/pdf', input_byte_size=len(paper),
                  input_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
                  extraction_payload=extracted.model_dump(mode='json'), source_results=[],
                  verification_summary=summary)
        session.add(job); session.flush()
        first = Report(job_id=job.id, report_version=1, report_json={'report_ids': [], 'citation_groups': []})
        session.add(first); session.flush()
        session.add(ReportPaperArtifactRecord(
            job_id=job.id, report_id=first.id, paper_version_id=job.paper_version_id, scope_id=job.scope_id,
            scope_type=job.scope_type, storage_key='paper', content_sha256=job.input_sha256,
            media_type='application/pdf', byte_size=len(paper), artifact_kind='submitted_pdf',
            presentation_status='page_faithful_ready', presentation_storage_key='paper',
            presentation_sha256=job.input_sha256, presentation_media_type='application/pdf',
            presentation_evidence={'page_dimensions': [{'page_index': 0, 'width': 612, 'height': 792}]},
            sanitization_evidence={}, expires_at=job.input_expires_at))
        session.commit()
        result = refresh_evidence_report_projection(session, backend, report_id=first.id)
        view = session.get(Report, uuid.UUID(result['report_id'])).report_json['evidence_report']
    # The upload's Title, when given, heads the report (owner request 2026-09-29).
    assert view['title'] == 'Remake'
    [passage] = view['patchwriting_passages']
    assert passage['number'] == 1 and passage['sources'][0]['wording'] == 'unquoted_verbatim'
    assert passage['paper_location']['localization_level'] == 'exact_rectangle'
    assert [summary_text(i) for i in view['summary']['academic_practice']
            if i.get('kind', '').startswith('passage_')] == [
        # This bare view has no citation or numbered reference to show the passage in.
        "1 passage uses a source's exact wording without quotation marks."]


from test_report_export import export_store  # noqa: E402,F401  (fixture)
