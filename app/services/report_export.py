"""Deterministic released-report exports over the retained paper PDF."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import uuid
import io
from html import escape

import fitz
from sqlalchemy.orm import Session

from app.models.report import Report
from app.services.evidence_report import (
    load_authorized_evidence_report_bundle,
    _render_citation_information,
    _inline_locator, _check_sentence, _panel_statement,
    project_reference_flags,
)
from app.services.storage.backend import StorageBackend
from app.services.report_member_navigation import member_targets, marker_words
from app.services.highlight_priority import LINK_MARKER_FINDINGS, SUBMITTED_LINK_FINDINGS


REPORT_EXPORT_VERSION = "released-report-pdf-v30"

_TONE_COLORS = {
    "evidence_available": (0.145, 0.388, 0.655),
    "limited_evidence": (0.086, 0.549, 0.549),
    "partial_evidence": (0.545, 0.486, 0.663),
    "retrieved_no_connection": (0.549, 0.420, 0.310),
    "attention": (0.851, 0.373, 0.008),
    "not_assessed": (0.722, 0.753, 0.784),
}
_AMBER = (0.851, 0.373, 0.008)
_QUOTE_FILL = (1.000, 0.824, 0.478)
# Soft red for "Cannot be verified"; bright solid blue for details that differ from
# the located record (both Evidence, owner decision 2026-09-24).
_UNVERIFIED_FILL = (0.949, 0.545, 0.510)
_DIFFERENCE_BLUE = (0.039, 0.486, 1.0)
_FIXED_PDF_DATE = "D:20000101000000Z"


class ReportExportError(ValueError):
    """Raised when an exact released export cannot be built safely."""


@dataclass(frozen=True)
class ReleasedReportExport:
    content: bytes
    manifest: dict
    manifest_sha256: str


def build_released_report_export(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
) -> ReleasedReportExport:
    """Build the one report's PDF derivative without persisting a partial object."""
    view, artifact, paper_content = load_authorized_evidence_report_bundle(
        session,
        backend,
        report_id=report_id,
        scope_type=scope_type,
        scope_id=scope_id,
    )
    try:
        parsed_report_id = (
            report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
        )
    except (TypeError, ValueError) as exc:
        raise ReportExportError("Report identity is invalid") from exc
    report = session.get(Report, parsed_report_id)
    if report is None:
        raise ReportExportError("Report is unavailable")

    paper_sha256 = hashlib.sha256(paper_content).hexdigest()
    if paper_sha256 != artifact.presentation_sha256:
        raise ReportExportError("Retained paper presentation does not match its hash")

    manifest = _base_manifest(
        report=report,
        view=view,
        artifact=artifact,
        paper_sha256=paper_sha256,
    )
    content, overlay_counts = _render_pdf(
        paper_content,
        citations=list(view.get("citations") or []),
        reference_practice=list(view.get("reference_practice") or []),
        view=view,
        export_binding=(
            f"report={manifest['report_id']};"
            f"report_version={manifest['report_version']};"
            f"paper_sha256={manifest['paper_presentation_sha256']}"
        ),
    )
    manifest["overlay_counts"] = overlay_counts
    manifest["page_count"] = _page_count(content)
    manifest["export_sha256"] = hashlib.sha256(content).hexdigest()
    manifest_sha256 = _digest(manifest)
    return ReleasedReportExport(
        content=content,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def _base_manifest(*, report, view, artifact, paper_sha256) -> dict:
    return {
        "export_version": REPORT_EXPORT_VERSION,
        "report_id": str(report.id),
        "report_version": int(report.report_version),
        "previous_report_id": (
            str(report.previous_report_id) if report.previous_report_id else None
        ),
        "amendment_reason": report.amendment_reason,
        "paper_artifact_id": str(artifact.id),
        "paper_version_id": str(view.get("paper_version_id") or ""),
        "paper_presentation_sha256": paper_sha256,
        "source_policy": {
            "source_bytes_embedded": False,
            "source_excerpts_embedded": True,
            "authorization_bound_source_actions_embedded": False,
        },
    }


def _render_pdf(
    paper_content: bytes,
    *,
    citations: list[dict],
    reference_practice: list[dict],
    export_binding: str,
    view: dict | None = None,
) -> tuple[bytes, dict[str, int]]:
    counts = {
        "citation_underlines": 0,
        "source_member_highlights": 0,
        "quotation_differences": 0,
        "reference_practice_findings": 0,
        "patchwriting_passages": 0,
    }
    try:
        document = fitz.open(stream=paper_content, filetype="pdf")
    except Exception as exc:
        raise ReportExportError("Retained paper presentation is not a readable PDF") from exc
    try:
        if view is not None:
            view = project_reference_flags(view, document, hashlib.sha256(paper_content).hexdigest())
            reference_practice = view.get('reference_practice') or []
            citations = view.get('citations') or citations
        words_by_page = marker_words(document)
        from app.services.report_member_navigation import missing_reference_targets
        from app.services.highlight_priority import (
            subtract_rectangles, ACADEMIC_FINDINGS, UNVERIFIED_FINDINGS, REFERENCE_DIFFERENCE_FINDINGS)
        from app.services.evidence_report import normalize_reference_findings
        reference_practice = normalize_reference_findings(reference_practice)
        paint_regions = []
        unverified_ids = {f.get('reference_id') for f in reference_practice
            if f.get('finding_type') in UNVERIFIED_FINDINGS}
        for finding in reference_practice:
            if (finding.get('finding_type') not in LINK_MARKER_FINDINGS
                    and finding.get('finding_type') not in REFERENCE_DIFFERENCE_FINDINGS):
                priority = (1.5 if finding.get('finding_type') == 'source_topical_mismatch' else
                            3 if finding.get('finding_type') in ACADEMIC_FINDINGS | UNVERIFIED_FINDINGS else 2)
                paint_regions.extend((page, tuple(rect), priority) for page,rect in
                                     _valid_rectangles(document, finding.get('rectangles')))
        for citation in citations:
            from app.services.report_layers import topical_mismatch
            paint_regions.extend((t['page_index'],tuple(t[k] for k in ('x0','y0','x1','y1')),1.5)
                for t in member_targets(citation, words_by_page)
                if topical_mismatch(citation['members'][t['member_index']], citation))
            paint_regions.extend((t['page_index'],tuple(t[k] for k in ('x0','y0','x1','y1')),3)
                for t in member_targets(citation, words_by_page)
                if citation['members'][t['member_index']].get('reference_id') in unverified_ids)
            paint_regions.extend((page,tuple(rect),3) for page,rect in
                _valid_rectangles(document, citation.get('quotation_difference_rectangles')))
            paint_regions.extend((t['page_index'],tuple(t[k] for k in ('x0','y0','x1','y1')),3)
                for t in missing_reference_targets(citation, words_by_page))
        passages = list((view or {}).get('patchwriting_passages') or [])
        for passage in passages:
            paint_regions.extend((page,tuple(rect),3) for page,rect in
                _valid_rectangles(document, (passage.get('paper_location') or {}).get('rectangles')))

        def paint_highlight(page_index, rectangle, color, opacity, priority):
            covers = [rect for page,rect,rank in paint_regions if page == page_index and rank > priority]
            for part in subtract_rectangles(tuple(rectangle), covers):
                document[page_index].draw_rect(part, color=None, fill=color, fill_opacity=opacity, overlay=True)

        for citation in citations:
            for target in missing_reference_targets(citation, words_by_page):
                rectangle = fitz.Rect(*(target[k] for k in ('x0','y0','x1','y1')))
                document[target['page_index']].draw_rect(rectangle, color=None, fill=_QUOTE_FILL, fill_opacity=.45, overlay=True)
            for page_index, rectangle in _valid_rectangles(
                document, (citation.get("paper_location") or {}).get("rectangles")
            ):
                page = document[page_index]
                page.draw_line(
                    (rectangle.x0, max(rectangle.y0, rectangle.y1 - 0.8)),
                    (rectangle.x1, max(rectangle.y0, rectangle.y1 - 0.8)),
                    color=(.77,.80,.82), width=.6, overlay=True,
                )
                counts["citation_underlines"] += 1
                from app.services.report_layers import citation_partial_relevance
                if citation_partial_relevance(citation):
                    page.draw_rect((rectangle + (-1,-1,1,1)) & page.rect,
                                   color=(.85,.37,.01), width=1.2, overlay=True)
            for target in member_targets(citation, words_by_page):
                rectangle=fitz.Rect(*(target[key] for key in ('x0','y0','x1','y1')))
                unverified = citation['members'][target['member_index']].get('reference_id') in unverified_ids
                paint_highlight(target['page_index'], rectangle,
                    _UNVERIFIED_FILL if unverified else _TONE_COLORS[target['tone']],
                    .42 if unverified else .23, 3 if unverified else 1)
                counts['source_member_highlights'] += 1
                from app.services.report_layers import topical_mismatch
                if topical_mismatch(citation['members'][target['member_index']], citation):
                    paint_highlight(target['page_index'], rectangle, (.937,.510,.729), .38, 1.5)
            for page_index, rectangle in _valid_rectangles(
                document, citation.get("quotation_difference_rectangles")
            ):
                document[page_index].draw_rect(
                    rectangle,
                    color=None,
                    fill=(1,.878,.4),
                    width=0.65,
                    fill_opacity=0.35,
                    overlay=True,
                )
                counts["quotation_differences"] += 1

        for finding in reference_practice:
            rendered = False
            rectangles = list(_valid_rectangles(document, finding.get("rectangles")))
            for page_index, rectangle in rectangles:
                if finding.get('finding_type') == 'source_topical_mismatch':
                    paint_highlight(page_index, rectangle, (.937,.510,.729), .38, 1.5)
                if finding.get('finding_type') in LINK_MARKER_FINDINGS:
                    if (page_index,rectangle) == rectangles[-1]:
                        cx,cy=rectangle.x1+10,(rectangle.y0+rectangle.y1)/2
                        if cx+8>document[page_index].rect.width:
                            cx=max(8,rectangle.x0-10)
                        document[page_index].draw_polyline([(cx,cy-6),(cx+6,cy),(cx,cy+6),(cx-6,cy),(cx,cy-6)],
                            color=(.463,.318,.659),fill=(.463,.318,.659),width=1,overlay=True)
                elif finding.get('finding_type') in REFERENCE_DIFFERENCE_FINDINGS:
                    document[page_index].draw_rect(rectangle, color=_DIFFERENCE_BLUE, width=1.5, overlay=True)
                elif finding.get('finding_type') in UNVERIFIED_FINDINGS:
                    paint_highlight(page_index, rectangle, _UNVERIFIED_FILL, .42, 3)
                elif finding.get('finding_type') != 'source_topical_mismatch':
                    color=(1,.895,.36) if finding.get('finding_type') in ACADEMIC_FINDINGS else _AMBER
                    paint_highlight(page_index, rectangle, color, .3,
                        3 if finding.get('finding_type') in ACADEMIC_FINDINGS else 2)
                rendered = True
            if rendered:
                counts["reference_practice_findings"] += 1

        # Patchwriting passages: the Academic Practice yellow, like the other
        # academic-practice findings.
        for passage in passages:
            placed = False
            for page_index, rectangle in _valid_rectangles(
                document, (passage.get('paper_location') or {}).get('rectangles')
            ):
                paint_highlight(page_index, rectangle, (1,.895,.36), .3, 3)
                placed = True
            if placed:
                counts["patchwriting_passages"] += 1

        counts.update(_append_evidence(document, citations, reference_practice, view or {}))
        document.set_metadata(
            {
                "title": "SourceFidelity report",
                "author": "",
                "subject": export_binding,
                "keywords": "SourceFidelity released export",
                "creator": "SourceFidelity",
                "producer": "SourceFidelity",
                "creationDate": _FIXED_PDF_DATE,
                "modDate": _FIXED_PDF_DATE,
            }
        )
        content = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    except ReportExportError:
        raise
    except Exception as exc:
        raise ReportExportError("Released report export could not be rendered") from exc
    finally:
        document.close()
    return content, counts


def _append_evidence(document, citations, findings, view) -> dict:
    """Append inspectable evidence with two-way links."""
    paper_pages = document.page_count
    blocks = [
        '<h1 id="evidence-start">Evidence</h1>',
        '<p>Click a marked passage in the paper to open its evidence here. Each entry links back to the paper. '
        'Evidence availability does not establish citation correctness. Missing text or a retrieval miss does not establish source absence.</p>',
        '<p>Translucent highlights identify individual sources: blue for full text, teal for abstracts, blue-mauve for limited text, '
        'and grey for no retrieved text or not verifiable. Light-grey underlines show the complete citation span. '
        'Select a source highlight to open its source-specific entry; the underline opens the complete citation entry. '
        'Pink highlights indicate a possible topical mismatch. Soft red marks a reference that cannot be verified and its citations; '
        'a blue outline marks reference details that differ from the located record. '
        'Orange highlights identify formatting issues, purple diamonds identify link issues, '
        'and yellow highlights indicate Academic Practice issues.</p>',
    ]
    blocks.append('<h2>Patterns and Issues</h2>')
    from app.services.report_style_guidance import guidance_links
    from app.services.evidence_report import _citation_guidance_kinds
    citation_format = str(view.get('citation_format') or '')
    from app.services.evidence_report import report_summary, summary_text
    for category, priorities in report_summary(view).items():
        if priorities:
            label = 'Citation and Reference Formatting' if category == 'reference_formatting' else category.replace('_',' ').capitalize()
            # The PDF keeps the complete count-only sentence: it has no side
            # window for instance links to open (ARCHITECTURE §8).
            blocks.append('<h3>'+escape(label)+'</h3><ul>'+''.join('<li>'+escape(summary_text(item))+'</li>' for item in priorities)+'</ul>')
    from app.services.evidence_report import _render_upload_priorities
    from app.services.highlight_priority import UNVERIFIED_FINDINGS as _UNVERIFIED
    blocks.append(_render_upload_priorities([{**c,'upload_action':{}} for c in citations],
        unverified=frozenset(f.get('reference_id') for f in findings if f.get('finding_type') in _UNVERIFIED)))
    targets = []
    words_by_page = marker_words(document)
    from app.services.evidence_report import _patchwriting_by_window, render_patchwriting
    passages = [p for p in view.get('patchwriting_passages') or []
                if isinstance(p, dict) and isinstance(p.get('number'), int)]
    patchwriting = _patchwriting_by_window(passages, citations)
    for index, citation in enumerate(citations, 1):
        key = f"citation-{index}"
        rectangles = (citation.get("paper_location") or {}).get("rectangles") or []
        targets.append((key, [{**r,'y0':max(r['y0'],r['y1']-1)} for r in rectangles], f"Citation {index}"))
        blocks.append(f'<h2 id="{key}">Citation {index}</h2><p class="return" id="return-{key}">Back to marked paper passage</p>')
        blocks.append('<blockquote class="student">'+escape(str(citation.get('display_student_text') or citation.get('student_text') or citation.get('citation_marker') or 'Citation text not retained'))+'</blockquote>')
        if citation.get('missing_reference_members'):
            blocks.append('<blockquote>'+escape(_panel_statement(citation.get('boundary_reason') or ''))+'</blockquote>')
        member_marks = member_targets(citation, words_by_page)
        for member_index, member in enumerate(citation.get('members', [])):
            member_key=f'{key}-source-{member_index+1}'
            label=f'Citation {index}, source {member_index+1}'
            marks=[r for r in member_marks if r['member_index']==member_index]
            targets.append((member_key,marks,label))
            blocks.append(f'<h2 id="{member_key}">{label}</h2><p class="return" id="return-{member_key}">Back to source citation</p>')
            blocks.append(_portable_member_html(member))
            if member.get('show_quotation_check'):
                from app.services.evidence_report import render_quotation_comparisons
                comparison = render_quotation_comparisons(
                    member.get('quotation_check') or {},
                    str(citation.get('display_student_text') or citation.get('student_text') or ''), portable=True)
                if comparison:
                    blocks.append('<div>' + comparison + '</div>')
            if ('citation', index, member_index) in patchwriting:
                # Patchwriting with the source's other Academic Practice checks.
                blocks.append('<h3>Academic Practice</h3>' + render_patchwriting(
                    patchwriting[('citation', index, member_index)], portable=True))
            from app.services.report_layers import topical_mismatch
            if topical_mismatch(member, citation):
                scope = member['abstract_relevance']['scope_assessment']
                blocks.append('<p>The abstract appears unrelated to the topic attributed to this source. '
                              +escape(scope['rationale'])+'</p>')
        from app.services.evidence_report import citation_after_punctuation
        if citation_after_punctuation(citation):
            blocks.append("<h3>Citation and Reference Formatting</h3><p>This parenthetical citation is placed after "
                          "the sentence's final punctuation.</p>")
        blocks.append(_render_citation_information(citation, index).replace('<details>', '<div>').replace('</details>', '</div>').replace('<summary>', '<h4>').replace('</summary>', '</h4>'))
        blocks.append(guidance_links(_citation_guidance_kinds(citation), citation_format))
    for index, finding in enumerate(findings, 1):
        key = f"reference-{index}"
        from app.services.highlight_priority import finding_category
        heading = {'evidence': 'Evidence', 'academic': 'Academic Practice'}.get(
            finding_category(finding.get('finding_type')), 'Citation and Reference Formatting') + f' {index}'
        if finding.get('finding_type') in SUBMITTED_LINK_FINDINGS:
            heading = f'Submitted-Link Issue {index}'
        if finding.get('finding_type') == 'source_topical_mismatch':
            heading = f'Potential Topical Mismatch {index}'
        rectangles = finding.get('rectangles') or []
        if finding.get('finding_type') in LINK_MARKER_FINDINGS:
            rectangles = [{**r,'x0':r['x1']+2,'x1':r['x1']+12,
                           'y0':(r['y0']+r['y1'])/2-5,'y1':(r['y0']+r['y1'])/2+5} for r in rectangles[-1:]]
        elif finding.get('finding_type') != 'source_topical_mismatch':
            rectangles = [{**r,'y1':r['y1']+3} for r in rectangles]
        targets.append((key, rectangles, heading))
        return_label = ('Back to marked passage' if finding.get('finding_type') in {'required_quotation_locator_missing', 'body_title_style'}
                        else 'Back to marked reference')
        blocks.append(f'<h2 id="{key}">{heading}</h2><p id="return-{key}" class="return">{return_label}</p>')
        from app.services.reference_credibility import credibility_finding_html, credibility_records_html
        from app.services.evidence_report import _panel_statement
        # No coaching text in the PDF either (owner decision 2026-09-30).
        blocks.append('<p>'+credibility_finding_html({**finding, 'finding': _panel_statement(finding.get('finding') or '')})+'</p>')
        if finding.get('finding_type') == 'required_quotation_locator_missing':
            blocks.append('<blockquote>'+escape(str(finding.get('quote_text') or ''))+'</blockquote>')
        raw_reference = escape(str((finding.get('source') or {}).get('raw_reference') or ''))
        if finding.get('finding_type') == 'reference_identifier_conflict':
            # "The submitted DOI identifies:" is completed by the identified
            # record, never by the student's own reference.
            blocks.append(credibility_records_html(finding))
            blocks.append('<p>Reference as submitted: '+raw_reference+'</p>')
        else:
            blocks.append('<p>'+raw_reference+'</p>')
        if finding.get('finding_type') in {'potentially_fabricated_reference', 'unverified_reference'}:
            blocks.append(credibility_records_html(finding))
        if finding.get('finding_type') == 'source_topical_mismatch':
            blocks.append('<h3>Selected Citation</h3><blockquote>'+escape(finding['citation_text'])+
                '</blockquote><h3>Abstract</h3><blockquote>'+escape(finding['abstract_text'])+
                '</blockquote>')
        difference = finding.get('field_difference') or {}
        if difference and finding.get('finding_type') == 'bibliographic_conflict':
            blocks.append('<p>In your reference: '+escape(str(difference.get('submitted_value') or 'Not retained'))+'</p>')
        if finding.get('related_references'):
            related_label = ('Entries identifying the same source:'
                             if finding.get('finding_type') == 'duplicate_reference_entry'
                             else 'References sharing this author and year:')
            blocks.append('<p>'+related_label+'</p><ul>' + ''.join(
                '<li>'+escape(str(peer.get('raw_reference') or ''))+'</li>'
                for peer in finding['related_references']) + '</ul>')
        if finding.get('located_record'):
            located = finding['located_record']
            fields = [', '.join(located.get('authors') or []), str(located.get('year') or ''), str(located.get('title') or ''), str(located.get('container_title') or ''), str(located.get('doi') or '')]
            blocks.append('<p>Located record: '+escape('. '.join(field for field in fields if field))+'</p>')
        blocks.append(guidance_links([finding.get('finding_type')], citation_format))
    # A patchwriting highlight links to the section that shows it: its
    # citation's source, else a Reference section for words no citation covers.
    from app.services.patchwriting_report import passage_windows
    from app.services.report_references import reference_numbers
    numbers = reference_numbers(view)
    written: set = set()
    for passage in passages:
        window = next(iter(passage_windows(passage, citations)), None)
        rectangles = [{**r,'y1':r['y1']+3} for r in (passage.get('paper_location') or {}).get('rectangles') or []]
        if window is None:
            continue
        if window['citation']:
            targets.append((f"passage-{passage['number']}", rectangles,
                            f"Citation {window['citation']}, source {window['member_index']+1}"))
            continue
        number = numbers.get(window['reference_id'])
        if not number:
            continue
        key, heading = f"reference-entry-{number}", f"Reference {number}"
        targets.append((f"passage-{passage['number']}", rectangles, heading))
        if key in written:
            continue
        written.add(key)
        source = window['item'].get('source') or {}
        blocks.append(f'<h2 id="{key}">{heading}</h2>'
                      '<p class="full-reference">'+escape(str(source.get('raw_reference') or source.get('title') or ''))+'</p>'
                      '<h3>Academic Practice</h3>' + render_patchwriting(patchwriting.get(('reference', window['reference_id'])),
                                                                          portable=True))
    positions = {}
    def position(element):
        if element.id and element.id not in positions and not fitz.Rect(element.rect).is_empty:
            positions[element.id] = (element.page_num - 1, fitz.Rect(element.rect))
    css = '''body {margin:0;font-family:sans-serif;font-size:10pt;line-height:1.4;color:#18212b}
        h1 {font-size:19pt} h2 {font-size:14pt;margin-top:20pt}
        h3 {font-size:11pt} h4 {font-size:9pt}
        h3.full_text {border-bottom:1pt solid #2563a7} h3.abstract_only,h3.partial_text {border-bottom:1pt solid #168c8c} h3.unavailable {border-bottom:1pt solid #b8c0c8}
        p {margin:6pt 0} .member {margin-bottom:12pt} .full-reference {font-size:8pt;color:#4c5660}
        blockquote {margin:8pt 12pt;padding:8pt;background-color:#f7f4ea}
        .student {background-color:#eef6ff} .return,a {color:#2563a7;text-decoration:underline;font-size:8pt}
        .metrics {font-size:7pt;color:#5b6672} .muted,.locator {font-size:9pt;color:#5b6672}
        .attention-text {color:#d95f02}'''
    appendix = _write_evidence_blocks(blocks, css, position)
    try:
        document.insert_pdf(appendix)
    finally:
        appendix.close()
    links = 0
    toc = [[1, "Submitted Paper", 1], [1, "Evidence", paper_pages + 1]]
    # Position callbacks can report an element that the layout engine never
    # paints. Bind destinations to actual rendered headings, not callbacks.
    rendered_headings = {}
    for page_index in range(paper_pages, document.page_count):
        for block in document[page_index].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                spans = line.get('spans', [])
                if spans and all(abs(s['size']-14)<.1 for s in spans):
                    rendered_headings[''.join(s['text'] for s in spans).strip()] = (
                        page_index-paper_pages, fitz.Rect(line['bbox']))
    for key, rectangles, label in targets:
        if label not in rendered_headings:
            raise ReportExportError(f"Evidence appendix destination is missing: {key}")
        positions[key] = rendered_headings[label]
        target_page, target_rect = positions[key]
        target_page += paper_pages
        toc.append([2, label, target_page + 1])
        valid = [(p,r) for p,r in _valid_rectangles(document, rectangles) if p < paper_pages]
        for page_index, rectangle in valid:
            document[page_index].insert_link({"kind":fitz.LINK_GOTO,"from":rectangle,"page":target_page,"to":target_rect.tl})
            links += 1
        if valid and 'return-'+key in positions:
            back_page, back_rect=positions['return-'+key]
            document[back_page+paper_pages].insert_link({"kind":fitz.LINK_GOTO,"from":back_rect,"page":valid[0][0],"to":valid[0][1].tl})
            links += 1
    document.set_toc(toc)
    # The PDF carries no processing or cost details; those are an
    # instructor-only section of the HTML report (ARCHITECTURE §8).
    for index in range(paper_pages, document.page_count):
        document[index].insert_text((42,775),f"Evidence appendix {index-paper_pages+1} / {document.page_count-paper_pages}",fontsize=8,color=(.36,.4,.45))
    return {"paper_pages":paper_pages,"evidence_appendix_pages":document.page_count-paper_pages,"internal_evidence_links":links}


def _write_evidence_blocks(blocks, css, position):
    """Flow independent blocks on shared pages, avoiding long-DOM text loss.

    The supported Story placement API keeps heading space and explicit page
    positions. A block can span pages, but one citation cannot swallow the next.
    """
    stream = io.BytesIO()
    writer = fitz.DocumentWriter(stream)
    page_number, y = 1, 40.0
    device = writer.begin_page(fitz.Rect(0, 0, 612, 792))
    for block in blocks:
        minimum = 150 if block.startswith('<h2') else 75 if block.startswith(('<h3', '<h4')) else 35
        if 750-y < minimum:
            writer.end_page()
            device = writer.begin_page(fitz.Rect(0, 0, 612, 792))
            page_number, y = page_number+1, 40.0
        story = fitz.Story(html='<html><body>'+block+'</body></html>', user_css=css)
        while True:
            more, filled = story.place(fitz.Rect(42, y, 570, 750))
            story.element_positions(position, {'page_num': page_number})
            story.draw(device)
            if not more:
                y = fitz.Rect(filled).y1 + 3
                break
            writer.end_page()
            device = writer.begin_page(fitz.Rect(0, 0, 612, 792))
            page_number, y = page_number+1, 40.0
            if page_number > 2000:
                writer.end_page()
                writer.close()
                raise ReportExportError('Evidence appendix exceeds the export page limit')
    writer.end_page()
    writer.close()
    return fitz.open(stream=stream.getvalue(), filetype='pdf')


def _portable_member_html(member: dict) -> str:
    """Use the PDF engine's supported block tags, with no interactive widgets."""
    source=member.get('source') or {}
    from app.services.evidence_report import (
        _member_is_media, _display_evidence_note, _member_evidence_extracts,
        _member_evidence_contexts, _panel_statement,
    )
    media = _member_is_media(member)
    parts = ['<h3>Media Reference - Cannot Retrieve</h3>'] if media else []
    parts.append('<p class="full-reference">'+escape(str(source.get('raw_reference') or source.get('title') or 'Reference unavailable'))+'</p>')
    if not media and member.get('availability') and member['availability'] != 'Source Not Retrieved':
        parts.append('<p>'+escape(_panel_statement(member['availability']))+'</p>')
    disagreement = member.get('scope_disagreement') or {}
    if not media and disagreement.get('note'):
        parts.append('<p class="muted">'+escape(_panel_statement(disagreement['note']))+'</p>')
    for primary in _member_evidence_extracts(member):
        parts.append('<blockquote>'+escape(str(primary.get('display_text', primary.get('text')) or ''))+_inline_locator(primary)+'</blockquote>')
        for key in ('evidence_note',):
            if primary.get(key):
                note = _display_evidence_note(member, primary)
                if note:
                    parts.append('<p class="muted">'+escape(note)+'</p>')
    contexts = _member_evidence_contexts(member)
    if contexts:
        parts.append('<h4>Additional Evidence and Context</h4>')
    for item in contexts:
        parts.append('<blockquote>'+escape(str(item['context_text']))+_inline_locator(item)+'</blockquote>')
        note = _display_evidence_note(member, item)
        if note:
            parts.append('<p class="muted">'+escape(note)+'</p>')
    from app.services.evidence_report import secondary_citation_line
    secondary = secondary_citation_line(member.get('secondary_citation'))
    if secondary:
        parts.append('<p>'+escape(secondary)+'</p>')
    for key,label in (('quotation','Quotation'),('locator','Locator')):
        check=member.get(key+'_check') or {}
        if member.get('show_'+key+'_check') and member.get('coverage_level') != 'unavailable':
            parts.append('<p>'+escape(_panel_statement(_check_sentence(label, check)))+'</p>')
    return '<div>'+''.join(parts)+'</div>'


def _valid_rectangles(document, items) -> list[tuple[int, fitz.Rect]]:
    result: list[tuple[int, fitz.Rect]] = []
    for item in items or []:
        try:
            page_index = int(item["page_index"])
            coordinates = tuple(
                float(item[key]) for key in ("x0", "y0", "x1", "y1")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in coordinates):
            continue
        rectangle = fitz.Rect(*coordinates)
        if not 0 <= page_index < document.page_count or rectangle.is_empty:
            continue
        if not document[page_index].rect.contains(rectangle):
            continue
        result.append((page_index, rectangle))
    return result


def _page_count(content: bytes) -> int:
    document = fitz.open(stream=content, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
