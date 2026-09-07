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
    load_authorized_evidence_report_bundle, citation_tones, grouped_members,
    _render_citation_information, _processing_label,
    _inline_locator, _check_sentence,
)
from app.services.paper_annotations import list_current_paper_annotations
from app.services.storage.backend import StorageBackend


REPORT_EXPORT_VERSION = "released-report-pdf-v2"

_TONE_COLORS = {
    "evidence_available": (0.145, 0.388, 0.655),
    "limited_evidence": (0.086, 0.549, 0.549),
    "retrieved_no_connection": (0.549, 0.420, 0.310),
    "attention": (0.851, 0.373, 0.008),
    "not_assessed": (0.722, 0.753, 0.784),
}
_VIOLET = (0.463, 0.318, 0.659)
_AMBER = (0.851, 0.373, 0.008)
_QUOTE_FILL = (1.000, 0.824, 0.478)
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
    """Build one released-only derivative without persisting a partial object."""
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

    annotations = list_current_paper_annotations(
        session,
        report_id=parsed_report_id,
        scope_type=scope_type,
        scope_id=scope_id,
        visibility="released",
    )
    _validate_annotation_bindings(
        annotations,
        artifact_id=str(artifact.id),
        paper_version_id=artifact.paper_version_id,
        paper_sha256=paper_sha256,
    )

    manifest = _base_manifest(
        report=report,
        view=view,
        artifact=artifact,
        paper_sha256=paper_sha256,
        annotations=annotations,
    )
    content, overlay_counts = _render_pdf(
        paper_content,
        citations=list(view.get("citations") or []),
        reference_practice=list(view.get("reference_practice") or []),
        annotations=annotations,
        view=view,
        export_binding=(
            f"report={manifest['report_id']};"
            f"report_version={manifest['report_version']};"
            f"paper_sha256={manifest['paper_presentation_sha256']};"
            f"released_annotations={manifest['released_annotation_set_sha256']}"
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


def _base_manifest(*, report, view, artifact, paper_sha256, annotations) -> dict:
    annotation_revisions = [
        {
            "annotation_id": item["annotation_id"],
            "revision": int(item["revision"]),
            "annotation_type": item["annotation_type"],
            "anchor_sha256": item["anchor_sha256"],
            "content_sha256": hashlib.sha256(
                str(item.get("content") or "").encode("utf-8")
            ).hexdigest(),
            "user_label_sha256": hashlib.sha256(
                str(item.get("user_label") or "").encode("utf-8")
            ).hexdigest(),
        }
        for item in annotations
    ]
    annotation_set_sha256 = _digest({"revisions": annotation_revisions})
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
        "released_annotation_revisions": annotation_revisions,
        "released_annotation_set_sha256": annotation_set_sha256,
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
    annotations: list[dict],
    export_binding: str,
    view: dict | None = None,
) -> tuple[bytes, dict[str, int]]:
    counts = {
        "citation_underlines": 0,
        "quotation_differences": 0,
        "reference_practice_findings": 0,
        "released_highlights": 0,
        "released_comments": 0,
    }
    try:
        document = fitz.open(stream=paper_content, filetype="pdf")
    except Exception as exc:
        raise ReportExportError("Retained paper presentation is not a readable PDF") from exc
    try:
        for citation in citations:
            tone = str(citation.get("tone") or "not_assessed")
            tones = citation_tones(citation)
            for page_index, rectangle in _valid_rectangles(
                document, (citation.get("paper_location") or {}).get("rectangles")
            ):
                page = document[page_index]
                for part, member_state in enumerate(tones):
                    page.draw_line(
                        (rectangle.x0 + rectangle.width * part / len(tones), max(rectangle.y0, rectangle.y1 - 0.8)),
                        (rectangle.x0 + rectangle.width * (part+1) / len(tones), max(rectangle.y0, rectangle.y1 - 0.8)),
                        color=_TONE_COLORS[member_state], width=0.9, overlay=True,
                    )
                counts["citation_underlines"] += 1
            for page_index, rectangle in _valid_rectangles(
                document, citation.get("quotation_difference_rectangles")
            ):
                document[page_index].draw_rect(
                    rectangle,
                    color=_AMBER,
                    fill=_QUOTE_FILL,
                    width=0.65,
                    fill_opacity=0.55,
                    overlay=True,
                )
                counts["quotation_differences"] += 1

        for finding in reference_practice:
            rendered = False
            for page_index, rectangle in _valid_rectangles(
                document, finding.get("rectangles")
            ):
                document[page_index].draw_rect(
                    rectangle,
                    color=_AMBER,
                    fill=_AMBER,
                    width=0.8,
                    fill_opacity=0.12,
                    overlay=True,
                )
                rendered = True
            if rendered:
                counts["reference_practice_findings"] += 1

        comment_offsets: dict[tuple[int, str], int] = {}
        for annotation in annotations:
            rectangles = _valid_rectangles(
                document, (annotation.get("anchor") or {}).get("rectangles")
            )
            if annotation.get("annotation_type") == "highlight":
                for page_index, rectangle in rectangles:
                    document[page_index].draw_rect(
                        rectangle,
                        color=None,
                        fill=_VIOLET,
                        fill_opacity=0.22,
                        overlay=True,
                    )
                    counts["released_highlights"] += 1
                continue
            if annotation.get("annotation_type") != "comment" or not rectangles:
                continue
            page_index, rectangle = rectangles[0]
            key = (page_index, str((annotation.get("anchor") or {}).get("anchor_id") or ""))
            offset = comment_offsets.get(key, 0)
            comment_offsets[key] = offset + 1
            point = fitz.Point(
                min(document[page_index].rect.x1 - 18, rectangle.x1 + offset * 10),
                min(document[page_index].rect.y1 - 18, rectangle.y0 + offset * 10),
            )
            note = document[page_index].add_text_annot(
                point, str(annotation.get("content") or "")
            )
            note.set_colors(stroke=_VIOLET)
            note.set_info(
                title="SourceFidelity",
                subject=str(annotation.get("user_label") or "Released instructor comment"),
                content=str(annotation.get("content") or ""),
                creationDate=_FIXED_PDF_DATE,
                modDate=_FIXED_PDF_DATE,
            )
            note.update()
            counts["released_comments"] += 1

        counts.update(_append_evidence(document, citations, reference_practice, annotations, view or {}))
        document.set_metadata(
            {
                "title": "SourceFidelity released annotated report",
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


def _append_evidence(document, citations, findings, annotations, view) -> dict:
    """Append inspectable evidence and printed comments with two-way links."""
    paper_pages = document.page_count
    blocks = [
        '<h1 id="evidence-start">Evidence and released comments</h1>',
        '<p>Click a marked passage in the paper to open its evidence here. Each entry links back to the paper. '
        'Evidence availability does not establish citation correctness. Missing text or a retrieval miss does not establish source absence.</p>',
        '<p>Blue: full text available; teal: abstract or limited text available; brown: completed passage check with no clear matching passage; '
        'orange: a specific issue to check; grey: no retrieved text or not verifiable. Each source occupies an equal share of the underline, in the order shown below.</p>',
    ]
    for category, priorities in (view.get("role_summaries", {}).get("student", {})).items():
        if priorities:
            blocks.append('<h3>'+escape(category.replace('_',' ').capitalize())+'</h3><ul>'+''.join('<li>'+escape(item)+'</li>' for item in priorities)+'</ul>')
    targets = []
    for index, citation in enumerate(citations, 1):
        key = f"citation-{index}"
        rectangles = (citation.get("paper_location") or {}).get("rectangles") or []
        targets.append((key, rectangles, f"Citation {index}"))
        blocks.append(f'<h2 id="{key}">Citation {index}</h2><p class="return" id="return-{key}">Back to marked paper passage</p>')
        blocks.append('<blockquote class="student">'+escape(str(citation.get('display_student_text') or citation.get('student_text') or citation.get('citation_marker') or 'Citation text not retained'))+'</blockquote>')
        for label, members in grouped_members(citation):
            level = members[0].get('coverage_level') or 'unavailable'
            blocks.append('<h3 class="'+escape(level, quote=True)+'">'+escape(label)+'</h3>')
            for member in members:
                # Public reference URLs can remain clickable; source access,
                # mutation URLs and complete source bytes are never portable.
                blocks.append(_portable_member_html(member))
        blocks.append(_render_citation_information(citation, index).replace('<details>', '<div>').replace('</details>', '</div>').replace('<summary>', '<h4>').replace('</summary>', '</h4>'))
    for index, finding in enumerate(findings, 1):
        key = f"reference-{index}"
        targets.append((key, finding.get("rectangles") or [], f"Reference practice {index}"))
        blocks.append(f'<h2 id="{key}">Reference practice {index}</h2><p id="return-{key}" class="return">Back to marked reference</p>')
        blocks.append('<p>'+escape(str(finding.get('finding') or 'Reference practice finding'))+'</p>')
        blocks.append('<p>'+escape(str((finding.get('source') or {}).get('raw_reference') or ''))+'</p>')
        difference = finding.get('field_difference') or {}
        if difference and finding.get('finding_type') == 'bibliographic_conflict':
            blocks.append('<p>In your reference: '+escape(str(difference.get('submitted_value') or 'Not retained'))+
                          '<br>In the located record: '+escape(str(difference.get('located_value') or 'Not retained'))+
                          '<br>Record provider: '+escape(str(difference.get('provider') or 'Not retained'))+'</p>')
        if finding.get('related_references'):
            blocks.append('<p>References sharing this author and year:</p><ul>' + ''.join(
                '<li>'+escape(str(peer.get('raw_reference') or ''))+'</li>'
                for peer in finding['related_references']) + '</ul>')
        if finding.get('located_record'):
            located = finding['located_record']
            fields = [', '.join(located.get('authors') or []), str(located.get('year') or ''), str(located.get('title') or ''), str(located.get('container_title') or ''), str(located.get('doi') or '')]
            blocks.append('<p>Located record: '+escape('. '.join(field for field in fields if field))+'</p>')
    for index, annotation in enumerate(annotations, 1):
        if annotation.get('annotation_type') != 'comment':
            continue
        key=f"comment-{index}"
        targets.append((key, (annotation.get('anchor') or {}).get('rectangles') or [], f"Released comment {index}"))
        blocks.append(f'<h2 id="{key}">Released comment {index}</h2><p id="return-{key}" class="return">Back to commented passage</p><p>'+escape(str(annotation.get('content') or ''))+'</p>')
    blocks.append('<p class="metrics">'+escape(_processing_label(view.get('processing_metrics') or {}))+'</p>')
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
    toc = [[1, "Submitted paper", 1], [1, "Evidence and released comments", paper_pages + 1]]
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
    parts = ['<p class="full-reference">'+escape(str(source.get('raw_reference') or source.get('title') or 'Reference unavailable'))+'</p>']
    if member.get('availability') and member['availability'] != 'Source not retrieved':
        parts.append('<p>'+escape(member['availability'])+'</p>')
    primary = member.get('best_evidence') or {}
    if primary:
        parts.append('<blockquote>'+escape(str(primary.get('display_text') or primary.get('text') or ''))+_inline_locator(primary)+'</blockquote>')
        for key in ('evidence_note',):
            if primary.get(key):
                parts.append('<p class="muted">'+escape(str(primary[key]))+'</p>')
        if primary.get('context_text'):
            parts.append('<h4>Full context for the selected excerpt</h4><blockquote>'+escape(primary['context_text'])+_inline_locator(primary)+'</blockquote>')
    for item in member.get('additional_evidence') or []:
        parts.append('<h4>Additional evidence and context</h4><blockquote>'+escape(str(item.get('context_text') or item.get('text') or ''))+_inline_locator(item)+'</blockquote>')
    for key,label in (('quotation','Quotation'),('locator','Locator')):
        check=member.get(key+'_check') or {}
        if member.get('show_'+key+'_check') and member.get('coverage_level') != 'unavailable':
            parts.append('<p>'+escape(_check_sentence(label, check))+'</p>')
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


def _validate_annotation_bindings(
    annotations: list[dict],
    *,
    artifact_id: str,
    paper_version_id: str,
    paper_sha256: str,
) -> None:
    for item in annotations:
        if (
            item.get("visibility") != "released"
            or item.get("paper_artifact_id") != artifact_id
            or item.get("paper_version_id") != paper_version_id
            or item.get("paper_content_sha256") != paper_sha256
        ):
            raise ReportExportError("Released annotation does not match the paper surface")


def _page_count(content: bytes) -> int:
    document = fitz.open(stream=content, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
