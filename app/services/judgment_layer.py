"""Judgment layout marks: where each judged claim is underlined on the paper (Phase C).

A claim is a clause-level verification candidate; its segments carry absolute
paper character offsets. Those offsets are mapped onto the retained PDF's own
word boxes inside the citation's located rectangles: both texts are reduced to
a stream of NFKC case-folded letters and digits (so line-break hyphens, quotes
and ligatures cannot break the match), the citation's stream must occur
exactly once in the page words, and each claim character selects its word.
When that is not possible the whole citation span is underlined and the
window says so; with no located citation the claim is reachable from the
window only. Geometry is presentation only, never evidence.
"""
from __future__ import annotations

import math
import unicodedata
import uuid

from sqlalchemy.orm import Session

from app.models.report import VerificationReportRecord
from app.services.judgment_input import judgment_eligibility
from app.services.judgment_report import candidate_paper_ranges, candidate_text

WHOLE_CITATION = "__citation__"


def _normalized(text: str, origin) -> tuple[str, list]:
    chars, origins = [], []
    for index, char in enumerate(str(text)):
        for piece in unicodedata.normalize("NFKC", char).casefold():
            if piece.isalnum():
                chars.append(piece)
                origins.append(origin(index))
    return "".join(chars), origins


def _citation_words(citation: dict, words_by_page: dict) -> list[tuple[int, tuple]]:
    words = []
    for rect in (citation.get("paper_location") or {}).get("rectangles") or []:
        try:
            page = int(rect["page_index"])
            x0, y0, x1, y1 = (float(rect[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        for word in words_by_page.get(page, words_by_page.get(str(page), [])):
            cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
            if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                item = (page, tuple(word[:5]))
                if item not in words:
                    words.append(item)
    return words


def _merge_per_line(boxes: list[tuple[int, tuple]]) -> list[dict]:
    merged: list[dict] = []
    for page, word in boxes:
        previous = merged[-1] if merged else None
        if (previous and previous["page_index"] == page and word[0] >= previous["x1"] - 1
                and min(previous["y1"], word[3]) - max(previous["y0"], word[1])
                > .5 * min(previous["y1"] - previous["y0"], word[3] - word[1])):
            previous.update(x1=word[2], y0=min(previous["y0"], word[1]), y1=max(previous["y1"], word[3]))
        else:
            merged.append({"page_index": page, "x0": word[0], "y0": word[1], "x1": word[2], "y1": word[3]})
    return merged


def claim_span_rectangles(citation: dict, ranges: list[tuple[int, int]],
                          words_by_page: dict) -> tuple[list[dict], str]:
    """(rectangles, placement): placement is exact, citation_span or none."""
    location = citation.get("paper_location") or {}
    if location.get("localization_level") != "exact_rectangle" or not location.get("rectangles"):
        return [], "none"
    whole = [{k: r[k] for k in ("page_index", "x0", "y0", "x1", "y1")} for r in location["rectangles"]]
    student = str(citation.get("student_text") or "")
    start = citation.get("paper_character_start")
    words = _citation_words(citation, words_by_page)
    if not student or not isinstance(start, int) or not words or not ranges:
        return whole, "citation_span"
    student_stream, student_origin = _normalized(student, lambda i: i)
    pdf_chars, pdf_origin = [], []
    for index, (_, word) in enumerate(words):
        stream, origin = _normalized(word[4], lambda _i, w=index: w)
        pdf_chars.append(stream)
        pdf_origin.extend(origin)
    pdf_stream = "".join(pdf_chars)
    at = pdf_stream.find(student_stream)
    if not student_stream or at < 0 or pdf_stream.find(student_stream, at + 1) >= 0:
        return whole, "citation_span"
    selected = set()
    for paper_start, paper_end in ranges:
        local_start, local_end = paper_start - start, paper_end - start
        if local_start < 0 or local_end > len(student):
            return whole, "citation_span"
        for position, origin in enumerate(student_origin):
            if local_start <= origin < local_end:
                selected.add(pdf_origin[at + position])
    if not selected:
        return whole, "citation_span"
    return _merge_per_line([words[i] for i in sorted(selected)]), "exact"


def _record(session: Session, record_id, scope_type: str, scope_id: str, paper_version_id):
    try:
        record = session.get(VerificationReportRecord, uuid.UUID(str(record_id)))
    except ValueError:
        return None
    if (record is None or (record.scope_type, record.scope_id) != (scope_type, scope_id)
            or record.paper_version_id != paper_version_id):
        return None
    return record


def build_judgment_layer(session: Session, view: dict, words_by_page: dict, *,
                         scope_type: str, scope_id: str) -> dict:
    """Every claim mark, in paper order; judged ones start pending."""
    marks = []
    for citation in view.get("citations") or []:
        number = citation.get("citation_number")
        members = [m for m in citation.get("members") or [] if m.get("verification_report_id")]
        whole_rects, whole_placement = claim_span_rectangles(
            citation, [(citation.get("paper_character_start") or 0,
                        (citation.get("paper_character_start") or 0) + len(citation.get("student_text") or ""))],
            words_by_page)
        static = {"key": f"{number}:static", "group": f"{number}:whole", "citation": number, "record": None,
                  "candidate": WHOLE_CITATION, "state": "not_judged", "static": True,
                  "claim": citation.get("student_text") or "",
                  "rects": whole_rects, "placement": whole_placement,
                  "order": citation.get("paper_character_start") or 0}
        if not members:
            # No stored evidence at all: not judged; the window heading says so.
            marks.append(static)
            continue
        before = len(marks)
        for member in members:
            record = _record(session, member["verification_report_id"], scope_type, scope_id,
                             view.get("paper_version_id"))
            if record is None:
                continue
            payload = record.report_payload or {}
            bundles = (payload.get("facet_evidence_foundation") or {}).get("candidate_bundles") or []
            if not judgment_eligibility(payload).eligible or not bundles:
                marks.append({"key": f"{number}:{record.id}:{WHOLE_CITATION}", "group": f"{number}:whole",
                              "citation": number, "record": str(record.id), "candidate": WHOLE_CITATION,
                              "claim": citation.get("student_text") or "",
                              "state": "pending", "rects": whole_rects, "placement": whole_placement,
                              "order": citation.get("paper_character_start") or 0})
                continue
            for bundle in bundles:
                ranges = candidate_paper_ranges(payload, bundle["candidate_id"])
                rects, placement = claim_span_rectangles(citation, ranges, words_by_page)
                marks.append({"key": f"{number}:{record.id}:{bundle['candidate_id']}",
                              "group": f"{number}:" + ",".join(f"{a}-{b}" for a, b in ranges),
                              "citation": number, "record": str(record.id),
                              "candidate": bundle["candidate_id"], "state": "pending",
                              "claim": candidate_text(payload, bundle["candidate_id"]),
                              "rects": rects, "placement": placement, "ranges": ranges,
                              "order": min((a for a, _ in ranges), default=citation.get("paper_character_start") or 0)})
        if len(marks) == before:
            # Every stored record was out of scope or missing: the citation is
            # still underlined, as not judged (the citation-span line is hidden).
            marks.append(static)
        _separate_contained(marks[before:], citation, words_by_page)
    marks.sort(key=lambda m: (m["order"], m["citation"] or 0))
    return {"marks": marks}


def _separate_contained(marks: list[dict], citation: dict, words_by_page: dict) -> None:
    """A proposition judged with the clause it modifies ("…, allowing her to
    grow …" with its main clause) contains that clause's words. It is
    underlined and bolded only where it adds words, so the two do not overlap
    (owner review 2026-09-29); what the judge read is unchanged."""
    def positions(ranges):
        return {p for a, b in ranges for p in range(a, b)}

    groups = {m["group"]: m["ranges"] for m in marks if m.get("ranges")}
    for mark in marks:
        own = positions(mark.get("ranges") or [])
        inner = set()
        for group, ranges in groups.items():
            other = positions(ranges)
            if group != mark["group"] and other and other < own:
                inner |= other
        if not inner:
            continue
        rest = sorted(own - inner)
        shown = []
        for p in rest:
            if shown and shown[-1][1] == p:
                shown[-1][1] = p + 1
            else:
                shown.append([p, p + 1])
        shown = [(a, b) for a, b in shown if b - a > 1]
        if not shown:
            continue
        rects, placement = claim_span_rectangles(citation, shown, words_by_page)
        if placement == "exact":
            mark["rects"], mark["display_ranges"] = rects, shown


JUDGMENT_CSS = (
    "/* Judgment marks and result blocks. */"
    ":root{--j-teal:#0f766e;--j-amber:#b45309;--j-purple:#7e22ce;--j-red:#b91c1c;--j-grey:#c5cbd1;--j-wait:#c5cbd1}"
    ".judgment-overlay{cursor:pointer}"
    ".judgment-line{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}"
    ".judgment-line.state-supported{stroke:var(--j-teal)}"
    ".judgment-line.state-qualified{stroke:var(--j-amber);stroke-dasharray:7 3}"
    ".judgment-line.state-contradicts{stroke:var(--j-purple);stroke-width:2.6;stroke-dasharray:0 3.4;stroke-linecap:round}"
    ".judgment-line.state-insufficient{stroke:var(--j-red);stroke-width:1.6}"
    # Not judged: the light-grey underline the report drew under every citation
    # span, which the judgment underlines now replace (owner request 2026-09-28).
    ".judgment-line.state-not_judged{stroke:var(--j-grey);stroke-width:1}"
    ".citation-overlay [data-citation-span]{display:none}"
    # Undecided: the not-judged grey, doubled (owner decision 2026-09-29).
    ".judgment-line.state-undecided{stroke:var(--j-grey);stroke-width:1}"
    ".judgment-line.state-pending{stroke:var(--j-wait);stroke-width:1;stroke-dasharray:1 2}"
    ".judgment-hit{fill:transparent;pointer-events:all}"
    # Light blue only, hover and selection alike (owner request 2026-09-28).
    ".judgment-overlay.hovered .judgment-hit,.judgment-overlay.selected .judgment-hit{fill:#b9dcff;fill-opacity:.22}"
    ".citation-overlay.proposition-mode.selected .selection-bg{fill-opacity:0}"
    ".judgment-overlay:focus{outline:none}.judgment-overlay:focus .judgment-hit{stroke:#2563eb;stroke-width:1;vector-effect:non-scaling-stroke}"
    # A result label is underlined in the same colour and style as the paper.
    ".jk{text-decoration-line:underline;text-decoration-thickness:2px;text-underline-offset:.3em}"
    ".jk.state-supported{text-decoration-color:var(--j-teal);text-decoration-style:solid}"
    ".jk.state-qualified{text-decoration-color:var(--j-amber);text-decoration-style:dashed}"
    ".jk.state-contradicts{text-decoration-color:var(--j-purple);text-decoration-style:dotted;text-decoration-thickness:3px}"
    ".jk.state-insufficient{text-decoration-color:var(--j-red);text-decoration-style:double}"
    ".jk.state-not_judged{text-decoration-color:var(--j-grey);text-decoration-style:solid}"
    ".jk.state-undecided{text-decoration-color:var(--j-grey);text-decoration-style:double}"
    ".jw-standin{background:#fef3c7;color:#78350f;padding:.3rem .5rem;border-radius:3px;font-size:.85rem}"
    ".jw-why{color:#334155}.jw-coaching{margin:.6rem 0}"
    ".judgment-window{margin:.4rem 0 .8rem}.judgment-window+.judgment-window{border-top:1px dashed #cbd5e1;padding-top:.6rem}"
    "#judgment-live{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}"
)


def render_judgment_assets(layer: dict, *, report_id: str, nonce: str) -> str:
    """The data block, live region and script for the Judgment layout."""
    import json
    from html import escape
    from pathlib import Path

    payload = {"report_id": report_id, "marks": layer.get("marks") or [], "fake_panel": bool(layer.get("fake_panel"))}
    if layer.get("static"):
        payload["static_results"] = layer.get("static_results") or []
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    script = Path(__file__).with_name("report_judgment.js").read_text()
    return (f'<script type="application/json" id="judgment-data">{data}</script>'
            f'<div id="judgment-live" role="status" aria-live="polite"></div>'
            f'<script nonce="{escape(nonce)}">{script}</script>')
