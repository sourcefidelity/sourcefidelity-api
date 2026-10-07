"""Checks that must hold in every rendered report (owner request 2026-10-07).

Unit tests check each rule on invented examples; these check the finished
report a reader sees, so defects that come from real files or from correct
parts not joined up are caught before a report is reviewed. Each check returns
violations as {"check", "window", "detail"}; an empty list passes. Paper-specific
expectations (owner-confirmed findings) are applied by ``check_expectations``.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString

_RETRIEVAL_PHRASE = re.compile(r"^\s*(?:retrieved?|available|accessed)\b[^.]{0,30}:?\s*$", re.IGNORECASE)
_ADDRESS_PIECE = re.compile(r"^[\w\-.~%/?#=&+:@;!*']+$")


def _windows(soup: BeautifulSoup):
    for template in soup.find_all("template"):
        inner = BeautifulSoup(template.decode_contents(), "html.parser")
        yield template.get("id") or "", inner


def split_links(soup: BeautifulSoup) -> list[dict]:
    """A reference's web address shown partly as a link and partly as plain
    text: text glued to the link's end, or a single address-like piece left
    after it at the end of the entry."""
    out = []
    places = list(_windows(soup)) + [("sources-to-upload", soup.select_one("section.upload-priorities") or BeautifulSoup("", "html.parser"))]
    for window, inner in places:
        # The student's own entry; a located record's page address is shown
        # as text on purpose when provider terms bar linking it.
        selector = "li" if window == "sources-to-upload" else ".full-reference"
        entries = [e for e in inner.select(selector) if not e.find_parent(class_="record-difference-block")]
        for entry in entries:
            for anchor in entry.find_all("a", href=re.compile(r"^https?://")):
                after = anchor.next_sibling
                text = str(after) if isinstance(after, NavigableString) else ""
                pieces = text.split()
                glued = bool(text) and not text[0].isspace() and bool(pieces) \
                    and _ADDRESS_PIECE.match(pieces[0].rstrip(".")) and pieces[0].rstrip(".")
                trailing = len(pieces) == 1 and anchor.find_next_sibling() is None \
                    and _ADDRESS_PIECE.match(pieces[0].rstrip(".")) and re.search(r"[=&/_%?#]", pieces[0])
                if glued or trailing:
                    out.append({"check": "split_link", "window": window,
                                "detail": f"{anchor.get('href')} | {text.strip()[:60]}"})
            for node in entry.find_all(string=re.compile(r"https?://")):
                if node.find_parent("a") is None:
                    out.append({"check": "unlinked_address", "window": window, "detail": str(node).strip()[:80]})
    return out


def missing_windows(soup: BeautifulSoup) -> list[dict]:
    """A summary item, button or mark pointing at a window that does not exist."""
    ids = {t.get("id") for t in soup.find_all("template")}
    window_id = re.compile(r"(?:citation|reference-entry)-panel-\d+$")
    out = []
    places = [("", soup)] + list(_windows(soup))
    for window, root in places:
        for element in root.select("[data-go-to], [data-panel-template]"):
            target = element.get("data-go-to") or element.get("data-panel-template")
            if target and window_id.match(target) and target not in ids:
                out.append({"check": "missing_window", "window": window or "page", "detail": f"points to {target}"})
    return out


def retrieval_phrase_titles(view: dict) -> list[dict]:
    """A reference whose parsed title is a retrieval phrase ("Retrieved from:")."""
    sources = [row.get("source") or {} for row in view.get("bibliography") or []]
    sources += [m.get("source") or {} for c in view.get("citations") or [] for m in c.get("members") or []]
    seen, out = set(), []
    for source in sources:
        title = str(source.get("title") or "")
        if title and _RETRIEVAL_PHRASE.match(title) and source.get("raw_reference") not in seen:
            seen.add(source.get("raw_reference"))
            out.append({"check": "retrieval_phrase_title", "window": "",
                        "detail": str(source.get("raw_reference") or "")[:80]})
    for finding in view.get("reference_practice") or []:
        if re.search(r"“\s*(?:retrieved?|available)\b[^”]{0,30}”", str(finding.get("finding") or ""), re.I):
            out.append({"check": "retrieval_phrase_title", "window": "",
                        "detail": str(finding.get("finding"))[:100]})
    return out


def hidden_findings(view: dict) -> list[dict]:
    """A reference finding the report made but could not place on the paper,
    so no window shows it."""
    out = []
    for finding in view.get("reference_practice") or []:
        if finding.get("rectangles") or finding.get("pervasive"):
            continue
        out.append({"check": "hidden_finding", "window": str(finding.get("reference_id") or ""),
                    "detail": f"{finding.get('finding_type')}: {str(finding.get('finding') or '')[:80]}"})
    return out


def check_report(html: str, view: dict) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    return split_links(soup) + missing_windows(soup) + retrieval_phrase_titles(view) + hidden_findings(view)


def check_expectations(html: str, expectations: list[dict]) -> list[dict]:
    """Owner-confirmed outcomes for one paper: {"window", "contains" | "absent", "note"}.
    A window of "" means the whole report text."""
    soup = BeautifulSoup(html, "html.parser")
    texts = {window: " ".join(inner.get_text(" ").split()) for window, inner in _windows(soup)}
    whole = " ".join(soup.get_text(" ").split()) + " " + " ".join(texts.values())
    out = []
    for item in expectations:
        window = item.get("window") or ""
        text = whole if not window else texts.get(window)
        if text is None:
            out.append({"check": "expectation", "window": window, "detail": f"window missing ({item.get('note', '')})"})
            continue
        if item.get("contains") and item["contains"] not in text:
            out.append({"check": "expectation", "window": window,
                        "detail": f"missing “{item['contains'][:60]}” ({item.get('note', '')})"})
        if item.get("absent") and item["absent"] in text:
            out.append({"check": "expectation", "window": window,
                        "detail": f"should not show “{item['absent'][:60]}” ({item.get('note', '')})"})
    return out


CHECK_VERSION = "report-check-v1"


def render_for_check(session, backend, report_id: str) -> tuple[str, dict]:
    """The report's HTML and projected findings, as the page builds them for a
    reader (paper actions, viewer scope and Judgment results aside)."""
    import hashlib

    import fitz

    from app.models.job import Job
    from app.models.report import Report
    from app.services.evidence_report import (load_authorized_evidence_report_bundle, project_reference_flags,
                                              render_evidence_report_html)
    from app.services.report_member_navigation import marker_words

    report = session.get(Report, report_id)
    job = session.get(Job, report.job_id)
    view, _artifact, paper = load_authorized_evidence_report_bundle(
        session, backend, report_id=str(report.id), scope_type=job.scope_type, scope_id=job.scope_id)
    with fitz.open(stream=paper, filetype="pdf") as document:
        view = project_reference_flags(view, document, hashlib.sha256(paper).hexdigest())
        view["paper_surface"]["selectable_words"] = {page.number: page.get_text("words", sort=True) for page in document}
        view["paper_surface"]["marker_words"] = marker_words(document)
    import secrets
    return render_evidence_report_html(view, csp_nonce=secrets.token_urlsafe(24)), view


def record_report_check(session, backend, report, job) -> dict | None:
    """Run the checks on a newly built report and keep the outcome on the job
    (check names and window ids only, never report text). Never raises."""
    import logging
    from collections import Counter
    from datetime import datetime, timezone

    logger = logging.getLogger(__name__)
    try:
        html, view = render_for_check(session, backend, report.id)
        violations = check_report(html, view)
        outcome = {"version": CHECK_VERSION, "report_id": str(report.id),
                   "checked_at": datetime.now(timezone.utc).isoformat(),
                   "passed": not violations,
                   "counts": dict(Counter(v["check"] for v in violations)),
                   "windows": sorted({v["window"] for v in violations if v["window"]})[:50]}
    except Exception as exc:  # noqa: BLE001 - a check problem must not fail the paper
        outcome = {"version": CHECK_VERSION, "report_id": str(report.id), "passed": None,
                   "error": type(exc).__name__}
    try:
        evidence = dict(job.upload_evidence or {})
        evidence["report_check"] = outcome
        job.upload_evidence = evidence
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
    if outcome.get("passed") is False:
        logger.warning("Report check failed for report %s: %s", outcome["report_id"], outcome["counts"])
    elif outcome.get("passed") is None:
        logger.warning("Report check could not run for report %s: %s", outcome["report_id"], outcome.get("error"))
    return outcome
