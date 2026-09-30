"""Reference numbering and per-reference window data for the evidence report.

Reference N follows bibliography order. Current reports carry that order in
their reference ids (``ref-0012-…`` from ``assign_reference_ids``); synthetic
or legacy ids fall back to the stored bibliography, then to paper position,
then to first appearance. The catalog is derived from the view on every
render and never stored, so numbering cannot drift between the summary and
the paper.
"""

from __future__ import annotations

import re

_ORDINAL_RE = re.compile(r"^ref-(\d+)-[0-9a-f]{12}$")

# Findings that sit in the paper's prose rather than on a reference entry.
# They keep their own windows instead of joining a Reference window.
BODY_TEXT_FINDINGS = frozenset({"body_title_style", "required_quotation_locator_missing"})

_COVERAGE_RANK = {"unavailable": 0, "abstract_only": 1, "partial_text": 2, "full_text": 3}


def reference_ordinal(reference_id) -> int | None:
    match = _ORDINAL_RE.match(str(reference_id or ""))
    return int(match.group(1)) if match else None


def reference_list_finding(finding: dict) -> bool:
    """A located finding that belongs on a reference-list entry."""
    return bool(finding.get("reference_id")) and finding.get("finding_type") not in BODY_TEXT_FINDINGS


def _ids_in_view(view: dict) -> list[str]:
    ids: list[str] = []
    for row in view.get("bibliography") or []:
        ids.append(row.get("reference_id"))
    for citation in view.get("citations") or []:
        for member in citation.get("members") or []:
            ids.append(member.get("reference_id"))
    for finding in view.get("reference_practice") or []:
        ids.append(finding.get("reference_id"))
        for peer in finding.get("related_references") or []:
            ids.append(peer.get("reference_id"))
    for row in view.get("uncited_link_checks") or []:
        for observation in row.get("observations") or []:
            ids.append(observation.get("reference_id"))
    return [rid for rid in dict.fromkeys(ids) if rid]


def reference_numbers(view: dict) -> dict[str, int]:
    """Map every reference id in the view to its displayed Reference N."""
    stored = view.get("reference_numbers")
    ids = _ids_in_view(view)
    if isinstance(stored, dict) and stored and all(rid in stored for rid in ids):
        return {str(rid): int(number) for rid, number in stored.items()}
    ordinals = {rid: reference_ordinal(rid) for rid in ids}
    values = list(ordinals.values())
    if ids and all(values) and len(set(values)) == len(values):
        return {rid: int(number) for rid, number in ordinals.items()}
    bibliography = [row.get("reference_id") for row in view.get("bibliography") or [] if row.get("reference_id")]
    locations = ((view.get("paper_surface") or {}).get("reference_locations") or {})

    def position(rid):
        rectangles = (locations.get(rid) or {}).get("rectangles") or []
        if not rectangles:
            return None
        first = min(rectangles, key=lambda r: (r["page_index"], r["y0"], r["x0"]))
        return (first["page_index"], first["y0"], first["x0"])

    if bibliography:
        order = list(dict.fromkeys(bibliography + ids))
    else:
        located = sorted((rid for rid in ids if position(rid) is not None), key=position)
        order = list(dict.fromkeys(located + ids))
    return {rid: number for number, rid in enumerate(order, 1)}


def reference_template_id(number: int) -> str:
    return f"reference-entry-panel-{int(number)}"


def reference_catalog(view: dict, numbers: dict[str, int] | None = None) -> list[dict]:
    """One entry per numbered reference, ordered by number."""
    numbers = numbers if numbers is not None else reference_numbers(view)
    citations = view.get("citations") or []
    findings = view.get("reference_practice") or []
    locations = ((view.get("paper_surface") or {}).get("reference_locations") or {})
    sources: dict[str, dict] = {}
    members: dict[str, dict] = {}
    citing: dict[str, list[int]] = {}
    first_citation: dict[str, tuple[int, int]] = {}
    for index, citation in enumerate(citations, 1):
        for member_index, member in enumerate(citation.get("members") or []):
            rid = member.get("reference_id")
            if not rid:
                continue
            citing.setdefault(rid, [])
            if index not in citing[rid]:
                citing[rid].append(index)
            first_citation.setdefault(rid, (index, member_index))
            source = member.get("source") or {}
            if source and (rid not in sources or (source.get("submitted_hyperlinks") and not sources[rid].get("submitted_hyperlinks"))):
                sources[rid] = source
            best = members.get(rid)
            if best is None or _COVERAGE_RANK.get(member.get("coverage_level"), 0) > _COVERAGE_RANK.get(best.get("coverage_level"), 0):
                members[rid] = member
    for finding in findings:
        rid = finding.get("reference_id")
        if rid and finding.get("source"):
            sources.setdefault(rid, finding["source"])
    for row in view.get("bibliography") or []:
        if row.get("reference_id") and row.get("source"):
            sources.setdefault(row["reference_id"], row["source"])
    for row in view.get("uncited_link_checks") or []:
        for observation in row.get("observations") or []:
            if observation.get("reference_id") and row.get("reference"):
                sources.setdefault(observation["reference_id"], row["reference"])
    entries = []
    for rid, number in sorted(numbers.items(), key=lambda pair: pair[1]):
        source = sources.get(rid)
        if not source:
            continue
        location = locations.get(rid)
        entries.append({
            "reference_id": rid,
            "number": number,
            "template_id": reference_template_id(number),
            "source": source,
            "location": location if location and location.get("rectangles") else None,
            "citation_numbers": citing.get(rid, []),
            "first_citation": first_citation.get(rid),
            "member": members.get(rid),
            "finding_indexes": [
                j for j, finding in enumerate(findings, 1)
                if finding.get("reference_id") == rid and reference_list_finding(finding)
                and finding.get("rectangles")
            ],
        })
    return entries
