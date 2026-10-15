"""Citation projection: the report's list of citations derived from the extracted claims.

Which citations a report shows, in what order and under which numbers, and
which references count as cited (owner decisions 2026-10-03 to 2026-10-14):
a sentence holding several sources' own markers becomes one citation per
clause, an unresolved marker becomes its own citation, a heading naming a work
is not a citation but its reference counts as cited, and a reference nothing
cites (save a film the paper names by its title) is uncited. Plain data in and
out, so these rules are tested together (2026-10-15: a clause mistaken for a
heading dropped four citations and called their references uncited).
"""
from __future__ import annotations

import re
from copy import deepcopy

from app.services.paper_extraction import unresolved_marker_report_groups


def heading_citation(citation: dict) -> bool:
    """A heading naming a work with its citation ("Classical Hollywood: The Day
    the Earth Stood Still (Wise, 1951)"): short, title-cased, holding the
    citation and with no sentence punctuation at its end. A clause cut at its
    citation is judged by its whole sentence (owner review 2026-10-14)."""
    text = " ".join(str((citation.get("clause_of") or citation).get("student_text") or "").split())
    marker = " ".join(str(citation.get("citation_marker") or "").split())
    # The citation may also open the heading: "Killer's Kiss (1955): Classical
    # Hollywood Film Analysis" (paper 6, 2026-10-04).
    if not marker or marker not in text or re.search(r"[.!?;][\"”’')]*$", text):
        return False
    lead = text.replace(marker, " ", 1).strip(" :–-")
    words = re.findall(r"[A-Za-z][\w'’\-]*", lead)
    if not 1 <= len(words) <= 15 or re.search(r"[.!?;][\"”’')]*$", lead):
        return False
    long_words = [word for word in words if len(word) >= 4]
    return bool(long_words) and sum(word[0].isupper() for word in long_words) >= 0.75 * len(long_words)


def project_citations(extraction, paper_surface: dict, groups: list, references: dict,
                      reference_layout: dict) -> tuple[list[dict], list[str]]:
    """The report's citations and the references only headings cite.

    `groups` holds one (claim, citation group, member views, has attention)
    per extracted claim, in the aggregate's order.
    """
    from app.services.evidence_report import (
        _citation_entry, _missing_reference_members, _normalize_display_text, _paper_location,
        _reason_label, _reference_mismatch_details, _reference_mismatch_members, _statement_text,
        _unresolved_marker_member)
    citations: list[dict] = []
    for claim, group, members, citation_has_attention in groups:
        from app.services.candidate_relationship_judgment import citation_clause_segments
        whole = {"student_text": claim.text, "citation_marker": claim.citation_marker}
        segments = (None if claim.citation_marker_type == "source_heading" or heading_citation(whole)
                    else citation_clause_segments(claim, [m.get("reference_id") for m in group.get("members", [])]))
        if not segments:
            citations.append(_citation_entry(
                extraction, paper_surface, claim, members, citation_has_attention,
                claim.passage_start, claim.passage_end, claim.text,
                _normalize_display_text(_statement_text(claim))))
            continue
        # One sentence, several sources' own clauses: each clause and its
        # marker is its own citation, shown with ellipses where the sentence
        # goes on (owner decision 2026-10-07). The highlight is narrowed to the
        # clause when the page is drawn.
        whole_location = _paper_location(paper_surface, passage_start=claim.passage_start,
                                         passage_end=claim.passage_end)
        for local_start, local_end, reference_ids, opens, closes in segments:
            part = [m for m in members if m.get("reference_id") in reference_ids]
            text = claim.text[local_start:local_end]
            display = ("" if opens else "…") + _normalize_display_text(text) + ("" if closes else "…")
            entry = _citation_entry(
                extraction, paper_surface, claim, part,
                any(f["level"] == "attention" for m in part for f in m.get("reference_findings") or [])
                or any((m.get(k) or {}).get("attention") for m in part for k in ("quotation_check", "locator_check")),
                claim.passage_start + local_start, claim.passage_start + local_end, text, display)
            entry["paper_location"] = deepcopy(whole_location)
            entry["clause_of"] = {"student_text": claim.text, "paper_character_start": claim.passage_start}
            citations.append(entry)


    for marker in unresolved_marker_report_groups(extraction.citation_marker_census):
        missing_members = _missing_reference_members(extraction, marker["passage_start"], marker["passage_end"])
        marker_members = [
            _unresolved_marker_member(
                references[reference_id],
                marker["reason_codes"],
                reference_layout.get(reference_id),
            )
            for reference_id in marker["reference_ids"]
            if reference_id in references
        ]
        citations.append(
            {
                "claim_id": marker["group_id"],
                "paper_character_start": marker["passage_start"],
                "paper_character_end": marker["passage_end"],
                "student_text": marker["text"],
                "display_student_text": _normalize_display_text(marker["text"]),
                "citation_marker": marker["citation_marker"],
                "claim_type": "not_assessed",
                "page_locator": "",
                "tone": "attention" if missing_members else "not_assessed",
                "display_coverage": "none",
                "paper_location": _paper_location(
                    paper_surface,
                    passage_start=marker["passage_start"],
                    passage_end=marker["passage_end"],
                ),
                "members": marker_members,
                "marker_only": not marker["recovered_sentence"],
                "missing_reference_members": missing_members,
                "reference_mismatch_members": _reference_mismatch_members(
                    extraction, marker["passage_start"], marker["passage_end"]),
                "reference_mismatch_details": _reference_mismatch_details(
                    extraction, marker["passage_start"], marker["passage_end"]),
                "boundary_reason": (
                    "No reference-list entry was found for " + "; ".join(missing_members) + "."
                    if missing_members else
                    "The sentence cites more than one source, but the app could not determine safely which parts belong to each source. Compare the listed sources manually."
                    if marker["recovered_sentence"] and marker_members
                    else _reason_label(marker["reason_codes"][0])
                ),
            }
        )

    citations.sort(
        key=lambda item: (
            item["paper_character_start"],
            item["paper_character_end"],
            item["claim_id"],
        )
    )
    # A heading that names a work with its citation states nothing to check
    # (owner request 2026-10-03): it is not a citation, but its reference
    # still counts as cited.
    heading_reference_ids = sorted({member.get("reference_id") for citation in citations
                                    if heading_citation(citation)
                                    for member in citation.get("members") or [] if member.get("reference_id")})
    citations[:] = [citation for citation in citations if not heading_citation(citation)]
    for citation_number, citation in enumerate(citations, 1):
        citation["citation_number"] = citation_number
    return citations, heading_reference_ids


def uncited_reference_ids(view: dict, reference_numbers: dict, paper_key: str) -> list[str]:
    """References no citation links to, except a film or programme the paper
    names by its title (owner decisions 2026-09-30); a reference a heading
    cites is cited (owner request 2026-10-03)."""
    from app.services.evidence_report import _media_named_in_paper
    from app.services.report_references import reference_catalog
    headed = set(view.get('heading_reference_ids') or [])
    return [
        entry['reference_id'] for entry in reference_catalog(view, reference_numbers)
        if not entry.get('citation_numbers') and entry['reference_id'] not in headed
        and not _media_named_in_paper(entry.get('source') or {}, paper_key)]
