"""Evidence-led individual-report projection and safe HTML rendering.

The projection deliberately excludes every experimental relationship field.
It is built only from the aggregate paper checkpoint, deterministic checks,
and immutable Evidence Packages.  Authentication remains an HTTP-layer
deployment gate; this module does not create a public report route.
"""

from __future__ import annotations

from html import escape
from copy import deepcopy
from difflib import SequenceMatcher
import re
from pathlib import Path
import hashlib
import json
import unicodedata
from typing import Iterable
import uuid
from urllib.parse import quote, urlsplit

from sqlalchemy.orm import Session
import fitz

from app.models.job import Job
from app.models.report import Report, ReportPaperArtifactRecord, VerificationReportRecord
from app.services.report_paper_artifact import (
    ReportPaperArtifactError,
    load_authorized_report_paper_artifact,
    paper_surface_descriptor,
)
from app.services.paper_extraction import (
    PaperExtractionArtifact,
    unresolved_marker_report_groups,
)
from app.services.sentence_splitter import split_sentences
from app.services.storage.backend import StorageBackend


REPORT_VIEW_VERSION = "evidence-led-report-v13"

_QUOTATION_ATTENTION = {"no_span_located", "some_spans_not_located"}
_LOCATOR_ATTENTION = {"located_span_outside_supplied_locator"}


class EvidenceReportError(ValueError):
    """The stored records cannot produce an exact, source-separated view."""


class EvidenceReportAuthorizationError(EvidenceReportError):
    """The requested aggregate report is outside the resolved viewer scope."""


def load_authorized_evidence_report_bundle(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    artifact_id: str | uuid.UUID | None = None,
    scope_type: str,
    scope_id: str,
) -> tuple[dict, ReportPaperArtifactRecord, bytes]:
    """Join one authorized report projection to its immutable presentation bytes.

    The HTTP layer must authenticate the viewer and resolve the supplied scope
    before calling this service. No route is intentionally created here.
    """
    view = get_authorized_evidence_report_view(
        session,
        report_id,
        scope_type=scope_type,
        scope_id=scope_id,
    )
    surface = view.get("paper_surface") or {}
    resolved_artifact_id = artifact_id or surface.get("artifact_id")
    if not resolved_artifact_id:
        raise EvidenceReportError("Report presentation is unavailable")
    try:
        record, content = load_authorized_report_paper_artifact(
            session,
            backend,
            report_id=report_id,
            artifact_id=resolved_artifact_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )
    except ReportPaperArtifactError as exc:
        raise EvidenceReportAuthorizationError(
            "Report does not exist in the authorized scope"
        ) from exc
    if (
        surface.get("artifact_id") != str(record.id)
        or surface.get("anchor_reason_code") != "presentation_hash_bound"
        or record.presentation_status != "page_faithful_ready"
        or record.presentation_media_type != "application/pdf"
    ):
        raise EvidenceReportError(
            "Report presentation is not bound to its immutable page surface"
        )
    return view, record, content


def get_authorized_evidence_report_view(
    session: Session,
    report_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
) -> dict:
    """Load only the already-persisted projection in one exact scope.

    The HTTP layer must resolve the authenticated viewer and capability before
    calling this function.  Supplying a scope here is not itself authentication.
    """
    try:
        parsed_id = (
            report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceReportAuthorizationError(
            "Report does not exist in the authorized scope"
        ) from exc
    report = session.get(Report, parsed_id)
    if report is None:
        raise EvidenceReportAuthorizationError(
            "Report does not exist in the authorized scope"
        )
    job = session.get(Job, report.job_id)
    if (
        job is None
        or job.scope_type != scope_type.strip()
        or job.scope_id != scope_id.strip()
    ):
        raise EvidenceReportAuthorizationError(
            "Report does not exist in the authorized scope"
        )
    view = (report.report_json or {}).get("evidence_report")
    if (
        not isinstance(view, dict)
        or view.get("report_id") != str(report.id)
        or view.get("paper_version_id") != job.paper_version_id
    ):
        raise EvidenceReportError("Persisted evidence report projection is invalid")
    return view


def build_evidence_report_view(
    *,
    report: Report,
    job: Job,
    extraction: PaperExtractionArtifact,
    verification_records: Iterable[VerificationReportRecord],
    paper_artifact: ReportPaperArtifactRecord | None = None,
) -> dict:
    """Build the renderer-facing projection without model judgment output."""
    if report.job_id != job.id:
        raise EvidenceReportError("Aggregate report does not belong to the paper job")
    if extraction.paper_version_id != job.paper_version_id:
        raise EvidenceReportError("Extraction does not belong to the paper version")

    from app.services.processing_metrics import report_processing_metrics
    aggregate = dict(report.report_json or {})
    records = list(verification_records)
    expected_ids = {str(value) for value in aggregate.get("report_ids", [])}
    records_by_id = {str(record.id): record for record in records}
    if set(records_by_id) != expected_ids:
        raise EvidenceReportError(
            "Verification records do not exactly match the aggregate report"
        )
    for record in records:
        if (
            record.paper_version_id != job.paper_version_id
            or record.scope_type != job.scope_type
            or record.scope_id != job.scope_id
        ):
            raise EvidenceReportError(
                "Verification record does not match the paper and authorization scope"
            )

    references = {item.reference_id: item for item in extraction.references}
    reference_layout = {
        item.reference_id: item
        for item in (
            extraction.reference_layout.entries
            if extraction.reference_layout is not None
            else []
        )
        if item.mapping_status == "matched"
    }
    claims = {item.claim_id: item for item in extraction.citation_claims}
    reference_findings = _reference_findings_by_id(
        aggregate.get("reference_consistency") or {}
    )
    paper_surface = paper_surface_descriptor(paper_artifact)
    anchor_correction = aggregate.get("citation_anchor_correction") or (getattr(job, "upload_evidence", None) or {}).get("citation_anchor_correction")
    if anchor_correction and paper_artifact is not None:
        from app.services.presentation_anchors import PresentationAnchorArtifact
        extraction_hash = hashlib.sha256(json.dumps(extraction.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if anchor_correction.get("extraction_sha256") != extraction_hash:
            raise EvidenceReportError("Corrected citation anchors do not match the extraction")
        corrected_anchors = PresentationAnchorArtifact.model_validate(anchor_correction["anchors"])
        if corrected_anchors.presentation_sha256 != paper_artifact.presentation_sha256:
            raise EvidenceReportError("Corrected citation anchors do not match the retained paper")
        paper_surface.update({
            "citation_anchors": [a.model_dump(mode="json") for a in corrected_anchors.anchors],
            "anchor_status": corrected_anchors.status,
            "anchor_reason_code": "presentation_hash_bound",
            "citation_anchor_count": corrected_anchors.citation_count,
            "matched_citation_anchor_count": corrected_anchors.matched_citation_count,
            "page_localized_citation_count": corrected_anchors.page_localized_citation_count,
            "structurally_localized_citation_count": corrected_anchors.structurally_localized_citation_count,
        })
    citations = []
    available_reference_ids: set[str] = set()
    limited_reference_ids: set[str] = set()
    quotation_attention = 0
    locator_attention = 0
    identity_attention_ids: set[str] = set()
    discovery_by_reference = {
        item.get("reference_id"): item.get("reference_discovery")
        for item in (job.source_results or [])
        if item.get("reference_id")
    }
    from app.services.reference_discovery import qualify_legacy_catalog_dates, qualify_book_edition_years
    original_discoveries = discovery_by_reference
    discovery_by_reference = {
        reference_id: qualify_book_edition_years(
            qualify_legacy_catalog_dates(discovery, getattr(references.get(reference_id), "url", None)),
            getattr(references.get(reference_id), "raw_ref", ""))
        for reference_id, discovery in discovery_by_reference.items()
    }
    identity_corrections = [
        {"reference_id": reference_id, "reason_code": "publication_date_and_edition_qualification",
         "previous_record_sha256": hashlib.sha256(json.dumps(original_discoveries[reference_id],sort_keys=True).encode()).hexdigest(),
         "corrected_record_sha256": hashlib.sha256(json.dumps(discovery,sort_keys=True).encode()).hexdigest()}
        for reference_id, discovery in discovery_by_reference.items()
        if discovery is not original_discoveries[reference_id]
    ]

    for group in aggregate.get("citation_groups", []):
        claim_id = group.get("claim_id")
        claim = claims.get(claim_id)
        if claim is None:
            raise EvidenceReportError("Citation group refers to an unknown claim")
        members = []
        citation_has_evidence = False
        citation_has_attention = False
        for member in group.get("members", []):
            reference_id = member.get("reference_id")
            reference = references.get(reference_id)
            if reference is None:
                raise EvidenceReportError("Citation member refers to an unknown reference")
            if member.get("status") == "evidence_package_persisted":
                record = records_by_id.get(str(member.get("report_id")))
                if record is None:
                    raise EvidenceReportError(
                        "Citation member refers to an unavailable verification record"
                    )
                item = _available_member(
                    member,
                    record,
                    reference,
                    claim,
                    reference_layout.get(reference_id),
                )
                if item["coverage_level"] == "full_text":
                    available_reference_ids.add(reference_id)
                else:
                    limited_reference_ids.add(reference_id)
                citation_has_evidence = citation_has_evidence or bool(
                    item["best_evidence"]
                )
                if item["quotation_check"]["attention"]:
                    quotation_attention += 1
                    citation_has_attention = True
                if item["locator_check"]["attention"]:
                    locator_attention += 1
                    citation_has_attention = True
            else:
                limited_reference_ids.add(reference_id)
                item = _unavailable_member(
                    member, reference, claim, reference_layout.get(reference_id)
                )
            citation_has_evidence = citation_has_evidence or bool(
                item.get("best_evidence")
            )
            item["reference_findings"] = reference_findings.get(reference_id, [])
            item["reference_identity"] = _identity_view(
                discovery_by_reference.get(reference_id)
            )
            if item["reference_identity"]["attention"]:
                identity_attention_ids.add(reference_id)
            if any(finding["level"] == "attention" for finding in item["reference_findings"]):
                citation_has_attention = True
            members.append(item)

        display_coverage = _citation_display_coverage(members)
        citation_has_connected_evidence = any(
            item.get("relevance_status") == "connected" for item in members
        )
        citation_has_retrieved_full_text = any(
            item.get("coverage_level") == "full_text" for item in members
        )
        tone = (
            "attention"
            if citation_has_attention
            else "evidence_available"
            if citation_has_connected_evidence and display_coverage == "full"
            else "limited_evidence"
            if citation_has_connected_evidence
            else "retrieved_no_connection"
            if citation_has_retrieved_full_text
            else "not_assessed"
        )
        quotation_differences = [
            difference
            for item in members
            for difference in (item.get("quotation_check") or {}).get(
                "differences", []
            )
        ]
        citations.append(
            {
                "claim_id": claim.claim_id,
                "paper_character_start": claim.passage_start,
                "paper_character_end": claim.passage_end,
                "student_text": claim.text,
                "display_student_text": _normalize_display_text(claim.text),
                "citation_marker": claim.citation_marker,
                "claim_type": claim.claim_type,
                "page_locator": claim.page_locator,
                "tone": tone,
                "display_coverage": display_coverage,
                "quotation_differences": quotation_differences,
                "paper_location": _paper_location(
                    paper_surface,
                    passage_start=claim.passage_start,
                    passage_end=claim.passage_end,
                ),
                "members": members,
            }
        )

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
                "boundary_reason": (
                    "No reference-list entry was found for " + "; ".join(missing_members) + ". Add the missing reference or correct the citation."
                    if missing_members else
                    "The sentence cites more than one source, but the app could not determine safely which parts belong to each source. Compare the listed sources manually."
                    if marker["recovered_sentence"]
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
    for citation_number, citation in enumerate(citations, 1):
        citation["citation_number"] = citation_number

    consistency = aggregate.get("reference_consistency") or {}
    duplicate_attention_ids = {
        reference_id
        for finding in consistency.get("findings", [])
        if finding.get("level") == "attention"
        for reference_id in finding.get("reference_ids", [])
    }
    formatting = aggregate.get("reference_formatting") or {}
    formatting_differences = [
        result
        for result in formatting.get("results", [])
        if result.get("status") == "difference"
    ]
    hanging_indent_differences = [
        result
        for result in formatting_differences
        if "hanging" in str(result.get("rule") or result.get("reason_code") or "").casefold()
        or "indent" in str(result.get("rule") or result.get("reason_code") or "").casefold()
    ]
    pervasive_hanging_indent = len(hanging_indent_differences) >= max(
        3, len(reference_layout) // 2
    )
    reference_practice = []
    for result in formatting_differences:
        if pervasive_hanging_indent and result in hanging_indent_differences:
            continue
        reference_id = result.get("reference_id")
        reference = references.get(reference_id)
        layout = reference_layout.get(reference_id)
        if reference is None or layout is None or not layout.rectangles:
            continue
        reference_practice.append(
            {
                "finding_type": "formatting",
                "reference_id": reference_id,
                "source": _reference_view(reference, layout),
                "finding": "The visible continuation indent differs from the 0.5-inch hanging-indent rule.",
                "rectangles": [item.model_dump(mode="json") for item in layout.rectangles],
            }
        )
    for reference_id in sorted(identity_attention_ids):
        reference = references.get(reference_id)
        layout = reference_layout.get(reference_id)
        discovery = discovery_by_reference.get(reference_id) or {}
        if reference is None:
            continue
        conflict = _reference_identity_conflict_view(discovery)
        for difference in conflict.get("field_differences") or []:
            field_name = difference["field_name"]
            reference_practice.append({
                "finding_type": "bibliographic_conflict",
                "reference_id": reference_id,
                "source": _reference_view(reference, layout),
                "finding": (
                    f"The {field_name.replace('_', ' ')} in this reference differs from the located record. "
                    "Compare the work and edition actually used before changing the reference."
                ),
                "located_record": conflict.get("located_record") or {},
                "conflicting_fields": [field_name],
                "field_difference": difference,
                "rectangles": [],
                "localization_status": "not_assessed",
            })
    for finding in consistency.get("findings", []):
        if finding.get("finding_type") != "duplicate_citation_key":
            continue
        peers = [references[rid] for rid in finding.get("reference_ids", []) if rid in references]
        for reference in peers:
            reference_practice.append({
                "finding_type": "duplicate_citation_key",
                "reference_id": reference.reference_id,
                "source": _reference_view(reference, reference_layout.get(reference.reference_id)),
                "finding": "These references share the same author and year. Use the citation style's distinguishing labels consistently in the references and in-text citations.",
                "related_references": [_reference_view(peer, reference_layout.get(peer.reference_id)) for peer in peers],
                "field_difference": {"field_name": "year", "submitted_value": reference.year},
                "rectangles": [], "localization_status": "not_assessed",
            })
    reference_practice_summary = []
    if pervasive_hanging_indent:
        reference_practice_summary.append(
            "The reference list consistently lacks the expected hanging indent. This is reported once rather than highlighting every reference."
        )
    unplaced_count = sum(
        (item.get("paper_location") or {}).get("localization_level")
        != "exact_rectangle"
        for item in citations
    )
    limits = [
        "Blue means inspectable source evidence is available; it does not mean the citation is correct.",
        "Orange identifies a named difference requiring review; it does not imply intent or misconduct.",
        "Light gray means evidence is unavailable, insufficient, or not assessed; it does not establish absence from the source.",
        "Automated citation-use judgment is disabled and is not part of this report.",
    ]
    if unplaced_count:
        limits.append(
            f"{unplaced_count} citation span(s) could not be placed on the page-faithful surface; they remain in the technical record and are not displayed speculatively."
        )
    overview = {
        "citations_analyzed": len(citations),
        "reference_count": len(extraction.references),
        "verified_full_text_sources": len(available_reference_ids),
        "limited_or_unavailable_sources": len(limited_reference_ids),
        "reference_identity_attention": len(
            identity_attention_ids | duplicate_attention_ids
        ),
        "quotation_differences_attention": quotation_attention,
        "locator_differences_attention": locator_attention,
    }
    overview.update(_reference_retrieval_counts(citations, len(extraction.references)))
    role_summaries = _build_role_summaries(
        citations=citations,
        overview=overview,
        pervasive_hanging_indent=pervasive_hanging_indent,
        reference_practice=reference_practice,
        require_paper_flags=True,
    )
    gauges = _build_report_gauges(
        citations=citations,
        references=references,
        formatting=formatting,
        discovery_by_reference=discovery_by_reference,
    )
    view = {
        "view_version": REPORT_VIEW_VERSION,
        "processing_metrics": aggregate.get("processing_metrics") or report_processing_metrics(job),
        "report_id": str(report.id),
        "paper_version_id": job.paper_version_id,
        "title": job.filename,
        "citation_format": extraction.citation_format.upper(),
        "word_counts": {
            "total": extraction.total_word_count,
            "body": extraction.body_word_count,
            "references": extraction.reference_word_count,
        },
        "paper_surface": paper_surface,
        "overview": overview,
        "role_summaries": role_summaries,
        "gauges": gauges,
        "citations": citations,
        "reference_practice": reference_practice,
        "reference_practice_summary": reference_practice_summary,
        "reference_identity_corrections": identity_corrections,
        "reference_consistency": {
            "status": consistency.get("status", "not_assessed"),
            "formatting_status": consistency.get("formatting_status", "not_assessed"),
            "findings": list(consistency.get("findings", [])),
        },
        "citation_use_assessment": None,
        "limits": limits,
    }
    return view


def _citation_display_coverage(members: list[dict]) -> str:
    """Describe retrieved source coverage independently of passage relevance."""
    if not members:
        return "none"
    levels = [str(member.get("coverage_level") or "unavailable") for member in members]
    if all(level == "full_text" for level in levels):
        return "full"
    if any(level in {"full_text", "abstract_only", "partial_text"} for level in levels):
        return "limited"
    return "none"


def _reference_retrieval_counts(citations: list[dict], reference_count: int) -> dict:
    """Count unique reference entries, never repeated citation memberships."""
    levels = {}
    ranks = {"unavailable": 0, "abstract_only": 1, "partial_text": 2, "full_text": 3}
    for citation in citations:
        for member in citation.get("members") or []:
            reference_id = member.get("reference_id")
            if reference_id:
                levels[reference_id] = max(levels.get(reference_id, 0), ranks.get(member.get("coverage_level"), 0))
    full = sum(level == 3 for level in levels.values())
    limited = sum(level in {1, 2} for level in levels.values())
    return {"verified_full_text_sources": full, "abstract_or_limited_sources": limited,
            "unavailable_sources": max(0, reference_count - full - limited)}


def _citation_list(numbers: list[int]) -> str:
    return "Citation " + str(numbers[0]) if len(numbers) == 1 else "Citations " + ", ".join(map(str, numbers))


def _build_role_summaries(
    *,
    citations: list[dict],
    overview: dict,
    pervasive_hanging_indent: bool,
    reference_practice: list[dict] | None = None,
    require_paper_flags: bool = False,
) -> dict:
    """Prefer repeated issues per category; otherwise show isolated findings."""
    if require_paper_flags:
        placed = {kind: {f.get("reference_id") for f in reference_practice or []
                        if f.get("finding_type") == kind and f.get("rectangles")}
                  for kind in ("bibliographic_conflict", "duplicate_citation_key")}
        citations = deepcopy(citations)
        for citation in citations:
            location = citation.get("paper_location") or {}
            if location.get("localization_level") != "exact_rectangle" or not location.get("rectangles"):
                citation["members"] = []
                citation.pop("missing_reference_members", None)
                continue
            for member in citation.get("members") or []:
                if member.get("reference_id") not in placed["bibliographic_conflict"]:
                    member["reference_identity"] = {**(member.get("reference_identity") or {}), "attention": False}
                member["reference_findings"] = [f for f in member.get("reference_findings") or []
                    if f.get("finding_type") != "duplicate_citation_key"
                    or (set(f.get("reference_ids") or []) <= placed["duplicate_citation_key"])]
    partial_ids = [
        index for index, citation in enumerate(citations, 1) if any(
            "only part" in str((member.get("best_evidence") or {}).get("evidence_note") or "")
            for member in citation.get("members") or []
        )
    ]
    partial_citations = len(partial_ids)
    no_excerpt_ids = [index for index, citation in enumerate(citations, 1) if any(
        member.get("coverage_level") == "full_text"
        and member.get("relevance_status") == "no_connection"
        and not member.get("best_evidence")
        for member in citation.get("members") or [])]
    indirect_citations = sum(
        any(
            (member.get("best_evidence") or {}).get("evidence_role")
            == "representation_of_other_work"
            for member in citation.get("members") or []
        )
        for citation in citations
    )
    quotation_attention_citations = sum(
        any(
            member.get("show_quotation_check")
            and (member.get("quotation_check") or {}).get("attention")
            for member in citation.get("members") or []
        )
        for citation in citations
    )
    conflicting_references = {
        member.get("reference_id")
        for citation in citations
        for member in citation.get("members") or []
        if (member.get("reference_identity") or {}).get("attention")
    }
    duplicate_key_references = {
        member.get("reference_id")
        for citation in citations
        for member in citation.get("members") or []
        if any(
            finding.get("finding_type") == "duplicate_citation_key"
            for finding in member.get("reference_findings") or []
        )
    }
    duplicate_groups = {
        tuple(sorted(finding.get("reference_ids") or []))
        for citation in citations for member in citation.get("members") or []
        for finding in member.get("reference_findings") or []
        if finding.get("finding_type") == "duplicate_citation_key"
    }
    missing = [(index, citation.get("missing_reference_members"))
               for index, citation in enumerate(citations, 1)
               if citation.get("missing_reference_members")]

    student = {
        "evidence": [],
        "reference_formatting": [],
        "academic_practice": [],
    }
    instructor = {
        "evidence": [],
        "reference_formatting": [],
        "academic_practice": [],
    }
    evidence_repeated = max(partial_citations, quotation_attention_citations) >= 2
    reference_repeated = pervasive_hanging_indent or len(conflicting_references) >= 2
    practice_repeated = max(indirect_citations, len(duplicate_groups), len(missing)) >= 2
    if partial_citations and (not evidence_repeated or partial_citations >= 2):
        student["evidence"].append(
            f"Check {_citation_list(partial_ids).lower()}: the source passages found address only part of each statement. "
            "Add evidence for the remaining parts or narrow the wording after checking the source."
        )
        instructor["evidence"].append(
            f"{partial_citations} {'citation has' if partial_citations == 1 else 'citations have'} source passages that address only part of the attributed statement "
            f"({_citation_list(partial_ids).lower()}). The earlier checks did not identify which specific parts still need evidence."
        )
    if quotation_attention_citations and (not evidence_repeated or quotation_attention_citations >= 2):
        student["evidence"].append(
            "Compare direct quotations with the retrieved source wording and correct any material difference. "
            f"Quotation checking identified differences needing attention in {quotation_attention_citations} {'citation' if quotation_attention_citations == 1 else 'citations'}."
        )
        instructor["evidence"].append(
            f"{quotation_attention_citations} {'quotation contains' if quotation_attention_citations == 1 else 'quotations contain'} wording differences that need comparison with the source. "
            "The changed words are marked in the paper and citation view."
        )
    if pervasive_hanging_indent:
        student["reference_formatting"].append(
            "Apply the required hanging indent consistently throughout the reference list. "
            "The visible reference entries repeatedly lack the expected continuation-line indent."
        )
        instructor["reference_formatting"].append(
            "The reference list repeatedly lacks the expected hanging indent. "
            "This pervasive difference is reported once instead of covering every reference with a highlight."
        )
    if conflicting_references and (not reference_repeated or len(conflicting_references) >= 2):
        student["reference_formatting"].append(
            "Check reference details against the works actually cited. "
            f"The title, author, date or other details differ from located records for {len(conflicting_references)} {'reference' if len(conflicting_references) == 1 else 'references'}."
        )
        instructor["reference_formatting"].append(
            f"{len(conflicting_references)} {'reference contains' if len(conflicting_references) == 1 else 'references contain'} bibliographic fields that conflict with located records. "
            "Compare the highlighted reference details with the located records."
        )
    if missing and (not practice_repeated or len(missing) >= 2):
        details = "; ".join(f"citation {index}: {', '.join(names)}" for index, names in missing)
        student["academic_practice"].append(
            f"Add the missing reference-list entry or correct the citation ({details}). "
            "The named source has no matching reference."
        )
        instructor["academic_practice"].append(
            f"An in-text source has no matching reference-list entry ({details})."
        )
    if indirect_citations and (not practice_repeated or indirect_citations >= 2):
        student["academic_practice"].append(
            "Identify when a cited article is reporting another study and cite the original work when appropriate. "
            f"In {indirect_citations} {'citation' if indirect_citations == 1 else 'citations'}, the source passage reports another work rather than the source's own finding."
        )
        instructor["academic_practice"].append(
            f"{indirect_citations} {'citation currently relies' if indirect_citations == 1 else 'citations currently rely'} on passages where the cited source represents another work. "
            "This is indirect citation rather than the source's own finding."
        )
    if duplicate_key_references and (not practice_repeated or len(duplicate_groups) >= 2):
        student["academic_practice"].append(
            "Make each in-text author-and-year citation identify one reference unambiguously. "
            f"{len(duplicate_key_references)} references currently share the same author and year; add the distinctions required by the citation style."
        )
        instructor["academic_practice"].append(
            f"{len(duplicate_key_references)} references share the same author and year, so their in-text citations do not distinguish the works. "
            "The references and corresponding citations need consistent distinguishing labels."
        )

    for section in student.values():
        del section[2:]
    for section in instructor.values():
        del section[2:]
    # Retrieval coverage is a review limitation, not an affirmative repeated
    # writing issue. Keep it visible even when two revision priorities exist.
    if no_excerpt_ids:
        unit = "citation" if len(no_excerpt_ids) == 1 else "citations"
        notice = (f"For {len(no_excerpt_ids)} {unit}, the app retrieved at least one full-text source but found no clearly relevant passage "
                  f"in it ({_citation_list(no_excerpt_ids).lower()}). ")
        student["evidence"].append(notice + "Check those sources and identify the relevant passages before revising; this does not mean evidence is absent.")
        instructor["evidence"].append(notice + "These sources need manual review; a retrieval miss does not establish that evidence is absent.")
    return {"student": student, "instructor": instructor}


def _build_report_gauges(
    *,
    citations: list[dict],
    references: dict,
    formatting: dict,
    discovery_by_reference: dict,
) -> list[dict]:
    """Build neutral count breakdowns; these are not scores or verdict gauges."""
    coverage_rank = {"unavailable": 0, "abstract_only": 1, "partial_text": 2, "full_text": 3}
    coverage_by_reference = {reference_id: "unavailable" for reference_id in references}
    for citation in citations:
        for member in citation.get("members") or []:
            reference_id = member.get("reference_id")
            level = str(member.get("coverage_level") or "unavailable")
            if reference_id in coverage_by_reference and coverage_rank.get(level, 0) > coverage_rank.get(
                coverage_by_reference[reference_id], 0
            ):
                coverage_by_reference[reference_id] = level
    coverage_counts = {
        level: sum(value == level for value in coverage_by_reference.values())
        for level in coverage_rank
    }
    claim_counts = {
        kind: sum(citation.get("claim_type") == kind for citation in citations)
        for kind in ("quotation", "paraphrase")
    }
    display_counts = {
        level: sum(citation.get("display_coverage") == level for citation in citations)
        for level in ("full", "limited", "none")
    }
    quotation_citations = [
        citation for citation in citations if citation.get("claim_type") == "quotation"
    ]
    quotation_assessed = sum(
        any(member.get("show_quotation_check") for member in citation.get("members") or [])
        for citation in quotation_citations
    )
    quotation_attention = sum(
        any(
            member.get("show_quotation_check")
            and (member.get("quotation_check") or {}).get("attention")
            for member in citation.get("members") or []
        )
        for citation in quotation_citations
    )
    formatting_counts = {
        status: sum(result.get("status") == status for result in formatting.get("results", []))
        for status in ("matches_rule", "difference", "not_assessed")
    }
    identity_statuses = {
        reference_id: _identity_view(discovery_by_reference.get(reference_id)).get("status")
        for reference_id in references
    }
    identity_counts = {
        "confirmed": sum(
            status in {"confirmed", "confirmed_with_minor_differences"}
            for status in identity_statuses.values()
        ),
        "conflict": sum(status == "bibliographic_conflict" for status in identity_statuses.values()),
        "limited": sum(
            status not in {"confirmed", "confirmed_with_minor_differences", "bibliographic_conflict"}
            for status in identity_statuses.values()
        ),
    }
    return [
        {
            "label": "Source availability",
            "total": len(references),
            "unit": "references",
            "parts": [
                {"label": "Full text", "value": coverage_counts["full_text"], "color": "#2563a7"},
                {"label": "Limited text", "value": coverage_counts["partial_text"], "color": "#735cb5"},
                {"label": "Abstract only", "value": coverage_counts["abstract_only"], "color": "#6b9fc4"},
                {"label": "Unavailable", "value": coverage_counts["unavailable"], "color": "#aeb8c4"},
            ],
        },
        {
            "label": "Citation forms",
            "total": len(citations),
            "unit": "citations",
            "parts": [
                {"label": "Quotations", "value": claim_counts["quotation"], "color": "#2563a7"},
                {"label": "Paraphrases", "value": claim_counts["paraphrase"], "color": "#69727d"},
            ],
        },
        {
            "label": "Displayed evidence",
            "total": len(citations),
            "unit": "citations",
            "parts": [
                {"label": "Full-text passage", "value": display_counts["full"], "color": "#2563a7"},
                {"label": "Limited passage", "value": display_counts["limited"], "color": "#735cb5"},
                {"label": "No passage shown", "value": display_counts["none"], "color": "#aeb8c4"},
            ],
        },
        {
            "label": "Quotation checking",
            "total": len(quotation_citations),
            "unit": "quotations",
            "parts": [
                {"label": "No difference detected", "value": max(0, quotation_assessed - quotation_attention), "color": "#2563a7"},
                {"label": "Needs attention", "value": quotation_attention, "color": "#d95f02"},
                {"label": "Not assessed", "value": max(0, len(quotation_citations) - quotation_assessed), "color": "#aeb8c4"},
            ],
        },
        {
            "label": "Reference identity",
            "total": len(references),
            "unit": "references",
            "parts": [
                {"label": "Confirmed record", "value": identity_counts["confirmed"], "color": "#2563a7"},
                {"label": "Bibliographic conflict", "value": identity_counts["conflict"], "color": "#d95f02"},
                {"label": "Limited outcome", "value": identity_counts["limited"], "color": "#aeb8c4"},
            ],
        },
        {
            "label": "Hanging-indent check",
            "total": sum(formatting_counts.values()),
            "unit": "observable references",
            "parts": [
                {"label": "Matches rule", "value": formatting_counts["matches_rule"], "color": "#2563a7"},
                {"label": "Difference", "value": formatting_counts["difference"], "color": "#d95f02"},
                {"label": "Not assessed", "value": formatting_counts["not_assessed"], "color": "#aeb8c4"},
            ],
        },
    ]


def _paper_location(
    paper_surface: dict,
    *,
    passage_start: int,
    passage_end: int,
) -> dict:
    """Project a hash-bound anchor into a non-authorizing viewer action contract."""
    matches = [
        anchor
        for anchor in paper_surface.get("citation_anchors", [])
        if anchor.get("passage_start") == passage_start
        and anchor.get("passage_end") == passage_end
    ]
    if len(matches) != 1:
        return {
            "localization_level": "semantic_only",
            "action": None,
            "reason_code": "presentation_anchor_unavailable",
        }
    anchor = matches[0]
    level = anchor.get("localization_level", "semantic_only")
    if level == "exact_rectangle":
        return {
            "anchor_id": anchor.get("anchor_id"),
            "localization_level": level,
            "page_indexes": list(anchor.get("page_indexes", [])),
            "rectangles": list(anchor.get("rectangles", [])),
            "action": {
                "label": "View in paper",
                "enabled": False,
                "status": "requires_authenticated_paper_delivery",
            },
        }
    if level == "page_only":
        return {
            "anchor_id": anchor.get("anchor_id"),
            "localization_level": level,
            "page_indexes": list(anchor.get("page_indexes", [])),
            "rectangles": [],
            "action": {
                "label": "View paper",
                "enabled": False,
                "status": "requires_authenticated_paper_delivery",
            },
        }
    return {
        "localization_level": (
            level if level in {"structural_only", "semantic_only"} else "semantic_only"
        ),
        "action": None,
        "reason_code": (
            "paragraph_known_presentation_location_unresolved"
            if level == "structural_only"
            else "presentation_anchor_unavailable"
        ),
    }


def _render_role_summaries(summaries: dict, *, audience: str) -> str:
    columns = (
        ("evidence", "Evidence"),
        ("reference_formatting", "Reference Formatting"),
        ("academic_practice", "Academic Practice"),
    )

    def render(audience: str, heading: str) -> str:
        values = summaries.get(audience) or {}
        sections = []
        for key, label in columns:
            items = list(values.get(key) or [])
            if items:
                body = "<ul>" + "".join(
                    f"<li>{escape(item)}</li>" for item in items
                ) + "</ul>"
            else:
                body = (
                    '<p class="empty-pattern">No issue was established '
                    "by the checks currently enabled.</p>"
                )
            sections.append(
                f'<section class="summary-column"><h2>{label}</h2>{body}</section>'
            )
        return (
            f'<div class="{audience}-summary"><h1>{heading}</h1>'
            f'<div class="summary-grid">'
            f'{"".join(sections)}</div></div>'
        )

    heading = "Patterns and issues" if audience == "instructor" else "Priorities for revision"
    return '<section class="summary">' + render(audience, heading) + "</section>"


def _render_how_to_read(view: dict) -> str:
    return (
        '<details class="read-guide"><summary><strong>How to read this report</strong></summary>'
        '<div class="guide-grid">'
        '<section><h2>Evidence, not a verdict</h2><p>Use the displayed passages to compare the paper with its cited sources. A blue line means relevant source text is available; it does not mean every part of the citation is correct.</p></section>'
        '<section><h2>Colors</h2><p>Dark blue means full text is available, teal means an abstract or limited text is available, brown means a completed passage check found no clear matching passage, orange identifies a specific issue to check, and light grey means no source text was retrieved or the citation was not verifiable. Availability alone does not establish support for the statement.</p></section>'
        '<section><h2>Limits</h2><p>An abstract cannot answer questions that require the complete source, and failure to retrieve or find a passage does not prove that the source lacks support. Open the source when the displayed evidence is incomplete or unclear.</p></section>'
        '</div></details>'
    )


def _render_report_gauges(gauges: list[dict]) -> str:
    rendered = []
    arc = "M 12 90 A 60 60 0 0 1 132 90"
    for gauge in gauges:
        parts = list(gauge.get("parts") or [])
        numeric_total = sum(max(0, int(part.get("value") or 0)) for part in parts)
        denominator = numeric_total or 1
        offset = 0.0
        segments = []
        legend = []
        for part in parts:
            value = max(0, int(part.get("value") or 0))
            percentage = value / denominator * 100
            color = str(part.get("color") or "#aeb8c4")
            if value:
                segments.append(
                    f'<path d="{arc}" pathLength="100" style="stroke:{escape(color, quote=True)};'
                    f'stroke-dasharray:{percentage:.3f} {100-percentage:.3f};stroke-dashoffset:-{offset:.3f}" />'
                )
            legend.append(
                f'<div><span><i class="swatch" style="background:{escape(color, quote=True)}"></i>'
                f'{escape(str(part.get("label") or "Other"))}</span><strong>{value}</strong></div>'
            )
            offset += percentage
        rendered.append(
            f'<section class="gauge" aria-label="{escape(str(gauge.get("label") or "Count breakdown"), quote=True)}">'
            f'<h3>{escape(str(gauge.get("label") or "Count breakdown"))}</h3>'
            f'<div class="gauge-graphic"><svg viewBox="0 0 145 105" aria-hidden="true">'
            f'<path class="gauge-base" d="{arc}" />{"".join(segments)}</svg>'
            f'<div class="gauge-total">{int(gauge.get("total") or 0)}'
            f'<span class="gauge-unit">{escape(str(gauge.get("unit") or "items"))}</span></div></div>'
            f'<div class="gauge-legend">{"".join(legend)}</div></section>'
        )
    return (
        '<details class="counts-disclosure"><summary><strong>Report Counts and Evidence Breakdown</strong></summary>'
        '<p class="muted">These are descriptive counts with explicit denominators, not scores or judgments.</p>'
        f'<div class="gauges">{"".join(rendered)}</div></details>'
    )


def render_evidence_report_html(view: dict, *, csp_nonce: str) -> str:
    """Render the page-faithful paper as the report's primary navigator."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", csp_nonce):
        raise EvidenceReportError("A valid CSP nonce is required for report rendering")
    nonce = escape(csp_nonce, quote=True)
    citations = list(view["citations"])
    panel_templates = "".join(
        _render_panel_template(item, index) for index, item in enumerate(citations, 1)
    )
    reference_practice = list(view.get("reference_practice") or [])
    audience = str(view.get("audience") or "student")
    if audience not in {"student", "instructor"}:
        raise EvidenceReportError("Report audience is invalid")
    annotations = [
        item
        for item in (view.get("annotations") or [])
        if audience == "instructor" or item.get("visibility") == "released"
    ]
    panel_templates += "".join(
        _render_reference_panel_template(item, index)
        for index, item in enumerate(reference_practice, 1)
    )
    panel_templates += _render_annotation_templates(annotations)
    paper_pages, placed_indexes = _render_continuous_paper(
        view["paper_surface"], citations, reference_practice, annotations
    )
    unplaced_references = "".join(
        f'<button type="button" data-panel-template="reference-panel-{index}">'
        f'{escape(str((item.get("source") or {}).get("author") or "Reference"))}: '
        f'{escape(", ".join(item.get("conflicting_fields") or []))}</button>'
        for index, item in enumerate(reference_practice, 1) if not item.get("rectangles")
    )
    if unplaced_references:
        paper_pages += ('<details><summary>Reference differences without exact page locations</summary>'
                        '<p>Select a difference to inspect it. No text has been highlighted speculatively.</p>'
                        + unplaced_references + '</details>')
    summaries = _render_role_summaries(
        view.get("role_summaries") or {}, audience=audience
    )
    how_to_read = _render_how_to_read(view)
    gauges = _render_report_gauges(view.get("gauges") or [])
    annotation_action = view.get("annotation_action") or {}
    annotation_create_href = str(annotation_action.get("create_href") or "")
    annotation_revision_template = str(
        annotation_action.get("revision_href_template") or ""
    )
    annotation_key = (
        '<span data-key="annotation"><i class="annotation-key"></i>instructor comment or highlight</span>'
        if annotations
        else ""
    )
    export_action = view.get("export_action") or {}
    export_mode = str(view.get("export_mode") or "interactive")
    if export_mode not in {"interactive", "released_print"}:
        raise EvidenceReportError("Report export mode is invalid")
    export_links = ""
    if export_action:
        export_links = (
            f'<a href="{escape(str(export_action.get("pdf_href") or "#"), quote=True)}">Download released PDF</a>'
            f'<a href="{escape(str(export_action.get("pdf_href") or "#"), quote=True)}?inline=true" target="_blank">Preview / print PDF</a>'

        )
    audience_links = (
        f'<a href="{escape(str(export_action.get("report_href") or "#"), quote=True)}">Interactive report</a>'
        if export_mode == "released_print"
        else (
            f'<a href="?audience=student"{' aria-current="page"' if audience == 'student' else ''}>Student</a>'
            f'<a href="?audience=instructor"{' aria-current="page"' if audience == 'instructor' else ''}>Instructor</a>'
        )
    )
    rendered = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(view['title'])}</title><style nonce="{nonce}">
:root{{--ink:#18212b;--muted:#5b6672;--line:#d8dee6;--blue:#2563a7;--teal:#168c8c;--no-connection:#8c6b4f;--blue-bg:#eef6ff;--source-bg:#f7f4ea;--amber:#d95f02;--amber-bg:#fff6e8;--gray:#b8c0c8;--violet:#7651a8;--paper:#fff;--page:#f3f5f7}}
*{{box-sizing:border-box}} [hidden]{{display:none!important}} body{{margin:0;background:var(--page);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
    header,main,footer{{width:100%;max-width:none}} header{{padding:1.5rem clamp(.75rem,2vw,1.5rem) 1rem}} h1{{margin:0 0 .25rem;font-size:1.8rem}} h2{{font-size:1.2rem}} .sub{{color:var(--muted)}}
.audience-switch{{display:flex;gap:.35rem;justify-content:flex-end;flex-wrap:wrap}} .audience-switch a{{border:1px solid var(--line);background:#fff;border-radius:5px;padding:.45rem .65rem;color:inherit;text-decoration:none}} .audience-switch a[aria-current="page"]{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.summary{{margin:1rem 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:1rem}} .summary-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}} .summary-column{{border-left:3px solid var(--blue);padding:0 .85rem}} .summary-column h2{{margin-top:0}} .summary-column li{{margin:.6rem 0}} .empty-pattern{{color:var(--muted)}} body[data-audience="student"] .instructor-summary,body[data-audience="instructor"] .student-summary{{display:none}}
.read-guide{{margin:1rem 0;background:#fff;border:1px solid var(--line);border-radius:7px;padding:.2rem .9rem}} .guide-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;padding:.7rem 0 1rem}} .guide-grid p{{margin:.25rem 0}}
    .layout{{--paper-share:68%;display:grid;grid-template-columns:minmax(20rem,var(--paper-share)) .75rem minmax(20rem,1fr);gap:0;padding:0 clamp(.75rem,2vw,1.5rem) 2rem;align-items:start}} .paper,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:1rem;min-width:0}} .panel>h2:first-child{{margin-top:0}} .source-group>h3{{font-size:.95rem;margin:.7rem 0 .2rem;border-bottom:2px solid var(--gray);padding-bottom:.2rem}} .source-group.full_text>h3{{border-color:var(--blue)}} .source-group.partial_text>h3,.source-group.abstract_only>h3{{border-color:var(--teal)}} .member.retrieved_no_connection .full-reference{{border-left:2px solid var(--no-connection);padding-left:.45rem}} .member.attention .full-reference{{border-left:2px solid var(--amber);padding-left:.45rem}} .panel{{position:sticky;top:1rem;align-self:start;max-height:calc(100vh - 2rem);overflow:auto;overflow-wrap:anywhere}} .splitter{{height:calc(100vh - 2rem);position:sticky;top:1rem;cursor:col-resize;touch-action:none;display:flex;align-items:center;justify-content:center}} .splitter::before{{content:"";width:3px;height:4rem;border-radius:2px;background:#aeb8c4}} .splitter:focus{{outline:2px solid var(--blue);outline-offset:-2px}} .paper-pages{{display:grid;gap:1.5rem}} .paper-page{{margin:0}} .paper-page figcaption{{color:var(--muted);font-size:.85rem;margin-bottom:.3rem}} .page-surface{{display:block;width:100%;height:auto;background:#fff;box-shadow:0 2px 12px #0002}}
    .paper-viewport{{overflow:auto}} .paper-pages{{width:var(--paper-zoom,100%);margin-inline:auto}} .toolbar{{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.8rem 0}} .toolbar button,.toolbar label{{border:1px solid var(--line);background:#fff;border-radius:5px;padding:.45rem .65rem;font:inherit}} .toolbar label{{display:flex;gap:.35rem;align-items:center}} .toolbar .group{{display:flex;gap:.3rem;padding-left:.45rem;border-left:1px solid var(--line)}} .toolbar button.active{{background:#f0e9fa;border-color:var(--violet)}} .paper-toolbar{{position:sticky;top:0;z-index:20;margin:-1rem -1rem .8rem;padding:.65rem 1rem;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 2px 5px #0001}} .annotation-status{{color:var(--muted);font-size:.88rem}}
    .citation-overlay,.reference-practice-overlay,.paper-annotation-overlay{{cursor:pointer}} .citation-overlay .selection-bg{{fill:transparent;stroke:none}} .citation-overlay .underline{{fill:none;vector-effect:non-scaling-stroke;stroke-width:1.5}} .citation-overlay.evidence_available{{stroke:var(--blue)}} .citation-overlay.limited_evidence{{stroke:var(--teal)}} .citation-overlay.retrieved_no_connection{{stroke:var(--no-connection)}} .citation-overlay.attention{{stroke:var(--amber)}} .citation-overlay.not_assessed{{stroke:var(--gray)}} .citation-overlay:focus .underline,.citation-overlay.selected .underline{{stroke-width:1.9}} .citation-overlay.selected .selection-bg{{fill:#b9dcff;fill-opacity:.52}} .citation-overlay.persisted-highlight .selection-bg{{fill:var(--violet);fill-opacity:.24}} .citation-overlay.selected.persisted-highlight .selection-bg{{fill:#b9dcff;fill-opacity:.52}} .citation-overlay .quote-difference-mark{{fill:#ffd27a;fill-opacity:.72;stroke:#8b5200;stroke-width:.65;vector-effect:non-scaling-stroke}} .citation-overlay:focus{{outline:none}} .instructor-comment-marker{{fill:var(--violet);stroke:#fff;stroke-width:1;vector-effect:non-scaling-stroke}} .paper-annotation-overlay .annotation-region-bg{{fill:transparent;stroke:var(--violet);stroke-width:1.25;stroke-dasharray:3 2;vector-effect:non-scaling-stroke}} .paper-annotation-overlay.persisted-highlight .annotation-region-bg{{fill:var(--violet);fill-opacity:.24;stroke:none}} .paper-annotation-overlay.selected .annotation-region-bg{{fill:#b9dcff;fill-opacity:.52;stroke:var(--blue);stroke-dasharray:none}} .region-selection{{fill:#b9dcff;fill-opacity:.25;stroke:var(--blue);stroke-width:1.25;stroke-dasharray:3 2;vector-effect:non-scaling-stroke;pointer-events:none}} body[data-audience="student"] .annotation-private-only{{display:none}} body[data-audience="student"] .citation-overlay.highlight-private-only .selection-bg{{fill:transparent}} body[data-audience="student"] .comment-private-only{{display:none}} body[data-audience="student"] .annotation-note[data-visibility="private"]{{display:none}} .reference-practice-overlay rect{{fill:var(--amber);fill-opacity:.14;stroke:var(--amber);stroke-width:1;stroke-dasharray:3 2;vector-effect:non-scaling-stroke}} .reference-practice-overlay.selected rect{{fill-opacity:.28}} .pen-active .page-surface{{cursor:crosshair}} .pen-stroke{{fill:none;stroke:var(--violet);stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke;pointer-events:none}}
    body.hide-evidence .citation-overlay,body.hide-reference-practice .reference-practice-overlay{{display:none}} body.paper-only .citation-overlay,body.paper-only .reference-practice-overlay{{display:none}}
    .selected-citation{{margin:.35rem 0 0;padding:.7rem .85rem;border-left:3px solid var(--blue);background:var(--blue-bg);white-space:normal;overflow-wrap:anywhere}} mark.quote-difference{{background:#ffd27a;color:inherit;padding:0 .05em;border-radius:2px}} .member{{border-top:1px solid var(--line);padding-top:.75rem;margin-top:.75rem;min-width:0}} .member:first-of-type{{border-top:0;padding-top:.2rem;margin-top:0}} .member h3{{margin:.25rem 0 .35rem}} blockquote.source-excerpt{{max-width:100%;margin:.35rem 0;padding:.7rem .9rem;border-left:3px solid #8b7a45;background:var(--source-bg);white-space:normal;overflow-wrap:anywhere;word-break:normal}} .locator,.muted{{color:var(--muted);font-size:.92rem}} .full-reference{{margin:.8rem 0;white-space:normal;overflow-wrap:anywhere;font-size:.8rem;line-height:1.35}} .reference-url{{color:var(--blue);text-decoration:underline}} details{{margin:.6rem 0}} .checks{{padding-left:1.2rem}} .attention-text{{color:var(--amber);font-weight:650}} .source-unavailable{{margin:.35rem 0}} .compact-action{{display:inline-block;padding:.18rem .38rem;border:0;border-radius:4px;background:var(--blue);color:#fff;font-size:.82rem;line-height:1.25;font-weight:650;text-decoration:none;cursor:pointer}} .actions{{margin:.45rem 0}} .source-upload button{{padding:.18rem .38rem;border:0;border-radius:4px;background:var(--violet);color:#fff;font-size:.82rem;line-height:1.25;font-weight:650}} .source-upload{{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem;margin:.5rem 0}} .source-upload label{{font-size:.86rem}} .source-upload input{{max-width:13rem;font-size:.8rem}} .upload-status{{display:block;width:100%;color:var(--muted);font-size:.88rem}} .annotation-note{{border-left:3px solid var(--violet);background:#f5f0fb;padding:.5rem .7rem;margin:.55rem 0}} .annotation-visibility{{color:var(--muted);font-size:.82rem;margin-left:.35rem}} .annotation-actions{{display:flex;gap:.35rem;flex-wrap:wrap}} .annotation-actions button{{border:1px solid var(--violet);background:#fff;border-radius:4px;padding:.18rem .38rem;font:inherit;font-size:.82rem;color:#4d3470}}
.legend{{display:flex;flex-wrap:wrap;gap:.55rem 1rem;background:#fff;border:1px solid var(--line);border-radius:6px;padding:.65rem .8rem}} .legend span{{display:inline-flex;align-items:center;gap:.35rem}} .line-key{{display:inline-block;width:2.1rem;border-bottom:3px solid #111}} .color-key{{display:inline-block;width:1.8rem;height:2px;border-radius:0}} .color-key.blue{{background:var(--blue)}} .color-key.teal{{background:var(--teal)}} .color-key.no-connection{{background:var(--no-connection)}} .color-key.amber{{background:var(--amber)}} .color-key.gray{{background:var(--gray)}} .reference-key{{display:inline-block;width:1.3rem;height:.75rem;background:var(--amber-bg);border:1px dashed var(--amber)}} .annotation-key{{display:inline-block;width:1.3rem;height:.75rem;background:#7651a83d;border:1px solid var(--violet)}}
.counts-disclosure{{margin:1rem clamp(.75rem,2vw,1.5rem) 2rem;background:#fff;border:1px solid var(--line);border-radius:8px;padding:.25rem .9rem}} .gauges{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;padding:1rem 0}} .gauge{{display:grid;grid-template-columns:145px 1fr;grid-template-areas:"label label" "graphic legend";gap:.35rem 1rem;border:1px solid var(--line);border-radius:7px;padding:.8rem}} .gauge h3{{grid-area:label;margin:0;text-align:center}} .gauge-graphic{{grid-area:graphic;position:relative;width:145px;height:105px}} .gauge-graphic svg{{width:145px;height:105px}} .gauge-graphic path{{fill:none;stroke-width:16}} .gauge-base{{stroke:#e0e5ea}} .gauge-total{{position:absolute;inset:58px 0 auto;text-align:center;font-size:1.2rem;font-weight:700}} .gauge-unit{{display:block;font-size:.72rem;color:var(--muted);font-weight:400}} .gauge-legend{{grid-area:legend;align-self:center}} .gauge-legend div{{display:flex;justify-content:space-between;gap:.8rem;font-size:.86rem}} .swatch{{display:inline-block;width:.65rem;height:.65rem;margin-right:.3rem}}
.page-container{{position:relative;container-type:inline-size;content-visibility:auto}} .paper-text-layer{{position:absolute;inset:0;pointer-events:none;user-select:text}} .paper-word{{position:absolute;display:inline-block;color:transparent;cursor:text;user-select:inherit;pointer-events:var(--word-pointer,auto);white-space:pre;line-height:1;font-family:serif}} .paper-word::selection{{color:transparent;background:#b9dcff88}} .page-surface image{{pointer-events:none}} .text-selection-off .paper-text-layer{{--word-pointer:none;user-select:none}} .annotation-editor textarea{{display:block;width:100%;min-height:7rem;font:inherit;margin:.5rem 0}} .processing-footer{{font-size:.65rem;color:var(--muted);padding:0 1.5rem 1rem}} .technical-export{{margin:0 1.5rem 1rem;font-size:.8rem}}
@media(max-width:900px){{.summary-grid,.guide-grid{{grid-template-columns:1fr}}}} @media(max-width:760px){{.layout{{grid-template-columns:1fr}} .splitter{{display:none}} .panel{{position:static;max-height:none}} .gauge{{grid-template-columns:1fr;grid-template-areas:"label" "graphic" "legend"}}}}
body[data-export-mode="released_print"] .paper-toolbar{{display:none}} body[data-export-mode="released_print"] .annotation-actions{{display:none}}
@media print{{body{{background:#fff}} header>.toolbar,.audience-switch,.read-guide,.counts-disclosure{{display:none}} .layout{{display:block;padding:0}} .paper{{border:0;padding:0}} .paper>h2,.paper-toolbar,.panel,.splitter{{display:none}} .paper-page{{break-after:page}} .paper-page figcaption{{display:none}} .page-surface{{box-shadow:none}}}}
</style></head><body data-audience="{audience}" data-export-mode="{escape(export_mode, quote=True)}" data-annotation-create="{escape(annotation_create_href, quote=True)}" data-annotation-revision-template="{escape(annotation_revision_template, quote=True)}">
<header><div class="audience-switch" aria-label="Report and export actions">{audience_links}{export_links}</div><h1>{escape(view['title'])}</h1><div class="sub">{escape(view['citation_format'])} · {int((view.get('word_counts') or {}).get('total') or 0):,} total words · {int((view.get('word_counts') or {}).get('body') or 0):,} body words · {int((view.get('word_counts') or {}).get('references') or 0):,} reference words · {int((view.get("overview") or {}).get("citations_analyzed") or len(view.get("citations") or [])):,} citations · {int((view.get("overview") or {}).get("reference_count") or 0):,} references · {int((view.get("overview") or {}).get("verified_full_text_sources") or 0):,} full-text references · {int((view.get("overview") or {}).get("abstract_or_limited_sources") or 0):,} abstract/limited-text references</div>
{summaries}{how_to_read}
<div class="toolbar"><button type="button" id="paper-only">Paper only</button><label><input id="layer-evidence" type="checkbox" checked>Evidence</label><label><input id="layer-reference" type="checkbox" checked>Reference practice</label><label title="Experimental citation-use judgment is disabled for this prototype"><input type="checkbox" disabled>Experimental judgment</label></div>
<div class="legend" id="active-key"><span data-key="evidence"><i class="color-key blue"></i>blue: full text available</span><span data-key="evidence"><i class="color-key teal"></i>teal: abstract or limited text available</span><span data-key="evidence"><i class="color-key no-connection"></i>brown: checked source; no clear matching passage</span><span data-key="evidence"><i class="color-key amber"></i>orange: a specific issue needs attention</span><span data-key="evidence"><i class="color-key gray"></i>light grey: no retrieved text or not verifiable</span><span data-key="reference"><i class="reference-key"></i>reference-formatting difference</span>{annotation_key}</div></header>
<main class="layout" id="report-layout"><section class="paper" aria-label="Submitted paper with SourceFidelity overlays"><h2>Submitted paper</h2><div class="toolbar paper-toolbar"><span class="group" aria-label="Paper zoom"><button type="button" id="zoom-out" aria-label="Zoom out">−</button><button type="button" id="zoom-in" aria-label="Zoom in">＋</button><span id="zoom-value">100%</span></span><span class="group" aria-label="Paper annotation tools"><button type="button" id="add-comment" disabled>Comment</button><button type="button" id="add-highlight" disabled>Highlight</button><button type="button" id="select-text" class="active">Select text</button><button type="button" id="select-region">Select area</button><button type="button" id="pen-tool">Pen (draft)</button><button type="button" id="undo-annotation" disabled>Undo</button><button type="button" id="redo-annotation" disabled>Redo</button></span><span class="annotation-status" id="annotation-status">Select text or a citation, then choose Comment or Highlight in Instructor view. Pen marks are drafts.</span></div><div class="paper-viewport">{paper_pages}</div></section>
<div class="splitter" id="report-splitter" role="separator" aria-label="Resize paper and evidence panels" aria-orientation="vertical" aria-valuemin="30" aria-valuemax="80" aria-valuenow="68" tabindex="0"></div>
<aside class="panel" id="evidence-panel" tabindex="-1" aria-live="polite"><h2>Select a citation</h2><p>Choose a marked span while reading the paper to inspect its source-specific evidence.</p></aside></main>
{gauges}<footer class="processing-footer">{escape(_processing_label(view.get("processing_metrics") or {}))}</footer>{_render_export_details(export_action)}<div hidden>{panel_templates}</div>
<script nonce="{nonce}">{Path(__file__).with_name("report_interactions.js").read_text()}</script>
</body></html>"""
    return _bind_inline_styles(rendered, nonce)



def _bind_inline_styles(rendered: str, nonce: str) -> str:
    """Move authored geometry/styles into the permitted nonce stylesheet."""
    from html import unescape
    rules = []
    def replace(match):
        value = unescape(match[1])
        if any(part in value.casefold() for part in ('<', '>', '@', 'url(', 'expression(')):
            raise EvidenceReportError('Report style is invalid')
        index = len(rules)
        rules.append(f'[data-report-style="s{index}"]'+'{'+value+'}')
        return f' data-report-style="s{index}"'
    rendered = re.sub(r' style="([^"]*)"', replace, rendered)
    return rendered.replace('</style>', '\n'+''.join(rules)+'</style>', 1)


def enable_authenticated_paper_actions(view: dict, *, report_id: str) -> dict:
    """Add same-origin paper/source links without mutating the projection."""
    result = deepcopy(view)
    result["paper_surface"] = {
        **result["paper_surface"],
        "report_id": report_id,
        "page_href_template": f"/report/{report_id}/paper/page/{{page_index}}",
        "action": {
            "label": "Open paper",
            "href": f"/report/{report_id}/paper",
        },
    }
    result["export_action"] = {
        "report_href": f"/report/{report_id}?audience=student",
        "pdf_href": f"/report/{report_id}/export.pdf",
        "print_href": f"/report/{report_id}/export/print",
        "manifest_href": f"/report/{report_id}/export/manifest",
    }
    for citation in result.get("citations", []):
        location = citation.get("paper_location") or {}
        action = location.get("action") or {}
        anchor_id = location.get("anchor_id")
        if (
            anchor_id
            and location.get("localization_level")
            in {"exact_rectangle", "page_only"}
        ):
            action.update(
                {
                    "enabled": True,
                    "status": "authenticated_paper_view_available",
                    "href": f"/report/{report_id}/paper/anchor/{anchor_id}",
                }
            )
            location["action"] = action
        if any(
            _member_accepts_upload(member)
            for member in citation.get("members", [])
        ):
            citation["upload_action"] = {
                "enabled": True,
                "label": "Upload source",
                "href": (
                    f"/report/{report_id}/citation/"
                    f"{quote(str(citation.get('claim_id') or ''), safe='')}/source/upload"
                ),
            }
        for member in citation.get("members", []):
            navigation = member.get("source_navigation") or {}
            verification_report_id = member.get("verification_report_id")
            if navigation.get("status") != "ready" or not verification_report_id:
                continue
            if str(navigation.get("representation_id") or "").startswith(
                "verification-run:"
            ):
                # The transient source is deleted after its bounded Evidence
                # Package is persisted. Never render a whole-source action that
                # will fail after report finalization.
                member["source_action"] = {
                    "enabled": False,
                    "status": "bounded_report_evidence_only",
                    "label": "Whole source is not retained",
                }
                continue
            source_action = dict(member.get("source_action") or {})
            source_action.update(
                {
                    "enabled": True,
                    "status": "authenticated_source_view_available",
                    "label": (
                        "Open abstract"
                        if member.get("coverage_level") == "abstract_only"
                        else "Open available text"
                        if member.get("coverage_level") not in {"full_text", "unavailable"}
                        else "Open source"
                    ),
                    "href": f"/report/{report_id}/source/{verification_report_id}",
                }
            )
            member["source_action"] = source_action
    return result


def _member_accepts_upload(member: dict) -> bool:
    from app.services.source_type import is_traditional_media
    source = member.get("source") or {}
    return (member.get("coverage_level") != "full_text"
            and source.get("source_kind") not in {"traditional_media", "video", "podcast_episode"}
            and not is_traditional_media(str(source.get("raw_reference") or "")))


def _field_characters(text: str) -> tuple[str, list[int]]:
    """Comparison-only typography folding with exact original character offsets."""
    chars, offsets = [], []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char).casefold():
            if folded.isalnum():
                chars.append(folded)
                offsets.append(index)
    return "".join(chars), offsets


def _attach_reference_field_geometry(view: dict, document, paper_hash: str) -> None:
    """Bind each differing field inside one uniquely matched complete reference.

    DOCX paragraph indexes are not PDF coordinates. Never fall back to marking
    the entire entry or to a year/title found elsewhere in the paper.
    """
    findings = [f for f in view.get("reference_practice") or [] if f.get("field_difference")]
    if not findings:
        return
    characters, geometry = [], []
    for page in document:
        for block_index, block in enumerate(page.get_text("rawdict", sort=True)["blocks"]):
            for line_index, line in enumerate(block.get("lines", [])):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        normalized, _ = _field_characters(char["c"])
                        for folded in normalized:
                            characters.append(folded)
                            geometry.append((page.number, block_index, line_index, char["bbox"]))
    surface = "".join(characters)
    for finding in findings:
        finding["rectangles"] = []
        finding["localization_status"] = "exact_field_not_located"
        raw = str((finding.get("source") or {}).get("raw_reference") or "")
        entry, _ = _field_characters(raw)
        difference = finding["field_difference"]
        field, _ = _field_characters(difference.get("submitted_value") or "")
        if len(entry) < 12 or not field:
            continue
        entry_matches = [m.start() for m in re.finditer(f"(?={re.escape(entry)})", surface)]
        field_matches = [m.start() for m in re.finditer(f"(?={re.escape(field)})", entry)]
        # A URL may repeat the publication year. Only a unique date in the
        # bibliographic prefix before an exactly identified title is eligible.
        if difference.get("field_name") == "year" and len(field_matches) > 1:
            title, _ = _field_characters((finding.get("source") or {}).get("title") or "")
            if title and entry.count(title) == 1:
                field_matches = [i for i in field_matches if i + len(field) <= entry.index(title)]
        if len(entry_matches) != 1 or len(field_matches) != 1:
            continue
        start = entry_matches[0] + field_matches[0]
        boxes = {}
        for page_index, block_index, line_index, bbox in geometry[start:start + len(field)]:
            key = (page_index, block_index, line_index)
            box = fitz.Rect(bbox)
            boxes[key] = boxes[key] | box if key in boxes else box
        finding["rectangles"] = [
            {"page_index": key[0], "x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}
            for key, box in boxes.items()
        ]
        finding["localization_status"] = "exact_field"
        finding["geometry_provenance"] = {
            "presentation_sha256": paper_hash,
            "reference_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "field_sha256": hashlib.sha256(str(difference["submitted_value"]).encode()).hexdigest(),
            "method": "unique_complete_reference_and_field_characters_v1",
        }


def attach_quotation_difference_geometry(view: dict, pdf_content: bytes) -> dict:
    """Bind report-only quotation differences to exact words on the paper PDF."""
    result = deepcopy(view)
    document = fitz.open(stream=pdf_content, filetype="pdf")
    try:
        _attach_reference_field_geometry(result, document, hashlib.sha256(pdf_content).hexdigest())
        for citation in result.get("citations") or []:
            _restore_visible_paper_hyphens(citation, document, hashlib.sha256(pdf_content).hexdigest())
            differences = list(citation.get("quotation_differences") or [])
            location = citation.get("paper_location") or {}
            if not differences or location.get("localization_level") != "exact_rectangle":
                continue
            by_page: dict[int, list[dict]] = {}
            for rectangle in location.get("rectangles") or []:
                try:
                    by_page.setdefault(int(rectangle["page_index"]), []).append(rectangle)
                except (KeyError, TypeError, ValueError):
                    continue
            quote_match = _QUOTE_PATTERN.search(str(citation.get("student_text") or ""))
            if quote_match is None:
                continue
            quote_tokens = [
                match.group(0).casefold()
                for match in _WORD_PATTERN.finditer(quote_match.group(1))
            ]
            if not quote_tokens:
                continue
            difference_indexes = {
                int(span["quote_token_index"])
                for difference in differences
                for span in difference.get("spans") or []
                if isinstance(span.get("quote_token_index"), int)
            }
            highlights = []
            for page_index, citation_rectangles in by_page.items():
                if not (0 <= page_index < document.page_count):
                    continue
                words = []
                for word in document[page_index].get_text("words"):
                    word_rect = fitz.Rect(word[:4])
                    if not any(
                        word_rect.intersects(
                            fitz.Rect(
                                float(rectangle["x0"]) - 1,
                                float(rectangle["y0"]) - 1,
                                float(rectangle["x1"]) + 1,
                                float(rectangle["y1"]) + 1,
                            )
                        )
                        for rectangle in citation_rectangles
                    ):
                        continue
                    for match in _WORD_PATTERN.finditer(str(word[4])):
                        words.append((match.group(0).casefold(), word_rect))
                normalized = [item[0] for item in words]
                start = next(
                    (
                        index
                        for index in range(0, len(normalized) - len(quote_tokens) + 1)
                        if normalized[index : index + len(quote_tokens)] == quote_tokens
                    ),
                    None,
                )
                if start is None:
                    continue
                for token_index in sorted(difference_indexes):
                    if not 0 <= token_index < len(quote_tokens):
                        continue
                    rectangle = words[start + token_index][1]
                    highlights.append(
                        {
                            "page_index": page_index,
                            "x0": round(rectangle.x0, 3),
                            "y0": round(rectangle.y0, 3),
                            "x1": round(rectangle.x1, 3),
                            "y1": round(rectangle.y1, 3),
                        }
                    )
            if highlights:
                citation["quotation_difference_rectangles"] = highlights
    finally:
        document.close()
    result['role_summaries'] = _build_role_summaries(
        citations=result.get('citations') or [], overview=result.get('overview') or {},
        pervasive_hanging_indent=bool(result.get('reference_practice_summary')),
        reference_practice=result.get('reference_practice') or [],
        require_paper_flags=True,
    )
    return result


def _missing_reference_members(extraction, start: int, end: int) -> list[str]:
    """Name exact parsed author/year members, not arbitrary unresolved years."""
    from app.services.citation_extractor import APA_MEMBER_RE
    missing = []
    for citation in extraction.citations:
        member = str(citation.marker_member or "").strip()
        if (
            citation.link_status == "missing_reference"
            and not citation.candidate_reference_ids
            and citation.marker_type == "parenthetical"
            and APA_MEMBER_RE.fullmatch(member)
            and citation.passage_start < end and start < citation.passage_end
        ):
            missing.append(member)
    return list(dict.fromkeys(missing))


def _available_member(
    member: dict,
    record: VerificationReportRecord,
    reference,
    claim,
    reference_layout=None,
) -> dict:
    package = (record.report_payload or {}).get("authoritative_evidence_package") or {}
    binding = package.get("source_binding") or {}
    if (
        package.get("package_id") != member.get("package_id")
        or package.get("package_sha256") != member.get("package_sha256")
        or package.get("claim_id") != member.get("claim_id")
        or binding.get("reference_id") != member.get("reference_id")
    ):
        raise EvidenceReportError("Evidence Package does not match its aggregate binding")
    passages_by_id = {item.get("passage_id"): item for item in package.get("passages", [])}
    from app.services.verification_evidence import _quotation_match
    targets = _member_quotation_targets(claim, reference.reference_id)
    exact_quote_passage_ids = [passage_id for passage_id, passage in passages_by_id.items()
                               if any(_quotation_match(str(passage.get("excerpt") or passage.get("text") or ""), target) for target in targets)]
    preferred_passage_ids = list(
        dict.fromkeys(
            list((package.get("quotation_check") or {}).get("evidence_passage_ids", []))
            + list((package.get("locator_check") or {}).get("evidence_passage_ids", [])) + exact_quote_passage_ids
        )
    )
    displayed_passage_ids = list(
        dict.fromkeys(
            list((package.get("retrieval") or {}).get("displayed_passage_ids", []))
            + preferred_passage_ids
        )
    )
    displayed = [
        passages_by_id[value]
        for value in displayed_passage_ids
        if value in passages_by_id
    ]
    relevance_gate = (
        (record.report_payload or {}).get("passage_relevance_gate") or {}
    )
    assessments = {
        item.get("passage_id"): item
        for item in relevance_gate.get("assessments", [])
        if item.get("passage_id")
    }
    displayed = _prioritize_display_passages(
        displayed,
        claim.text,
        relevance_gate,
        preferred_passage_ids=preferred_passage_ids,
    )
    ordered = displayed
    assessed_primary = _eligible_display_passages(
        ordered,
        relevance_gate,
        preferred_passage_ids=preferred_passage_ids,
    )
    excluded_only = bool(ordered) and not assessed_primary and all(
        _display_passage_is_metadata(item) for item in ordered
    )
    best_item = assessed_primary[0] if assessed_primary else None
    best = (
        _passage_view(
            best_item,
            assessments.get(best_item.get("passage_id")),
            claim_text=claim.text,
        )
        if best_item
        else None
    )
    additional_items = [
        item for item in assessed_primary if item is not best_item
    ][:2]
    additional = [
        _passage_view(
            item,
            assessments.get(item.get("passage_id")),
            claim_text=claim.text,
        )
        for item in additional_items
    ]
    targets = _member_quotation_targets(claim, reference.reference_id)
    quotation = _check_view(package.get("quotation_check") or {}, _QUOTATION_ATTENTION)
    positive = _positive_quote_check(targets, [str(item.get("text") or item.get("excerpt") or "") for item in passages_by_id.values()])
    if positive:
        quotation = positive
    if targets and not positive and quotation.get("outcome") in _QUOTATION_ATTENTION:
        differences = _quotation_difference_diagnostics(
            _normalize_display_text(claim.text),
            [
                str(item.get("text") or item.get("excerpt") or "")
                for item in passages_by_id.values()
            ],
        )
        if differences:
            quotation["differences"] = differences
            material = any(item["severity"] == "material" for item in differences)
            quotation["attention"] = material
            quotation["label"] = _quotation_difference_label(differences)
    locator = _check_view(package.get("locator_check") or {}, _LOCATOR_ATTENTION)
    limitations = _material_limitations(package, [item for item in [best_item, *additional_items] if item])
    relevance_connected = bool(best_item) and (
        best_item.get("passage_id") in preferred_passage_ids
        or assessments.get(best_item.get("passage_id"), {}).get("relevance")
        in {"relevant", "partially_relevant"}
    )
    completeness = (package.get("coverage") or {}).get("completeness_verdict")
    availability_reason = {
        "uncertain": "Source completeness is uncertain. Check the source manually.",
        "incomplete": "Only part of the source is available. Evidence applies only to the available portion.",
        "not_assessed": "Source completeness was not established. Check the source manually.",
    }.get(completeness, "")
    quotation_ready = quotation.get("status") in {"complete", "incomplete"}
    locator_ready = locator.get("status") in {"complete", "incomplete"}
    return {
        "status": "evidence_available",
        "coverage_level": (package.get("coverage") or {}).get(
            "level", "unavailable"
        ),
        "reference_id": reference.reference_id,
        "source": _reference_view(reference, reference_layout),
        "availability": (
            "The retained extract contains publisher or catalog information, not a passage from the work's contents. The cited content could not be checked; obtain the source for manual review."
            if excluded_only else
            availability_reason
            if availability_reason
            else "No passage has been selected for this statement; compare the source manually."
            if relevance_gate.get("status") != "complete" and best is None
            else (
                "No clearly relevant passage was found for this source's part of the citation. Its title, study setting or metadata may be relevant; check the source manually."
                if len(getattr(claim, "reference_ids", []) or []) > 1
                else "No clearly relevant passage was found. Check the source manually; this does not establish that evidence is absent."
            )
            if ordered and best is None
            else ""
        ),
        "best_evidence": best,
        "additional_evidence": additional,
        "relevance_status": (
            "connected"
            if relevance_connected
            else "not_assessed" if excluded_only
            else "no_connection"
            if relevance_gate.get("status") == "complete"
            else "not_assessed"
        ),
        "quotation_check": quotation,
        "quotation_targets": targets,
        "locator_check": locator,
        "source_action": _reference_source_action(reference),
        "source_navigation": (record.report_payload or {}).get("source_navigation"),
        "verification_report_id": str(record.id),
        "source_version": (package.get("source_identity") or {}).get(
            "edition_or_version"
        ),
        "source_completeness": (package.get("coverage") or {}).get(
            "completeness_verdict"
        ),
        "show_quotation_check": bool(targets) and quotation_ready,
        "show_locator_check": bool(claim.page_locator) and locator_ready,
        "limitations": limitations,
    }


def _render_paper_surface(surface: dict) -> str:
    action = surface.get("action") or {}
    action_html = ""
    if action.get("href") and action.get("label"):
        action_html = (
            f' <a href="{escape(action["href"], quote=True)}" target="_blank" '
            f'rel="noopener">{escape(action["label"])}</a>'
        )
    return (
        '<div class="notice"><strong>Paper view</strong><br>'
        f'{escape(surface["message"])}{action_html}</div>'
    )


def _unavailable_member(member: dict, reference, claim, reference_layout=None) -> dict:
    from app.services.verification_evidence import _quotation_targets, _quotation_match
    from app.services.passage_relevance import _complete_abstract_excerpt

    abstract_available = bool(member.get("abstract_available"))
    abstract = member.get("abstract_evidence") or {}
    abstract_text = _normalize_display_text(abstract.get("text") or "")
    abstract_relevance = member.get("abstract_relevance") or {}
    abstract_relevance_value = str(abstract_relevance.get("relevance") or "")
    relevance_connected = abstract_relevance_value in {
        "relevant",
        "partially_relevant",
    }
    # A lexical best-of-list is not a relevance decision. Preserve the whole
    # bounded abstract when no responsive span has been established.
    excerpt = str(abstract_relevance.get("related_excerpt") or "")
    abstract_hash = hashlib.sha256(abstract_text.encode()).hexdigest()
    claim_hash = hashlib.sha256(str(getattr(claim, "text", "")).encode()).hexdigest()
    bound_assessment = (abstract_relevance.get("status") == "complete"
                        and abstract_relevance.get("abstract_sha256") == abstract_hash
                        and abstract_relevance.get("claim_sha256") == claim_hash)
    abstract_display = (excerpt if bound_assessment and relevance_connected
                        and _complete_abstract_excerpt(abstract_text, excerpt) else abstract_text)
    scope = abstract_relevance.get("scope_assessment") or {}
    scope_attention = bool(
        bound_assessment and not abstract.get("truncated") and scope.get("status") == "complete"
        and scope.get("relevance") == "apparent_mismatch" and scope.get("attention")
        and scope.get("confidence") == "high"
        and scope.get("discrepancy") in {"different_subject", "incompatible_stated_scope"}
        and scope.get("abstract_sha256") == abstract_hash and scope.get("claim_sha256") == claim_hash
        and scope.get("abstract_span") and scope["abstract_span"] in abstract_text
        and scope.get("claim_span") and scope["claim_span"] in str(getattr(claim, "text", ""))
        and str(scope.get("rationale") or "").strip()
    )
    evidence_note = (
        "The abstract addresses only part of the attributed statement."
        if abstract_relevance_value == "partially_relevant"
        else ""
    )
    abstract_evidence = (
        {
            "text": abstract_text,
            "display_text": abstract_display,
            "context_text": abstract_text if abstract_display != abstract_text else "",
            "locator": "",
            "evidence_kind": "abstract",
            "evidence_role": "unclear",
            "evidence_note": evidence_note,
            "excerpt_truncated": bool(abstract.get("truncated")),
        }
        if abstract_available and abstract_text
        else None
    )
    targets = _member_quotation_targets(claim, reference.reference_id)
    quotation = _check_view({}, _QUOTATION_ATTENTION)
    if targets:
        matches = [_quotation_match(abstract_text, target) for target in targets]
        located = bool(abstract_text) and all(match is not None for match in matches)
        quotation = {
            "status": "complete" if located else "not_assessed",
            "outcome": "all_spans_located_in_abstract" if located else "full_source_required",
            "attention": False,
            "label": (
                "Complete quoted wording located in the abstract"
                if located else
                "Quotation cannot be checked without the full source; the complete quoted wording was not located in the available abstract."
            ),
            "limitations": ["Abstract-only wording check; source pagination is not assessed."],
            "matches": [
                {"start": match[0], "end": match[1], "method": match[2]}
                for match in matches if match
            ],
        }
        if located and abstract_evidence:
            abstract_evidence["display_text"] = " … ".join(
                abstract_text[match[0]:match[1]] for match in matches if match
            )
            abstract_evidence["context_text"] = abstract_text
            relevance_connected = True
    return {
        "status": "source_unavailable",
        "coverage_level": "abstract_only" if abstract_available else "unavailable",
        "reference_id": reference.reference_id,
        "source": _reference_view(reference, reference_layout),
        "availability": (
            "This is a media reference. Automated source-text checks are not available."
            if getattr(reference, "source_kind", "") == "traditional_media"
            else "" if abstract_available else "Source not retrieved"
        ),
        "best_evidence": abstract_evidence,
        "additional_evidence": [],
        "relevance_status": "connected" if relevance_connected else "not_assessed",
        "abstract_relevance": abstract_relevance,
        "abstract_scope_attention": scope_attention,
        "quotation_check": quotation,
        "quotation_targets": targets,
        "locator_check": _check_view({}, _LOCATOR_ATTENTION),
        "source_action": _reference_source_action(reference),
        "source_navigation": None,
        "show_quotation_check": bool(targets) and abstract_available and bool(abstract_text),
        "show_locator_check": False,
        # The absence of full text already explains why quotation and locator
        # details are omitted. Repeating that fact in Citation Information
        # adds noise without giving the reader another actionable fact.
        "limitations": [],
    }


def _member_quotation_targets(claim, reference_id: str) -> list[str]:
    """Bind quotes to their exact next parenthetical or preceding narrative marker."""
    from app.services.verification_evidence import _quotation_targets
    text = str(getattr(claim, "text", ""))
    markers = sorted(getattr(claim, "citation_markers", []) or [], key=lambda m:m.local_start)
    if not markers:
        return _quotation_targets(text) if len(getattr(claim, "reference_ids", [reference_id])) <= 1 else []
    result = []
    for quote_span in _QUOTE_PATTERN.finditer(text):
        following = next((m for m in markers if m.local_start >= quote_span.end()), None)
        preceding = next((m for m in reversed(markers) if m.local_end <= quote_span.start()), None)
        marker = following if following and following.marker_type == "parenthetical" else preceding if preceding and preceding.marker_type == "narrative" else None
        if marker and reference_id in marker.reference_ids and text[marker.local_start:marker.local_end] == marker.text:
            result.append(quote_span[1].strip())
    return result


def _unresolved_marker_member(
    reference, reason_codes: list[str], reference_layout=None
) -> dict:
    return {
        "status": "citation_not_assessed",
        "coverage_level": "unavailable",
        "reference_id": reference.reference_id,
        "source": _reference_view(reference, reference_layout),
        "availability": "",
        "best_evidence": None,
        "additional_evidence": [],
        "relevance_status": "not_assessed",
        "quotation_check": _check_view({}, _QUOTATION_ATTENTION),
        "locator_check": _check_view({}, _LOCATOR_ATTENTION),
        "source_action": {
            "status": "citation_boundary_requires_review",
            "label": "Source action unavailable",
            "enabled": False,
        },
        "source_navigation": None,
        "show_quotation_check": False,
        "show_locator_check": False,
        "limitations": [],
        "reference_findings": [],
        "reference_identity": {
            "status": "not_assessed",
            "label": "Not assessed because citation membership is unresolved",
            "attention": False,
        },
    }


def _reference_view(reference, reference_layout=None) -> dict:
    return {
        "author": reference.author,
        "year": reference.year,
        "title": reference.title,
        "doi": getattr(reference, "doi", ""),
        "source_kind": getattr(reference, "source_kind", "unknown"),
        "raw_reference": reference.raw_ref,
        "text_style_spans": (
            [item.model_dump(mode="json") for item in reference_layout.text_style_spans]
            if reference_layout is not None
            else []
        ),
    }


def _reference_source_action(reference) -> dict:
    """Expose only a validated DOI/HTTPS route when stored bytes are unavailable."""
    if reference.doi:
        return {
            "status": "canonical_landing_available",
            "label": "Open source record",
            "enabled": True,
            "href": f"https://doi.org/{quote(reference.doi.strip(), safe='/():._-')}",
        }
    candidate = (reference.url or "").strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        parsed = None
    if parsed and parsed.scheme == "https" and parsed.netloc and not parsed.username:
        return {
            "status": "cited_https_route_available",
            "label": "Open cited source link",
            "enabled": True,
            "href": candidate,
        }
    return {
        "status": "source_route_unavailable",
        "label": "Source route unavailable",
        "enabled": False,
    }


def _passage_view(
    passage: dict,
    assessment: dict | None = None,
    *,
    claim_text: str = "",
) -> dict:
    label = passage.get("page_label")
    if label:
        locator = f"Page {label}"
    elif passage.get("page_index") is not None:
        locator = f"PDF page {int(passage['page_index']) + 1}"
    else:
        locator = "Text location available in the technical record"
    evidence_role = (assessment or {}).get("evidence_role", "unclear")
    relevance = (assessment or {}).get("relevance")
    role_note = {
        "representation_of_other_work": (
            "This passage reports another work rather than this source's own finding. Check whether the original work should also be cited."
        ),
        "methods_or_background": (
            "This passage appears to provide methods or background rather than a direct finding."
        ),
        "source_synthesis_or_conclusion": (
            "This passage states the source author's synthesis or conclusion."
        ),
        "document_level_member_evidence": (
            "This information helps identify the cited study's topic, date, population, or design; one source alone may not establish a broader claim about the literature."
        ),
    }.get(evidence_role, "")
    relevance_note = (
        "This passage addresses only part of the citation; it does not establish every material detail."
        if relevance == "partially_relevant"
        else ""
    )
    note = " ".join(value for value in (relevance_note, role_note) if value)
    normalized_text = _normalize_display_text(passage.get("excerpt", ""))
    display_text = _responsive_display_excerpt(normalized_text, claim_text) if assessment else normalized_text
    if assessment is None:
        note = "Retrieved passage for manual comparison; its connection to this statement has not been assessed."
    return {
        "passage_id": passage.get("passage_id"),
        "text": passage.get("excerpt", ""),
        "display_text": display_text,
        "context_text": normalized_text if normalized_text != display_text else "",
        "locator": locator,
        "evidence_kind": "full_text",
        "evidence_role": evidence_role,
        "evidence_note": note,
        "excerpt_truncated": bool(passage.get("excerpt_truncated")),
    }


def _restore_visible_paper_hyphens(citation: dict, document, paper_hash: str) -> None:
    """Repair display only where exact retained paper geometry proves a hyphen.

    Historical extraction, offsets and packages remain intact. A changed quote
    cannot reuse a check performed on its old extraction.
    """
    raw = []
    for rectangle in (citation.get("paper_location") or {}).get("rectangles") or []:
        page_index = rectangle.get("page_index", -1)
        if not isinstance(page_index, int) or not 0 <= page_index < document.page_count:
            continue
        box = fitz.Rect(*(float(rectangle[key]) for key in ("x0", "y0", "x1", "y1")))
        raw.append(document[page_index].get_textbox(box).replace('\u00ad','-'))
    variants = {}
    wrapped_forms = set()
    for match in re.finditer(r"\b([^\W_]+)-\s*([^\W_]+)\b", '\n'.join(raw)):
        joined = match[1] + match[2]
        variants.setdefault(joined, set()).add(match[1]+'-'+match[2])
        if '\n' in match[0]:
            wrapped_forms.add(match[1]+'-'+match[2])
    original = str(citation.get('display_student_text') or citation.get('student_text') or '')
    repaired = original
    for joined, forms in variants.items():
        if len(forms) == 1:
            repaired = re.sub(r'\b'+re.escape(joined)+r'\b', next(iter(forms)), repaired)
    if repaired == original:
        return
    citation['display_student_text'] = repaired
    citation['paper_wording_provenance'] = {
        'method':'retained_pdf_hyphen_restoration', 'paper_sha256':paper_hash,
        'original_extraction_sha256':hashlib.sha256(str(citation.get('student_text') or '').encode()).hexdigest(),
        'display_sha256':hashlib.sha256(repaired.encode()).hexdigest(),
    }
    from app.services.verification_evidence import _quotation_targets, _quotation_match
    if _quotation_targets(original) == _quotation_targets(repaired):
        return
    citation['quotation_differences'] = []
    citation['quotation_difference_rectangles'] = []
    for member in citation.get('members') or []:
        if member.get('coverage_level') == 'unavailable':
            member['show_quotation_check'] = False
            continue
        targets = list(member.get('quotation_targets') or [])
        if not targets and len(citation.get('members') or []) == 1:
            targets = _quotation_targets(original)
        if not targets:
            member['show_quotation_check'] = False
            continue
        for joined, forms in variants.items():
            if len(forms) == 1:
                targets = [re.sub(r'\b'+re.escape(joined)+r'\b', next(iter(forms)), target) for target in targets]
        member['quotation_targets'] = targets
        comparison_targets = list(targets)
        # Preserve actual line-wrap evidence for the quotation check. A
        # displayed cultivat-ing must not become an alleged changed word when
        # the retained PDF proves that the hyphen occurred at a line ending.
        for form in wrapped_forms:
            comparison_targets = [re.sub(r'\b'+re.escape(form)+r'\b', form.replace('-', '-\n', 1), target)
                                  for target in comparison_targets]
        member['quotation_comparison_targets'] = comparison_targets
        texts = [str(p.get('text') or '') for p in [member.get('best_evidence') or {}, *(member.get('additional_evidence') or [])]]
        positive = _positive_quote_check(comparison_targets, texts)
        member['quotation_check'] = positive or {
            'status':'not_assessed', 'attention':False,
            'outcome':'restored_paper_quote_requires_recheck',
            'label':'The quotation needs a fresh check against the source after a PDF text-extraction correction.',
        }
        member['show_quotation_check'] = True


def _positive_quote_check(targets: list[str], texts: list[str]) -> dict | None:
    """A positive wording check over retained exact evidence, never a negative."""
    from app.services.verification_evidence import _quotation_match
    if not targets:
        return None
    matches = []
    for target in targets:
        found = next(((text, match) for text in texts if (match := _quotation_match(text, target))), None)
        if found is None:
            return None
        text, match = found
        matches.append({"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "quote_sha256": hashlib.sha256(target.encode()).hexdigest(),
                        "start": match[0], "end": match[1], "method": match[2]})
    editorial = any(m["method"] in {"marked_editorial_match", "ellipsis_normalized"} for m in matches)
    return {"status": "complete", "outcome": "retained_evidence_editorial_match" if editorial else "retained_evidence_wording_match",
            "attention": False, "matches": matches,
            "label": ("Unchanged quotation wording matches the source around the marked brackets or omissions. Check the changes against the full context; their meaning has not been automatically verified."
                      if editorial else "Complete quoted wording matches the source.")}


def _normalize_display_text(value: str) -> str:
    """Make extracted PDF prose readable without changing stored evidence."""
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "-", text)
    return re.sub(r"\s+", " ", text).strip()


_QUOTE_PATTERN = re.compile(r'[“"]([^”"]{2,})[”"]')
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_CRITICAL_QUOTATION_TOKENS = {
    "no", "not", "never", "none", "neither", "nor", "without",
    "all", "always", "only", "must", "cannot", "can't",
}


def _quotation_difference_diagnostics(
    claim_text: str, source_passages: list[str]
) -> list[dict]:
    """Describe a close quotation mismatch without weakening exact matching.

    This is a report-only diagnostic over immutable passages. It never changes
    the authoritative exact-quotation result. Mechanical normalization remains
    handled by the authoritative checker; this layer distinguishes a small,
    inspectable wording difference from a quotation that was simply not found.
    """
    results: list[dict] = []
    source_token_sets = [
        [(match.group(0).casefold(), match.group(0)) for match in _WORD_PATTERN.finditer(text)]
        for text in source_passages
        if text
    ]
    for quote_match in _QUOTE_PATTERN.finditer(claim_text):
        quote = quote_match.group(1)
        from app.services.verification_evidence import _quotation_match
        if any(_quotation_match(text, quote) for text in source_passages if text):
            continue
        target_matches = list(_WORD_PATTERN.finditer(quote))
        target = [match.group(0).casefold() for match in target_matches]
        if len(target) < 3:
            continue
        best: tuple[float, list[tuple[str, str]], list[str]] | None = None
        for source_tokens in source_token_sets:
            source = [normalized for normalized, _raw in source_tokens]
            for width in range(max(2, len(target) - 2), len(target) + 3):
                if width > len(source):
                    continue
                for start in range(0, len(source) - width + 1):
                    window = source[start : start + width]
                    matcher = SequenceMatcher(a=target, b=window, autojunk=False)
                    score = matcher.ratio()
                    if best is None or score > best[0]:
                        changes: list[tuple[str, str]] = []
                        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
                            if tag == "equal":
                                continue
                            paper = " ".join(target[a0:a1])
                            source_value = " ".join(window[b0:b1])
                            changes.append((paper, source_value))
                        best = (score, changes, window)
        if best is None or best[0] < 0.58 or not best[1]:
            continue
        score, changes, _window = best
        changed_target_tokens = sum(
            max(1, len(_WORD_PATTERN.findall(paper))) for paper, _source in changes
        )
        difference_ratio = min(1.0, changed_target_tokens / len(target))
        critical = any(
            token in _CRITICAL_QUOTATION_TOKENS or token.isdigit()
            for paper, source in changes
            for token in _WORD_PATTERN.findall(f"{paper} {source}".casefold())
        )
        severity = "material" if difference_ratio > 0.25 or critical else "minor"
        spans = []
        for paper, source in changes:
            paper_tokens = _WORD_PATTERN.findall(paper)
            if not paper_tokens:
                continue
            wanted = [token.casefold() for token in paper_tokens]
            for index in range(0, len(target) - len(wanted) + 1):
                if target[index : index + len(wanted)] != wanted:
                    continue
                local_start = quote_match.start(1) + target_matches[index].start()
                local_end = quote_match.start(1) + target_matches[index + len(wanted) - 1].end()
                spans.append(
                    {
                        "local_start": local_start,
                        "local_end": local_end,
                        "paper_text": claim_text[local_start:local_end],
                        "source_text": source,
                        "quote_token_index": index,
                    }
                )
                break
        results.append(
            {
                "severity": severity,
                "difference_ratio": round(difference_ratio, 4),
                "matching_ratio": round(score, 4),
                "word_count": len(target),
                "changed_word_count": changed_target_tokens,
                "critical_token_changed": critical,
                "spans": spans,
            }
        )
    return results


def _quotation_difference_label(differences: list[dict]) -> str:
    difference = max(differences, key=lambda item: item.get("difference_ratio", 0.0))
    changed = int(difference.get("changed_word_count") or 0)
    total = int(difference.get("word_count") or 0)
    spans = list(difference.get("spans") or [])
    substitution = ""
    if len(spans) == 1 and spans[0].get("source_text"):
        substitution = (
            f': the paper uses “{spans[0]["paper_text"]}” where the source uses '
            f'“{spans[0]["source_text"]}”'
        )
    if difference.get("severity") == "minor":
        return f"Quotation has a minor wording difference ({changed} of {total} words){substitution}"
    return f"Quotation wording differs materially ({changed} of {total} words){substitution}"


_FRONT_MATTER_CUES = re.compile(
    r"\b(email|university|faculty|department|school of|college|institute|centre|center|hospital|professor)\b",
    re.IGNORECASE,
)


def _responsive_display_excerpt(value: str, claim_text: str) -> str:
    """Show the responsive sentence neighborhood from a hash-bound passage.

    Retrieval windows remain unchanged in the Evidence Package. This display
    projection removes identifiable title/byline material before an abstract
    and avoids forcing a reader through an unrelated leading paragraph when a
    later complete sentence is the part connected to the citation.
    """
    text = _normalize_display_text(value)
    text = re.sub(
        r"^(?:(?:[A-Z]\s+){5,}[A-Z])\s+\d+\s+",
        "",
        text,
    )
    abstract = re.search(r"\babstract\b", text, re.IGNORECASE)
    if (
        abstract
        and 0 < abstract.start() <= 700
        and _FRONT_MATTER_CUES.search(text[: abstract.start()])
    ):
        text = f"Abstract {text[abstract.end():].lstrip(' :-')}"
    if not claim_text:
        return text
    sentences = [item.strip() for item in split_sentences(text) if item.strip()]
    if len(sentences) < 2:
        return text
    claim_terms = _display_terms(claim_text)
    claim_set = set(claim_terms)
    claim_bigrams = set(zip(claim_terms, claim_terms[1:]))

    def score(sentence: str) -> tuple[float, int, int]:
        terms = _display_terms(sentence)
        term_set = set(terms)
        overlap = len(claim_set & term_set) / max(1, len(claim_set))
        bigrams = len(claim_bigrams & set(zip(terms, terms[1:])))
        return overlap, bigrams, min(len(sentence), 500)

    best_index = max(range(len(sentences)), key=lambda index: score(sentences[index]))
    if re.match(r"^[.…,;:)\]]", sentences[best_index]):
        return text
    best_score = score(sentences[best_index])
    selected = [sentences[best_index]]
    total = len(selected[0])
    for offset in (1, -1):
        index = best_index + offset
        if not 0 <= index < len(sentences):
            continue
        candidate = sentences[index]
        candidate_score = score(candidate)
        if (
            candidate_score[1] == 0
            and candidate_score[0] < max(0.10, best_score[0] * 0.5)
        ):
            continue
        if len(selected) >= 2 or total + 1 + len(candidate) > 620:
            continue
        if index < best_index:
            selected.insert(0, candidate)
        else:
            selected.append(candidate)
        total += 1 + len(candidate)
    result = " ".join(selected)
    return re.sub(
        r"^\((?:instructor|participant|teacher|student)[^)]*\)\s+",
        "",
        result,
        flags=re.IGNORECASE,
    )


_DISPLAY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "et", "al", "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "in", "is", "it", "its", "of", "on", "or", "our", "she",
    "that", "the", "their", "them", "they", "this", "those", "to", "was",
    "were", "which", "who", "with", "would",
}
_DISPLAY_RELEVANCE = {
    "relevant": 4,
    "partially_relevant": 3,
    "uncertain": 2,
    "not_relevant": 1,
}
_DISPLAY_CONFIDENCE = {"high": 3, "medium": 2, "low": 1, "none": 0}
_DISPLAY_EVIDENCE_ROLE = {
    "source_own_claim_or_finding": 5,
    "source_synthesis_or_conclusion": 4,
    "document_level_member_evidence": 4,
    "unclear": 3,
    "methods_or_background": 2,
    "representation_of_other_work": 1,
}


def _prioritize_display_passages(
    passages: list[dict],
    claim_text: str,
    relevance_gate: dict | None = None,
    *,
    preferred_passage_ids: list[str] | None = None,
) -> list[dict]:
    """Rank an authorized union for viewing without changing evidence identity.

    Retrieval order remains authoritative in the Evidence Package.  This
    report-only projection first uses the already persisted, shadow-only
    relevance assessment when present, then a bounded lexical responsiveness
    score and the recorded retrieval score.  It never removes a passage or
    changes the evidence considered by another stage.
    """
    assessments = {
        item.get("passage_id"): item
        for item in (relevance_gate or {}).get("assessments", [])
        if item.get("passage_id")
    }
    preferred_rank = {
        passage_id: len(preferred_passage_ids or []) - index
        for index, passage_id in enumerate(preferred_passage_ids or [])
    }
    claim_terms = _display_terms(claim_text)
    claim_bigrams = set(zip(claim_terms, claim_terms[1:]))
    indexed = list(enumerate(passages))

    def key(item: tuple[int, dict]) -> tuple[float, ...]:
        index, passage = item
        assessment = assessments.get(passage.get("passage_id"), {})
        passage_terms = _display_terms(str(passage.get("excerpt") or ""))
        passage_term_set = set(passage_terms)
        overlap = sum(term in passage_term_set for term in set(claim_terms))
        overlap_ratio = overlap / max(1, len(set(claim_terms)))
        bigram_overlap = len(claim_bigrams & set(zip(passage_terms, passage_terms[1:])))
        relevance = assessment.get("relevance")
        display_tier = (
            0
            if relevance == "not_relevant"
            else _DISPLAY_EVIDENCE_ROLE.get(assessment.get("evidence_role"), 0)
        )
        return (
            float(preferred_rank.get(passage.get("passage_id"), 0)),
            float(display_tier),
            float(_DISPLAY_RELEVANCE.get(assessment.get("relevance"), 0)),
            float(_DISPLAY_CONFIDENCE.get(assessment.get("confidence"), 0)),
            float(passage.get("boundary_status") == "sentence_complete"),
            overlap_ratio,
            float(bigram_overlap),
            float(passage.get("retrieval_score") or 0.0),
            float(-index),
        )

    return [passage for _index, passage in sorted(indexed, key=key, reverse=True)]


def _display_terms(value: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", value.casefold())
    return [_display_stem(term) for term in terms if term not in _DISPLAY_STOPWORDS]


def _display_passage_is_metadata(passage: dict) -> bool:
    from app.services.verification_evidence import passage_role_from_text
    return passage_role_from_text(str(passage.get("text") or passage.get("excerpt") or "")) == "publication_metadata"


def _eligible_display_passages(
    passages: list[dict],
    relevance_gate: dict | None,
    *,
    preferred_passage_ids: list[str] | None = None,
) -> list[dict]:
    """Keep unassessed retrieved passages inspectable; never imply relevance."""
    # Retain the immutable package, but do not present catalog/publisher
    # boilerplate as substantive source evidence in the ordinary report.
    passages = [passage for passage in passages if not _display_passage_is_metadata(passage)
                or passage.get("passage_id") in (preferred_passage_ids or [])]
    if (relevance_gate or {}).get("status") != "complete":
        return list(passages)
    assessments = {
        item.get("passage_id"): item
        for item in (relevance_gate or {}).get("assessments", [])
        if item.get("passage_id")
    }
    preferred = set(preferred_passage_ids or [])
    eligible = [
        passage
        for passage in passages
        if passage.get("passage_id") in preferred
        or (assessments.get(passage.get("passage_id")) or {}).get("relevance")
        in {"relevant", "partially_relevant"}
    ]
    has_direct = any(
        (assessments.get(passage.get("passage_id")) or {}).get("evidence_role")
        in {
            "source_own_claim_or_finding",
            "source_synthesis_or_conclusion",
            "document_level_member_evidence",
        }
        for passage in eligible
        if passage.get("passage_id") not in preferred
    )
    if has_direct:
        eligible = [
            passage
            for passage in eligible
            if passage.get("passage_id") in preferred
            or (assessments.get(passage.get("passage_id")) or {}).get("evidence_role")
            not in {"representation_of_other_work", "methods_or_background"}
        ]
    has_fully_relevant = any(
        (assessments.get(passage.get("passage_id")) or {}).get("relevance")
        == "relevant"
        for passage in eligible
        if passage.get("passage_id") not in preferred
    )
    if has_fully_relevant:
        return [
            passage
            for passage in eligible
            if passage.get("passage_id") in preferred
            or (assessments.get(passage.get("passage_id")) or {}).get("relevance")
            == "relevant"
        ]
    partial = [
        passage
        for passage in eligible
        if passage.get("passage_id") not in preferred
    ]
    preferred_passages = [
        passage
        for passage in eligible
        if passage.get("passage_id") in preferred
    ]
    return preferred_passages + partial[:1]


def _display_stem(term: str) -> str:
    """Small dependency-free normalizer used only for report ordering."""
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ness", "ing", "ied", "ies", "ed", "es", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def _check_view(check: dict, attention_outcomes: set[str]) -> dict:
    status = check.get("status", "not_run")
    outcome = check.get("outcome")
    return {
        "status": status,
        "outcome": outcome,
        "attention": status == "complete" and outcome in attention_outcomes,
        "label": _reason_label(outcome or status),
        "limitations": list(check.get("limitations", [])),
    }


def _reference_findings_by_id(consistency: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for finding in consistency.get("findings", []):
        for reference_id in finding.get("reference_ids", []):
            result.setdefault(reference_id, []).append(finding)
    return result


def _identity_view(discovery: dict | None) -> dict:
    if not discovery:
        return {
            "status": "not_assessed",
            "label": "Reference identity search was not attached",
            "attention": False,
        }
    outcome = discovery.get("outcome") or "not_assessed"
    return {
        "status": outcome,
        "label": _reason_label(outcome),
        "attention": outcome == "bibliographic_conflict",
        "edition_year_unresolved": any(
            comparison.get("reason_code") == "book_edition_year_unresolved"
            for candidate in discovery.get("candidates") or []
            for comparison in candidate.get("comparisons") or []
        ),
    }


def _reference_identity_conflict_view(discovery: dict) -> dict:
    """Expose only the located candidate fields needed to inspect a conflict."""
    candidates = list(discovery.get("candidates") or [])
    candidate = next(
        (
            item
            for item in candidates
            if (item.get("plausible_identity_match") or item.get("authoritative_identifier_match")) and any(
                comparison.get("outcome") == "material_conflict"
                for comparison in item.get("comparisons") or []
            )
        ),
        {},
    )
    observed = candidate.get("observed") or {}
    expected = discovery.get("expected") or {}
    def value(record, field):
        raw = record.get("authors" if field == "author" else field) or ""
        return ", ".join(raw) if isinstance(raw, list) else str(raw)
    return {
        "located_record": {
            "title": str(observed.get("title") or ""),
            "authors": list(observed.get("authors") or []),
            "year": str(observed.get("year") or ""),
            "doi": str(observed.get("doi") or ""),
            "container_title": str(observed.get("container_title") or ""),
        },
        "conflicting_fields": [
            str(item.get("field_name") or "")
            for item in candidate.get("comparisons") or []
            if item.get("outcome") == "material_conflict"
        ],
        "field_differences": [
            {"field_name": item["field_name"],
             "submitted_value": value(expected, item["field_name"]),
             "located_value": value(observed, item["field_name"]),
             "candidate_id": candidate.get("candidate_id"),
             "provider": candidate.get("provider"),
             "reason_code": item.get("reason_code")}
            for item in candidate.get("comparisons") or []
            if item.get("outcome") == "material_conflict"
        ],
    }


def _material_limitations(package: dict, displayed: list[dict]) -> list[str]:
    """Keep the ordinary panel concise; technical retrieval limits stay in audit."""
    candidates = [
        *((package.get("source_identity") or {}).get("limitations", [])),
        *((package.get("coverage") or {}).get("limitations", [])),
    ]
    if any(item.get("excerpt_truncated") for item in displayed):
        candidates.append(
            "At least one displayed excerpt was shortened. Open the source if more context is needed."
        )
    return list(
        dict.fromkeys(
            _user_facing_limitation(item)
            for item in candidates
            if item
        )
    )


def _user_facing_limitation(value: str) -> str:
    normalized = str(value).casefold()
    if "representation completeness is 'uncertain'" in normalized:
        return "Source completeness is uncertain. Check the source manually."
    if "representation completeness is 'incomplete'" in normalized:
        return "Only part of the source is available. Evidence applies only to the available portion."
    if "representation completeness is 'not_assessed'" in normalized:
        return "Source completeness was not established. Check the source manually."
    if "policy-bounded subset of the source text" in normalized:
        return "Only part of the source text could be processed. Check the complete source manually."
    return str(value)


def _reason_label(value: str | None) -> str:
    labels = {
        "all_spans_literal_match": "Quoted wording located exactly",
        "all_spans_normalized_match": "Complete quoted wording matches the source",
        "all_spans_ocr_token_sequence_match": "Quoted wording located in OCR text",
        "no_span_located": "Quoted wording was not located in the retrieved evidence",
        "some_spans_not_located": "Some quoted wording was not located",
        "located_span_matches_supplied_locator": "Provided page/paragraph number matches the retrieved source",
        "located_span_outside_supplied_locator": "Located quotation falls outside the supplied locator",
        "source_version_pagination_may_differ": "Published-page locator cannot be checked against this source version",
        "not_applicable": "Not applicable",
        "not_run": "Not run",
        "not_assessable": "Not assessable",
        "full_text_unavailable": "Full text was not retrieved",
        "source_not_found": "The cited route did not provide retrievable source content, and no authorized replacement was located",
        "cited_webpage_unavailable": "The cited webpage could not be retrieved, and no replacement source content was located",
        "cited_webpage_unavailable_metadata_only": "The cited webpage could not be retrieved; bibliographic or search records were found, but no full source content was acquired",
        "identity_not_verified": "Source identity was not verified",
        "confirmed": "Reference identity confirmed",
        "confirmed_with_minor_differences": "Reference identity confirmed with minor differences",
        "possible_match": "Possible bibliographic match; review may be useful",
        "bibliographic_conflict": "Bibliographic fields conflict with the located record",
        "unlocated_after_search": "Reference was not located after a completed search",
        "insufficient_metadata": "Reference metadata was insufficient for a completed search",
        "search_incomplete": "Reference search was incomplete",
        "source_result_missing": "No source result was available",
        "requires_authenticated_source_delivery": "Available only through an authenticated source view",
        "source_upload_or_correction_needed": "Source upload/correction is not yet connected in this view",
        "citation_scope_or_member_unresolved": "Citation scope or source membership is unresolved",
        "citation_reference_ambiguous": "Citation could match more than one submitted reference",
        "citation_reference_missing": "Citation was not linked to a submitted reference",
        "citation_marker_not_parsed": "Citation marker was detected but could not be parsed safely",
        "citation_member_bound": "Citation marker is bound to its source member",
        "citation_boundary_requires_review": "Citation boundary must be resolved before source evidence can be assessed",
    }
    if value in labels:
        return labels[value]
    return str(value or "Unavailable").replace("_", " ").capitalize()


def _render_continuous_paper(
    surface: dict,
    citations: list[dict],
    reference_practice: list[dict] | None = None,
    annotations: list[dict] | None = None,
) -> tuple[str, set[int]]:
    dimensions = surface.get("page_dimensions") or []
    href_template = surface.get("page_href_template")
    if not dimensions or not href_template:
        return (
            '<p class="muted">The page-faithful paper surface is unavailable in this view.</p>',
            set(),
        )
    overlays_by_page: dict[int, list[str]] = {}
    annotations_by_anchor: dict[str, list[dict]] = {}
    for annotation in annotations or []:
        anchor_id = str((annotation.get("anchor") or {}).get("anchor_id") or "")
        if anchor_id:
            annotations_by_anchor.setdefault(anchor_id, []).append(annotation)
    placed: set[int] = set()
    for index, citation in enumerate(citations, 1):
        location = citation.get("paper_location") or {}
        if location.get("localization_level") != "exact_rectangle":
            continue
        rectangles = location.get("rectangles") or []
        anchor_id = str(location.get("anchor_id") or "")
        anchor_annotations = annotations_by_anchor.get(anchor_id, [])
        persisted_highlight = any(
            item.get("annotation_type") == "highlight" for item in anchor_annotations
        )
        released_highlight = any(
            item.get("annotation_type") == "highlight"
            and item.get("visibility") == "released"
            for item in anchor_annotations
        )
        persisted_comments = [
            item for item in anchor_annotations if item.get("annotation_type") == "comment"
        ]
        released_comments = [
            item for item in persisted_comments if item.get("visibility") == "released"
        ]
        valid_rectangles: dict[int, list[tuple[float, float, float, float]]] = {}
        for item in rectangles:
            try:
                page_index = int(item["page_index"])
                x0, y0, x1, y1 = (
                    float(item["x0"]),
                    float(item["y0"]),
                    float(item["x1"]),
                    float(item["y1"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= x0 < x1 and 0 <= y0 < y1:
                valid_rectangles.setdefault(page_index, []).append((x0, y0, x1, y1))
        for page_index, page_rectangles in valid_rectangles.items():
            tones = citation_tones(citation)
            palette = {"evidence_available":"#2563a7", "limited_evidence":"#168c8c", "retrieved_no_connection":"#8c6b4f", "attention":"#d95f02", "not_assessed":"#b8c0c8"}
            marks = "".join(
                f'<rect class="selection-bg" x="{x0:.3f}" y="{y0:.3f}" '
                f'width="{x1-x0:.3f}" height="{y1-y0:.3f}" rx="1" />'
                + "".join(
                    f'<line class="underline" style="stroke:{palette[tone]}" x1="{x0+(x1-x0)*part/len(tones):.3f}" y1="{y1-0.8:.3f}" '
                    f'x2="{x0+(x1-x0)*(part+1)/len(tones):.3f}" y2="{y1-0.8:.3f}" />'
                    for part, tone in enumerate(tones)
                )
                for x0, y0, x1, y1 in page_rectangles
            )
            difference_marks = "".join(
                f'<rect class="quote-difference-mark" x="{float(item["x0"]):.3f}" '
                f'y="{float(item["y0"]):.3f}" width="{float(item["x1"])-float(item["x0"]):.3f}" '
                f'height="{float(item["y1"])-float(item["y0"]):.3f}" rx="1" />'
                for item in citation.get("quotation_difference_rectangles") or []
                if int(item.get("page_index", -1)) == page_index
                and float(item.get("x1", 0)) > float(item.get("x0", 0))
                and float(item.get("y1", 0)) > float(item.get("y0", 0))
            )
            tone = _safe_tone(citation.get("tone"))
            coverage = (
                citation.get("display_coverage")
                if citation.get("display_coverage") in {"full", "limited", "none"}
                else "none"
            )
            annotation_marks = ""
            if persisted_comments and page_rectangles:
                marker_x = max(item[2] for item in page_rectangles) - 5
                marker_y = min(item[1] for item in page_rectangles) + 5
                comment_class = (
                    "instructor-comment-marker"
                    if released_comments
                    else "instructor-comment-marker comment-private-only"
                )
                annotation_marks = (
                    f'<circle class="{comment_class}" cx="{marker_x:.3f}" '
                    f'cy="{marker_y:.3f}" r="4.5"><title>'
                    f'{len(persisted_comments)} saved instructor comment'
                    f'{"s" if len(persisted_comments) != 1 else ""}</title></circle>'
                )
            annotation_class = " persisted-highlight" if persisted_highlight else ""
            if persisted_highlight and not released_highlight:
                annotation_class += " highlight-private-only"
            annotation_template = (
                f' data-annotation-template="annotation-panel-{anchor_id}"'
                if anchor_annotations
                else ""
            )
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="citation-overlay {tone} coverage_{coverage}{annotation_class}" '
                f'role="button" tabindex="0" aria-pressed="false" '
                f'aria-label="Citation {index}: {escape(", ".join(_tone_label(t) for t in tones), quote=True)}" '
                f'data-hover-label="{escape(_tone_label(tone), quote=True)} — select to inspect evidence" '
                f'data-anchor-id="{escape(anchor_id, quote=True)}"{annotation_template} '
                f'data-panel-template="citation-panel-{index}">{marks}{difference_marks}{annotation_marks}</a>'
            )
            placed.add(index)

    for anchor_id, anchor_annotations in annotations_by_anchor.items():
        anchor = anchor_annotations[0].get("anchor") or {}
        if anchor.get("anchor_kind") not in {"page_region", "text_selection"}:
            continue
        by_page: dict[int, list[tuple[float, float, float, float]]] = {}
        for item in anchor.get("rectangles") or []:
            try:
                page_index = int(item["page_index"])
                coords = tuple(float(item[key]) for key in ("x0", "y0", "x1", "y1"))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= coords[0] < coords[2] and 0 <= coords[1] < coords[3]:
                by_page.setdefault(page_index, []).append(coords)
        if not by_page:
            continue
        persisted_highlight = any(
            item.get("annotation_type") == "highlight" for item in anchor_annotations
        )
        released = any(
            item.get("visibility") == "released" for item in anchor_annotations
        )
        comments = [
            item for item in anchor_annotations if item.get("annotation_type") == "comment"
        ]
        released_comments = [
            item for item in comments if item.get("visibility") == "released"
        ]
        classes = "paper-annotation-overlay"
        if persisted_highlight:
            classes += " persisted-highlight"
        if not released:
            classes += " annotation-private-only"
        for page_index, rectangles in by_page.items():
            marks = "".join(
                f'<rect class="annotation-region-bg" x="{x0:.3f}" y="{y0:.3f}" '
                f'width="{x1-x0:.3f}" height="{y1-y0:.3f}" rx="1" />'
                for x0, y0, x1, y1 in rectangles
            )
            if comments:
                marker_x = max(item[2] for item in rectangles) - 5
                marker_y = min(item[1] for item in rectangles) + 5
                marker_class = (
                    "instructor-comment-marker"
                    if released_comments
                    else "instructor-comment-marker comment-private-only"
                )
                marks += (
                    f'<circle class="{marker_class}" cx="{marker_x:.3f}" '
                    f'cy="{marker_y:.3f}" r="4.5"><title>'
                    f'{len(comments)} saved instructor comment'
                    f'{"s" if len(comments) != 1 else ""}</title></circle>'
                )
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="{classes}" role="button" tabindex="0" '
                f'aria-pressed="false" aria-label="Instructor annotation" '
                f'data-anchor-id="{escape(anchor_id, quote=True)}" '
                f'data-panel-template="annotation-panel-{escape(anchor_id, quote=True)}">'
                f'{marks}</a>'
            )

    for index, finding in enumerate(reference_practice or [], 1):
        by_page: dict[int, list[tuple[float, float, float, float]]] = {}
        for item in finding.get("rectangles") or []:
            try:
                page_index = int(item["page_index"])
                coords = tuple(float(item[key]) for key in ("x0", "y0", "x1", "y1"))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= coords[0] < coords[2] and 0 <= coords[1] < coords[3]:
                by_page.setdefault(page_index, []).append(coords)
        for page_index, rectangles in by_page.items():
            marks = "".join(
                f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{x1-x0:.3f}" height="{y1-y0:.3f}" rx="1" />'
                for x0, y0, x1, y1 in rectangles
            )
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="reference-practice-overlay" role="button" '
                f'tabindex="0" aria-pressed="false" aria-label="Reference-practice difference" '
                f'data-panel-template="reference-panel-{index}">{marks}</a>'
            )

    pages: list[str] = []
    for item in dimensions:
        try:
            page_index = int(item["page_index"])
            width = float(item["width"])
            height = float(item["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        href = href_template.format(page_index=page_index)
        text_layer = "".join(
            f'<span class="paper-word" data-word-index="{index}" style="left:{word[0]/width*100:.5f}%;'
            f'top:{(word[1]+(word[3]-word[1])*.12)/height*100:.5f}%;width:{(word[2]-word[0])/width*100:.5f}%;'
            f'height:{(word[3]-word[1])/height*100:.5f}%;font-size:{(word[3]-word[1])/width*77:.5f}cqw">{escape(str(word[4]))} </span>'
            for index, word in enumerate((surface.get("selectable_words") or {}).get(page_index, []))
        )
        pages.append(
            f'<figure class="paper-page"><figcaption>Page {page_index + 1}</figcaption>'
            f'<div class="page-container" style="aspect-ratio:{width:.3f}/{height:.3f}" data-page-index="{page_index}"><svg class="page-surface" data-page-index="{page_index}" viewBox="0 0 {width:.3f} {height:.3f}" '
            f'role="img" aria-label="Submitted paper page {page_index + 1}">'
            f'<image href="{escape(href, quote=True)}" width="{width:.3f}" height="{height:.3f}" />'
            f'{"".join(overlays_by_page.get(page_index, []))}</svg><div class="paper-text-layer">{text_layer}</div></div></figure>'
        )
    return f'<div class="paper-pages">{"".join(pages)}</div>', placed


def _render_annotation_templates(annotations: list[dict]) -> str:
    grouped: dict[str, list[dict]] = {}
    for annotation in annotations:
        anchor_id = str((annotation.get("anchor") or {}).get("anchor_id") or "")
        if anchor_id:
            grouped.setdefault(anchor_id, []).append(annotation)
    templates = []
    for anchor_id, values in grouped.items():
        notes = []
        for item in values:
            annotation_id = escape(str(item.get("annotation_id") or ""), quote=True)
            revision = int(item.get("revision") or 0)
            annotation_type = str(item.get("annotation_type") or "")
            visibility = str(item.get("visibility") or "private")
            content = str(item.get("content") or "")
            label = "Comment" if annotation_type == "comment" else "Highlight"
            text = (
                f'<p>{escape(content)}</p>'
                if content
                else '<p class="muted">Highlighted by the report owner.</p>'
            )
            notes.append(
                f'<section class="annotation-note" data-annotation-id="{annotation_id}" '
                f'data-revision="{revision}" data-visibility="{escape(visibility, quote=True)}" '
                f'data-content="{escape(content, quote=True)}"><strong>{label}</strong> '
                f'<span class="annotation-visibility">{escape(visibility)}</span>{text}'
                '<div class="annotation-actions">'
                + (
                    '<button type="button" data-annotation-operation="edit">Edit</button>'
                    if annotation_type == "comment"
                    else ""
                )
                + f'<button type="button" data-annotation-operation="visibility">'
                f'{"Make private" if visibility == "released" else "Release"}</button>'
                '<button type="button" data-annotation-operation="delete">Delete</button>'
                '</div></section>'
            )
        templates.append(
            f'<template id="annotation-panel-{escape(anchor_id, quote=True)}">'
            '<h2>Instructor annotations</h2>'
            f'{"".join(notes)}</template>'
        )
    return "".join(templates)


def _render_unplaced_citations(citations: list[dict], placed: set[int]) -> str:
    rows = "".join(
        f'<button type="button" data-panel-template="citation-panel-{index}" '
        f'aria-pressed="false" data-hover-label="Citation location not resolved — select to inspect">'
        f'{escape(_citation.get("student_text") or "Citation location not resolved")}</button>'
        for index, _citation in enumerate(citations, 1)
        if index not in placed
    )
    if not rows:
        return ""
    return (
        '<details class="unplaced"><summary>Citations without exact page geometry</summary>'
        f'<p class="muted">These items remain not assessed and are not placed speculatively.</p>{rows}</details>'
    )


def _safe_tone(value: str | None) -> str:
    return value if value in {
        "evidence_available", "limited_evidence", "retrieved_no_connection",
        "attention", "not_assessed"
    } else "not_assessed"


def _tone_label(tone: str) -> str:
    return {
        "evidence_available": "full text available",
        "limited_evidence": "abstract or limited text available",
        "retrieved_no_connection": "source retrieved; no clear matching passage",
        "attention": "needs attention",
        "not_assessed": "not verifiable",
    }[tone]


def grouped_members(citation: dict) -> list[tuple[str, list[dict]]]:
    groups = (
        ("Full text sources", {"full_text"}),
        ("Limited text sources", {"partial_text"}),
        ("Abstract only sources", {"abstract_only"}),
        ("Not retrieved sources", {"unavailable"}),
    )
    return [
        (label, [m for m in citation.get("members", []) if (m.get("coverage_level") or "unavailable") in levels])
        for label, levels in groups
        if any((m.get("coverage_level") or "unavailable") in levels for m in citation.get("members", []))
    ]


def member_tone(member: dict) -> str:
    if member.get("abstract_scope_attention"):
        return "attention"
    if any((member.get(check) or {}).get("attention") for check in ("quotation_check", "locator_check")):
        return "attention"
    if member.get("relevance_status") == "connected":
        return "evidence_available" if member.get("coverage_level") == "full_text" else "limited_evidence"
    if member.get("coverage_level") in {"abstract_only", "partial_text"}:
        return "limited_evidence"
    if member.get("coverage_level") == "full_text" and member.get("relevance_status") == "not_assessed":
        return "evidence_available"
    return "retrieved_no_connection" if member.get("coverage_level") == "full_text" else "not_assessed"


def citation_tones(citation: dict) -> list[str]:
    if citation.get("missing_reference_members"):
        return ["attention"]
    tones = [member_tone(m) for _label, members in grouped_members(citation) for m in members]
    if not tones:
        return [_safe_tone(citation.get("tone"))]
    return tones


def _render_upload(upload: dict) -> str:
    if not upload.get("enabled") or not upload.get("href"):
        return ""
    return (
        f'<form class="source-upload" method="post" enctype="multipart/form-data" action="{escape(upload["href"], quote=True)}">'
        '<input type="file" name="file" accept="application/pdf,.pdf" aria-label="Choose source PDF" hidden required>'
        '<button type="button" data-choose-source>Upload source</button>'
        '<span class="upload-status" aria-live="polite"></span></form>'
    )


def _processing_label(metrics: dict) -> str:
    def seconds(key):
        value = metrics.get(key)
        return f"{value:,.2f} s" if isinstance(value, (int, float)) else "not recorded"
    calls = metrics.get("llm_calls")
    tokens = metrics.get("total_tokens")
    return (
        f"Processing time: {seconds('wall_seconds')} · CPU time: {seconds('cpu_seconds')} · "
        f"LLM calls: {calls if calls is not None else 'not recorded'} · "
        f"LLM tokens: {tokens if tokens is not None else 'not recorded'}"
        + (" · Recorded stages only" if metrics.get("partial") else "")
    )


def _render_export_details(action: dict) -> str:
    if not action.get("manifest_href"):
        return ""
    return (
        '<details class="technical-export"><summary>Technical export record</summary>'
        '<p>For verification and support: identifies the report version, original paper, released annotation revisions and PDF fingerprint. '
        'These details help an owner or administrator establish which version was shared; they are not academic findings.</p>'
        f'<a href="{escape(action["manifest_href"], quote=True)}">Download verification record (JSON)</a></details>'
    )


def _render_panel_template(citation: dict, index: int) -> str:
    upload = citation.get("upload_action") or {}
    members = "".join(
        f'<section class="source-group {escape(str(values[0].get("coverage_level") or "unavailable"), quote=True)}"><h3>{label}</h3>'
        + "".join(_render_member({**item, "upload_action": upload if _member_accepts_upload(item) else {}}, grouped=True) for item in values)
        + '</section>'
        for label, values in grouped_members(citation)
    )
    if not members:
        members = (
            '<p class="muted">No source-specific evidence could be attached to this citation.</p>'
        )
    information = _render_citation_information(citation, index)
    practice = (
        f'<p class="attention-text">{escape(citation["boundary_reason"])}</p>'
        if citation.get("missing_reference_members") else ""
    )
    return (
        f'<template id="citation-panel-{index}"><h2>Selected citation</h2>'
        f'<p class="selected-citation">{_render_citation_text(citation)}</p>'
        f'{practice}{members}{information}</template>'
    )


def _render_citation_text(citation: dict) -> str:
    text = str(citation.get("display_student_text") or citation.get("student_text") or "")
    # Difference spans are computed against the normalized display text for
    # current extracted citations; reject any stale or overlapping span rather
    # than highlighting the wrong words.
    spans = sorted(
        (
            span
            for difference in citation.get("quotation_differences") or []
            for span in difference.get("spans") or []
        ),
        key=lambda item: (int(item.get("local_start", -1)), int(item.get("local_end", -1))),
    )
    valid = []
    cursor = 0
    for span in spans:
        try:
            start, end = int(span["local_start"]), int(span["local_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (cursor <= start < end <= len(text)):
            continue
        if text[start:end] != span.get("paper_text"):
            continue
        valid.append((start, end))
        cursor = end
    if not valid:
        return escape(text)
    output = []
    cursor = 0
    for start, end in valid:
        output.append(escape(text[cursor:start]))
        output.append(
            f'<mark class="quote-difference">{escape(text[start:end])}</mark>'
        )
        cursor = end
    output.append(escape(text[cursor:]))
    return "".join(output)


def _render_citation_information(citation: dict, index: int) -> str:
    notes = []
    if citation.get("boundary_reason"):
        notes.append(str(citation["boundary_reason"]))
    for member in citation.get("members") or []:
        notes.extend(str(item) for item in member.get("limitations") or [] if item)
        version = str(member.get("source_version") or "").strip()
        if version:
            notes.append(f"Source version: {_source_version_label(version)}")
    notes = list(dict.fromkeys(notes))
    detail_items = [f"Citation {int(citation.get('citation_number') or index)}"]
    detail_items.extend(notes)
    rows = "".join(f"<li>{escape(item)}</li>" for item in detail_items)
    return f"<details><summary>Citation Information</summary><ul>{rows}</ul></details>"


def _source_version_label(value: str) -> str:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {
        "accepted_manuscript",
        "author_accepted_manuscript",
        "accepted_version",
    }:
        return (
            "author-accepted manuscript; wording can be verified, but publisher "
            "pagination and layout may differ"
        )
    if normalized in {"preprint", "submitted_manuscript"}:
        return "pre-publication manuscript; compare version-specific wording manually"
    return value.replace("_", " ")


def _render_reference_panel_template(finding: dict, index: int) -> str:
    source = finding["source"]
    fallback = ". ".join(
        value for value in (source["author"], source["year"], source["title"]) if value
    ) or source["raw_reference"]
    if finding.get("finding_type") == "bibliographic_conflict":
        difference = finding.get("field_difference") or {}
        located = finding.get("located_record") or {}
        located_parts = [
            ", ".join(located.get("authors") or []),
            str(located.get("year") or ""),
            str(located.get("title") or ""),
            str(located.get("container_title") or ""),
            str(located.get("doi") or ""),
        ]
        located_text = ". ".join(value for value in located_parts if value)
        fields = ", ".join(finding.get("conflicting_fields") or []) or "bibliographic fields"
        return (
            f'<template id="reference-panel-{index}"><h2>Reference information</h2>'
            f'<p>{escape(finding["finding"])}</p>'
            f'<p><strong>Fields needing comparison:</strong> {escape(fields)}</p>'
            + (f'<p><strong>In your reference:</strong> {escape(str(difference.get("submitted_value") or "Not retained"))}<br>'
               f'<strong>In the located record:</strong> {escape(str(difference.get("located_value") or "Not retained"))}</p>'
               f'<p class="muted">Record provider: {escape(str(difference.get("provider") or "Not retained"))}. '
               'A difference does not by itself establish that your reference is wrong.</p>' if difference else '') +
            '<h3>Submitted reference</h3>'
            f'<p class="full-reference">{_render_formatted_reference(source, fallback)}</p>'
            '<h3>Located record</h3>'
            f'<p class="full-reference">{escape(located_text or "No displayable located-record fields were retained.")}</p>'
            '</template>'
        )
    related = "".join(
        f'<p class="full-reference">{_render_formatted_reference(peer, peer["raw_reference"])}</p>'
        for peer in finding.get("related_references") or []
    )
    return (
        f'<template id="reference-panel-{index}"><h2>Reference practice</h2>'
        f'<p>{escape(finding["finding"])}</p>'
        + (related or f'<p class="full-reference">{_render_formatted_reference(source, fallback)}</p>') + '</template>'
    )


def _render_member(member: dict, *, grouped: bool = False) -> str:
    source = member["source"]
    source_label = ". ".join(
        value for value in (source["author"], source["year"], source["title"]) if value
    ) or source["raw_reference"]
    evidence = ""
    if member.get("best_evidence"):
        item = member["best_evidence"]
        heading = (
            "Abstract evidence"
            if item.get("evidence_kind") == "abstract"
            else "Source evidence"
        )
        evidence_note = (
            f'<p class="muted">{escape(item["evidence_note"])}</p>'
            if item.get("evidence_note")
            else ""
        )
        evidence = (
            ("" if grouped else f'<h3>{heading}</h3>') + '<blockquote class="source-excerpt">'
            f'{escape(item.get("display_text") or item["text"])}{_inline_locator(item)}</blockquote>'
            f'{evidence_note}'
        )
    additional = ""
    additional_items = []
    primary = member.get("best_evidence") or {}
    if primary.get("context_text"):
        additional_items.append(
            '<h4>Full context for the selected excerpt</h4>'
            f'<blockquote class="source-excerpt">{escape(primary["context_text"])}{_inline_locator(primary)}</blockquote>'
        )
    additional_items.extend(
        f'<blockquote class="source-excerpt">{escape(item.get("context_text") or item.get("display_text") or item["text"])}{_inline_locator(item)}</blockquote>'
            + (
                f'<p class="muted">{escape(item["evidence_note"])}</p>'
                if item.get("evidence_note")
                else ""
            )
            for item in member.get("additional_evidence") or []
        )
    if additional_items:
        additional = (
            "<details><summary>Additional evidence and context</summary>"
            f'{"".join(additional_items)}</details>'
        )
    check_items = []
    if member.get("show_quotation_check") and member.get("coverage_level") != "unavailable":
        check_items.append(("Quotation", member.get("quotation_check") or {}, False))
    if member.get("show_locator_check"):
        check_items.append(("Locator", member.get("locator_check") or {}, False))
    checks = ""
    if check_items:
        rows = "".join(
            f'<li class="{"attention-text" if check.get("attention") else ""}">'
            f'{escape(_check_sentence(label, check))}</li>'
            for label, check, _identity_item in check_items
        )
        checks = f'<ul class="checks">{rows}</ul>'
    if any(f.get("finding_type") == "duplicate_citation_key" for f in member.get("reference_findings") or []):
        checks += '<p class="attention-text">Two or more references share this author and year. Add the citation style\'s distinguishing labels to the references and in-text citations.</p>'
    if (member.get("reference_identity") or {}).get("edition_year_unresolved"):
        checks += '<p class="muted">Located book records have different publication years. The edition used has not been established; compare its publication page and ISBN. This is not a confirmed reference error.</p>'
    action = member.get("source_action") or {}
    action_html = ""
    if action.get("enabled") and action.get("href") and action.get("href") not in str(source.get("raw_reference") or ""):
        action_html = (
            f'<p class="actions"><a class="compact-action" href="{escape(action["href"], quote=True)}" '
            f'target="_blank" rel="noopener">{escape(action.get("label", "Open source"))}</a></p>'
        )
    reference_html = (
        f'<p class="full-reference">{_render_formatted_reference(source, source_label)}</p>'
    )
    scope_notice = ""
    if member.get("abstract_scope_attention"):
        scope = (member.get("abstract_relevance") or {}).get("scope_assessment") or {}
        scope_notice = (
            '<p class="attention">The abstract appears unrelated to the topic attributed to this source. '
            'Manual review of the full source is necessary.</p>'
            f'<p>{escape(str(scope.get("rationale") or ""))}</p>'
            f'<p><strong>Abstract wording:</strong> {escape(str(scope.get("abstract_span") or ""))}<br>'
            f'<strong>Citation wording:</strong> {escape(str(scope.get("claim_span") or ""))}</p>'
        )
    availability = (
        '<p class="source-unavailable"><strong>Source not retrieved</strong></p>'
        if member.get("availability") == "Source not retrieved" and not grouped
        else f'<p>{escape(member["availability"])}</p>'
        if member.get("availability") and member.get("availability") != "Source not retrieved"
        else ""
    )
    return (
        f'<section class="member {member_tone(member)}" data-reference-id="{escape(str(member.get("reference_id") or ""), quote=True)}">'
        f'{reference_html}{availability}{scope_notice}{evidence}{additional}{checks}{action_html}{_render_upload(member.get("upload_action") or {})}</section>'
    )


def _inline_locator(item: dict) -> str:
    if item.get("evidence_kind") == "abstract":
        return ""
    value = str(item.get("locator") or "")
    if value.startswith("PDF page "):
        value = "PDF p. " + value[9:]
    elif value.startswith("Page "):
        value = "p. " + value[5:]
    else:
        return ""
    return ' <span class="locator">(' + escape(value) + ')</span>'


def _check_sentence(label: str, check: dict) -> str:
    value = str(check.get("label") or "Not assessed")
    if label.casefold() in value.casefold() or (label == "Quotation" and "quoted" in value.casefold()) or (label == "Locator" and "page/paragraph" in value.casefold()):
        return value
    return label + ": " + value


def _render_formatted_reference(source: dict, fallback: str) -> str:
    """Render only hash-bound submitted font ranges over escaped reference text."""
    text = str(source.get("raw_reference") or fallback)
    flags = [(False, False)] * len(text)
    for item in source.get("text_style_spans") or []:
        try:
            start, end = int(item["start"]), int(item["end"])
            italic, bold = bool(item.get("italic")), bool(item.get("bold"))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(text)) or not (italic or bold):
            continue
        for index in range(start, end):
            flags[index] = (italic, bold)
    chunks = []
    cursor = 0
    while cursor < len(text):
        italic, bold = flags[cursor]
        end = cursor + 1
        while end < len(text) and flags[end] == (italic, bold):
            end += 1
        value = _link_reference_text(text[cursor:end])
        if italic:
            value = f"<em>{value}</em>"
        if bold:
            value = f"<strong>{value}</strong>"
        chunks.append(value)
        cursor = end
    return "".join(chunks)


def _link_reference_text(text: str) -> str:
    chunks, cursor = [], 0
    for match in re.finditer(r"https?://[^\s<>\"\u201c\u201d]+", text):
        url = match.group().rstrip(".,;)")
        chunks.append(escape(text[cursor:match.start()]))
        chunks.append(f'<a class="reference-url" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(url)}</a>')
        cursor = match.start() + len(url)
    chunks.append(escape(text[cursor:]))
    return "".join(chunks)
