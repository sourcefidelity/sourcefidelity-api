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
from app.services.highlight_priority import (
    SUBMITTED_LINK_FINDINGS, LINK_MARKER_FINDINGS, REFERENCE_DIFFERENCE_FINDINGS, UNVERIFIED_FINDINGS, finding_category,
)


REPORT_VIEW_VERSION = "evidence-led-report-v31"

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
    from app.services.submitted_link_display import saved_link_view
    link_observations = saved_link_view(extraction.references, aggregate)
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
        aggregate.get("reference_consistency") or {}, references
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
                source_result = next((s for s in job.source_results or []
                    if s.get('reference_id') == reference_id), {})
                access = source_result.get('public_source_access') or {}
                provisional = source_result.get('provisional_source') or {}
                if (not access and provisional.get('policy_version') == 'submission-possible-match-v1'
                        and provisional.get('source_url')
                        and provisional.get('content_sha256')
                        and provisional.get('content_sha256') == (item.get('source_navigation') or {}).get('content_sha256')
                        and _safe_reference_href(str(provisional['source_url']))):
                    # Independently downloaded candidate URL, not a verified
                    # work identity. Brave-unretained URLs remain absent.
                    item['source_action'] = dict(enabled=True, status='public_candidate_available',
                        href=provisional['source_url'], label='Open retrieved candidate')
                if (access.get('version') == 'verified-public-source-access-v1'
                        and access.get('content_sha256')
                        and access.get('content_sha256') == (item.get('source_navigation') or {}).get('content_sha256')
                        and _safe_reference_href(str(access.get('href') or ''))):
                    item['source_action'] = dict(enabled=True, status='verified_public_source_available',
                        href=access['href'], label='Open available text' if item['coverage_level'] != 'full_text' else 'Open source')
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
            elif member.get("status") == "citation_not_assessed":
                item = _unresolved_marker_member(
                    reference, [member.get("reason_code", "citation_boundary_requires_review")],
                    reference_layout.get(reference_id),
                )
                item["availability"] = (
                    "Continuation not assessed: its source attribution is unresolved."
                )
            else:
                limited_reference_ids.add(reference_id)
                item = _unavailable_member(
                    member, reference, claim, reference_layout.get(reference_id)
                )
                # Operational only: lets the connected report offer a new search.
                item["reason_code"] = member.get("reason_code")
            item["reference_identity"] = _identity_view(
                discovery_by_reference.get(reference_id)
            )
            item = _withhold_unidentified_abstract(item)
            item = _withhold_mismatched_evidence(item, {"student_text": claim.text})
            # One flag, not a gradient: where the judgment stopped short
            # of a mark, the comparison and the reason go in the evidence
            # window instead of beside the citation.
            from app.services.report_layers import scope_disagreement
            disagreement = scope_disagreement(item, {"student_text": claim.text})
            if disagreement:
                item["scope_disagreement"] = disagreement
            citation_has_evidence = citation_has_evidence or bool(
                item.get("best_evidence")
            )
            item["reference_findings"] = reference_findings.get(reference_id, [])
            item['submitted_link_observations'] = [row for row in link_observations
                if row['reference_id'] == reference_id]
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
    from app.services.reference_credibility import assess_reference_credibility
    source_observations = {item.get('reference_id'): item for item in (job.source_results or [])}
    from app.services.reference_verification import assess_reference_verification
    credibility = {}
    verification = {}
    for reference_id, reference in references.items():
        observation = source_observations.get(reference_id) or {}
        assessment = assess_reference_credibility(reference, observation.get('reference_discovery'),
                                                   observation.get('reference_discovery_trace'))
        credibility[reference_id] = assessment
        checked = assess_reference_verification(reference, observation.get('reference_discovery'))
        verification[reference_id] = {k: v for k, v in checked.items() if k != 'findings'}
        # "Cannot be verified" replaces the former fabrication review flag;
        # that assessment stays in the audit record but is not reported.
        for finding in [f for f in assessment['findings']
                        if f['finding_type'] != 'potentially_fabricated_reference'] + checked['findings']:
            reference_practice.append({**finding, 'source': _reference_view(reference, reference_layout.get(reference_id))})
    credibility_ids = {f['reference_id'] for f in reference_practice}
    from app.services.reference_formatting import (
        bibliographic_field_conflicts, book_publication_year_discrepancy,
    )
    for reference_id, reference in references.items():
        discovery = discovery_by_reference.get(reference_id)
        for builder in (book_publication_year_discrepancy, bibliographic_field_conflicts):
            finding = builder(reference, discovery)
            if not finding:
                continue
            # The book year rule already names the year for a monograph;
            # do not tell a reader the same thing twice in two voices.
            if (finding['finding_type'] == 'bibliographic_field_conflict'
                    and any(existing.get('reference_id') == reference_id
                            and existing.get('finding_type') == 'publication_year_discrepancy'
                            for existing in reference_practice)):
                finding['field_differences'] = [d for d in finding['field_differences']
                                               if d['field_name'] != 'year']
                if not finding['field_differences']:
                    continue
            finding['source'] = _reference_view(reference, reference_layout.get(reference_id))
            reference_practice.append(finding)
    from app.services.quotation_locator_requirement import quotation_locator_findings
    for finding in quotation_locator_findings(extraction.quotation_locator_requirements, extraction.citation_claims, references):
        reference = references[finding['reference_id']]
        finding['source'] = _reference_view(reference, reference_layout.get(reference.reference_id))
        reference_practice.append(finding)
    from app.services.reference_formatting import reference_style_findings
    from app.services.body_title_formatting import body_title_findings
    for finding in body_title_findings(
        extraction.body_title_formatting, references,
        extraction.reference_layout.content_sha256 if extraction.reference_layout else None,
    ):
        finding['source'] = _reference_view(references[finding['reference_id']], None)
        reference_practice.append(finding)
    for finding in reference_style_findings(formatting, references):
        reference = references[finding['reference_id']]
        finding['source'] = _reference_view(reference, reference_layout.get(reference.reference_id))
        reference_practice.append(finding)
    for reference_id, discovery in discovery_by_reference.items():
        mismatch = (discovery or {}).get('doi_title_mismatch') or {}
        reference = references.get(reference_id)
        if not mismatch or reference is None:
            continue
        layout = reference_layout.get(reference_id)
        registered = str(mismatch.get('registered_title') or '').strip()
        # An identifier that resolves is not the same as an identifier that
        # resolves to the cited work. State the discrepancy, not a motive.
        reference_practice.append({
            'finding_type': 'doi_registers_a_different_title',
            'reference_id': reference_id,
            'finding': (
                'The DOI given in this reference is registered to a different title'
                + (f': "{registered}".' if registered else '.')
                + ' Check the identifier against the work cited; an identifier that'
                  ' belongs to another work does not confirm this reference.'),
            'rule_id': mismatch.get('policy_version') or 'doi-title-mismatch-v1',
            'rule_source': 'https://www.doi.org/the-identifier/resources/handbook/',
            'field_difference': {'field_name': 'doi',
                                 'submitted_value': str(mismatch.get('doi') or '')},
            'located_record': {'title': registered, 'provider': mismatch.get('provider')},
            'source': _reference_view(reference, layout),
            'rectangles': ([item.model_dump(mode='json') for item in layout.rectangles]
                           if layout is not None and layout.rectangles else []),
            'localization_status': ('exact_rectangle'
                                    if layout is not None and layout.rectangles
                                    else 'not_assessed'),
        })
    from app.services.reference_formatting import contribution_editor_findings
    container_records = {
        reference_id: (discovery or {}).get('container_identity') or {}
        for reference_id, discovery in discovery_by_reference.items()
    }
    for finding in contribution_editor_findings(references.values(), container_records):
        reference = references[finding['reference_id']]
        layout = reference_layout.get(reference.reference_id)
        finding['source'] = _reference_view(reference, layout)
        if layout is not None and layout.rectangles:
            # Mark it on the reference itself, like the other reference-form
            # findings; a panel nobody opens is not a report.
            finding['rectangles'] = [item.model_dump(mode='json') for item in layout.rectangles]
            finding['localization_status'] = 'exact_rectangle'
        reference_practice.append(finding)
    from app.services.required_reference_locator import required_doi_omissions
    from app.services.required_reference_author import required_author_omissions
    from app.services.assessment_configuration import assessment_link_omissions
    locator_inventory = _locator_inventory_view(extraction, paper_surface)
    assessment_omissions = assessment_link_omissions(extraction.assessment_configuration.model_dump(), locator_inventory)
    assessment_ids = {f['reference_id'] for f in assessment_omissions}
    style_omissions = required_doi_omissions(
        policy_version=extraction.required_doi_policy_version,
        citation_format=extraction.citation_format,
        inventory=locator_inventory,
        references=references, discoveries=discovery_by_reference,
    )
    author_omissions = required_author_omissions(
        policy_version=extraction.required_author_policy_version,
        citation_format=extraction.citation_format, inventory=locator_inventory,
        references=references, discoveries=discovery_by_reference)
    for omission in assessment_omissions + [f for f in style_omissions if f['reference_id'] not in assessment_ids] + author_omissions:
        reference = references[omission['reference_id']]
        omission['source'] = _reference_view(reference, reference_layout.get(reference.reference_id))
        reference_practice.append(omission)
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
        if reference_id in credibility_ids:
            # One evidence-bound credibility explanation, not duplicate field
            # warnings for the same identifier/compound discrepancy.
            continue
        reference = references.get(reference_id)
        layout = reference_layout.get(reference_id)
        discovery = discovery_by_reference.get(reference_id) or {}
        if reference is None:
            continue
        conflict = _reference_identity_conflict_view(discovery)
        for difference in conflict.get("field_differences") or []:
            field_name = difference["field_name"]
            if not _reference_field_difference_is_reportable(
                field_name, difference, reference
            ):
                continue
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
    # A same-titled work credited to someone else (owner decision 2026-09-30):
    # the existing Source Record Conflict on the author, in its existing words.
    from app.services.reference_verification import same_title_author_difference
    author_conflicts = {f.get('reference_id') for f in reference_practice
                        if f.get('finding_type') == 'bibliographic_conflict'
                        and 'author' in (f.get('conflicting_fields') or [])}
    for reference_id, checked in verification.items():
        reference = references.get(reference_id)
        if (reference is None or checked.get('status') != 'possible_match'
                or reference_id in author_conflicts):
            continue
        difference = same_title_author_difference(reference, discovery_by_reference.get(reference_id))
        if difference is None:
            continue
        reference_practice.append({
            "finding_type": "bibliographic_conflict",
            "reference_id": reference_id,
            "source": _reference_view(reference, reference_layout.get(reference_id)),
            "finding": ("The author in this reference differs from the located record. "
                        "Compare the work and edition actually used before changing the reference."),
            "located_record": {"title": difference["located_title"],
                               "authors": [difference["located_value"]]},
            "conflicting_fields": ["author"],
            "field_difference": {k: difference[k] for k in
                                 ("field_name", "submitted_value", "located_value", "provider", "candidate_id")},
            "same_title_record": True,
            "rectangles": [],
            "localization_status": "not_assessed",
        })
        publisher = difference.get("publisher_difference")
        if publisher:
            reference_practice.append({
                "finding_type": "bibliographic_conflict",
                "reference_id": reference_id,
                "source": _reference_view(reference, reference_layout.get(reference_id)),
                "finding": ("The publisher in this reference differs from the located record. "
                            "Compare the work and edition actually used before changing the reference."),
                "located_record": {"title": difference["located_title"], "authors": [difference["located_value"]]},
                "conflicting_fields": ["publisher"],
                "field_difference": {**publisher, "provider": difference["provider"],
                                     "candidate_id": difference["candidate_id"]},
                "same_title_record": True,
                "rectangles": [],
                "localization_status": "not_assessed",
            })
    for finding in consistency.get("findings", []):
        if finding.get("finding_type") == "duplicate_reference_entry":
            peers = [references[rid] for rid in finding.get("reference_ids", []) if rid in references]
            for reference in peers:
                reference_practice.append({
                    "finding_type": "duplicate_reference_entry", "reference_id": reference.reference_id,
                    "retained_finding_id": finding["finding_id"],
                    "source": _reference_view(reference, reference_layout.get(reference.reference_id)),
                    "finding": finding["explanation"],
                    "related_references": [{"reference_id": peer.reference_id, **_reference_view(peer, reference_layout.get(peer.reference_id))} for peer in peers],
                    "field_difference": {"field_name": "entry", "submitted_value": reference.raw_ref},
                    "rectangles": [], "localization_status": "not_assessed",
                })
            continue
        if finding.get("finding_type") != "duplicate_citation_key":
            continue
        peers = [references[rid] for rid in finding.get("reference_ids", []) if rid in references]
        from app.services.reference_consistency import apa_in_text_form
        if len({apa_in_text_form(peer) for peer in peers}) == len(peers):
            continue  # their in-text citations already differ (2026-09-30)
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
    from app.services.reference_formatting import reference_order_projection
    from app.services.report_references import reference_ordinal
    reference_practice, pervasive_reference_order = reference_order_projection(
        reference_practice, reference_ordinal)
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
    # Patchwriting findings shown in owner wording (2026-09-29); geometry is
    # bound with the other paper spans in attach_quotation_difference_geometry.
    from app.services.patchwriting_report import build_passages, stored_block
    patchwriting_passages = build_passages(
        stored_block(job, aggregate),
        lambda reference_id: (_reference_view(references[reference_id], reference_layout.get(reference_id))
                              if reference_id in references else None))
    summary = _build_report_summary(
        citations=citations,
        overview=overview,
        pervasive_hanging_indent=pervasive_hanging_indent,
        reference_practice=reference_practice,
        require_paper_flags=True,
        pervasive_reference_order=pervasive_reference_order,
        patchwriting_passages=patchwriting_passages,
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
        # The upload's Title when one was given, else the file name (owner request 2026-09-29).
        "title": (getattr(job, "title", None) or "").strip() or job.filename,
        "citation_format": extraction.citation_format.upper(),
        "word_counts": {
            "total": extraction.total_word_count,
            "body": extraction.body_word_count,
            "references": extraction.reference_word_count,
        },
        "paper_surface": paper_surface,
        "overview": overview,
        "summary": summary,
        "gauges": gauges,
        "citations": citations,
        "patchwriting_passages": patchwriting_passages,
        "reference_practice": reference_practice,
        "reference_credibility": credibility,
        "reference_verification": verification,
        "submitted_link_observations": link_observations,
        "uncited_link_checks": [
            {'reference': _reference_view(reference, reference_layout.get(reference_id)),
             'observations': [row for row in link_observations if row['reference_id'] == reference_id]}
            for reference_id, reference in references.items()
            if reference_id not in {rid for claim in claims.values() for rid in claim.reference_ids}
            and any(row['reference_id'] == reference_id for row in link_observations)
        ],
        "assessment_configuration": extraction.assessment_configuration.model_dump(mode='json'),
        "submitted_locator_inventory": _locator_inventory_view(extraction, paper_surface),
        "reference_practice_summary": reference_practice_summary,
        "pervasive_reference_order": pervasive_reference_order,
        # Every reference in bibliography order, so an uncited reference with
        # no findings still receives a Reference N window.
        "bibliography": [
            {"reference_id": reference.reference_id,
             "source": _reference_view(reference, reference_layout.get(reference.reference_id))}
            for reference in extraction.references
        ],
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


def _locator_inventory_view(extraction, surface=None) -> dict | None:
    inventory = extraction.submitted_locator_inventory
    layout = extraction.reference_layout
    if inventory is None or layout is None or inventory.paper_sha256 != layout.content_sha256:
        return None
    references = {r.reference_id: r for r in extraction.references}
    if set(references) != {e.reference_id for e in inventory.entries}:
        return None
    if any(hashlib.sha256(references[e.reference_id].raw_ref.encode()).hexdigest() != e.reference_text_sha256
           for e in inventory.entries):
        return None
    value = inventory.model_dump(mode="json")
    navigation = (surface or {}).get('submitted_reference_navigation') or {}
    projected = {}
    if (layout.location_kind == 'paragraph'
            and navigation.get('input_sha256') == inventory.paper_sha256):
        from app.services.reference_layout import ReferenceLayoutArtifact
        try:
            rendered = ReferenceLayoutArtifact.model_validate(navigation.get('layout'))
            if rendered.content_sha256 == (surface or {}).get('presentation_sha256'):
                projected = {e.reference_id: e for e in rendered.entries
                             if e.mapping_status == 'matched' and e.match_confidence >= 0.95}
        except ValueError:
            pass
    for entry in value['entries']:
        entry['submitted_reference'] = references[entry['reference_id']].raw_ref
        mapped = projected.get(entry['reference_id'])
        if mapped and mapped.reference_text_sha256 == entry['reference_text_sha256']:
            entry['rectangles'] = [b.model_dump(mode='json') for b in mapped.rectangles]
    return value


def _render_locator_count(value: dict | None) -> str:
    if not value:
        return ''  # Historical reports without a saved inventory stay unchanged.
    counts = value['counts']
    coverage = (' Counts cover extracted references only.'
                if value['reference_list_coverage'] != 'complete' else '')
    uncertain = f' {counts["unknown"]} could not be assessed.' if counts['unknown'] else ''
    statement = (f'Link inventory only: {counts["not_observed"]} of {value["assessed_entries"]} assessed references contain neither a DOI nor a URL.'
                 if value['assessed_entries'] else 'DOI/URL absence could not be assessed.')
    return ('<p class="locator-count">' + escape(statement + uncertain + coverage +
            ' This does not establish that a link is required. Required-link omissions are marked separately in the paper.') + '</p>')


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


SUMMARY_INSTANCE_LIMIT = 5


def summary_text(item) -> str:
    """The complete legacy sentence for a summary item or legacy string."""
    return item if isinstance(item, str) else str((item or {}).get("text") or "")


def _summary_item(kind: str, text: str, *, lead: str | None = None, rest: str = "",
                  instances: list[dict] | None = None, count: int | None = None,
                  lead_counts: bool = True, pervasive: bool = False) -> dict:
    """One summary finding: the exact legacy sentence plus linkable instances.

    ``text`` stays the complete sentence used by the PDF, style guidance and
    older consumers. The HTML report renders ``lead``, the instance links, then
    ``rest``; ``lead + rest`` omits only the instance list.
    """
    return {"kind": kind, "text": text, "lead": lead, "rest": rest,
            "instances": list(instances or []), "count": count,
            "lead_counts": lead_counts, "pervasive": pervasive}


def _finding_target(finding: dict, index: int, citations: list[dict], numbers: dict[str, int]) -> str:
    """The window a finding opens, shared by the summary and the paper."""
    from app.services.report_references import reference_list_finding, reference_template_id
    rid = finding.get("reference_id")
    if reference_list_finding(finding) and numbers.get(rid):
        return reference_template_id(numbers[rid])
    overlap = [i for i, c in enumerate(citations, 1) if _formatting_overlaps(c, finding)]
    return f"citation-panel-{overlap[0]}" if len(overlap) == 1 else f"reference-panel-{index}"


def _summary_numbers(citations: list[dict], reference_practice: list[dict] | None,
                     numbers: dict[str, int] | None) -> dict[str, int]:
    if numbers is not None:
        return numbers
    from app.services.report_references import reference_numbers
    return reference_numbers({"citations": citations, "reference_practice": reference_practice or []})


def citation_after_punctuation(citation: dict) -> bool:
    """A parenthetical citation written after its sentence's final punctuation.

    "…the status quo. (Hess,1974)" or, before the boundary repair, a claim that
    opens with the marker and runs into the next sentence (Paper 2, 2026-09-30).
    """
    marker = str(citation.get("citation_marker") or "").strip()
    text = str(citation.get("student_text") or "").strip()
    if not (marker.startswith("(") and marker.endswith(")") and marker in text):
        return False
    # APA places a block quotation's citation after its final punctuation; a
    # quotation claim without quotation marks is a block quotation.
    if citation.get("claim_type") == "quotation" and not re.search(r'[“”"]', text):
        return False
    if text.endswith(marker):
        return re.search(r'[.!?]["”’\']?\s*$', text[:-len(marker)]) is not None
    return text.startswith(marker) and re.match(r'\s*[A-Z“"‘]', text[len(marker):]) is not None


def _passage_instances(passages: list[dict], citations: list[dict], numbers: dict[str, int]) -> list[dict]:
    """One link per window showing a patchwriting passage: citations, then references."""
    from app.services.patchwriting_report import passage_windows
    from app.services.report_references import reference_template_id
    rows, seen = [], set()
    for passage in passages:
        for window in passage_windows(passage, citations):
            if window["citation"]:
                row = {"type": "citation", "number": window["citation"],
                       "target": f"citation-panel-{window['citation']}"}
            elif numbers.get(window["reference_id"]):
                number = numbers[window["reference_id"]]
                row = {"type": "reference", "number": number, "target": reference_template_id(number)}
            else:
                continue
            if row["target"] not in seen:
                seen.add(row["target"])
                rows.append({**row, "mark": f"patchwriting-mark-{passage['number']}"})
    return sorted(rows, key=lambda row: (row["type"] != "citation", row["number"]))


def _instance_list(instances: list[dict]) -> str:
    """"citation 3", "citations 3, 7", or "citation 3; reference 9" for the PDF sentence."""
    kinds = {row["type"] for row in instances}
    if len(kinds) == 1:
        kind = kinds.pop()
        return (kind if len(instances) == 1 else kind + "s") + " " + ", ".join(str(r["number"]) for r in instances)
    return "; ".join(f"{row['type']} {row['number']}" for row in instances)


def _build_report_summary(
    *,
    citations: list[dict],
    overview: dict,
    pervasive_hanging_indent: bool,
    reference_practice: list[dict] | None = None,
    require_paper_flags: bool = False,
    pervasive_reference_order: bool = False,
    reference_numbers: dict[str, int] | None = None,
    patchwriting_passages: list[dict] | None = None,
    uncited_reference_ids: list[str] | None = None,
) -> dict:
    """Every issue per category, in the owner's order and wording."""
    from app.services.report_references import reference_template_id
    reference_practice = normalize_reference_findings(reference_practice or [])
    numbers = _summary_numbers(citations, reference_practice, reference_numbers)
    all_citations = citations

    def cite(indexes) -> list[dict]:
        return [{"type": "citation", "number": i, "target": f"citation-panel-{i}"}
                for i in sorted(dict.fromkeys(indexes))]

    def refs(ids) -> list[dict]:
        found = sorted({numbers[rid] for rid in ids if rid in numbers})
        return [{"type": "reference", "number": n, "target": reference_template_id(n)} for n in found]

    def groups(id_groups) -> list[dict]:
        rows = []
        for group in id_groups:
            found = sorted({numbers[rid] for rid in group if rid in numbers})
            if found:
                rows.append({"type": "group", "numbers": found, "target": reference_template_id(found[0])})
        return sorted(rows, key=lambda row: row["numbers"])
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
                    or bool(set(f.get("reference_ids") or []) & placed["duplicate_citation_key"])]
    indirect_ids = [
        index for index, citation in enumerate(citations, 1)
        if any(member.get("secondary_citation") for member in citation.get("members") or [])
    ]
    indirect_citations = len(indirect_ids)
    quotation_ids = [
        index for index, citation in enumerate(citations, 1)
        if any(
            member.get("show_quotation_check")
            and (member.get("quotation_check") or {}).get("attention")
            for member in citation.get("members") or []
        )
    ]
    quotation_attention_citations = len(quotation_ids)
    conflicting_references = {
        member.get("reference_id")
        for citation in citations
        for member in citation.get("members") or []
        if (member.get("reference_identity") or {}).get("attention")
    } | {
        # A same-titled work credited to someone else (2026-09-30), counted
        # with the other source-record conflicts once it is placed.
        f.get("reference_id") for f in reference_practice or []
        if f.get("finding_type") == "bibliographic_conflict" and f.get("same_title_record") and f.get("rectangles")
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
    for finding in reference_practice or []:
        if finding.get('finding_type') == 'duplicate_citation_key' and finding.get('rectangles'):
            ids = {f.get('reference_id') for f in reference_practice or []
                   if f.get('finding_type') == 'duplicate_citation_key'
                   and f.get('retained_finding_id') == finding.get('retained_finding_id')}
            duplicate_key_references.update(ids - {None})
            duplicate_groups.add(tuple(sorted(ids - {None})))
    missing = [(index, citation.get("missing_reference_members"))
               for index, citation in enumerate(citations, 1)
               if citation.get("missing_reference_members")]

    summary = {
        "evidence": [],
        "reference_formatting": [],
        "academic_practice": [],
    }

    def add(category, kind, text, *, lead=None, rest="", instances=None, count=None,
            lead_counts=True, pervasive=False):
        summary[category].append(_summary_item(
            kind, text, lead=lead, rest=rest, instances=instances,
            count=count, lead_counts=lead_counts, pervasive=pervasive))

    from app.services.report_layers import topical_mismatch
    mismatch_ids = [index for index, citation in enumerate(citations, 1)
                    if any(topical_mismatch(member, citation)
                           for member in citation.get('members', []))]
    if mismatch_ids:
        lead = (f"{len(mismatch_ids)} source is possibly not related to the citation" if len(mismatch_ids) == 1
                else f"{len(mismatch_ids)} sources are possibly not related to citations")
        add('evidence', 'topical_mismatch', f"{lead} ({_citation_list(mismatch_ids).lower()}).",
            lead=lead, rest='.', instances=cite(mismatch_ids), count=len(mismatch_ids))
    for kind, category, one, many in (
        ('unverified_reference', 'evidence', 'reference(s) cannot be verified', 'reference(s) cannot be verified'),
        ('reference_identifier_placeholder', 'reference_formatting',
         'reference contains an unfinished identifier placeholder', 'references contain unfinished identifier placeholders'),
    ):
        ids = {f.get('reference_id') for f in reference_practice or []
               if f.get('finding_type') == kind and f.get('rectangles')}
        if ids:
            lead = f'{len(ids)} {one if len(ids) == 1 else many}'
            add(category, kind, lead + '.', lead=lead, rest='.', instances=refs(ids), count=len(ids))
            conflicting_references -= ids
    confirmed_duplicates = {f.get('retained_finding_id') for f in reference_practice or []
                            if f.get('finding_type') == 'duplicate_reference_entry'
                            and f.get('rectangles') and f.get('retained_finding_id')}
    if confirmed_duplicates:
        lead = (f"{len(confirmed_duplicates)} duplicate reference group repeats the same source in the reference list"
                if len(confirmed_duplicates) == 1 else
                f"{len(confirmed_duplicates)} duplicate reference groups repeat the same sources in the reference list")
        entry_groups = [[f.get('reference_id') for f in reference_practice or []
                         if f.get('finding_type') == 'duplicate_reference_entry'
                         and f.get('retained_finding_id') == retained]
                        for retained in confirmed_duplicates]
        add('academic_practice', 'duplicate_reference_entry', lead + '.', lead=lead, rest='.',
            instances=groups(entry_groups), count=len(confirmed_duplicates))
    author_ids = {f.get('reference_id') for f in reference_practice or []
                  if f.get('finding_type') == 'reference_author_conflict' and f.get('rectangles')}
    if author_ids:
        lead = (f'{len(author_ids)} reference names an author different from the source byline' if len(author_ids) == 1
                else f'{len(author_ids)} references name authors different from the source bylines')
        add('academic_practice', 'reference_author_conflict', lead + '.', lead=lead, rest='.',
            instances=refs(author_ids), count=len(author_ids))
    link_ids={f.get('reference_id') for f in reference_practice or [] if f.get('finding_type') in SUBMITTED_LINK_FINDINGS and f.get('rectangles')}
    if link_ids:
        conflicting_references -= link_ids
        homepage_ids={f.get('reference_id') for f in reference_practice or []
                      if f.get('finding_type')=='submitted_link_issue' and f.get('rectangles')
                      and f.get('link_outcome') in {'website_level_address','site_homepage'}}
        other_ids = link_ids - homepage_ids
        if other_ids:
            n = len(other_ids)
            lead = (f"{n} reference has a submitted link that returned a missing page or conflicting destination details"
                    if n == 1 else
                    f"{n} references have submitted links that returned missing pages or conflicting destination details")
            add('academic_practice', 'submitted_link_issue', lead + '.', lead=lead, rest='.',
                instances=refs(other_ids), count=n)
        if homepage_ids:
            n = len(homepage_ids)
            lead = (f"{n} reference links to a website address rather than a page for the cited work" if n == 1
                    else f"{n} references link to website addresses rather than pages for the cited works")
            add('academic_practice', 'submitted_link_homepage', lead + '.', lead=lead, rest='.',
                instances=refs(homepage_ids), count=n)
    # Every issue is listed; the former rule that hid a single issue when
    # another pattern repeated was removed (owner decision 2026-09-30).
    if quotation_attention_citations:
        # Quoted wording that does not match the source is an academic-practice
        # question about how the source was used, not a limit on the evidence
        # retrieved for it (ARCHITECTURE §8).
        lead = (f"{quotation_attention_citations} {'quotation contains' if quotation_attention_citations == 1 else 'quotations contain'} "
                "wording differences from the source")
        add('academic_practice', 'quotation_difference', lead + '. ', lead=lead, rest='.',
            instances=cite(quotation_ids), count=quotation_attention_citations)
    from app.services.patchwriting_report import summary_counts
    passages_by_kind = summary_counts(patchwriting_passages)
    # Owner wording 2026-09-29, singular then plural; a passage whose wording
    # the source itself quotes counts under its own kind. Each passage links to
    # the window that shows it: its citation, else its source's Reference window.
    for kind, summary_kind, one, many in (
        ('close_paraphrase', 'passage_close_wording',
         "1 passage closely follows a source's wording", "{n} passages closely follow sources' wording"),
        ('unquoted_verbatim', 'passage_exact_wording',
         "1 passage uses a source's exact wording without quotation marks",
         "{n} passages use sources' exact wording without quotation marks"),
    ):
        found = passages_by_kind[kind]
        if found:
            n = len(found)
            lead = one if n == 1 else many.format(n=n)
            instances = _passage_instances(found, all_citations, numbers)
            add('academic_practice', summary_kind,
                f"{lead} ({_instance_list(instances)})." if instances else f"{lead}.",
                lead=lead, rest='.', count=n, instances=instances)
    if pervasive_hanging_indent:
        add('reference_formatting', 'hanging_indent_pervasive',
            "The reference list repeatedly lacks the expected hanging indent. ", pervasive=True)
    else:
        indent_ids = {f.get('reference_id') for f in reference_practice or []
                      if f.get('finding_type') == 'formatting' and f.get('rectangles')} - {None}
        if indent_ids:
            lead = f'{len(indent_ids)} reference(s) lack the expected hanging indent'
            add('reference_formatting', 'hanging_indent', lead + '.', lead=lead, rest='.',
                instances=refs(indent_ids), count=len(indent_ids))
    if pervasive_reference_order:
        add('reference_formatting', 'reference_order_pervasive',
            'The reference list is not in alphabetical order by first-author surname.', pervasive=True)
    author_count_ids = {f['reference_id'] for f in reference_practice or []
                        if f.get('finding_type') == 'required_author_missing' and f.get('rectangles')}
    if author_count_ids:
        n = len(author_count_ids)
        lead = (f'{n} reference omits a verified article author' if n == 1
                else f'{n} references omit verified article authors')
        add('reference_formatting', 'required_author_missing', lead + '.', lead=lead, rest='.',
            instances=refs(author_count_ids), count=n)
    doi_ids = {f['reference_id'] for f in reference_practice or []
               if f.get('finding_type') == 'required_doi_missing' and f.get('rectangles')}
    if doi_ids:
        n = len(doi_ids)
        lead = f'{n} reference omits a DOI for the cited work' if n == 1 else f'{n} references omit DOIs for the cited works'
        add('reference_formatting', 'required_doi_missing', lead + '.', lead=lead, rest='.',
            instances=refs(doi_ids), count=n)
    assessment_ids = {f['reference_id'] for f in reference_practice or []
                      if f.get('finding_type') == 'assessment_link_missing' and f.get('rectangles')}
    if assessment_ids:
        lead = (f'{len(assessment_ids)} reference lacks the DOI, URL, or library link required by this assessment'
                if len(assessment_ids) == 1 else
                f'{len(assessment_ids)} references lack the DOI, URL, or library link required by this assessment')
        add('reference_formatting', 'assessment_link_missing', lead + '.', lead=lead, rest='.',
            instances=refs(assessment_ids), count=len(assessment_ids))
    # Owner wording 2026-09-29, singular then plural.
    for kind, message in (
        ('publication_year_discrepancy', ('{count} reference has a publication year that differs from catalog records.',
                                          '{count} references have publication years that differ from catalog records.')),
        ('bibliographic_field_conflict', ('{count} reference gives details that differ from the record for the source.',
                                          '{count} references give details that differ from the records for the sources.')),
        ('reference_title_style', ('{count} reference title uses incorrect title formatting.',
                                   '{count} reference titles use incorrect title formatting.')),
        ('body_title_style', ('{count} film or book title lacks required italics in the paper.',
                              '{count} film or book titles lack required italics in the paper.')),
        ('reference_order', ('{count} reference is not in alphabetical order.',
                             '{count} references are not in alphabetical order.')),
        ('contribution_author_is_volume_editor',
         ('{count} reference cites part of a book like it was an edited collection.',
          '{count} references cite parts of books like they were edited collections.')),
        ('required_quotation_locator_missing', ('', '')),
    ):
        placed = [(j, f) for j, f in enumerate(reference_practice or [], 1)
                  if f.get('finding_type') == kind and f.get('rectangles')]
        if kind == 'body_title_style':
            # One work left unitalicized throughout is one habit, not one
            # finding per occurrence: count distinct works and link each to
            # its first flagged occurrence in paper order.
            works: dict = {}
            for j, f in placed:
                key = f.get('reference_id') or re.sub(r'\W+', ' ', str((f.get('field_difference') or {}).get('submitted_value') or '')).strip().casefold()
                first = min(f['rectangles'], key=lambda r: (r.get('page_index', 0), r.get('y0', 0), r.get('x0', 0)))
                position = (first.get('page_index', 0), first.get('y0', 0), first.get('x0', 0))
                if key not in works or position < works[key][0]:
                    works[key] = (position, j, f)
            ordered = sorted(works.values(), key=lambda row: row[0])
            count = len(ordered)
            instances = [{"type": "named",
                          "label": str((f.get('field_difference') or {}).get('submitted_value')
                                       or (f.get('source') or {}).get('title') or 'Title'),
                          "target": _finding_target(f, j, all_citations, numbers)}
                         for _, j, f in ordered]
        elif kind == 'required_quotation_locator_missing':
            claims = {c.get('claim_id'): i for i, c in enumerate(all_citations, 1) if c.get('claim_id')}
            count = len({f.get('claim_id') for _, f in placed})
            instances = []
            seen = set()
            for j, f in placed:
                if f.get('claim_id') in seen:
                    continue
                seen.add(f.get('claim_id'))
                if f.get('claim_id') in claims:
                    instances.extend(cite([claims[f['claim_id']]]))
                else:
                    instances.append({"type": "named", "label": str(f.get('citation_marker') or 'Passage'),
                                      "target": _finding_target(f, j, all_citations, numbers)})
            instances.sort(key=lambda row: (row.get('type') != 'citation', row.get('number', 0)))
        else:
            ids = {f.get('reference_id') for _, f in placed}
            count = len(ids)
            instances = refs(ids)
        if count:
            text = (f"{count} {'quotation lacks' if count == 1 else 'quotations lack'} page or paragraph locators "
                    "in the parenthetical citation."
                    if kind == 'required_quotation_locator_missing' else message[count != 1].format(count=count))
            add('evidence' if kind in REFERENCE_DIFFERENCE_FINDINGS else 'reference_formatting',
                kind, text, lead=text.removesuffix('.'), rest='.', instances=instances, count=count)
    if conflicting_references:
        n = len(conflicting_references)
        lead = f"{n} {'reference contains' if n == 1 else 'references contain'} bibliographic fields that conflict with source records"
        add('evidence', 'identity_conflict', lead + '. ', lead=lead, rest='.',
            instances=refs(conflicting_references), count=n)
    if missing:
        first_index: dict[str, int] = {}
        for index, names in missing:
            for name in names:
                first_index.setdefault(re.sub(r'^as cited in\s+', '', name, flags=re.I).strip(), index)
        names = sorted(first_index)
        lead = f"{len(names)} in-text {'source has' if len(names) == 1 else 'sources have'} no matching reference-list entry"
        instances = [{"type": "named", "label": name, "target": f"citation-panel-{first_index[name]}"}
                     for name in sorted(first_index, key=lambda n: (first_index[n], n))]
        add('academic_practice', 'missing_reference_entry', f"{lead} ({'; '.join(names)}).",
            lead=lead, rest='.', instances=instances, count=len(names))
    if indirect_citations:
        lead = (f"{indirect_citations} {'citation currently relies' if indirect_citations == 1 else 'citations currently rely'} "
                "on passages where the cited source represents another work")
        add('academic_practice', 'indirect_source', lead + '. ', lead=lead, rest='.',
            instances=cite(indirect_ids), count=indirect_citations)
    uncited = refs(uncited_reference_ids or [])
    if uncited:
        # Owner wording 2026-09-30, singular then plural.
        n = len(uncited)
        lead = "1 reference is not cited in the paper" if n == 1 else f"{n} references are not cited in the paper"
        add('academic_practice', 'uncited_reference', f"{lead} ({_instance_list(uncited)}).",
            lead=lead, rest='.', instances=uncited, count=n)
    misplaced = [index for index, citation in enumerate(all_citations, 1) if citation_after_punctuation(citation)]
    if misplaced:
        # Owner wording 2026-09-30, singular then plural.
        n = len(misplaced)
        lead = ("1 parenthetical citation is placed after the sentence's final punctuation" if n == 1 else
                f"{n} parenthetical citations are placed after the sentences' final punctuation")
        instances = cite(misplaced)
        add('reference_formatting', 'citation_after_punctuation', f"{lead} ({_instance_list(instances)}).",
            lead=lead, rest='.', instances=instances, count=n)
    if duplicate_key_references:
        n = len(duplicate_key_references)
        lead = f"{n} references share the same author and year, so their in-text citations do not distinguish the works"
        add('academic_practice', 'duplicate_citation_key', lead + '. ', lead=lead, rest='.',
            instances=groups(duplicate_groups), count=n)

    return summary


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


def _and_join(values: list) -> str:
    values = [str(value) for value in values]
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def _render_summary_instances(item: dict, *, placed: frozenset = frozenset(),
                              located: frozenset = frozenset()) -> str:
    """Link the first five instances; an ellipsis signals that more exist."""
    instances = list(item.get("instances") or [])
    shown = instances[:SUMMARY_INSTANCE_LIMIT]
    truncated = len(instances) > SUMMARY_INSTANCE_LIMIT

    marks = {row.get("target"): row["mark"] for row in shown if row.get("mark")}

    def link(label: str, target: str, aria: str, href: str) -> str:
        # A patchwriting instance also names its paper highlight, which opens the window.
        mark = f' data-go-to-mark="{escape(marks[target], quote=True)}"' if target in marks else ''
        return (f'<a class="summary-instance" href="{escape(href, quote=True)}" '
                f'data-go-to="{escape(target, quote=True)}"{mark} aria-label="{escape(aria, quote=True)}">'
                f'{escape(label)}</a>')

    def citation_href(number):
        return f"#citation-location-{number}" if number in placed else "#evidence-panel"

    def reference_href(number):
        return f"#reference-location-{number}" if number in located else "#evidence-panel"

    kinds = {row.get("type") for row in shown}
    if kinds == {"passage"}:
        # Summaries stored before passages moved into citation and reference
        # windows (2026-09-29) name passages that have no window of their own.
        body = ("passage " if len(instances) == 1 else "passages ") + ", ".join(
            escape(str(row.get("number"))) for row in shown) + (", …" if truncated else "")
    elif kinds == {"citation"}:
        body = ("citation " if len(instances) == 1 else "citations ") + ", ".join(
            link(str(row["number"]), row["target"], f"Citation {row['number']}", citation_href(row["number"]))
            for row in shown) + (", …" if truncated else "")
    elif kinds == {"reference"}:
        body = ("reference " if len(instances) == 1 else "references ") + ", ".join(
            link(str(row["number"]), row["target"], f"Reference {row['number']}", reference_href(row["number"]))
            for row in shown) + (", …" if truncated else "")
    elif kinds == {"group"}:
        body = "references " + "; ".join(
            link(_and_join(row["numbers"]), row["target"], "References " + _and_join(row["numbers"]),
                 reference_href(row["numbers"][0]))
            for row in shown) + ("; …" if truncated else "")
    else:
        parts = []
        for row in shown:
            if row.get("type") == "citation":
                parts.append(link(f"citation {row['number']}", row["target"], f"Citation {row['number']}",
                                  citation_href(row["number"])))
            elif row.get("type") == "reference":
                parts.append(link(f"reference {row['number']}", row["target"], f"Reference {row['number']}",
                                  reference_href(row["number"])))
            else:
                parts.append(link(str(row.get("label") or "Instance"), row["target"],
                                  str(row.get("label") or "Instance"), "#evidence-panel"))
        body = "; ".join(parts) + ("; …" if truncated else "")
    if truncated and not item.get("lead_counts", True) and item.get("count"):
        body = f"{int(item['count'])} in total: " + body
    return body


def _render_summary_item(item, *, placed: frozenset = frozenset(), located: frozenset = frozenset()) -> str:
    if isinstance(item, str):
        return escape(item)
    if item.get("lead") is None or not item.get("instances"):
        return escape(summary_text(item))
    return (f'{escape(str(item["lead"]))} '
            f'({_render_summary_instances(item, placed=placed, located=located)})'
            f'{escape(str(item.get("rest") or ""))}')


def report_summary(view: dict) -> dict:
    """The report's one summary; older views stored one per audience."""
    if isinstance(view.get('summary'), dict):
        return view['summary']
    legacy = view.get('role_summaries') or {}
    return legacy.get('instructor') or {}


def _retrieval_sentence(overview: dict) -> str:
    """Owner wording 2026-09-27: each kind of retrieval out of the total references."""
    total = int(overview.get("reference_count") or 0)
    if not total:
        return ""
    full = int(overview.get("verified_full_text_sources") or 0)
    limited = int(overview.get("abstract_or_limited_sources") or 0)
    none = int(overview.get("unavailable_sources") or 0)
    return (f"{full}/{total} full-texts retrieved, {limited}/{total} abstract/limited-texts retrieved, "
            f"and {none}/{total} unretrieved texts.")


# Owner wording 2026-09-29 (second revision; "not supported" 2026-09-30); a part is shown only when its count is not zero.
# "statements" appears once, in the first part shown (owner request 2026-09-29).
_JUDGMENT_PARTS = (
    ("supported", "{n}/{x}{s} are supported by the sources"),
    ("qualified", "{n}/{x}{s} have qualified or mixed support in the sources"),
    ("contradicts", "{n}/{x}{s} contradict the sources"),
    ("insufficient", "{n}/{x}{s} are not supported by the sources"),
    ("undecided", "{n}/{x}{s} cannot be decided upon by the LLM"),
)


def judgment_summary_sentence(states: dict, without_full_text: int, citation_total: int) -> str:
    """X for the statement parts is the number of statements the judge
    answered (each proposition and source, undecided included); the last
    part counts citations."""
    judged = sum(int(states.get(key) or 0) for key, _ in _JUDGMENT_PARTS)
    shown = [(key, text) for key, text in _JUDGMENT_PARTS if states.get(key)]
    parts = [text.format(n=int(states[key]), x=judged, s=" statements" if i == 0 else "")
             for i, (key, text) in enumerate(shown)]
    if without_full_text:
        parts.append(f"{without_full_text}/{citation_total} citations lack full texts and are not judged")
    if not parts:
        return ""
    return (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]) + "."


def _render_report_summary(summary: dict, *, placed_citations: frozenset = frozenset(),
                           located_references: frozenset = frozenset(), retrieval: str = "",
                           judgments: dict | None = None) -> str:
    """One summary for everyone (owner decision 2026-09-25): neutral wording,
    no revision coaching and no cap; style-guide links live in the windows."""
    columns = (   # owner order and names, 2026-09-28
        ("evidence", "Sources"),
        ("academic_practice", "Academic Practice"),
        ("reference_formatting", "Citation and Reference Formatting"),
    )
    sections = []
    for key, label in columns:
        items = list(summary.get(key) or [])
        # The retrieval counts are the first Evidence item (owner request 2026-09-27).
        lead = (f'<li class="retrieval-summary">{escape(retrieval)}</li>'
                if key == "evidence" and retrieval else "")
        if key == "evidence" and judgments is not None:
            text = judgment_summary_sentence(judgments["states"], judgments["without_full_text"],
                                             judgments["citations"])
            lead += (f'<li class="judgment-summary" data-judgment-summary data-without-full-text='
                     f'"{judgments["without_full_text"]}" data-citation-total="{judgments["citations"]}"'
                     f'{"" if text else " hidden"}>{escape(text)}</li>')
        def row(item) -> str:
            return (f'<li data-summary-kind="{escape(str(item.get("kind") or ""), quote=True) if isinstance(item, dict) else ""}">'
                    f"{_render_summary_item(item, placed=placed_citations, located=located_references)}</li>")
        # References that cannot be verified head the Sources column, before
        # the retrieval and judgment lines (owner request 2026-09-30).
        first = [item for item in items if isinstance(item, dict) and item.get("kind") == "unverified_reference"]
        items = [item for item in items if item not in first]
        lead = "".join(row(item) for item in first) + lead
        if items:
            body = "<ul>" + lead + "".join(row(item) for item in items) + "</ul>"
        else:
            body = f"<ul>{lead}</ul>" if lead else ""
        sections.append(f'<section class="summary-column"><h2>{label}</h2>{body}</section>')
    return ('<section class="summary"><div class="report-summary">'
            f'<div class="summary-grid">{"".join(sections)}</div></div></section>')


HOW_TO_READ_TITLE = 'How to Read This Report'
# Patchwriting passages: the Academic Practice yellow; light-blue hover and
# selection like citations, drawn over the yellow; no outline. In the window,
# the student's and the source's words carry no background (owner request 2026-09-29).
PASSAGE_CSS = (
    '.patchwriting-overlay{cursor:pointer}.patchwriting-overlay:focus{outline:none}'
    '.patchwriting-overlay .patchwriting-hit{fill:#ffe45c;fill-opacity:.4;stroke:none}'
    '.patchwriting-overlay .patchwriting-selection{fill:transparent;stroke:none}'
    '.patchwriting-overlay.hovered .patchwriting-selection,.patchwriting-overlay.selected .patchwriting-selection,'
    '.patchwriting-overlay:focus .patchwriting-selection{fill:#b9dcff;fill-opacity:.22}'
    'body.hide-reference-practice .patchwriting-overlay,body.layout-judgment .patchwriting-overlay{display:none}'
    '.patchwriting-finding+.patchwriting-finding{margin-top:.75rem}'
    '.patchwriting-student,.patchwriting-source{margin:.3rem 0}.passage-source-label{margin:.5rem 0 0}'
)
HOW_TO_READ_CSS = (
    '.how-to-read-dialog{max-width:44rem;width:calc(100% - 2rem);border:1px solid var(--line);border-radius:8px;'
    'padding:1.2rem 1.5rem;color:var(--ink)}.how-to-read-dialog::backdrop{background:#18212b99}'
    '.how-to-read-dialog h1{font-size:1.4rem;margin:0 0 .5rem}.how-to-read-dialog h2{font-size:1.05rem;margin:1rem 0 .3rem}'
    '.how-to-read-dialog p{margin:.3rem 0}.how-to-read-dialog .how-to-read-actions{text-align:right;margin-top:1rem}'
    '.how-to-read-close{border:0;border-radius:5px;background:var(--blue);color:#fff;padding:.45rem 1rem;font:inherit;cursor:pointer}'
    '.upload-heading{font-weight:400}.upload-heading strong{font-weight:700}'
    '.upload-heading .upload-intro{font-size:1rem}.verification-record{margin:0 0 .8rem}'
    'body>main.layout:last-of-type{margin-bottom:0}'
)


def _how_to_read_script(nonce: str, *, persist: bool) -> str:
    """Show How to read before the report: once per browser in the app; on every
    opening of an exported report (owner request 2026-09-28)."""
    remember = 'true' if persist else 'false'
    return (f'<script nonce="{escape(nonce, quote=True)}">(()=>{{const d=document.getElementById("how-to-read-dialog");'
            f'if(!d||typeof d.showModal!=="function")return;const remember={remember},key="sourcefidelity-how-to-read-seen";'
            'let seen=false;if(remember){try{seen=localStorage.getItem(key)==="1";}catch(e){}}'
            'if(seen)return;d.addEventListener("close",()=>{if(remember){try{localStorage.setItem(key,"1");}catch(e){}}});'
            'd.querySelector("[data-close-guide]")?.addEventListener("click",()=>d.close());'
            'd.showModal();})();</script>')


def _how_to_read_sections() -> str:
    """The owner's text (revised 2026-09-28, second revision; patchwriting sentence 2026-09-29)."""
    return (
        '<h2>Source Identification, Verification and Retrieval</h2>'
        '<p>Each cited work is identified, its reference details are checked against scholarly indexes, '
        'library catalogues and web search, and the source is retrieved where possible. Soft red highlights '
        'mark a reference that cannot be verified and its citations; a blue outline marks reference '
        'details that differ from the located record; and pink highlights indicate a possible topical mismatch '
        'between citation and source.</p>'
        '<h2>Poor Academic Practice</h2>'
        '<p>Yellow highlights mark attribution issues, missing or duplicate reference entries, patchwriting, '
        'secondary citation, and quoted wording that differs from the source. Patchwriting is checked only '
        'against sources whose full text was retrieved.</p>'
        '<h2>Citation and Reference Format Checking</h2>'
        '<p>Orange highlights mark citation and reference style and layout issues. '
        'A purple diamond marks a link or DOI issue: a link that is dead or incorrect, a DOI registered to a '
        'different source, or a DOI or link the reference should include but does not.</p>'
        '<h2>Source Use Judgment</h2>'
        '<p>When the full text of a source has been retrieved, an AI model assesses whether each citation '
        'statement is backed by that source. The underline styles show the results: supports, qualified or '
        'mixed, contradicts, not supported, and not judged.</p>'
        '<p>AI can make mistakes. Check the sources to verify judgments.</p>'
    )


def _render_how_to_read(view: dict) -> str:
    """The collapsible guide, and the same text shown once before the report."""
    return (
        f'<details class="read-guide"><summary><strong>{HOW_TO_READ_TITLE}</strong></summary>'
        f'{_how_to_read_sections()}</details>'
        '<dialog class="how-to-read-dialog" id="how-to-read-dialog" aria-labelledby="how-to-read-title">'
        f'<h1 id="how-to-read-title">{HOW_TO_READ_TITLE}</h1>{_how_to_read_sections()}'
        '<p class="how-to-read-actions"><button type="button" class="how-to-read-close" data-close-guide>Close</button></p></dialog>'
    )


_ADAPTER_NAMES = {
    'openalex': 'OpenAlex', 'crossref': 'Crossref', 'semantic_scholar': 'Semantic Scholar', 'core': 'CORE',
    'elsevier': 'Elsevier', 'datacite': 'DataCite', 'eric': 'ERIC', 'open_library': 'Open Library',
    'google_books': 'Google Books', 'gutenberg': 'Project Gutenberg', 'wikisource': 'Wikisource',
    'internet_archive': 'Internet Archive',
}
_SEARCH_NAMES = {
    'brave': 'Brave Search', 'exa': 'Exa', 'searxng': 'SearXNG (self-hosted)', 'tavily': 'Tavily',
    'google custom search': 'Google Custom Search', 'brightdata': 'Bright Data', 'duckduckgo': 'DuckDuckGo',
}


def _render_technical_details(metrics: dict, manifest_href: str = '') -> str:
    """Operational record for one run of the report: runtime, requests, tokens and cost.

    Figures cover a single run (owner request 2026-09-28): the first complete
    check and the report's Judgment run, not the later refreshes. Unknown
    values read "not recorded" and unpriced usage "no price on record";
    nothing unknown is shown as zero (ARCHITECTURE §8).
    """
    unrecorded = 'not recorded for this report'

    def seconds(value):
        return f'{value:,.2f} s' if isinstance(value, (int, float)) else 'not recorded'

    def usd(value):
        return f'US${value:,.4f}' if isinstance(value, (int, float)) else None

    def table(headings, rows):
        head = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headings)
        body = ''.join('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows)
        return f'<table class="tech-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    recorded = metrics.get('metrics_version') == 'processing-metrics-v2'
    sections = []
    runtime = table(('Measure', 'Value'), [('Processing Time', seconds(metrics.get('wall_seconds'))),
                                           ('CPU Time', seconds(metrics.get('cpu_seconds')))])
    sections.append('<section><h2>Runtime</h2>' + runtime + '</section>')
    if recorded:
        adapters = [(_ADAPTER_NAMES.get(row['provider'], row['provider']), f"{int(row['requests']):,}")
                    for row in metrics.get('adapter_requests') or []]
        body = table(('Source', 'Requests'), sorted(adapters)) if adapters else '<p>No academic-source requests were recorded.</p>'
    else:
        body = f'<p>Per-source requests: {unrecorded}.</p>'
    sections.append('<section><h2>Academic Sources</h2>' + body + '</section>')
    if recorded:
        fetches = [(str(row['kind']).capitalize(), f"{int(row['requests']):,}") for row in metrics.get('direct_fetches') or []]
        body = table(('Kind', 'Requests'), sorted(fetches)) if fetches else '<p>No direct web fetches were recorded.</p>'
        sections.append('<section><h2>Direct Web Fetches</h2>' + body + '</section>')
    if recorded:
        rows = []
        for row in metrics.get('search_by_provider') or []:
            if row.get('cost_basis') == 'free':
                cost = 'no charge'
            elif row.get('cost_usd') is None:
                cost = 'no price on record'
            else:
                cost = usd(row['cost_usd'])
            rows.append((_SEARCH_NAMES.get(row['provider'], row['provider']), f"{int(row['calls']):,}", cost))
        body = table(('Search API', 'Requests', 'Estimated Cost'), rows) if rows else '<p>No search API requests were recorded.</p>'
    else:
        legacy = metrics.get('search_calls_by_provider') or {}
        rows = [(_SEARCH_NAMES.get(name, name), f'{int(count):,}', unrecorded) for name, count in sorted(legacy.items())]
        body = (table(('Search API', 'Requests', 'Estimated Cost'), rows) if rows else
                f'<p>Search API requests: {metrics.get("search_api_calls", "not recorded")}.</p>')
    sections.append('<section><h2>Search APIs</h2>' + body + '</section>')
    if recorded:
        rows = []
        for row in metrics.get('llm_by_model') or []:
            cost = usd(row.get('cost_usd'))
            cost = ('no price on record' if cost is None else cost + ('' if row.get('cost_complete') else ' (partial)'))
            calls = f"{int(row.get('calls') or 0):,}"
            if row.get('missing_usage_calls'):
                calls += f" ({int(row['missing_usage_calls']):,} without usage)"
            rows.append((row.get('model') or 'unnamed model', row.get('endpoint_host') or 'not recorded', calls,
                         f"{int(row.get('prompt_tokens') or 0):,}", f"{int(row.get('prompt_cache_hit_tokens') or 0):,}",
                         f"{int(row.get('completion_tokens') or 0):,}", cost))
        body = (table(('Model', 'Endpoint', 'Calls', 'Input Tokens', 'Of Which Cached', 'Output Tokens', 'Estimated Cost'), rows)
                if rows else '<p>No language-model calls were recorded.</p>')
    else:
        calls, tokens = metrics.get('llm_calls'), metrics.get('total_tokens')
        body = (f'<p>Calls: {calls if calls is not None else "not recorded"} · Tokens: '
                f'{f"{tokens:,}" if isinstance(tokens, int) else "not recorded"} · Per-model detail: {unrecorded}.</p>')
    sections.append('<section class="wide"><h2>Language Models</h2>' + body + '</section>')
    total = usd(metrics.get('estimated_cost_usd'))
    sections.append('<section><h2>Estimated Cost (Before Credits)</h2>'
                    f'<p><strong>{total or "not recorded"}</strong></p></section>')
    link = (f'<p class="verification-record"><a href="{escape(manifest_href.split("?")[0], quote=True)}">'
            'Download Verification Record (JSON)</a></p>' if manifest_href else '')
    return ('<details class="technical-details"><summary><strong>Technical Details</strong></summary>'
            f'<div class="tech-grid">{"".join(sections)}</div>{link}</details>')


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


def _formatting_overlaps(citation: dict, finding: dict) -> bool:
    if finding.get('finding_type') in LINK_MARKER_FINDINGS | {'source_topical_mismatch'}:
        return False
    return any(a.get('page_index') == b.get('page_index')
        and max(a['x0'],b['x0']) < min(a['x1'],b['x1'])
        and max(a['y0'],b['y0']) < min(a['y1'],b['y1'])
        for a in (citation.get('paper_location') or {}).get('rectangles',[])
        for b in finding.get('rectangles') or [])


def _upload_priorities(citations: list[dict], unverified: frozenset = frozenset()) -> list[dict]:
    sources = {}
    for index, citation in enumerate(citations, 1):
        for member_index, member in enumerate(citation.get('members', [])):
            rid = member.get('reference_id')
            if not rid:
                continue
            row = sources.setdefault(rid, {'member':member, 'citation_index':index,
                'member_index':member_index, 'citations':set(), 'full_text':False})
            row['citations'].add(index)
            row['full_text'] |= member.get('coverage_level') == 'full_text'
    # A reference that cannot be verified is not suggested for upload
    # (owner decision 2026-09-30: Gerbner, Paper 1).
    eligible = [row for row in sources.values() if not row['full_text']
        and not (row['member'].get('unverified') or row['member'].get('reference_id') in unverified)
        and row['member'].get('coverage_level') in {'unavailable','abstract_only','partial_text'}
        and _member_accepts_upload(row['member'])]
    return sorted(eligible,key=lambda row:(-len(row['citations']),row['citation_index'],row['member_index']))[:3]


# The key (owner layout 2026-09-28): each mark is drawn on its own label text.
KEY_HTML = (
    '<div class="legend" id="active-key">'
    '<div class="key-line" data-key="judgment"><span class="key-title">Judgment:</span>'
    '<span class="jk state-supported">Supported</span>'
    '<span class="jk state-qualified">Qualified or Mixed</span>'
    '<span class="jk state-contradicts">Contradicts</span>'
    '<span class="jk state-insufficient">Not Supported</span>'
    '<span class="jk state-undecided">LLM Undecided</span>'
    '<span class="jk state-not_judged">Not Judged</span></div>'
    '<div class="key-line"><span class="key-title">Issues:</span>'
    '<span class="key-mark key-practice">Academic-Practice</span>'
    '<span class="key-mark key-reference">Citation/Reference Issue</span>'
    '<span class="key-mark key-unverified">Unverifiable Reference</span>'
    '<span class="key-mark key-record">Source Record Conflict</span>'
    '<span class="key-link"><i class="submitted-link-key"></i>Link</span></div></div>'
)
KEY_CSS = (
    '.legend{display:flex;flex-direction:column;gap:.45rem}'
    '.legend .key-line{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1rem}'
    '.legend .key-title{font-weight:650}'
    '.legend .key-mark{padding:0 .2em;border-radius:2px}'
    '.legend .key-practice{background:#ffe45c66}.legend .key-reference{background:#ff9a384d}'
    '.legend .key-unverified{background:#f28b826b}'
    '.legend .key-record{border:2px solid #0a7cff}'
    '.legend .key-link{display:inline-flex;align-items:center;gap:.45rem}'
    '.legend .submitted-link-key{display:inline-block;width:.7rem;height:.7rem;transform:rotate(45deg);background:#7651a8}'
)


def _render_upload_priorities(citations: list[dict], locations: dict | None = None, *,
                              judgments: int | None = None, unverified: frozenset = frozenset()) -> str:
    rows = []
    upload_href = None
    affected_pairs = 0
    priorities = _upload_priorities(citations, unverified)
    for row in priorities:
        source = row['member'].get('source') or {}
        label = str(source.get('raw_reference') or source.get('title') or 'Source')
        count = len(row['citations'])
        reference = _render_formatted_reference(source, label)
        href = str(source.get('url') or '')
        if href and _safe_reference_href(href) and escape(href,quote=True) not in reference:
            reference += f' <a href="{escape(href,quote=True)}" target="_blank" rel="noopener">{escape(href)}</a>'
        affected_pairs += len(row['citations'])
        # Owner layout 2026-09-28: the citation count, then the reference.
        rows.append(f'<li>{count} {"citation" if count == 1 else "citations"} – {reference}</li>')
        upload = citations[row['citation_index']-1].get('upload_action') or {}
        if upload.get('enabled') and upload.get('href'):
            upload_href = str(upload['href']).split('/citation/', 1)[0] + '/source/upload'
    if not rows:
        return ''
    upload_html = ''
    if upload_href:
        upload_html = (f'<form class="source-upload upload-priority-form" method="post" enctype="multipart/form-data" action="{escape(upload_href, quote=True)}">'
            '<input type="file" name="file" accept="application/pdf,.pdf" aria-label="Choose source PDF" hidden required>'
            '<button type="button" data-choose-source>Upload Sources</button>'
            '<span class="upload-status" aria-live="polite"></span></form>')
    intro = ''
    if judgments is not None:
        # Owner wording 2026-09-29: judgments, not judged citations. Each
        # citation of an uploaded source gains at least one judgment.
        these = (f'these {len(priorities)} sources are' if len(priorities) != 1 else 'this source is')
        # On the heading's line, not bold.
        def judgments_text(n):
            return f"{n:,} judgment{'' if n == 1 else 's'}"
        intro = (f'<span class="upload-intro"> - This paper has <span data-judgment-count>{judgments_text(judgments)}'
                 f'</span>. If {these} uploaded, the paper will have {judgments_text(affected_pairs).replace(" ", " more ", 1)}.</span>')
    return ('<section class="upload-priorities"><h2 class="upload-heading"><strong>Sources to Upload</strong>'
            + intro + '</h2>' +
        '<ol>'+''.join(rows)+'</ol>'+upload_html+'</section>')


def render_evidence_report_html(view: dict, *, csp_nonce: str) -> str:
    """Render the page-faithful paper as the report's primary navigator."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", csp_nonce):
        raise EvidenceReportError("A valid CSP nonce is required for report rendering")
    nonce = escape(csp_nonce, quote=True)
    from app.services.report_member_navigation import member_targets
    from app.services.report_layers import topical_mismatch
    citations = [{**item, 'members': [{**member, 'abstract_scope_attention': topical_mismatch(member, item)}
                   for member in item.get('members', [])], 'source_member_targets':member_targets(
        item, (view.get('paper_surface') or {}).get('marker_words') or (view.get('paper_surface') or {}).get('selectable_words') or {})}
        for item in view['citations']]
    reference_practice = normalize_reference_findings(view.get("reference_practice") or [])
    # Every index below (windows, overlays, summaries) must count this list.
    view = {**view, 'reference_practice': reference_practice}
    unverified_ids = {f.get('reference_id') for f in reference_practice
                      if f.get('finding_type') in UNVERIFIED_FINDINGS}
    for citation in citations:
        for member in citation.get('members', []):
            member['unverified'] = member.get('reference_id') in unverified_ids
    export_mode = str(view.get("export_mode") or "interactive")
    # One layout (owner decision 2026-09-28): Judgment is part of every
    # interactive report; exports carry no Judgment.
    # An exported report carries finished Judgment results (owner request 2026-09-29).
    static_judgment = bool((view.get('judgment_layer') or {}).get('static'))
    judgment_enabled = export_mode == 'interactive' and (not view.get('portable_export') or static_judgment)
    if judgment_enabled and view.get('judgment_layer') is not None:
        attach_propositions(citations, view['judgment_layer'])
    panel_templates = ''
    citation_format = str(view.get('citation_format') or '')
    from app.services.report_style_guidance import guidance_links
    passages = [p for p in view.get('patchwriting_passages') or []
                if isinstance(p, dict) and isinstance(p.get('number'), int)]
    patchwriting = _patchwriting_by_window(passages, citations)
    for index, item in enumerate(citations, 1):
        template = _render_panel_template(item, index, patchwriting=patchwriting)
        overlapping = [(f, j) for j, f in enumerate(reference_practice, 1) if _formatting_overlaps(item, f)]
        # Owner wording 2026-09-30 for a parenthetical after the final punctuation.
        placement = ({'formatting': '<p class="reference-issue">This parenthetical citation is placed after '
                                    'the sentence\'s final punctuation.</p>'}
                     if citation_after_punctuation(item) else None)
        related = _grouped_findings(overlapping, combined=True, guidance=False, extra=placement)
        # One style-guide paragraph at the bottom of the citation window.
        related += guidance_links(_citation_guidance_kinds(item) + [f.get('finding_type') for f, _ in overlapping],
                                  citation_format)
        panel_templates += template.replace('</template>', related + '</template>')
    from app.services.report_references import reference_catalog, reference_list_finding, reference_numbers
    numbers = reference_numbers(view)
    catalog = reference_catalog({**view, 'citations': citations}, numbers)
    for index, item in enumerate(reference_practice, 1):
        if reference_list_finding(item) and item.get('reference_id') in numbers:
            continue  # Shown inside its Reference N window.
        template = _render_reference_panel_template(item,index)
        related = [i for i,c in enumerate(citations,1) if _formatting_overlaps(c,item)]
        if len(related)>1:
            links = '<p>This issue overlaps these citations:</p>'+''.join(
                f'<button type="button" data-panel-template="citation-panel-{i}">Citation {i}</button>' for i in related)
            template = template.replace('</template>',links+'</template>')
        panel_templates += template
    paper_key = _paper_key(view.get('paper_surface') or {})
    for entry in catalog:
        panel_templates += _render_reference_window_template(
            entry, reference_practice, citations, citation_format=citation_format,
            patchwriting=patchwriting.get(('reference', entry.get('reference_id'))), paper_key=paper_key)
    paper_pages, placed_indexes = _render_continuous_paper(
        view["paper_surface"], citations, reference_practice,
        numbers=numbers, catalog=catalog, passages=passages,
    )
    # Keep unresolved geometry in retained diagnostics, not a new report section.
    summaries = _render_report_summary(
        report_summary(view),
        placed_citations=frozenset(i for i, c in enumerate(citations, 1)
                                   if (c.get('paper_location') or {}).get('rectangles')),
        located_references=frozenset(entry['number'] for entry in catalog if entry.get('location')),
        retrieval=_retrieval_sentence(view.get("overview") or {}),
        judgments=({"states": (view.get('judgment_summary') or {}).get('states') or {},
                    "without_full_text": sum(1 for c in citations if not any(
                        m.get('coverage_level') == 'full_text' for m in c.get('members') or [])),
                    "citations": len(citations)} if judgment_enabled else None),
    )
    if (view.get('assessment_configuration') or {}).get('require_reference_links'):
        summaries += '<p class="assessment-requirement">Assessment requirement: every reference must contain a DOI, URL, or library link. Link presence is checked separately from validity or access.</p>'
    # Shown to everyone for now; the PDF carries none (ARCHITECTURE §8).
    how_to_read = _render_how_to_read(view) + _render_technical_details(
        view.get('processing_metrics') or {}, str((view.get('export_action') or {}).get('manifest_href') or ''))
    gauges = ''  # Retain diagnostic counts in the snapshot, not the user report.
    export_action = view.get("export_action") or {}
    if export_mode not in {"interactive", "released_print"}:
        raise EvidenceReportError("Report export mode is invalid")
    export_links = ""
    if export_action:
        export_links = (
            f'<a href="{escape(str(export_action.get("report_href") or "#").split("?")[0], quote=True)}/export.html">Download Interactive Report</a>'
            f'<a href="{escape(str(export_action.get("pdf_href") or "#").split("?")[0], quote=True)}">Download PDF</a>'

        )
    print_links = (
        f'<a href="{escape(str(export_action.get("report_href") or "#"), quote=True)}">Interactive Report</a>'
        if export_mode == "released_print" else ''
    )
    # Select text is the one paper tool: copying a passage. Comment, highlight
    # and pen tools were removed (owner decision 2026-09-25).
    paper_tools = ('<span class="group" aria-label="Paper tools">'
        '<button type="button" id="select-text" aria-pressed="false">Select Text</button></span>')
    # Judgments: judged statements (each proposition and source).
    judgment_count = sum(int(v or 0) for v in ((view.get('judgment_summary') or {}).get('states') or {}).values())
    rendered = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(view['title'])}</title><style nonce="{nonce}">
:root{{--ink:#18212b;--muted:#5b6672;--line:#d8dee6;--blue:#2563a7;--teal:#168c8c;--no-connection:#8c6b4f;--blue-bg:#eef6ff;--source-bg:#f7f4ea;--amber:#d95f02;--amber-bg:#fff6e8;--gray:#b8c0c8;--violet:#7651a8;--paper:#fff;--page:#f3f5f7}}
*{{box-sizing:border-box}} [hidden]{{display:none!important}} body{{margin:0;background:var(--page);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
    header,main,footer{{width:100%;max-width:none}} header{{padding:1.5rem clamp(.75rem,2vw,1.5rem) 1rem}} h1{{margin:0 0 .25rem;font-size:1.8rem}} h2{{font-size:1.2rem}} .sub{{color:var(--muted)}}
.report-actions{{display:flex;gap:.35rem;justify-content:flex-end;flex-wrap:wrap}} .report-actions a{{border:1px solid var(--line);background:#fff;border-radius:5px;padding:.45rem .65rem;color:inherit;text-decoration:none}} .report-actions a[aria-current="page"]{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.summary{{margin:1rem 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:1rem}} .summary-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}} .summary-column{{border-left:3px solid var(--blue);padding:0 .85rem}} .summary-column h2{{margin-top:0}} .summary-column li{{margin:.6rem 0}} .empty-pattern{{color:var(--muted)}}
.technical-details{{margin:1rem 0;background:#fff;border:1px solid var(--line);border-radius:7px;padding:.2rem .9rem}}.tech-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr));gap:.5rem 1.25rem;padding:.2rem 0 1rem}}.tech-grid section.wide{{grid-column:1/-1;overflow-x:auto}}.tech-grid h2{{font-size:1rem;margin:.6rem 0 .3rem}}.tech-table{{border-collapse:collapse;font-size:.86rem;width:100%}}.tech-table th,.tech-table td{{text-align:left;padding:.2rem .5rem .2rem 0;border-bottom:1px solid var(--line);white-space:nowrap}}.tech-table th{{color:var(--muted);font-weight:600}}@media print{{.technical-details{{display:none}}}} .read-guide{{margin:1rem 0;background:#fff;border:1px solid var(--line);border-radius:7px;padding:.2rem .9rem}} .guide-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;padding:.7rem 0 1rem}} .guide-grid p{{margin:.25rem 0}}
    .layout{{--paper-share:68%;width:auto;display:grid;grid-template-columns:minmax(20rem,var(--paper-share)) minmax(20.75rem,1fr);grid-template-rows:auto minmax(0,1fr);height:100dvh;overflow:clip;margin:0 clamp(.75rem,2vw,1.5rem);background:var(--page);border:1px solid var(--line);border-radius:8px}} .paper,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:1rem;min-width:0}} .panel>h2:first-child{{margin-top:0}} .source-group>h3{{font-size:.95rem;margin:.7rem 0 .2rem;border-bottom:2px solid var(--gray);padding-bottom:.2rem}} .source-group.full_text>h3{{border-color:var(--blue)}} .source-group.partial_text>h3,.source-group.abstract_only>h3{{border-color:var(--teal)}} .panel{{overflow:auto;overflow-wrap:anywhere}} .splitter{{cursor:col-resize;touch-action:none;display:flex;align-items:center;justify-content:center}} .splitter::before{{content:"";width:3px;height:4rem;border-radius:2px;background:#aeb8c4}} .splitter:focus{{outline:2px solid var(--blue);outline-offset:-2px}} .paper-pages{{display:grid;gap:1.5rem}} .paper-page{{margin:0}} .paper-page figcaption{{color:var(--muted);font-size:.85rem;margin-bottom:.3rem}} .page-surface{{display:block;width:100%;height:auto;background:#fff;box-shadow:0 2px 12px #0002}}
    .paper-viewport{{overflow:auto}} .paper-pages{{width:var(--paper-zoom,100%);margin-inline:auto}} .toolbar{{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.8rem 0}} .toolbar button,.toolbar label{{border:1px solid var(--line);background:#fff;border-radius:5px;padding:.45rem .65rem;font:inherit}} .toolbar label{{display:flex;gap:.35rem;align-items:center}} .toolbar .group{{display:flex;gap:.3rem;padding-left:.45rem;border-left:1px solid var(--line)}} .toolbar button.active{{background:#f0e9fa;border-color:var(--violet)}} .paper-toolbar{{position:static}} .report-status{{color:var(--muted);font-size:.88rem}}
    .citation-overlay,.reference-practice-overlay{{cursor:pointer}} .citation-overlay .selection-bg{{fill:transparent;stroke:none}} .citation-overlay .underline{{fill:none;vector-effect:non-scaling-stroke;stroke-width:1.5}} .citation-overlay.evidence_available{{stroke:var(--blue)}} .citation-overlay.limited_evidence{{stroke:var(--teal)}} .citation-overlay.retrieved_no_connection{{stroke:var(--no-connection)}} .citation-overlay.attention{{stroke:var(--amber)}} .citation-overlay.not_assessed{{stroke:var(--gray)}} .citation-overlay:focus .underline,.citation-overlay.selected .underline{{stroke-width:1.9}} .citation-overlay.selected .selection-bg{{fill:#b9dcff;fill-opacity:.52}} .citation-overlay .quote-difference-mark{{fill:#ffe45c;fill-opacity:.4;stroke:none}} .citation-overlay:focus{{outline:none}} .reference-practice-overlay rect{{fill:var(--amber);fill-opacity:.14;stroke:var(--amber);stroke-width:1;stroke-dasharray:3 2;vector-effect:non-scaling-stroke}} .reference-practice-overlay.selected rect{{fill-opacity:.28}}
    body.hide-evidence .citation-overlay,body.hide-reference-practice .reference-practice-overlay{{display:none}} body.layout-paper .citation-overlay,body.layout-paper .reference-practice-overlay{{display:none}}
    .selected-citation{{margin:.35rem 0 0;padding:.7rem .85rem;border-left:3px solid var(--blue);background:var(--blue-bg);white-space:normal;overflow-wrap:anywhere}} mark.quote-difference{{background:#ffe45c66;color:inherit;padding:0 .05em;border-radius:2px}} .member{{border-top:1px solid var(--line);padding-top:.75rem;margin-top:.75rem;min-width:0}} .member:first-of-type{{border-top:0;padding-top:.2rem;margin-top:0}} .member h3{{margin:.25rem 0 .35rem}} blockquote.source-excerpt{{max-width:100%;margin:.35rem 0;padding:.7rem .9rem;border-left:3px solid #8b7a45;white-space:normal;overflow-wrap:anywhere;word-break:normal}} .locator,.muted{{color:var(--muted);font-size:.92rem}} .full-reference{{margin:.8rem 0;white-space:normal;overflow-wrap:anywhere;font-size:.8rem;line-height:1.35}} .reference-url{{color:var(--blue);text-decoration:underline}} details{{margin:.6rem 0}} .checks{{padding-left:1.2rem}} .attention-text{{color:var(--amber);font-weight:650}} .source-unavailable{{margin:.35rem 0}} .compact-action{{display:inline-block;padding:.18rem .38rem;border:0;border-radius:4px;background:var(--blue);color:#fff;font-size:.82rem;line-height:1.25;font-weight:650;text-decoration:none;cursor:pointer}} .actions{{margin:.45rem 0}} .source-upload button,.search-again button{{padding:.18rem .38rem;border:0;border-radius:4px;background:var(--violet);color:#fff;font-size:.82rem;line-height:1.25;font-weight:650}} .source-upload,.search-again{{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem;margin:.5rem 0}} .source-upload label{{font-size:.86rem}} .source-upload input{{max-width:13rem;font-size:.8rem}} .upload-status{{display:block;width:100%;color:var(--muted);font-size:.88rem}}
.legend{{display:flex;flex-wrap:wrap;gap:.55rem 1rem;background:#fff;border:1px solid var(--line);border-radius:6px;padding:.65rem .8rem}} .legend span{{display:inline-flex;align-items:center;gap:.35rem}} .line-key{{display:inline-block;width:2.1rem;border-bottom:3px solid #111}} .color-key{{display:inline-block;width:1.8rem;height:2px;border-radius:0}} .color-key.blue{{background:var(--blue)}} .color-key.teal{{background:var(--teal)}} .color-key.no-connection{{background:var(--no-connection)}} .color-key.amber{{background:var(--amber)}} .color-key.gray{{background:var(--gray)}} .reference-key{{display:inline-block;width:1.3rem;height:.75rem;background:var(--amber-bg);border:1px dashed var(--amber)}}
.counts-disclosure{{margin:1rem clamp(.75rem,2vw,1.5rem) 2rem;background:#fff;border:1px solid var(--line);border-radius:8px;padding:.25rem .9rem}} .gauges{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;padding:1rem 0}} .gauge{{display:grid;grid-template-columns:145px 1fr;grid-template-areas:"label label" "graphic legend";gap:.35rem 1rem;border:1px solid var(--line);border-radius:7px;padding:.8rem}} .gauge h3{{grid-area:label;margin:0;text-align:center}} .gauge-graphic{{grid-area:graphic;position:relative;width:145px;height:105px}} .gauge-graphic svg{{width:145px;height:105px}} .gauge-graphic path{{fill:none;stroke-width:16}} .gauge-base{{stroke:#e0e5ea}} .gauge-total{{position:absolute;inset:58px 0 auto;text-align:center;font-size:1.2rem;font-weight:700}} .gauge-unit{{display:block;font-size:.72rem;color:var(--muted);font-weight:400}} .gauge-legend{{grid-area:legend;align-self:center}} .gauge-legend div{{display:flex;justify-content:space-between;gap:.8rem;font-size:.86rem}} .swatch{{display:inline-block;width:.65rem;height:.65rem;margin-right:.3rem}}
.page-container{{position:relative;container-type:inline-size;content-visibility:auto}} .paper-text-layer{{position:absolute;inset:0;pointer-events:none;user-select:text}} .paper-word{{position:absolute;display:inline-block;color:transparent;cursor:text;user-select:inherit;pointer-events:var(--word-pointer,auto);white-space:pre;line-height:1;font-family:serif}} .paper-word::selection{{color:transparent;background:#b9dcff88}} .page-surface image{{pointer-events:none}} .text-selection-off .paper-text-layer{{--word-pointer:none;user-select:none}} .technical-export{{margin:0 1.5rem 1rem;font-size:.8rem}}
.citation-overlay .citation-span{{stroke:#c5cbd1;stroke-width:1;fill:none}} .citation-overlay.selected .citation-span,.citation-overlay:focus .citation-span{{stroke:#aeb7c0;stroke-width:1}} .source-highlight{{stroke:none;fill-opacity:.23}} .member-target.evidence_available .source-highlight{{fill:var(--blue)}} .member-target.limited_evidence .source-highlight{{fill:var(--teal)}} .member-target.not_assessed .source-highlight{{fill:var(--gray)}} .member-target.selected .source-highlight,.member-target:focus .source-highlight{{fill-opacity:.36;stroke:var(--blue);stroke-width:1;vector-effect:non-scaling-stroke}} .member-navigation{{display:flex;align-items:center;gap:.7rem;margin:.5rem 0}} .member-navigation button{{background:#fff;border:1px solid var(--line);border-radius:4px;padding:.25rem .6rem;cursor:pointer}} [data-source-member][hidden]{{display:none}} .color-key{{height:.75rem;opacity:.28;border-radius:2px}}
.citation-overlay [data-citation-span],.citation-overlay.selected [data-citation-span],.citation-overlay:focus [data-citation-span]{{stroke:#c5cbd1;stroke-width:1}}
.citation-overlay.selected .selection-bg{{fill-opacity:.22}}

.member-target.selected .source-highlight,.member-target:focus .source-highlight{{fill-opacity:.26}}
.paper-word::selection{{background:#b9dcff44}}
.source-group>h3{{border-bottom:0;color:var(--ink)}}
.paper-toolbar{{flex-wrap:nowrap;overflow-x:auto}} .paper-toolbar .group{{flex:none;align-items:center;white-space:nowrap}} .paper-toolbar button{{white-space:nowrap}} .paper-toolbar .report-status{{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}}
.paper-pages{{user-select:none}} .paper-text-layer{{user-select:text}} .style-guidance{{text-decoration:underline}}
@media print{{.page-container{{content-visibility:visible}}}}
.member-target.partial_evidence .source-highlight,.color-key.mauve{{fill:#8b7ca9;background:#8b7ca9}}
.relevance-key,.practice-key,.copying-key{{display:inline-block;width:1.3rem;height:.75rem}} .relevance-key{{background:#ef82ba}} .practice-key{{background:#ffe066}} .copying-key{{background:#efb0b0}}
.layer-mark{{display:none;vector-effect:non-scaling-stroke;stroke-width:1.2}} .mark-relevance{{fill:#ef82ba;fill-opacity:.38;stroke:none;pointer-events:all}} .mark-reference{{fill:none;stroke:#d95f02;stroke-dasharray:3 2}} .mark-practice{{fill:#ffe066;fill-opacity:.3;stroke:none}}
.layer-indicator{{stroke:white;stroke-width:.5;vector-effect:non-scaling-stroke}} .indicator-relevance{{fill:#b08b00}} .indicator-reference{{fill:#d95f02}} .indicator-practice{{fill:#cfaa00}}
.reference-practice-overlay .reference-hit{{fill:transparent;stroke:none;pointer-events:all}} .reference-field-marker{{fill:var(--amber);stroke:white;stroke-width:.4;vector-effect:non-scaling-stroke}} .reference-practice-overlay:focus{{outline:none}} .reference-practice-overlay:focus .reference-field-marker{{stroke:var(--blue);stroke-width:1}} .reference-key{{width:.4rem;height:.4rem;background:var(--amber);border:0;border-radius:50%}} .report-status:empty{{display:none}}
.toolbar button[aria-pressed="true"]{{background:var(--blue-bg);border-color:var(--blue)}}
.quote-difference-mark{{display:block;fill:#ffe066;fill-opacity:.35;stroke:none}} body.focus-practice .mark-practice,body.focus-reference .mark-reference,body.focus-relevance .mark-relevance{{display:block}}
body:not(.focus-practice) .indicator-practice,body:not(.focus-reference) .indicator-reference,body:not(.focus-relevance) .indicator-relevance{{display:none}}
body.hide-evidence .citation-overlay{{display:revert}} body.hide-evidence .source-highlight{{visibility:hidden}} body.hide-reference-practice .reference-practice-overlay{{display:none}}
body.layout-paper .citation-overlay{{display:revert}} body.layout-paper .source-highlight,body.layout-paper .layer-indicator,body.layout-paper .layer-mark,body.layout-paper .quote-difference-mark{{display:none}}
@media(max-width:900px){{.summary-grid,.guide-grid{{grid-template-columns:1fr}}}} @media(max-width:760px){{.gauge{{grid-template-columns:1fr;grid-template-areas:"label" "graphic" "legend"}}}}
body[data-export-mode="released_print"] .paper-toolbar{{display:none}}
@media print{{body{{background:#fff}} header>.toolbar,.report-actions,.read-guide,.counts-disclosure{{display:none}} .layout{{display:block;padding:0}} .paper{{border:0;padding:0}} .paper>h2,.paper-toolbar,.panel,.splitter{{display:none}} .paper-page{{break-after:page}} .paper-page figcaption{{display:none}} .page-surface{{box-shadow:none}}}}
.reference-key,.submitted-link-key{{display:inline-block;width:1.8rem;height:2px;border:0;border-radius:0;transform:none;background:#d95f02}} .reference-hit{{pointer-events:none!important}}
.reference-field-marker{{fill:none;stroke:#d95f02;stroke-width:2;vector-effect:none;pointer-events:stroke}} .submitted-link-marker{{stroke:#7651a8}} .submitted-link-key{{background:#7651a8}}
.indicator-reference{{fill:none;stroke:#d95f02;stroke-width:2;vector-effect:none}}
.reference-formatting-hit{{fill:transparent;stroke:none;pointer-events:all}}
.upload-priorities{{background:white;border:1px solid var(--line);padding:.8rem 1rem;margin:1rem 0}} .upload-priorities li{{margin:.5rem 0}} .upload-priorities button{{text-align:left;max-width:100%;white-space:normal}}
</style></head><body class="layout-sources" data-export-mode="{escape(export_mode, quote=True)}">
<header><div class="report-actions" aria-label="Report and export actions">{print_links}{export_links}</div><h1>{escape(view['title'])}</h1><div class="sub">{escape(view['citation_format'])} · {int((view.get('word_counts') or {}).get('total') or 0):,} total words · {int((view.get('word_counts') or {}).get('body') or 0):,} body words · {int((view.get('word_counts') or {}).get('references') or 0):,} reference words</div>
{summaries}{_render_upload_priorities(citations, (view.get('paper_surface') or {}).get('reference_locations'), judgments=judgment_count if judgment_enabled else None)}{how_to_read}
{KEY_HTML}</header>
<main class="layout" id="report-layout"><div class="workspace-bar"><div class="toolbar paper-toolbar"><span class="group" aria-label="Paper zoom"><button type="button" id="zoom-out" aria-label="Zoom out">−</button><button type="button" id="zoom-in" aria-label="Zoom in">＋</button><span id="zoom-value">100%</span></span>{paper_tools}</div><div class="toolbar evidence-controls" aria-label="Report layout"><span class="group" aria-label="Citation and flag navigation"><button type="button" data-report-step="-1" aria-label="Previous citation or flag">←</button><button type="button" data-report-step="1" aria-label="Next citation or flag">→</button></span></div></div>
<section class="paper" aria-label="Submitted paper with SourceFidelity overlays"><p class="report-status" id="report-status" role="status"></p><div class="paper-viewport">{paper_pages}</div></section>
<div class="side-pane"><div class="splitter" id="report-splitter" role="separator" aria-label="Resize paper and evidence panels" aria-orientation="vertical" aria-valuemin="30" aria-valuemax="80" aria-valuenow="68" tabindex="0"></div><div class="evidence-column"><aside class="panel" id="evidence-panel" tabindex="-1" aria-live="polite"><h2>Select a Citation</h2><p>Choose a marked span while reading the paper to inspect its source-specific evidence.</p></aside></div></div></main>
{gauges}<div hidden>{panel_templates}</div>
<script nonce="{nonce}">{Path(__file__).with_name("report_interactions.js").read_text()}</script>
</body></html>"""
    rendered = rendered.replace('</style>', '.practice-notice{margin:.35rem 0} body:not(.focus-practice) .missing-reference-target{display:none}</style>', 1)
    rendered = rendered.replace('</style>', EVIDENCE_LABEL_CSS + KEY_CSS + HOW_TO_READ_CSS + PASSAGE_CSS + '</style>', 1)
    if judgment_enabled and view.get('judgment_layer') is not None:
        # Judgment marks, styles and script; results fill each window's slots.
        from app.services.judgment_layer import JUDGMENT_CSS, render_judgment_assets
        rendered = rendered.replace('</style>', JUDGMENT_CSS + '</style>', 1)
        rendered = rendered.replace('</body></html>', render_judgment_assets(
            view['judgment_layer'], report_id=str(view.get('report_id') or ''), nonce=nonce) + '</body></html>', 1)
    if not judgment_enabled:
        rendered = rendered.replace('</style>', '.legend [data-key=judgment],.judgment-part,.judgment-sep,'
                                    '.judgment-slot{display:none}</style>', 1)
    rendered = rendered.replace('</body></html>', _how_to_read_script(
        nonce, persist=export_mode == 'interactive' and not view.get('portable_export')) + '</body></html>', 1)
    if static_judgment:
        rendered = rendered.replace('</style>', '.jw-retry{display:none}</style>', 1)
    rendered = rendered.replace('orange indicator: citation/reference issue', 'orange highlight: citation/reference issue')
    rendered = rendered.replace('</style>', '.reference-practice-overlay .reference-formatting-hit.academic-highlight{fill:#ffe45c;fill-opacity:.4;stroke:none}</style>', 1)
    rendered = rendered.replace('orange underlines indicate', 'orange highlights indicate').replace('issue underlines and affected-word highlights', 'affected-word highlights')
    rendered = rendered.replace(' Availability is not a correctness judgment.', '')
    rendered = rendered.replace(' full-text references', ' full-text retrieval(s)').replace(' abstract/limited-text references', ' abstract/limited-text retrieval(s)').replace(' unretrieved references', ' unretrieved sources')
    rendered = rendered.replace('Purple underline: submitted-link issue.', 'Purple diamond: link issue.').replace('Purple underline', 'Purple diamond')
    rendered = rendered.replace('purple underlines indicate', 'purple diamonds indicate').replace('an orange or purple underline', 'an orange highlight or purple diamond')
    rendered = rendered.replace('</style>', '.reference-practice-overlay .reference-formatting-hit{fill:#ff9a38;fill-opacity:.3;stroke:none;pointer-events:all}.reference-practice-overlay .mark-relevance{fill:#ef82ba;fill-opacity:.38;stroke:none;pointer-events:all}.reference-practice-overlay .topical-reference-hit{fill:transparent;fill-opacity:1;stroke:none;pointer-events:all}.issue-heading.topical{text-decoration:none;background:rgba(239,130,186,.38)}.reference-key{height:.75rem;background:#ff9a384d}.indicator-reference{fill:#ff9a38;fill-opacity:.3;stroke:none}.issue-heading.formatting{text-decoration:none}</style>', 1)
    rendered = rendered.replace('</style>', '.issue-heading{font-weight:700;text-decoration:none;padding:.1em .2em;box-decoration-break:clone;color:inherit}.issue-heading.formatting{text-decoration-color:#d95f02;background:rgba(255,154,56,.3)}.issue-heading.academic{text-decoration-color:#b99b00;background:rgba(255,228,92,.4)}.submitted-link-marker{fill:#7651a8;stroke:white;stroke-width:1;pointer-events:all}.submitted-link-key{width:.7rem;height:.7rem;transform:rotate(45deg);background:#7651a8}.paper-reference-link rect{fill:transparent;pointer-events:all}</style>', 1)
    rendered = rendered.replace('<h2>Submitted Paper</h2>', '')
    rendered = rendered.replace('<span>Light-grey underline:', '<span data-key="reference"><i class="submitted-link-key"></i>purple diamond: link issue</span><span>Light-grey underline:')
    rendered = rendered.replace('</style>', '.citation-overlay.member-target .source-highlight.unverified-highlight,.reference-practice-overlay .reference-formatting-hit.unverified-highlight{fill:#f28b82;fill-opacity:.42;stroke:none}.reference-practice-overlay .reference-formatting-hit.reference-difference-highlight{fill:transparent;stroke:#0a7cff;stroke-width:1.8;stroke-dasharray:none;pointer-events:all}.issue-heading.unverified{text-decoration-color:#c5221f;background:rgba(242,139,130,.42)}.issue-heading.evidence{text-decoration-color:#2f62a8}.unverified-key,.difference-key{display:inline-block;width:1.3rem;height:.75rem}.unverified-key{background:#f28b826b}.difference-key{border:2px solid #0a7cff}.reference-finding .finding-item+.finding-item{margin-top:.6rem;padding-top:.6rem;border-top:1px solid #e4e7ec}</style>', 1)
    rendered = rendered.replace('</style>', '.citation-overlay.member-target .source-highlight.academic-highlight{fill:#ffe45c;fill-opacity:.4;stroke:none}.submitted-link-hit{fill:transparent;pointer-events:all}.submitted-link-key{width:.85rem;height:.85rem;transform:rotate(45deg);background:#7651a8}.focus-relevance .topical-reference-hit{display:block}</style>', 1)
    rendered = rendered.replace('</style>', '/* One connected workspace: a single control bar over the paper and the evidence window. */.workspace-bar{grid-column:1/-1;grid-row:1;display:flex;align-items:center;justify-content:space-between;gap:.75rem;padding:.3rem .6rem;background:#fff;border-bottom:1px solid var(--line);min-width:0}.workspace-bar .toolbar{position:static;margin:0;padding:0;background:transparent;border:0;box-shadow:none;height:auto;flex-wrap:nowrap;overflow-x:auto;gap:.25rem;min-width:0}.evidence-controls{margin-left:auto}.evidence-controls>*{flex:none;white-space:nowrap}.evidence-controls .group{display:inline-flex;gap:.3rem}.toolbar button,.toolbar label{padding:.35rem .45rem}.toolbar button:disabled{opacity:.45;cursor:default}.layout>.paper{grid-row:2;min-height:0;min-width:0;display:flex;flex-direction:column;padding:0;border:0;border-radius:0;background:transparent}.layout .paper-viewport{flex:1;min-height:0;overflow:auto;padding:.75rem}/* Skipped content-visibility pages report their last-rendered width; a flexible track stops it holding the column open. */.paper-pages{grid-template-columns:minmax(0,1fr)}.paper>.report-status{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);margin:0}.side-pane{grid-row:2;min-height:0;min-width:0;display:grid;grid-template-columns:.75rem minmax(0,1fr);background:#fff;border-left:1px solid var(--line);transition:transform .22s ease}.side-pane .splitter{height:auto;position:static;background:var(--page)}.side-pane .evidence-column{min-height:0;min-width:0;display:flex;flex-direction:column}.side-pane .panel{flex:1;min-height:0;max-height:none;position:static;margin:0;border:0;border-radius:0}.layout.panel-sliding .side-pane{transform:translateX(100%)}.layout.panel-collapsed{grid-template-columns:minmax(0,1fr)}.layout.panel-collapsed .side-pane{display:none}@media(prefers-reduced-motion:reduce){.side-pane{transition:none}}.page-container{content-visibility:auto}.text-selection-off .page-container,.text-selection-off .page-container *{user-select:none!important;-webkit-user-select:none!important}@media(max-width:760px){.layout{grid-template-columns:1fr;grid-template-rows:auto auto auto;height:auto;overflow:visible}.workspace-bar{flex-wrap:wrap}.layout>.paper{grid-row:auto}.layout .paper-viewport{max-height:calc(100dvh - 3rem)}.side-pane{grid-row:auto;grid-template-columns:1fr;border-left:0;border-top:1px solid var(--line)}.side-pane .splitter{display:none}.side-pane .panel{max-height:none}}@media print{.workspace-bar,.side-pane{display:none}.layout{display:block;height:auto;overflow:visible;border:0;margin:0}.layout .paper-viewport{max-height:none;overflow:visible;padding:0}.page-container{content-visibility:visible}}/* Margin numbers and whole-entry reference targets. */.paper-badge{cursor:pointer}.paper-badge .badge-bg{fill:#fff;stroke:#7b8794;stroke-width:.6;vector-effect:non-scaling-stroke}.paper-badge .badge-number{font:600 7px system-ui,-apple-system,sans-serif;fill:#34404c;pointer-events:none}.paper-badge:hover .badge-bg,.paper-badge.selected .badge-bg{fill:#dcecff;stroke:var(--blue)}body.layout-paper .paper-badge{display:none}.style-guidance-links{margin-top:.8rem;padding-top:.5rem;border-top:1px solid #e4e7ec;font-size:.85rem;color:var(--muted)}body.layout-judgment .citation-overlay,body.layout-judgment .reference-practice-overlay,body.layout-judgment .reference-entry-overlay,body.layout-judgment .paper-badge{display:none}@media print{.paper-badge{display:none}}.reference-entry-overlay{cursor:pointer}.reference-entry-hit{fill:transparent;stroke:none;pointer-events:all}.reference-entry-overlay:hover .reference-entry-hit{fill:#b9dcff;fill-opacity:.12}.reference-entry-overlay.selected .reference-entry-hit{fill:#b9dcff;fill-opacity:.22}.reference-entry-overlay:focus{outline:none}.reference-entry-overlay:focus .reference-entry-hit{stroke:var(--blue);stroke-width:1;vector-effect:non-scaling-stroke}.badge-key{display:inline-flex;align-items:center;justify-content:center;min-width:1.05rem;height:.85rem;padding:0 .2rem;border:1px solid #7b8794;background:#fff;font-size:.62rem;font-style:normal;font-weight:600;color:#34404c}.citation-badge-key{border-radius:.45rem}.reference-badge-key{border-radius:2px}/* Linked summary instances and the Reference window. */.summary-instance{color:var(--blue)}.reference-availability{font-size:.95rem;margin:.6rem 0 .3rem}.reference-citations button{border:1px solid var(--line);background:#fff;border-radius:4px;padding:.1rem .45rem;margin:.1rem .15rem;font:inherit;font-size:.86rem;cursor:pointer}.reference-finding{border-top:1px solid var(--line);margin-top:.75rem;padding-top:.25rem}' + '</style>', 1)
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


def enable_authenticated_paper_actions(
    view: dict, *, report_id: str, search_again_enabled: bool = True
) -> dict:
    """Add same-origin paper/source links without mutating the projection.

    ``search_again_enabled`` is False once marks are released for the paper's
    assessment (assessment_marks.py): no "Search Again" action is attached.
    """
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
        "report_href": f"/report/{report_id}",
        "pdf_href": f"/report/{report_id}/export.pdf",
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
                "label": "Upload Source",
                "href": (
                    f"/report/{report_id}/citation/"
                    f"{quote(str(citation.get('claim_id') or ''), safe='')}/source/upload"
                ),
            }
        for member in citation.get("members", []):
            if (search_again_enabled and member.get("reference_id")
                    and member_search_incomplete(member)):
                member["search_again_action"] = {
                    "enabled": True,
                    "href": (
                        f"/report/{report_id}/reference/"
                        f"{quote(str(member['reference_id']), safe='')}/search-again"
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
                if (member.get('source_action') or {}).get('status') in {'verified_public_source_available', 'public_candidate_available'}:
                    continue
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


def member_search_incomplete(member: dict) -> bool:
    """A retrieval member whose search did not finish and can be run again."""
    if (member.get("status") != "source_unavailable"
            or member.get("coverage_level") not in {"unavailable", "abstract_only"}
            or _member_is_media(member)):
        return False
    return (member.get("reason_code") == "full_text_search_incomplete"
            or (member.get("reference_identity") or {}).get("status") == "search_incomplete")


def _member_accepts_upload(member: dict) -> bool:
    # A reference that cannot be verified keeps its upload, so the reader can
    # supply the source if the search was wrong; it is only left out of the
    # suggested uploads (owner decision 2026-09-30).
    return member.get('coverage_level') != 'full_text' and not _member_is_media(member)


def _member_is_media(member: dict) -> bool:
    from app.services.source_type import is_traditional_media
    source = member.get("source") or {}
    return (source.get("source_kind") in {"traditional_media", "video", "podcast_episode"}
            or is_traditional_media(str(source.get("raw_reference") or "")))


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
        raw = (str(finding.get('citation_text') or '') if finding.get('finding_type') in {'required_quotation_locator_missing', 'body_title_style'}
               else str((finding.get("source") or {}).get("raw_reference") or ""))
        entry, _ = _field_characters(raw)
        difference = finding["field_difference"]
        field, _ = _field_characters(difference.get("submitted_value") or "")
        if finding.get('finding_type') == 'required_quotation_locator_missing':
            # Mark the retained citation marker, never the borrowed passage.
            marker = str(finding.get('citation_marker') or '')
            candidates = re.findall(r'\([^()\n]*\)', marker or raw)
            if not candidates and marker and marker in raw:
                # An MLA narrative citation may consist of the author alone.
                candidates = [marker]
            if len(candidates) != 1:
                continue
            difference = {**difference, 'submitted_value': candidates[0]}
            field, _ = _field_characters(candidates[0])
        if len(entry) < 12 or not field:
            continue
        entry_matches = [m.start() for m in re.finditer(f"(?={re.escape(entry)})", surface)]
        field_matches = [m.start() for m in re.finditer(f"(?={re.escape(field)})", entry)]
        literal_field = str(difference.get('submitted_value') or '')
        if len(field_matches) > 1 and literal_field and raw.count(literal_field) == 1:
            # A URL slug can repeat a title/year after normalization. The
            # exact literal submitted field disambiguates its original offset.
            prefix, _ = _field_characters(raw[:raw.index(literal_field)])
            field_matches = [len(prefix)]
        # A URL may repeat the publication year. Only a unique date in the
        # bibliographic prefix before an exactly identified title is eligible.
        if difference.get("field_name") == "year" and len(field_matches) > 1:
            title, _ = _field_characters((finding.get("source") or {}).get("title") or "")
            if title and entry.count(title) == 1:
                field_matches = [i for i in field_matches if i + len(field) <= entry.index(title)]
        if finding.get('finding_type') == 'duplicate_reference_entry' and len(entry_matches) > 1:
            # Identical repeated entries cannot be uniquely located singly.
            # Require a complete occurrence census matching the retained peers;
            # each receives the same group finding at its distinct occurrence.
            peers = [p for p in finding.get('related_references') or []
                     if _field_characters(p.get('raw_reference') or '')[0] == entry]
            peer_ids = [p.get('reference_id') for p in peers]
            if len(entry_matches) == len(peer_ids) == len(set(peer_ids)) and finding.get('reference_id') in peer_ids:
                entry_matches = [entry_matches[peer_ids.index(finding['reference_id'])]]
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
            ("citation_sha256" if finding.get('finding_type') in {'required_quotation_locator_missing', 'body_title_style'} else "reference_sha256"): hashlib.sha256(raw.encode()).hexdigest(),
            "field_sha256": hashlib.sha256(str(difference["submitted_value"]).encode()).hexdigest(),
            "method": ("unique_complete_citation_and_field_characters_v1" if finding.get('finding_type') in {'required_quotation_locator_missing', 'body_title_style'} else "unique_complete_reference_and_field_characters_v1"),
        }


_REFERENCE_DOI_TEXT_RE = re.compile(r"\bdoi:\s*(10\.\d{4,9}/\S+)", re.IGNORECASE)


def _derived_doi_link_overlays(document, rectangles, raw, existing_hrefs):
    """Locate a typed `doi:` address so the reader can open it.

    Paper-side reference links normally come from native PDF/DOCX annotations.
    A reference that writes its identifier as plain text has no annotation, so
    the address is present but unopenable. This finds that exact text inside
    the reference's own geometry and offers the canonical resolver. It adds no
    submitted-link provenance and never fires when a native link already
    carries the same identifier. Wrapped text that cannot be located simply
    yields no overlay.
    """
    overlays: list[dict] = []
    for match in _REFERENCE_DOI_TEXT_RE.finditer(raw or ""):
        label = match.group().rstrip(".,;)")
        identifier = match.group(1).rstrip(".,;)")
        href = "https://doi.org/" + identifier
        if not _safe_reference_href(href):
            continue
        if any(identifier.casefold() in str(value).casefold() for value in existing_hrefs):
            continue
        for rectangle in rectangles:
            box = fitz.Rect(*(rectangle[key] for key in ("x0", "y0", "x1", "y1")))
            try:
                areas = document[rectangle["page_index"]].search_for(label, clip=box)
            except (ValueError, RuntimeError):
                areas = []
            for area in areas or []:
                overlays.append({
                    "href": href,
                    "derived_address": "reference_doi_text",
                    "rectangle": {
                        "page_index": rectangle["page_index"],
                        "x0": area.x0, "y0": area.y0, "x1": area.x1, "y1": area.y1,
                    },
                })
            if overlays:
                break
    return overlays


def normalize_reference_findings(findings: list[dict]) -> list[dict]:
    """Present stored reference findings under the current categories.

    Display only; stored findings are not rewritten. The former "potentially
    fabricated" flag is shown as "Cannot be verified" (a subset of it: every
    reference it flagged had also gone unlocated). One wrong DOI is one
    submitted-link issue however many checks noticed it. A reference that
    cannot be verified has no located record of its own, so a comparison with
    whatever its DOI resolved to is not a difference in its details.
    """
    from app.services.reference_verification import FINDING_TEXT, FINDING_TYPE, POLICY
    rows = []
    for finding in findings or []:
        if finding.get('finding_type') == 'potentially_fabricated_reference':
            finding = {**finding, 'finding_type': FINDING_TYPE, 'finding': FINDING_TEXT,
                       'legacy_finding_type': 'potentially_fabricated_reference',
                       'presentation_policy_version': POLICY}
        rows.append(finding)
    identifier = {f.get('reference_id') for f in rows if f.get('finding_type') == 'reference_identifier_conflict'}
    wrong_doi = identifier | {f.get('reference_id') for f in rows
                              if f.get('finding_type') == 'doi_registers_a_different_title'}
    unverified = {f.get('reference_id') for f in rows if f.get('finding_type') in UNVERIFIED_FINDINGS}
    return [f for f in rows
            if not (f.get('finding_type') == 'doi_registers_a_different_title' and f.get('reference_id') in identifier)
            and not (f.get('finding_type') in REFERENCE_DIFFERENCE_FINDINGS and f.get('reference_id') in unverified)
            # The identity-conflict comparison may be with the work the wrong
            # DOI names; the title-anchored field comparison cannot be.
            and not (f.get('finding_type') == 'bibliographic_conflict' and f.get('reference_id') in wrong_doi)]


def project_reference_flags(view: dict, document, paper_hash: str) -> dict:
    """Display retained findings on exact fields; never rewrite historical output."""
    result = deepcopy(view)
    result['reference_practice'] = normalize_reference_findings(result.get('reference_practice'))
    withheld = [f for f in result.get('reference_practice', [])
        if f.get('finding_type') == 'bibliographic_conflict'
        and not _reportable_bibliographic_difference(f.get('field_difference') or {})]
    if withheld:
        result['withheld_bibliographic_conflicts'] = withheld
        result['reference_practice'] = [f for f in result.get('reference_practice', []) if f not in withheld]
    members = [m for c in result.get('citations', []) for m in c.get('members', [])]
    from app.services.sentence_splitter import split_sentences
    paper_sentences = None
    # Recover display geometry at exact character boundaries when PDF word
    # tokenization joins a citation to the next sentence. Never alter claims.
    for citation in result.get('citations', []):
        text = str(citation.get('student_text') or '')
        recovered_context = False
        if re.fullmatch(r'\(as cited in [^()\n]+\)\.', text, re.I):
            if paper_sentences is None:
                paper_sentences = split_sentences('\n'.join(p.get_text(sort=True) for p in document))
            normalized, _ = _field_characters(text)
            matches = [s for s in paper_sentences if _field_characters(s)[0].endswith(normalized)
                       and re.search(r'[?!]\s*\(as cited in ', s, re.I) and len(s) <= 4000]
            if len(matches) == 1:
                text = matches[0]
                recovered_context = True
        if (citation.get('paper_location') or {}).get('rectangles') and not recovered_context:
            continue
        if len(text) < 30:
            continue
        probe = {'finding_type':'body_title_style', 'citation_text':text,
                 'field_difference':{'submitted_value':text}}
        _attach_reference_field_geometry({'reference_practice':[probe]}, document, paper_hash)
        if probe.get('rectangles'):
            if recovered_context:
                citation['display_student_text'] = _normalize_display_text(text)
                citation['display_context_provenance'] = probe['geometry_provenance']
            citation['paper_location'] = {**(citation.get('paper_location') or {}),
                'localization_level':'exact_rectangle', 'rectangles':probe['rectangles'],
                'geometry_provenance':probe['geometry_provenance']}
    sources = {m['reference_id']: m.get('source') or {} for m in members if m.get('reference_id')}
    for finding in result.get('reference_practice') or []:
        if finding.get('reference_id') and finding.get('source'):
            sources.setdefault(finding['reference_id'], finding['source'])
    for row in result.get('uncited_link_checks') or []:
        for observation in row.get('observations') or []:
            sources.setdefault(observation.get('reference_id'), row.get('reference') or {})
    # Locate every bibliography entry so each receives a Reference N window,
    # including uncited references without findings. Older views predate the
    # stored bibliography; their verified locator inventory carries the raw
    # text of every reference. Finding logic below keeps using ``sources``.
    catalog_sources = dict(sources)
    for row in result.get('bibliography') or []:
        if row.get('reference_id') and row.get('source'):
            catalog_sources.setdefault(row['reference_id'], row['source'])
    if not result.get('bibliography'):
        entries = [entry for entry in (result.get('submitted_locator_inventory') or {}).get('entries') or []
                   if entry.get('reference_id') and entry.get('submitted_reference')]
        for entry in entries:
            catalog_sources.setdefault(entry['reference_id'], {'raw_reference': entry['submitted_reference']})
        if entries:
            result['bibliography'] = [{'reference_id': entry['reference_id'],
                                       'source': catalog_sources[entry['reference_id']]} for entry in entries]
    locations = {}
    probes = [{'source':source, 'field_difference':{'submitted_value':source.get('raw_reference') or ''}}
              for source in catalog_sources.values()]
    _attach_reference_field_geometry({'reference_practice':probes}, document, paper_hash)
    sources_by_probe = catalog_sources
    for rid, probe in zip(sources_by_probe, probes):
        if probe.get('rectangles'):
            locations[rid] = {'index':len(locations), 'rectangles':probe['rectangles'],
                              'geometry_provenance':probe['geometry_provenance']}
            links, observations = [], []
            raw = sources_by_probe[rid].get('raw_reference') or ''
            normalized, offsets = _field_characters(raw)
            for rectangle in probe['rectangles']:
                box = fitz.Rect(*(rectangle[k] for k in ('x0','y0','x1','y1')))
                for link in document[rectangle['page_index']].get_links():
                    href = str(link.get('uri') or '')
                    area = fitz.Rect(link['from'])
                    if not box.intersects(area) or not _safe_reference_href(href):
                        continue
                    if href not in links:
                        links.append(href)
                    label = document[rectangle['page_index']].get_textbox(area).strip()
                    folded, _ = _field_characters(label)
                    starts = [m.start() for m in re.finditer(f'(?={re.escape(folded)})', normalized)] if len(folded)>=3 else []
                    repeated = (len(starts)>1 and all(
                        not raw[offsets[a+len(folded)-1]+1:offsets[b]].strip(' .\n\t')
                        for a,b in zip(starts,starts[1:])))
                    if len(starts)==1 or repeated:
                        start=offsets[starts[0]]; end=offsets[starts[-1]+len(folded)-1]+1
                        observation={'href':href,'start':start,'end':end,'label':raw[start:end],
                            'repeated_label':repeated,
                            'rectangle':{'page_index':rectangle['page_index'],'x0':area.x0,'y0':area.y0,'x1':area.x1,'y1':area.y1}}
                        if observation not in observations:
                            observations.append(observation)
            if len(links)>1:
                observations=[o for o in observations if not o.get('repeated_label')]
            sources_by_probe[rid]['submitted_hyperlinks'] = links
            sources_by_probe[rid]['submitted_hyperlink_labels'] = observations
            # A derived overlay makes a typed `doi:` address openable. It is
            # deliberately kept out of submitted_hyperlinks/labels: the student
            # wrote text, not a hyperlink, and those fields carry submitted-link
            # provenance that drives link findings.
            locations[rid]['links'] = observations + _derived_doi_link_overlays(
                document, probe['rectangles'], raw, links
            )
            sources_by_probe[rid]['hyperlink_provenance'] = probe['geometry_provenance']
    result.setdefault('paper_surface', {})['reference_locations'] = locations
    for member in members:
        source = sources.get(member.get('reference_id')) or {}
        if source.get('raw_reference') == (member.get('source') or {}).get('raw_reference'):
            member['source'].update({key:source[key] for key in ('submitted_hyperlinks','submitted_hyperlink_labels','hyperlink_provenance') if key in source})
    sources.update({row['reference_id']:row['source'] for row in result.get('reference_practice') or [] if row.get('reference_id') and row.get('source')})
    for row in result.get('uncited_link_checks') or []:
        for observation in row.get('observations') or []:
            sources.setdefault(observation.get('reference_id'),row.get('reference') or {})
    findings = result.setdefault('reference_practice', [])
    from app.services.report_layers import topical_mismatch
    findings[:] = [f for f in findings if f.get('finding_type') != 'source_topical_mismatch']
    for citation in result.get('citations', []):
        for member in citation.get('members', []):
            source = member.get('source') or {}
            if not source.get('raw_reference') or not topical_mismatch(member, citation):
                continue
            scope = member['abstract_relevance']['scope_assessment']
            findings.append({
                'finding_type': 'source_topical_mismatch',
                'reference_id': member.get('reference_id'), 'source': deepcopy(source),
                'finding': 'The abstract appears unrelated to the topic attributed to this source. ' + scope['rationale'],
                'citation_text': citation['student_text'],
                'abstract_text': member['best_evidence']['text'],
                'scope_rationale': scope['rationale'],
                'field_difference': {'submitted_value': source['raw_reference']},
                'rectangles': [],
            })
    from app.services.submitted_link_display import reference_link_findings, without_repeated_identifier_conflicts
    findings[:]=[f for f in findings if f.get('finding_type') not in {'submitted_link_issue', 'reference_author_conflict'}]
    findings.extend(without_repeated_identifier_conflicts(
        reference_link_findings(result.get('submitted_link_observations'), sources), findings))
    existing = {f.get('reference_id') for f in findings if f.get('finding_type') == 'duplicate_citation_key'}
    for member in members:
        for finding in member.get('reference_findings') or []:
            if finding.get('finding_type') != 'duplicate_citation_key':
                continue
            peers = finding.get('reference_ids') or []
            for rid in peers:
                source = sources.get(rid)
                if rid in existing or not source:
                    continue
                findings.append({
                    'finding_type': 'duplicate_citation_key', 'reference_id': rid,
                    'source': deepcopy(source),
                    'finding': 'These references share the same author and year. Distinguish the works consistently in the references and in-text citations using the required citation style.',
                    'related_references': [deepcopy(sources[r]) for r in peers if r in sources],
                    'field_difference': {'field_name': 'year', 'submitted_value': source.get('year')},
                    'rectangles': [], 'localization_status': 'not_assessed',
                    'retained_finding_id': finding.get('finding_id'),
                })
                existing.add(rid)
    # Flag only entries that must move; a pervasively unsorted list is stated
    # once. Historical findings recover the observed order from reference ids.
    from app.services.reference_formatting import reference_order_projection
    from app.services.report_references import reference_numbers, reference_ordinal
    findings[:], pervasive_order = reference_order_projection(findings, reference_ordinal)
    result['pervasive_reference_order'] = bool(result.get('pervasive_reference_order')) or pervasive_order
    _attach_reference_field_geometry(result, document, paper_hash)
    result.pop('reference_numbers', None)
    result['reference_numbers'] = reference_numbers(result)
    # References no citation links to, except a film or programme the paper
    # names by its title (owner decisions 2026-09-30).
    from app.services.report_references import reference_catalog
    words = (result.get('paper_surface') or {}).get('selectable_words')
    paper_key = (_paper_key(result['paper_surface']) if words
                 else _fold_for_titles(' '.join(page.get_text() for page in document)))
    result['uncited_reference_ids'] = [
        entry['reference_id'] for entry in reference_catalog(result, result['reference_numbers'])
        if not entry.get('citation_numbers') and not _media_named_in_paper(entry.get('source') or {}, paper_key)]
    result.pop('role_summaries', None)   # the former per-audience summaries
    summary = result['summary'] = _build_report_summary(
        citations=result.get('citations') or [], overview=result.get('overview') or {},
        pervasive_hanging_indent=bool(result.get('reference_practice_summary')),
        reference_practice=findings, require_paper_flags=True,
        pervasive_reference_order=result['pervasive_reference_order'],
        reference_numbers=result['reference_numbers'],
        patchwriting_passages=result.get('patchwriting_passages'),
        uncited_reference_ids=result.get('uncited_reference_ids'),
    )
    # These are citation/reference consistency recommendations, not intent findings.
    for item in list(summary['academic_practice']):
        if (item.get('kind') == 'duplicate_citation_key' if isinstance(item, dict)
                else 'same author and year' in item):
            summary['academic_practice'].remove(item)
            summary['reference_formatting'].append(item)
    result['presentation_projection'] = {'version': 'reference-flags-v9', 'paper_sha256': paper_hash}
    return result


def attach_quotation_difference_geometry(view: dict, pdf_content: bytes) -> dict:
    """Bind report-only quotation differences to exact words on the paper PDF."""
    result = deepcopy(view)
    document = fitz.open(stream=pdf_content, filetype="pdf")
    try:
        _attach_reference_field_geometry(result, document, hashlib.sha256(pdf_content).hexdigest())
        for citation in result.get("citations") or []:
            _restore_visible_paper_hyphens(citation, document, hashlib.sha256(pdf_content).hexdigest())
            differences = list(citation.get("quotation_differences") or [])
            # Quoted wording that was not located in the source at all produces
            # no per-token difference, yet it is the clearest academic-practice
            # case to show. Mark the whole quoted span rather than nothing.
            unlocated = any(
                (member.get("quotation_check") or {}).get("outcome") == "no_span_located"
                and (member.get("quotation_check") or {}).get("attention")
                for member in citation.get("members") or []
            )
            location = citation.get("paper_location") or {}
            if (not differences and not unlocated) or location.get(
                "localization_level"
            ) != "exact_rectangle":
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
            if not difference_indexes and unlocated:
                difference_indexes = set(range(len(quote_tokens)))
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
    if result.get('patchwriting_passages'):
        from app.services.patchwriting_report import attach_passage_geometry
        attach_passage_geometry(result['patchwriting_passages'], pdf_content, result.get('citations') or [])
    result.pop('role_summaries', None)
    result['summary'] = _build_report_summary(
        citations=result.get('citations') or [], overview=result.get('overview') or {},
        pervasive_hanging_indent=bool(result.get('reference_practice_summary')),
        reference_practice=result.get('reference_practice') or [],
        require_paper_flags=True,
        pervasive_reference_order=bool(result.get('pervasive_reference_order')),
        patchwriting_passages=result.get('patchwriting_passages'),
        uncited_reference_ids=result.get('uncited_reference_ids'),
    )
    return result


def _missing_reference_members(extraction, start: int, end: int) -> list[str]:
    """Name exact parsed author/year members, not arbitrary unresolved years."""
    from app.services.citation_extractor import APA_MEMBER_RE, APA_NARRATIVE_RE
    missing = []
    for citation in extraction.citations:
        member = str(citation.marker_member or "").strip()
        narrative = APA_NARRATIVE_RE.fullmatch(member) if citation.marker_type == "narrative" else None
        if narrative:
            retained = getattr(getattr(extraction, 'reference_consistency', None), 'findings', [])
            if getattr(citation, 'is_secondary', False) or not any(
                finding.finding_type == 'missing_reference_entry'
                and finding.passage_start == citation.passage_start
                and finding.passage_end == citation.passage_end
                and finding.marker_text_sha256 == hashlib.sha256(citation.citation_marker.encode()).hexdigest()
                and not finding.candidate_reference_ids
                for finding in retained
            ):
                continue
            member = f"{narrative[1]}, {narrative[2]}"
        if (
            citation.link_status == "missing_reference"
            and not citation.candidate_reference_ids
            and (citation.marker_type == "parenthetical" or narrative is not None)
            and APA_MEMBER_RE.fullmatch(member)
            and citation.passage_start < end and start < citation.passage_end
        ):
            missing.append(member)
    # Sentence grouping may retain only the first citation's marker_member.
    # The immutable marker census preserves each independently unresolved
    # marker; use only exact single-member parentheticals inside this passage.
    for marker in getattr(extraction, 'citation_marker_census', []) or []:
        text = str(marker.text or '')
        member = text[1:-1].strip() if text.startswith('(') and text.endswith(')') else ''
        if (marker.link_status == 'missing_reference'
                and not marker.reference_ids and not marker.candidate_reference_ids
                and marker.marker_type == 'parenthetical' and marker.member_count == 1
                and start <= marker.passage_start < marker.passage_end <= end
                and APA_MEMBER_RE.fullmatch(member)):
            missing.append(member)
    return list(dict.fromkeys(missing))


def _restore_retained_continuations(passages: list[dict]) -> list[dict]:
    """Reassemble only exact overlapping package slices; never fetch new text."""
    result = []
    for parent in passages:
        if parent.get('parent_passage_id'):
            continue
        text = str(parent.get('excerpt') or '')
        start = parent.get('character_start')
        end = parent.get('character_end')
        if isinstance(start, int) and isinstance(end, int):
            children = sorted((p for p in passages if p.get('parent_passage_id') == parent.get('passage_id')),
                              key=lambda p:p.get('character_start', -1))
            for child in children:
                if any(child.get(k) != parent.get(k) for k in ('representation_id','content_sha256','page_index')):
                    continue
                offset = child.get('character_start', -1) - start
                fragment = str(child.get('excerpt') or '')
                overlap = len(text) - offset
                if offset < 0 or overlap <= 0 or offset + len(fragment) > end - start:
                    continue
                if text[offset:] != fragment[:overlap]:
                    continue
                text += fragment[overlap:]
            # The parent's own hash must agree before widening its display.
            if len(text) == end-start and hashlib.sha256(text.encode()).hexdigest() == parent.get('passage_text_sha256'):
                parent = {**parent, 'excerpt':text, 'excerpt_truncated':False}
        result.append(parent)
    return result


def _joint_report_selection(payload, claim_text, passages, eligible, preferred, baseline, trace, *, source_title=''):
    """Consume only a fresh joint receipt; never weaken protected display priority."""
    receipt = payload.get('joint_evidence_selection')
    if not receipt or isinstance(receipt, dict) and receipt.get('status') == 'not_run':
        return baseline, None, trace

    def fallback(reason):
        return baseline, None, {**trace, 'joint_selection_status': 'fallback',
                                'joint_selection_reason': reason}

    if not isinstance(receipt, dict) or receipt.get('source_title') != source_title:
        return fallback('joint_receipt_invalid')
    package = payload.get('authoritative_evidence_package') or {}
    if any(payload.get(key) != package.get(key) for key in ('source_identity', 'coverage', 'source_binding')):
        return fallback('joint_package_binding_mismatch')
    originals = {pid: str(p.get('excerpt') or p.get('text') or '')
                 for pid, p in passages.items() if not p.get('parent_passage_id')}
    joint_context = payload.get('joint_selection_context') or {}
    original_ids = joint_context.get('passage_ids') if isinstance(joint_context, dict) else None
    if (not isinstance(original_ids, list) or any(not isinstance(pid, str) for pid in original_ids)
            or len(original_ids) != len(set(original_ids))
            or set(original_ids) != set(originals)):
        return fallback('joint_original_passage_ids_mismatch')
    originals = {pid: originals[pid] for pid in original_ids}
    if any(hashlib.sha256(text.encode()).hexdigest() != passages[pid].get('passage_text_sha256')
           for pid, text in originals.items()):
        return fallback('joint_original_text_unavailable')
    try:
        from app.services.joint_evidence_selection import validate_joint_projection
        valid = validate_joint_projection(
            receipt, claim_text=claim_text,
            passages=originals,
            source_identity=payload.get('source_identity') or {},
            coverage=payload.get('coverage') or {}, gate=payload.get('passage_relevance_gate') or {},
            source_binding=payload.get('source_binding'), claim_context=payload.get('joint_selection_context') or {},
        )
    except (ImportError, ValueError, TypeError, KeyError, AttributeError):
        return fallback('joint_receipt_invalid')
    if receipt.get('status') != 'complete' or not valid:
        return fallback('joint_receipt_invalid')
    rows = receipt.get('selected') or []
    ids = [row.get('passage_id') for row in rows]
    eligible_by_id = {p['passage_id']: p for p in eligible}
    if len(ids) > 3 or len(ids) != len(set(ids)) or any(pid not in eligible_by_id for pid in ids):
        return fallback('joint_selection_ineligible')
    protected = set(preferred) & set(eligible_by_id)
    if protected and (not ids or ids[0] not in protected or not protected <= set(ids)):
        return fallback('joint_protected_priority_conflict')
    candidates = {row['passage_id']: row['source_span'] for row in rows}
    if any(pid not in candidates for pid in ids):
        return fallback('joint_receipt_invalid')
    return [eligible_by_id[pid] for pid in ids], {pid: candidates[pid] for pid in ids}, {
        **trace, 'version': receipt['version'], 'joint_selection_status': 'applied',
        'legacy_selection': trace, 'selected': [{'passage_id': row['passage_id'], 'reason': row['reason']} for row in rows],
        'omitted': [{'passage_id': pid, 'reason': 'joint_not_selected'} for pid in eligible_by_id if pid not in ids],
        'support_assessed': False,
    }


def _available_member(
    member: dict,
    record: VerificationReportRecord,
    reference,
    claim,
    reference_layout=None,
) -> dict:
    view = _with_sentence_evidence(_available_member_view(member, record, reference, claim, reference_layout),
                                   record, claim)
    # Secondary citation from GLM's sentences (owner request 2026-09-29).
    from app.services.secondary_citation import secondary_citation
    flag = secondary_citation(view.get("evidence_sentences") or [], view.get("source") or {})
    return {**view, "secondary_citation": flag} if flag else view


def evidence_sentence_key(page_index, absolute_start: int, absolute_end: int) -> str:
    """One sentence's identity across the selector and the judge: page and exact position."""
    return f"{page_index}:{absolute_start}:{absolute_end}"


def _with_sentence_evidence(view: dict, record, claim) -> dict:
    """The citation's one evidence list, when GLM chose it (owner decision 2026-09-28)."""
    payload = record.report_payload or {}
    view = {**view, "verification_report_id": str(record.id)}
    selection = payload.get("sentence_evidence") or {}
    if selection.get("status") not in {"selected", "empty"}:
        return view
    from app.services.text_quality import readable_text
    passages = {item.get("passage_id"): item for item in payload.get("passages") or []}
    items = []
    for item in selection.get("items") or []:
        passage = passages.get(item.get("passage_id")) or {}
        start = int(passage.get("character_start") or 0)
        page = passage.get("page_label") or (item["page_index"] + 1 if item.get("page_index") is not None else None)
        items.append({"key": evidence_sentence_key(item.get("page_index"), start + item["passage_start"],
                                                   start + item["passage_end"]),
                      "text": readable_text(item["text"]), "page": page, "reason": item.get("reason")})
    view["evidence_sentences"] = items
    if not items and not view.get("availability"):
        # Existing approved wording for no fitting evidence.
        view["availability"] = (
            "No clearly relevant passage was found for this source's part of the citation. Its title, study setting or metadata may be relevant; check the source manually."
            if len(getattr(claim, "reference_ids", []) or []) > 1
            else "No clearly relevant passage was found. Check the source manually; this does not establish that evidence is absent.")
    return view


def _available_member_view(
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
    passages_by_id.update({item['passage_id']:item for item in _restore_retained_continuations(package.get('passages', []))})
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
            + [item['passage_id'] for item in ((record.report_payload or {}).get('passage_relevance_gate') or {}).get('assessments',[])
               if item.get('passage_id') in passages_by_id and item.get('relevance') in {'relevant','partially_relevant'}]
        )
    )
    gate = (record.report_payload or {}).get('passage_relevance_gate') or {}
    if gate.get('status') != 'complete':
        # A failed advisory pass must not hide responsive retained union entries.
        # They remain explicitly unassessed, not promoted to relevance findings.
        displayed_passage_ids = list(dict.fromkeys([*displayed_passage_ids, *passages_by_id]))
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
    # A title/byline that merely repeats this source's reference is not content
    # evidence. Preserve document-level study metadata for aggregate citations.
    if len(getattr(claim, 'reference_ids', []) or []) == 1:
        identity_terms = set(_display_terms(' '.join(str(getattr(reference,k,'') or '') for k in ('title','author'))))
        title_terms = set(_display_terms(str(getattr(reference, 'title', '') or '')))
        assessed_primary = [p for p in assessed_primary if not (
            p.get('passage_id') not in preferred_passage_ids
            and len(str(p.get('excerpt') or '')) < 350
            and (set(_display_terms(str(p.get('excerpt') or ''))) <= identity_terms
                 or (assessments.get(p.get('passage_id'), {}).get('evidence_role') == 'document_level_member_evidence'
                     and len(title_terms) >= 3
                     and title_terms <= set(_display_terms(str(p.get('excerpt') or '')))
                     and not re.search(r'[.!?]\s', str(p.get('excerpt') or ''))))
        )]
    excluded_only = bool(ordered) and not assessed_primary and all(
        _display_passage_is_metadata(item) for item in ordered
    )
    from app.services.evidence_display_selection import select_display_passages
    source_vocabulary = '\n'.join(str(p.get('excerpt') or '') for p in passages_by_id.values())
    def displayed_context(p):
        view = _passage_view(p, assessments.get(p.get('passage_id')), claim_text=claim.text,
                             source_vocabulary=source_vocabulary)
        return view.get('context_text') or view['display_text']
    selected_items, selection_trace = select_display_passages(
        assessed_primary, claim.text, assessments, preferred_passage_ids,
        terms=_display_terms, excerpt=_responsive_display_excerpt,
        context=displayed_context,
    )
    selection_trace['package_passage_ids'] = list(passages_by_id)
    selected_items, joint_spans, selection_trace = _joint_report_selection(
        record.report_payload or {}, claim.text, passages_by_id, assessed_primary,
        preferred_passage_ids, selected_items, selection_trace, source_title=str(getattr(reference, 'title', '') or ''),
    )
    best_item = selected_items[0] if selected_items else None
    best = (
        _passage_view(
            best_item,
            assessments.get(best_item.get("passage_id")),
            claim_text=claim.text,
            source_vocabulary=source_vocabulary,
        )
        if best_item
        else None
    )
    additional_items = selected_items[1:]
    additional = [
        _passage_view(
            item,
            assessments.get(item.get("passage_id")),
            claim_text=claim.text,
            source_vocabulary=source_vocabulary,
        )
        for item in additional_items
    ]
    evidence_extracts = None
    if joint_spans is not None:
        evidence_extracts = []
        for item, view in zip(selected_items, [best, *additional]):
            span = _normalize_source_display_text(joint_spans[item['passage_id']], source_vocabulary)
            blocks = [_normalize_source_display_text(block, source_vocabulary)
                      for block in re.split(r'\n\s*\n', item.get('excerpt', '')) if block.strip()]
            containing = [block for block in blocks if span in block]
            context = _bounded_display_context(containing[0] if len(containing) == 1 else
                _normalize_source_display_text(item.get('excerpt', ''), source_vocabulary), span)
            view.update(display_text=span, context_text=context if context != span else '')
            evidence_extracts.append(view)
    targets = _member_quotation_targets(claim, reference.reference_id)
    quotation = _check_view(package.get("quotation_check") or {}, _QUOTATION_ATTENTION)
    positive = _positive_quote_check(targets, [str(item.get("text") or item.get("excerpt") or "") for item in passages_by_id.values()])
    if positive:
        quotation = positive
    if targets and not positive and quotation.get("outcome") in _QUOTATION_ATTENTION:
        differences = _quotation_difference_diagnostics(
            _normalize_display_text(claim.text), list(passages_by_id.values()),
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
    provisional = (package.get('source_identity') or {}).get('status') == 'uncertain'
    if provisional:
        quotation = _check_view({}, _QUOTATION_ATTENTION)
        locator = _check_view({}, _LOCATOR_ATTENTION)
        quotation_ready = locator_ready = False
        identity_notes = (package.get('source_identity') or {}).get('limitations') or []
        availability_reason = ' '.join(dict.fromkeys([
            'Possible source match—identity not confirmed', *identity_notes, availability_reason,
        ]))
    # The scope record sits at the payload root, beside the evidence package
    # rather than inside it; `package` is the authoritative-evidence sub-dict.
    scope_record = (record.report_payload or {}).get("source_scope_assessment") or {}
    scope_excerpt = str(scope_record.get("excerpt") or "")
    scope_assessment = scope_record.get("assessment") or {}
    if scope_assessment.get("scope_assessment"):
        # The whole-document term counts are evidence about the judgment,
        # not part of it, so they travel beside the model's own fields
        # where the report layer reads them.
        scope_assessment = {**scope_assessment, "scope_assessment": {
            **scope_assessment["scope_assessment"],
            "claim_terms_present": scope_record.get("claim_terms_present"),
            "claim_terms_total": scope_record.get("claim_terms_total"),
        }}
    return {
        "status": "evidence_available",
        "identity_status": "possible_match" if provisional else "verified",
        "coverage_level": (package.get("coverage") or {}).get(
            "level", "unavailable"
        ),
        # The text the scope comparison was made against, carried separately
        # from the displayed passage: one says what this work is about, the
        # other says what was cited. The report layer binds the judgment to
        # this text, so an empty excerpt simply yields no comparison.
        "scope_source": {"text": scope_excerpt} if scope_excerpt else {},
        "abstract_relevance": scope_assessment if scope_excerpt else {},
        "reference_id": reference.reference_id,
        "source": _reference_view(reference, reference_layout),
        "alternate_edition": package.get("alternate_edition"),
        "availability": (
            availability_reason if provisional else
            "The retained extract contains publisher or catalog information, not a passage from the work's contents. The cited content could not be checked."
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
        **({"evidence_extracts": evidence_extracts} if evidence_extracts is not None else {}),
        "display_selection": selection_trace,
        "additional_evidence": additional,
        "relevance_status": (
            "not_assessed" if provisional else
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
        '<div class="notice"><strong>Paper View</strong><br>'
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
    # An abstract is already short and bounded. Collapsing it to the single
    # responsive sentence hid most of what the reader needs to judge the
    # citation, so always show the whole abstract and retain the responsive
    # sentence separately for emphasis.
    responsive_excerpt = (excerpt if bound_assessment and relevance_connected
                          and _complete_abstract_excerpt(abstract_text, excerpt) else "")
    abstract_display = abstract_text
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
    evidence_note = ''  # Abstract coverage is not a partial-relevance flag.
    abstract_evidence = (
        {
            "text": abstract_text,
            "display_text": abstract_display,
            "context_text": abstract_text if abstract_display != abstract_text else "",
            "responsive_excerpt": responsive_excerpt,
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
            "This is a media reference. Automated source retrieval and checks are not available."
            if getattr(reference, "source_kind", "") == "traditional_media"
            else "" if abstract_available else "Source Not Retrieved"
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
        "url": getattr(reference, "url", ""),
        "source_kind": getattr(reference, "source_kind", "unknown"),
        "raw_reference": reference.raw_ref,
        "text_style_spans": (
            [item.model_dump(mode="json") for item in reference_layout.text_style_spans]
            if reference_layout is not None
            else []
        ),
    }


def _reference_source_action(reference) -> dict:
    """Expose a submitted DOI/HTTPS route; this is not identity validation."""
    from app.services.ref_field_extractor import decode_doi
    if reference.doi:
        return {
            "status": "canonical_landing_available",
            "label": "Open source record",
            "enabled": True,
            "href": f"https://doi.org/{quote(decode_doi(reference.doi.strip()), safe='/():._-')}",
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


def _remaining_source_context(ordered: list[dict], selected: list[dict]) -> list[dict]:
    """Expose retained, non-metadata context without a new relevance decision.

    Inputs belong to one already validated member package. Do not join sources,
    retrieve text, or promote retained-but-unselected passages to primary evidence.
    """
    seen = {p.get("passage_id") for p in selected}
    result = []
    for passage in ordered:
        pid = passage.get("passage_id")
        if not pid or pid in seen or _display_passage_is_metadata(passage):
            continue
        seen.add(pid)
        view = _passage_view(passage)
        view["evidence_note"] = "Retained source context for manual comparison; inclusion does not establish that it addresses this statement."
        if passage.get("retrieval_method") == "explanatory_note_context":
            view["evidence_note"] = "Explanatory source note. " + view["evidence_note"]
        result.append(view)
    return result


def _passage_view(
    passage: dict,
    assessment: dict | None = None,
    *,
    claim_text: str = "",
    source_vocabulary: str = "",
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
    role_note = ''  # Role interpretation wording is deferred post-prototype.
    relevance_note = ''  # Advisory selection labels are not ordinary report judgments.
    note = " ".join(value for value in (relevance_note, role_note) if value)
    normalized_text = _normalize_source_display_text(passage.get("excerpt", ""), source_vocabulary)
    from app.services.evidence_display_selection import bound_observation
    observation = bound_observation(passage, assessment or {}, claim_text)
    if observation and not _closed_display_delimiters(observation['source_span']):
        observation = None
    raw_text = str(passage.get('excerpt') or '')
    if observation and not any(_normalize_display_text(observation['source_span']) in block
                               for block in _display_prose_blocks(raw_text)):
        observation = None
    display_text = (_normalize_display_text(observation['source_span'])
                    if observation else _responsive_display_excerpt(raw_text, claim_text))
    # A retrieval window can cross a column/article boundary. Show the full
    # original paragraph containing the chosen sentence, not unrelated blocks
    # that happened to share that search window.
    blocks = _display_prose_blocks(raw_text)
    matches = [block for block in blocks if display_text and display_text in block]
    context_text = _bounded_display_context(matches[0] if len(matches)==1 else raw_text, display_text)
    display_text = _normalize_source_display_text(display_text, source_vocabulary)
    context_text = _normalize_source_display_text(context_text, source_vocabulary)
    if assessment is None:
        note = "Retrieved passage for manual comparison; its connection to this statement has not been assessed."
    if normalized_text and not display_text:
        note = (note + ' No complete excerpt could be displayed within the passage boundary.').strip()
    if passage.get("retrieval_method") == "explanatory_note_context":
        note = "Explanatory source note. " + note
    return {
        "passage_id": passage.get("passage_id"),
        "text": passage.get("excerpt", ""),
        "display_text": display_text,
        "context_text": context_text if context_text != display_text else "",
        "locator": locator,
        "evidence_kind": "full_text",
        "evidence_role": evidence_role,
        "relevance": relevance,
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
        if member.get('identity_status') == 'possible_match':
            member['show_quotation_check'] = False
            continue
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


def _join_line_break_hyphens(quote: str, source_texts: list[str]) -> str:
    """Rejoin a word the student's PDF split at a line end ("cultivat-ing").

    Only when the source has the joined word and not the hyphenated one, so a
    real hyphenated compound is never changed (2026-09-30).
    """
    source = ' '.join(source_texts).casefold()
    words = set(re.findall(r"[^\W_]+", source))

    def join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        hyphenated = f"{left}-{right}".casefold()
        return left + right if (left + right).casefold() in words and hyphenated not in source else match.group(0)
    return re.sub(r"([^\W\d_]+)-([^\W\d_]+)", join, quote)


def _quote_match_allowing_line_breaks(texts: list[str], target: str):
    """The quotation checker's match, retried with line-break hyphens rejoined."""
    from app.services.verification_evidence import _quotation_match
    found = next(((text, match) for text in texts if (match := _quotation_match(text, target))), None)
    if found is None:
        joined = _join_line_break_hyphens(target, texts)
        if joined != target:
            found = next(((text, match) for text in texts if (match := _quotation_match(text, joined))), None)
    return found


def _positive_quote_check(targets: list[str], texts: list[str]) -> dict | None:
    """A positive wording check over retained exact evidence, never a negative."""
    if not targets:
        return None
    matches = []
    for target in targets:
        found = _quote_match_allowing_line_breaks(texts, target)
        if found is None:
            return None
        text, match = found
        matches.append({"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "quote_sha256": hashlib.sha256(target.encode()).hexdigest(),
                        "start": match[0], "end": match[1], "method": match[2]})
    editorial = any(m["method"] in {"marked_editorial_match", "ellipsis_normalized"} for m in matches)
    return {"status": "complete", "outcome": "retained_evidence_editorial_match" if editorial else "retained_evidence_wording_match",
            "attention": False, "matches": matches,
            "label": ("Unchanged quotation wording matches the source around the marked brackets or omissions."  # no coaching (owner decision 2026-09-30)
                      if editorial else "Complete quoted wording matches the source.")}


def _normalize_display_text(value: str) -> str:
    """Make extracted PDF prose readable without changing stored evidence."""
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_source_display_text(value: str, source_vocabulary: str = '') -> str:
    """Repair split source words only when the same source corroborates spelling.

    Never apply this to student wording or immutable quotation-check inputs.
    Ambiguous compounds retain their hyphen.
    """
    words = set(re.findall(r'(?<![\w-])[A-Za-z]+(?![\w-])', source_vocabulary.casefold()))
    def join(match):
        word = match[1] + match[2]
        return word if word.casefold() in words else match[0]
    return _normalize_display_text(re.sub(r'\b([A-Za-z]+)-[ \t\n]*([a-z]+)\b',join,value))


_QUOTE_PATTERN = re.compile(r'[“"]([^”"]{2,})[”"]')
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_CRITICAL_QUOTATION_TOKENS = {
    "no", "not", "never", "none", "neither", "nor", "without",
    "all", "always", "only", "must", "cannot", "can't",
}


def _quotation_difference_diagnostics(
    claim_text: str, source_passages: list
) -> list[dict]:
    """Describe a close quotation mismatch without weakening exact matching.

    This is a report-only diagnostic over immutable passages. It never changes
    the authoritative exact-quotation result. Mechanical normalization remains
    handled by the authoritative checker; this layer distinguishes a small,
    inspectable wording difference from a quotation that was simply not found.
    """
    results: list[dict] = []
    # Passages may be plain text or package passages carrying their page.
    passages = [(item, None) if isinstance(item, str) else
                (str(item.get("text") or item.get("excerpt") or ""),
                 item.get("page_label") or (int(item["page_index"]) + 1
                                            if isinstance(item.get("page_index"), int) else None))
                for item in source_passages or []]
    passages = [(text, page) for text, page in passages if text]
    source_texts = [text for text, _page in passages]
    source_token_sets = [
        [(match.group(0).casefold(), match.group(0), match.start(), match.end())
         for match in _WORD_PATTERN.finditer(text)]
        for text in source_texts
    ]
    for quote_match in _QUOTE_PATTERN.finditer(claim_text):
        quote = quote_match.group(1)
        if _quote_match_allowing_line_breaks(source_texts, quote):
            continue
        target_matches = list(_WORD_PATTERN.finditer(quote))
        target = [match.group(0).casefold() for match in target_matches]
        brackets = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]", quote)]
        bracketed = {k for k, m in enumerate(target_matches) if any(lo < m.start() < hi for lo, hi in brackets)}
        if len(target) < 3:
            continue
        best: tuple | None = None
        for passage_index, source_tokens in enumerate(source_token_sets):
            source = [token[0] for token in source_tokens]
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
                            # A line-break hyphen split, or wording inside the
                            # student's square brackets (an editorial
                            # clarification), is not a wording difference.
                            if paper.replace(" ", "") == source_value.replace(" ", ""):
                                continue
                            if a1 > a0 and all(k in bracketed for k in range(a0, a1)):
                                continue
                            if a1 == a0 and (a0 in bracketed or a0 - 1 in bracketed):
                                continue
                            changes.append((paper, source_value))
                        best = (score, changes, window, passage_index, start, width)
        if best is None or best[0] < 0.58 or not best[1]:
            continue
        score, changes, _window, passage_index, window_start, width = best
        tokens = source_token_sets[passage_index]
        excerpt_text, excerpt_page = passages[passage_index]
        # The source's own wording for the matched stretch, for display below
        # the student's quotation (owner request 2026-09-30).
        source_excerpt = {"text": excerpt_text[tokens[window_start][2]:tokens[window_start + width - 1][3]],
                          "page": str(excerpt_page) if excerpt_page not in (None, "") else None}
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
                "quote_start": quote_match.start(1),
                "quote_end": quote_match.end(1),
                "source_excerpt": source_excerpt,
            }
        )
    return results


def _quotation_difference_label(differences: list[dict]) -> str:
    difference = max(differences, key=lambda item: item.get("difference_ratio", 0.0))
    changed = int(difference.get("changed_word_count") or 0)
    total = int(difference.get("word_count") or 0)
    # The window shows both wordings below this line, so the line no longer
    # repeats a substituted word (owner request 2026-09-30).
    substitution = ""
    if difference.get("severity") == "minor":
        return f"Quotation has a minor wording difference ({changed} of {total} words){substitution}"
    return f"Quotation wording differs materially ({changed} of {total} words){substitution}"


_FRONT_MATTER_CUES = re.compile(
    r"\b(email|university|faculty|department|school of|college|institute|centre|center|hospital|professor)\b",
    re.IGNORECASE,
)


def _sentence_start_context(text: str) -> str:
    """Remove a clear leading continuation only if a later sentence survives."""
    if re.match(r'^[a-z]|^[.…,;:)\]]', text):
        sentences = split_sentences(text)
        if len(sentences) > 1:
            first = text.find(sentences[1])
            if first > 0:
                return text[first:]
    return text


def _closed_display_delimiters(text: str) -> bool:
    """Display syntax only; never a quotation-fidelity or relevance judgment."""
    pairs = {'(': ')', '[': ']', '{': '}', '“': '”'}
    stack = []
    for char in text:
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    return not stack and text.count('"') % 2 == 0


def _display_prose_blocks(value: str) -> list[str]:
    """Keep raw line boundaries long enough to separate compact headings.

    Only standalone title-case/lowercase lines after a sentence or at the
    window start qualify. Never treat an ordinary wrapped prose line as a
    boundary merely because it starts with a capital. Retained text is intact.
    """
    lines = str(value).splitlines()
    blocks, pending = [], []
    at_boundary = True
    heading_run = False
    minor = {'and', 'or', 'of', 'the', 'in', 'to', 'for', 'a', 'an'}
    for index, line in enumerate(lines):
        words = re.findall(r"[A-Za-z]+", line)
        following = lines[index + 1].lstrip() if index + 1 < len(lines) else ''
        title_case = words and all(w[0].isupper() or w in minor for w in words)
        heading = (at_boundary and (1 if heading_run else 2) <= len(words) <= 9 and len(line.strip()) <= 70
                   and not re.search(r'[.!?;,：:]', line) and not line.rstrip().endswith('-')
                   and (title_case or line.strip().islower())
                   and bool(re.match(r'[A-Z]', following)))
        if heading or not line.strip():
            if pending:
                blocks.append(_normalize_display_text('\n'.join(pending)))
                pending = []
            at_boundary = True
            heading_run = heading
            continue
        pending.append(line)
        heading_run = False
        at_boundary = bool(re.search(r'[.!?][”"\')\]]*\d*\s*$', line))
    if pending:
        blocks.append(_normalize_display_text('\n'.join(pending)))
    return blocks


def _readable_display_units(value: str) -> list[str]:
    return [unit for block in _display_prose_blocks(value)
            for unit in _closed_sentence_units(block)]


def _closed_sentence_units(text: str) -> list[str]:
    """Keep adjacent sentences inside a quotation together, without invented text."""
    units = []
    cursor = 0
    pending_start = None
    for sentence in split_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            return []
        if pending_start is None:
            # A lower-case page/window continuation is not a sentence start.
            if start == 0 and re.match(r'^[a-z.…,;:)\]]', sentence):
                cursor = start + len(sentence)
                continue
            pending_start = start
        cursor = start + len(sentence)
        unit = text[pending_start:cursor]
        if _closed_display_delimiters(unit):
            # A page-ending continuation is not a complete display unit.
            if re.search(r'[.!?][\s”"\')\]]*$', unit):
                units.append(unit)
                pending_start = None
    return units


def _bounded_display_context(value: str, excerpt: str) -> str:
    """At most one complete neighboring unit on either side, within 1,400 chars."""
    text = _normalize_display_text(value)
    if not excerpt or text.count(excerpt) != 1:
        return ''
    units = _readable_display_units(value)
    offsets = []
    cursor = 0
    for unit in units:
        start = text.find(unit, cursor)
        offsets.append((start, start + len(unit)))
        cursor = start + len(unit)
    start = text.index(excerpt)
    end = start + len(excerpt)
    covered = [i for i, (lo, hi) in enumerate(offsets) if lo < end and hi > start]
    if not covered:
        return ''
    first, last = covered[0], covered[-1]
    lo, hi = offsets[first][0], offsets[last][1]
    if lo > start or hi < end or hi - lo > 1400:
        return ''
    # Add adjacent context only; never skip a large intervening unit.
    if first and hi - offsets[first - 1][0] <= 1400 and not text[offsets[first - 1][1]:lo].strip():
        lo = offsets[first - 1][0]
    if last + 1 < len(offsets) and offsets[last + 1][1] - lo <= 1400 and not text[hi:offsets[last + 1][0]].strip():
        hi = offsets[last + 1][1]
    return text[lo:hi]


def _responsive_display_excerpt(value: str, claim_text: str) -> str:
    """Show the responsive sentence neighborhood from a hash-bound passage.

    Retrieval windows remain unchanged in the Evidence Package. This display
    projection removes identifiable title/byline material before an abstract
    and avoids forcing a reader through an unrelated leading paragraph when a
    later complete sentence is the part connected to the citation.
    """
    text = _sentence_start_context(_normalize_display_text(value))
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
    sentences = [unit for unit in _readable_display_units(value) if len(unit) <= 1400]
    if not sentences:
        # Short title/document-level evidence need not end with punctuation.
        return text if len(text) <= 1400 and _closed_display_delimiters(text) else ''
    if len(sentences) < 2:
        return sentences[0]
    claim_terms = _display_terms(claim_text)
    claim_set = set(claim_terms)
    claim_bigrams = set(zip(claim_terms, claim_terms[1:]))
    named_terms = _claim_named_terms(claim_text)
    sentence_terms = [set(_display_terms(sentence)) for sentence in sentences]
    # Repeated names/title words must not outweigh the distinctive proposition.
    weights = {term: 1 / (1 + sum(term in terms for terms in sentence_terms)) for term in claim_set}

    def score(sentence: str) -> tuple[float, int, int]:
        terms = _display_terms(sentence)
        term_set = set(terms)
        overlap = sum(weights[term] for term in claim_set & term_set) / max(sum(weights.values()), 0.001)
        bigrams = len(claim_bigrams & set(zip(terms, terms[1:])))
        named = len(named_terms & set(re.findall(r'\b\w+\b', sentence.casefold())))
        return named + overlap / max(8, len(terms)), bigrams, -len(sentence)

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
        if candidate_score[0] < best_score[0] * 0.8:
            continue
        if len(selected) >= 2 or total + 1 + len(candidate) > 620:
            continue
        if index < best_index:
            selected.insert(0, candidate)
        else:
            selected.append(candidate)
        total += 1 + len(candidate)
    result = " ".join(selected)
    if result not in text:
        result = sentences[best_index]
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


def _claim_named_terms(claim_text: str) -> set[str]:
    body = re.sub(r'\([^)]*\d{4}[^)]*\)', '', claim_text)
    names = {t.casefold() for t in re.findall(r'\b[A-Z][a-z]{2,}\b', body)}
    first = re.match(r'\w+', body)
    if first:
        names.discard(first.group().casefold())
    return names


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
    # Prefer a retained passage addressing a named actor over generic theory
    # when both have the same advisory relevance. Citation author is excluded.
    actors = _claim_named_terms(claim_text)
    applies_framework = bool(re.search(
        r'\b(?:appl\w*|using|draw\w* on|through)\b.{0,90}\b(?:theor\w*|framework|concept|model)\b',
        claim_text, re.I))
    from app.services.verification_evidence import _passage_boundary_status

    def key(item: tuple[int, dict]) -> tuple[float, ...]:
        index, passage = item
        assessment = assessments.get(passage.get("passage_id"), {})
        passage_terms = _display_terms(_responsive_display_excerpt(str(passage.get("excerpt") or ""), claim_text))
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
        named_overlap = len(actors & set(re.findall(r'\b\w+\b', str(passage.get('excerpt') or '').casefold())))
        proposition_overlap = len((set(claim_terms) & passage_term_set) - set(_display_terms(' '.join(actors))))
        connected_account = relevance in {'relevant', 'partially_relevant'} and assessment.get('evidence_role') in {
            'source_own_claim_or_finding', 'source_synthesis_or_conclusion'}
        specific_attribution = named_overlap if not applies_framework and (proposition_overlap >= 2 or connected_account) else 0
        return (
            float(preferred_rank.get(passage.get("passage_id"), 0)),
            float(_DISPLAY_RELEVANCE.get(assessment.get("relevance"), 0)),
            float(specific_attribution),
            float(_DISPLAY_CONFIDENCE.get(assessment.get("confidence"), 0)),
            overlap_ratio,
            float(bigram_overlap),
            # Parent completeness does not make its truncated visible prefix
            # complete. Keep conservative parent/continuation status as well.
            float(
                passage.get("boundary_status") == "sentence_complete"
                and _passage_boundary_status(str(passage.get("excerpt") or ""))
                == "sentence_complete"
            ),
            float(display_tier),
            float(passage.get("retrieval_score") or 0.0),
            float(-index),
        )

    return [passage for _index, passage in sorted(indexed, key=key, reverse=True)]


def _display_terms(value: str) -> list[str]:
    # Comparison only: permit discretionary line-wrap spelling without
    # rewriting either the source excerpt or the student's authoritative text.
    value = unicodedata.normalize('NFKC', value)
    value = re.sub(r"(?<=[A-Za-z])-\s*(?=[a-z])", "", value)
    terms = re.findall(r"[a-z0-9]+", value.casefold())
    return [_display_stem(term) for term in terms if term not in _DISPLAY_STOPWORDS]


def _display_passage_is_metadata(passage: dict) -> bool:
    from app.services.verification_evidence import passage_role_from_text
    text = str(passage.get("text") or passage.get("excerpt") or "")
    if passage_role_from_text(text) == 'publication_metadata':
        return True
    # A standalone bibliographic footnote can lack its parent Notes heading.
    # Do not present a numbered external citation as the source's argument.
    # Exact quotation/locator priorities remain exempt in the caller.
    own_prose=re.sub(r'https?://\S+', '', text)
    own_prose=re.sub(r'“[^”]*”|"[^"]*"', '', own_prose)
    return bool(re.match(r'^\s*\d{1,4}\.\s+(?:See|Cf\.)\s', text)
                and re.search(r'https?://', text)
                and re.search(r'\b(?:19|20)\d{2}\b', text)
                and not re.search(r'\b(?:is|are|was|were|has|have|had|because|however|argues?|shows?|demonstrates?|suggests?)\b',
                                  own_prose)
                and not re.search(r'\n\s*\n', text))


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
    return eligible


def _display_stem(term: str) -> str:
    """Small dependency-free normalizer used only for report ordering."""
    for suffix in ("ically", "ingly", "edly", "ation", "ments", "ment", "ness", "ism", "ity", "ing", "ied", "ies", "ed", "es", "s"):
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


def _reference_findings_by_id(consistency: dict, references: dict | None = None) -> dict[str, list[dict]]:
    from app.services.reference_consistency import apa_in_text_form
    result: dict[str, list[dict]] = {}
    for finding in consistency.get("findings", []):
        ids = finding.get("reference_ids", [])
        if (finding.get("finding_type") == "duplicate_citation_key" and references
                and all(rid in references for rid in ids)
                and len({apa_in_text_form(references[rid]) for rid in ids}) == len(ids)):
            continue  # stored before in-text forms were compared (2026-09-30)
        for reference_id in finding.get("reference_ids", []):
            result.setdefault(reference_id, []).append(finding)
    return result


# A work the search never identified has no abstract of its own to show.
# Belton's chapter title merged any record containing "studio" and "system",
# and the report served the abstract of a review of a different book as his.
_IDENTIFIED_FOR_SOURCE_EVIDENCE = frozenset({
    "confirmed", "confirmed_with_minor_differences",
})


def _withhold_unidentified_abstract(item: dict) -> dict:
    """Show no abstract when the cited work was never identified.

    The abstract belongs to whichever record the search merged, which is only
    the cited work when identity was established. Withholding it is the
    difference between "we could not confirm this source" and presenting a
    stranger's summary as the student's source.
    """
    if item.get("coverage_level") != "abstract_only":
        return item
    if (item.get("reference_identity") or {}).get("status") in _IDENTIFIED_FOR_SOURCE_EVIDENCE:
        return item
    return {
        **item,
        "coverage_level": "unavailable",
        "best_evidence": None,
        "additional_evidence": [],
        "abstract_relevance": None,
        "abstract_scope_attention": False,
        "relevance_status": "not_assessed",
        "availability": (
            "The cited work could not be identified, so no summary is shown. "
            "Any record the search returned may describe a different work."
        ),
    }


def _withhold_mismatched_evidence(item: dict, citation: dict) -> dict:
    """Show no supporting passages for a source marked topically mismatched.

    Selecting passages from a work and presenting them as evidence for a claim
    the same report marks as unsupported by that work tells the reader two
    opposite things about one source. The scope comparison stays: it is the
    explanation, and the excerpt it was made against remains so the reader can
    check it.

    Abstract members are untouched. There the displayed abstract IS the text
    the judgment was made on, so withholding it would remove both the mark and
    any way to verify it; only a retrieved document keeps the judgment's basis
    in a separate field.
    """
    from app.services.report_layers import topical_mismatch
    if item.get("coverage_level") not in {"full_text", "partial_text"}:
        return item
    if not topical_mismatch(item, citation):
        return item
    withheld = len(item.get("additional_evidence") or []) + bool(item.get("best_evidence"))
    return {
        **item,
        "best_evidence": None,
        "additional_evidence": [],
        "relevance_status": "not_assessed",
        "topical_mismatch_withheld_evidence": withheld,
        "availability": (
            "This source is marked as a possible topical mismatch, so no passage "
            "from it is shown as evidence for this statement. The comparison that "
            "raised it is shown above; read the source directly to judge whether "
            "it supports the citation."
        ),
    }


def _identity_view(discovery: dict | None) -> dict:
    if not discovery:
        return {
            "status": "not_assessed",
            "label": "Reference identity search was not attached",
            "attention": False,
        }
    outcome = discovery.get("outcome") or "not_assessed"
    from app.services.reference_discovery import ReferenceDiscoveryRecord, credible_field_candidates
    try:
        compatible_year = any(c.observed.year == record.expected.year
                              for record in [ReferenceDiscoveryRecord.model_validate(discovery)]
                              for c in credible_field_candidates(record))
    except (ValueError, TypeError):
        compatible_year = False
    return {
        "status": outcome,
        "label": _reason_label(outcome),
        "attention": outcome == "bibliographic_conflict",
        "edition_year_unresolved": not compatible_year and any(
            comparison.get("reason_code") == "book_edition_year_unresolved"
            for candidate in discovery.get("candidates") or []
            for comparison in candidate.get("comparisons") or []
        ),
        # Whether any credible candidate agreed on BOTH title and author. This
        # answers a narrower question than the outcome status: not "is every
        # field consistent" but "do we know which work this is". A reference
        # whose year is wrong still names its work; one that was never located
        # does not, and only the second makes the source's subject unsafe to
        # compare against the citation.
        "title_and_author_agree": _title_and_author_agree(discovery),
    }


def _agreed_fields(candidate: dict) -> set[str]:
    return {
        str(comparison.get("field_name") or "")
        for comparison in candidate.get("comparisons") or []
        if comparison.get("outcome") == "agreement"
    }


def _title_and_author_agree(discovery: dict) -> bool:
    return any(
        {"title", "author"} <= _agreed_fields(candidate)
        for candidate in discovery.get("candidates") or []
        if candidate.get("plausible_identity_match")
        or candidate.get("authoritative_identifier_match")
    )


def _reportable_bibliographic_difference(difference: dict) -> bool:
    """Presentation abstention, not identity promotion or a historical rewrite.

    Missing fields, mechanical word spacing, and a title followed by separately
    delimited metadata/subtitle cannot establish an affirmative field error.
    Keep the underlying comparison and discovery outcome unchanged.
    """
    left = str(difference.get('submitted_value') or '').strip()
    right = str(difference.get('located_value') or '').strip()
    if not left or not right:
        return False
    field = difference.get('field_name')
    if field not in {'title', 'author', 'container_title'}:
        return True
    compact = lambda value: ''.join(ch for ch in value.casefold() if ch.isalnum())
    if compact(left) == compact(right):
        return False
    if field == 'title':
        for short, full in ((left, right), (right, left)):
            if len(compact(short)) < 16:
                continue
            for boundary in re.finditer(r'[.:]', full):
                if compact(full[:boundary.start()]) == compact(short):
                    return False
    return True


def _reference_field_difference_is_reportable(
    field_name: str, difference: dict, reference
) -> bool:
    """Decide whether a record difference is the student's problem to fix.

    A metadata record is one registry's view of a work, not the work itself.
    Two differences are routinely registry artefacts rather than reference
    errors, so neither becomes an orange citation/reference finding:

    ``doi``
        One work can carry several registered DOIs, such as an aggregator's
        alongside the publisher's, and a reference may legitimately cite either.
        A DOI that fails to resolve, or resolves to a different work, is a
        submitted-link issue and keeps its purple diamond (ARCHITECTURE §8).

    ``year``
        A journal issue dated one year and registered the next is an ordinary
        publication-date difference. This reuses the premise already accepted
        for acquisition in ``journal-title-author-near-year-v1``: an exact
        one-year gap on an otherwise agreeing journal article is not evidence
        that the reference is wrong. Larger gaps and non-journal works keep
        their finding.
    """
    if field_name == "doi":
        return False
    if field_name == "author" and _same_people_other_name_order(difference):
        return False
    if field_name == "year":
        submitted = str(difference.get("submitted_value") or "").strip()
        located = str(difference.get("located_value") or "").strip()
        source_kind = str(getattr(reference, "source_kind", "") or "")
        if (
            source_kind == "journal_article"
            and re.fullmatch(r"\d{4}", submitted)
            and re.fullmatch(r"\d{4}", located)
            and abs(int(submitted) - int(located)) == 1
        ):
            return False
    return True


def _same_people_other_name_order(difference: dict) -> bool:
    """Every cited surname appears among the record's names ("Le-Ha, P." and
    "Le-Ha Phan"; "Nguyen, P.A." and "Anh Nguyen Phuong"): the same people with
    given and family names in another order, not a different author (2026-09-30)."""
    submitted = str(difference.get("submitted_value") or "")
    located = str(difference.get("located_value") or "")
    cited = [n.strip().casefold() for n in re.findall(r"([^\W\d_][\w'’\-]+)\s*,", submitted)]
    tokens = {t.casefold() for t in re.findall(r"[^\W\d_][\w'’\-]+", located)}
    return bool(cited) and all(name in tokens for name in cited)


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
    result = {
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
    result['field_differences'] = [d for d in result['field_differences']
        if _reportable_bibliographic_difference(d)]
    result['conflicting_fields'] = [d['field_name'] for d in result['field_differences']]
    return result


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
        # Owner-approved wording (2026-09-29): a required full-text search was skipped or failed.
        "full_text_search_incomplete": "The full-text search did not finish, so the full text may still be available",
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
    *,
    numbers: dict[str, int] | None = None,
    catalog: list[dict] | None = None,
    passages: list[dict] | None = None,
) -> tuple[str, set[int]]:
    numbers = numbers or {}
    catalog = catalog or []
    occupied_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    dimensions = surface.get("page_dimensions") or []
    href_template = surface.get("page_href_template")
    if not dimensions or not href_template:
        return (
            '<p class="muted">The page-faithful paper surface is unavailable in this view.</p>',
            set(),
        )
    overlays_by_page: dict[int, list[str]] = {}
    placed: set[int] = set()
    for index, citation in enumerate(citations, 1):
        location = citation.get("paper_location") or {}
        if location.get("localization_level") != "exact_rectangle":
            continue
        rectangles = location.get("rectangles") or []
        anchor_id = str(location.get("anchor_id") or "")
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
            from app.services.report_layers import citation_partial_relevance
            from app.services.report_member_navigation import availability_tone
            tones = list(dict.fromkeys(availability_tone(member) for member in citation.get('members', []))) or ['not_assessed']
            palette = {"evidence_available":"#2563a7", "limited_evidence":"#168c8c", "retrieved_no_connection":"#8c6b4f", "attention":"#d95f02", "not_assessed":"#b8c0c8"}
            marks = "".join(
                f'<rect class="selection-bg" x="{x0:.3f}" y="{y0:.3f}" '
                f'width="{x1-x0:.3f}" height="{y1-y0:.3f}" rx="1" />'
                + f'<line class="underline" data-citation-span="true" stroke="#c5cbd1" x1="{x0:.3f}" y1="{y1-0.8:.3f}" '
                  f'x2="{x1:.3f}" y2="{y1-0.8:.3f}" />'
                for x0, y0, x1, y1 in page_rectangles
            )
            if citation_partial_relevance(citation):
                marks += ''.join(
                    f'<rect class="partial-relevance-outline" x="{x0-1:.3f}" y="{y0-1:.3f}" '
                    f'width="{x1-x0+2:.3f}" height="{y1-y0+2:.3f}"><title>Partial relevance: full-text passage addresses only part of this citation</title></rect>'
                    for x0,y0,x1,y1 in page_rectangles)
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
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="citation-overlay {tone} coverage_{coverage}" '
                f'role="button" tabindex="0" aria-pressed="false" '
                f'aria-label="Citation {index}: {escape(", ".join(_tone_label(t) for t in tones), quote=True)}" '
                f'data-hover-label="Complete citation span — select to inspect sources" '
                f'data-anchor-id="{escape(anchor_id, quote=True)}" '
                f'data-panel-template="citation-panel-{index}">{marks}{difference_marks}</a>'
            )
            placed.add(index)

        from app.services.report_member_navigation import member_targets, missing_reference_targets
        for target in missing_reference_targets(citation, surface.get('marker_words') or surface.get('selectable_words') or {}):
            overlays_by_page.setdefault(target['page_index'], []).append(
                f'<a href="#evidence-panel" class="citation-overlay missing-reference-target" role="button" tabindex="0" '
                f'aria-label="Citation {index}: source missing from reference list" data-panel-template="citation-panel-{index}">'
                f'<rect class="quote-difference-mark" x="{target["x0"]:.3f}" y="{target["y0"]:.3f}" '
                f'width="{target["x1"]-target["x0"]:.3f}" height="{target["y1"]-target["y0"]:.3f}" /></a>')
        targets = citation.get('source_member_targets')
        if targets is None:
            targets = member_targets(citation, surface.get('marker_words') or surface.get('selectable_words') or {})
        if citation_after_punctuation(citation):
            # The misplaced parenthetical, marked in the formatting orange (2026-09-30).
            for target in targets:
                overlays_by_page.setdefault(target['page_index'], []).append(
                    f'<a href="#evidence-panel" class="citation-overlay citation-format-target" role="button" tabindex="0" '
                    f'aria-label="Citation {index}" data-panel-template="citation-panel-{index}">'
                    f'<rect class="citation-format-mark" style="fill:#ff9a38;fill-opacity:.3;stroke:none" '
                    f'x="{target["x0"]:.3f}" y="{target["y0"]:.3f}" '
                    f'width="{target["x1"]-target["x0"]:.3f}" height="{target["y1"]-target["y0"]:.3f}" /></a>')
        for target in targets:
            member_index = target['member_index']
            member = citation['members'][member_index]
            label = str((member.get('source') or {}).get('raw_reference') or 'Source')
            from app.services.report_layers import member_marks
            issue = member_marks(member, citation, target, (surface.get('selectable_words') or {}).get(target['page_index'], []))
            overlays_by_page.setdefault(target['page_index'], []).append(
                f'<a href="#evidence-panel" class="citation-overlay member-target {target["tone"]}" '
                f'role="button" tabindex="0" aria-pressed="false" '
                f'aria-label="Citation {index}, source {member_index+1}: {escape(label, quote=True)}; {_tone_label(target["tone"])}" '
                f'data-member-index="{member_index}" data-anchor-id="{escape(anchor_id, quote=True)}" '
                f'data-panel-template="citation-panel-{index}">'
                f'<rect class="source-highlight{" unverified-highlight" if member.get("unverified") else " academic-highlight" if member.get("secondary_citation") else ""}" '
                + ('style="fill:#f28b82;fill-opacity:.42;stroke:none" ' if member.get('unverified') else
                   'style="fill:#ffe45c;fill-opacity:.4;stroke:none" ' if member.get('secondary_citation') else '') +
                f'x="{target["x0"]:.3f}" y="{target["y0"]:.3f}" '
                f'width="{target["x1"]-target["x0"]:.3f}" height="{target["y1"]-target["y0"]:.3f}" rx="1" />{issue}</a>'
            )

    indexed_findings = list(enumerate(reference_practice or [], 1))
    # Broad topical hit targets sit behind exact formatting/academic targets.
    # Keep original indices so existing evidence-panel bindings do not change.
    indexed_findings.sort(key=lambda pair: pair[1].get('finding_type') != 'source_topical_mismatch')
    for index, finding in indexed_findings:
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
            marker_class = ('reference-field-marker submitted-link-marker' if finding.get('finding_type') in LINK_MARKER_FINDINGS
                            else 'reference-field-marker')
            label = {'required_doi_missing':'Required DOI Missing', 'required_author_missing':'Required Author Missing',
                     'duplicate_reference_entry':'Academic Practice: Duplicate Reference Entry',
                     'reference_author_conflict':'Academic Practice: Incorrect Author Attribution',
                     'unverified_reference':'Cannot Be Verified',
                     'bibliographic_conflict':'Differs from the Located Record',
                     'bibliographic_field_conflict':'Differs from the Located Record',
                     'publication_year_discrepancy':'Differs from the Located Record',
                     'reference_identifier_conflict':'DOI Identifies a Different Work',
                     'doi_registers_a_different_title':'DOI Identifies a Different Work',
                     'reference_identifier_placeholder':'Unfinished Identifier',
                     'source_topical_mismatch':'Potential Topical Mismatch',
                     'assessment_link_missing':'Assessment-Required Link Missing', 'submitted_link_issue':'Submitted-Link Issue'
                     }.get(finding.get('finding_type'), 'Citation/Reference Formatting Issue')
            marks = ''
            if finding.get('finding_type') in LINK_MARKER_FINDINGS:
                if page_index != max(by_page):
                    continue
                x0,y0,x1,y1=rectangles[-1]
                cx,cy=x1+10,(y0+y1)/2
                page_width=next((p['width'] for p in surface.get('page_dimensions', []) if p['page_index']==page_index),x1+20)
                if cx+8>page_width:
                    cx=max(8,x0-10)
                occupied_by_page.setdefault(page_index, []).append((cx-8, cy-8, cx+8, cy+8))
                marks=(f'<path class="reference-field-marker submitted-link-marker" '
                    f'd="M {cx:.3f} {cy-6:.3f} L {cx+6:.3f} {cy:.3f} L {cx:.3f} {cy+6:.3f} L {cx-6:.3f} {cy:.3f} Z"><title>{escape(label)}</title></path>'
                    f'<rect class="submitted-link-hit" x="{cx-8:.3f}" y="{cy-8:.3f}" width="16" height="16" />')
            if finding.get('finding_type') == 'source_topical_mismatch':
                marks = ''.join(
                    f'<rect class="topical-reference-hit layer-mark" style="fill:transparent;stroke:none;pointer-events:all" x="{x0-4:.3f}" y="{y0-4:.3f}" width="{x1-x0+8:.3f}" height="{y1-y0+8:.3f}" />'
                    f'<rect class="reference-field-marker layer-mark mark-relevance" style="fill:#ef82ba;fill-opacity:.38;stroke:none;pointer-events:all" '
                    f'x="{x0-1.5:.3f}" y="{y0-1.5:.3f}" width="{x1-x0+3:.3f}" height="{y1-y0+3:.3f}" />'
                    for x0,y0,x1,y1 in rectangles)
            elif finding.get('finding_type') not in LINK_MARKER_FINDINGS:
                from app.services.highlight_priority import ACADEMIC_FINDINGS
                kind = finding.get('finding_type')
                # Soft red with a darker outline for "Cannot be verified", so
                # it stays distinct from topical-mismatch pink; a bright solid blue
                # outline for details that differ from the located record (2026-09-30).
                paint = (' unverified-highlight', ' style="fill:#f28b82;fill-opacity:.42;stroke:none"')\
                    if kind in UNVERIFIED_FINDINGS else \
                    (' reference-difference-highlight', ' style="fill:transparent;stroke:#0a7cff;stroke-width:1.8;stroke-dasharray:none"')\
                    if kind in REFERENCE_DIFFERENCE_FINDINGS else \
                    (' academic-highlight', ' style="fill:#ffe45c;fill-opacity:.4;stroke:none"')\
                    if kind in ACADEMIC_FINDINGS else ('', '')
                marks += ''.join(f'<rect class="reference-formatting-hit{paint[0]}"{paint[1]} x="{x0:.3f}" y="{y0:.3f}" '
                    f'width="{x1-x0:.3f}" height="{y1-y0:.3f}"><title>{escape(label)}</title></rect>' for x0,y0,x1,y1 in rectangles)
            target = _finding_target(finding, index, citations, numbers)
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="reference-practice-overlay" role="button" '
                f'tabindex="0" aria-pressed="false" aria-label="{escape(label, quote=True) if finding.get("finding_type") == "duplicate_reference_entry" else "Reference-practice difference"}" '
                f'data-panel-template="{target}">{marks}</a>'
            )

    pages: list[str] = []
    badge_requests: dict[int, list[dict]] = {}

    def first_rectangle(rectangles):
        valid = [r for r in rectangles or [] if all(k in r for k in ('page_index', 'x0', 'y0', 'x1', 'y1'))]
        return min(valid, key=lambda r: (int(r['page_index']), float(r['y0']), float(r['x0']))) if valid else None

    # Patchwriting passages: the Academic Practice highlight on the student's
    # words, no badge; drawn after the citations so a click on the highlight
    # opens the window that shows it: the citation (at the matched source) or,
    # for words no citation covers, the source's Reference window.
    from app.services.patchwriting_report import passage_windows
    from app.services.report_references import reference_template_id
    for passage in passages or []:
        number = int(passage['number'])
        window = next(iter(passage_windows(passage, citations)), None)
        if window is None:
            continue
        if window['citation']:
            target = (f'data-panel-template="citation-panel-{window["citation"]}" '
                      f'data-member-index="{window["member_index"]}"')
        elif (numbers or {}).get(window['reference_id']):
            target = f'data-panel-template="{reference_template_id(numbers[window["reference_id"]])}"'
        else:
            continue
        by_page: dict[int, list[tuple[float, float, float, float]]] = {}
        for item in (passage.get('paper_location') or {}).get('rectangles') or []:
            try:
                page_index = int(item["page_index"])
                coords = tuple(float(item[key]) for key in ("x0", "y0", "x1", "y1"))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= coords[0] < coords[2] and 0 <= coords[1] < coords[3]:
                by_page.setdefault(page_index, []).append(coords)
        for position, (page_index, rectangles) in enumerate(sorted(by_page.items())):
            marks = ''.join(
                f'<rect class="patchwriting-hit academic-highlight" style="fill:#ffe45c;fill-opacity:.4;stroke:none" '
                f'x="{x0:.3f}" y="{y0:.3f}" width="{x1-x0:.3f}" height="{y1-y0:.3f}" />'
                for x0, y0, x1, y1 in rectangles) + ''.join(
                f'<rect class="patchwriting-selection" x="{x0:.3f}" y="{y0:.3f}" '
                f'width="{x1-x0:.3f}" height="{y1-y0:.3f}" />'
                for x0, y0, x1, y1 in rectangles)
            mark_id = f'id="patchwriting-mark-{number}" ' if position == 0 else ''
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" {mark_id}class="patchwriting-overlay" role="button" tabindex="0" '
                f'aria-pressed="false" aria-label="Academic Practice" data-passage="{number}" '
                f'{target}>{marks}</a>')

    for index, citation in enumerate(citations, 1):
        rectangles = (citation.get('paper_location') or {}).get('rectangles') or []
        if rectangles:
            rectangle = rectangles[0]
            overlays_by_page.setdefault(rectangle['page_index'], []).append(
                f'<rect id="citation-location-{index}" x="{rectangle["x0"]}" y="{rectangle["y0"]}" '
                'width="1" height="1" fill="none" pointer-events="none" />')
        first = (first_rectangle(rectangles) if index in placed
                 else first_rectangle(citation.get('source_member_targets') or []))
        if first:
            badge_requests.setdefault(int(first['page_index']), []).append({
                'template': f'citation-panel-{index}', 'kind': 'citation', 'number': index,
                'line': (float(first['x0']), float(first['y0']), float(first['x1']), float(first['y1']))})
    # Every located reference entry opens its Reference N window. Its DOI/URL
    # text is not a separate paper link: the window lists the student's links.
    for entry in catalog:
        location = entry.get('location') or {}
        by_page: dict[int, list[dict]] = {}
        for rectangle in location.get('rectangles') or []:
            by_page.setdefault(int(rectangle['page_index']), []).append(rectangle)
        for page_index, rectangles in by_page.items():
            hits = ''.join(
                f'<rect class="reference-entry-hit" x="{float(r["x0"]):.3f}" y="{float(r["y0"]):.3f}" '
                f'width="{float(r["x1"])-float(r["x0"]):.3f}" height="{float(r["y1"])-float(r["y0"]):.3f}" />'
                for r in rectangles)
            overlays_by_page.setdefault(page_index, []).insert(0,
                f'<a href="#evidence-panel" class="reference-entry-overlay" role="button" tabindex="0" '
                f'aria-pressed="false" aria-label="Reference {int(entry["number"])}" '
                f'data-panel-template="{escape(entry["template_id"], quote=True)}">{hits}</a>')
        first = first_rectangle(location.get('rectangles') or [])
        if first:
            overlays_by_page.setdefault(int(first['page_index']), []).append(
                f'<rect id="reference-location-{int(entry["number"])}" '
                f'x="{first["x0"]}" y="{first["y0"]}" '
                f'width="{float(first["x1"])-float(first["x0"])}" height="{float(first["y1"])-float(first["y0"])}" '
                'fill="none" stroke="none" pointer-events="none" />')
            badge_requests.setdefault(int(first['page_index']), []).append({
                'template': entry['template_id'], 'kind': 'reference', 'number': int(entry['number']),
                'line': (float(first['x0']), float(first['y0']), float(first['x1']), float(first['y1']))})
    from app.services.report_badges import BADGE_FONT, place_badges, text_block_edges
    words = surface.get('selectable_words') or {}
    widths = {int(item['page_index']): float(item['width']) for item in dimensions
              if 'page_index' in item and 'width' in item}
    for page_index, requests in badge_requests.items():
        page_words = words.get(page_index) or words.get(str(page_index)) or []
        edges = text_block_edges(page_words, [r['line'] for r in requests])
        if edges is None or page_index not in widths:
            continue
        for badge in place_badges(requests, text_left=edges[0], text_right=edges[1],
                                  page_width=widths[page_index],
                                  occupied=occupied_by_page.get(page_index, [])):
            x0, y0, x1, y1 = badge['box']
            radius = (y1 - y0) / 2 if badge['kind'] == 'citation' else 1.2
            label = 'Citation' if badge['kind'] == 'citation' else 'Reference'
            overlays_by_page.setdefault(page_index, []).append(
                f'<a href="#evidence-panel" class="paper-badge {badge["kind"]}-badge" '
                f'data-panel-template="{escape(badge["template"], quote=True)}" tabindex="-1" aria-hidden="true">'
                f'<title>{label} {int(badge["number"])}</title>'
                f'<rect class="badge-bg" x="{x0:.3f}" y="{y0:.3f}" width="{x1-x0:.3f}" height="{y1-y0:.3f}" rx="{radius:.3f}" />'
                f'<text class="badge-number" x="{(x0+x1)/2:.3f}" y="{(y0+y1)/2 + BADGE_FONT*0.35:.3f}" '
                f'text-anchor="middle">{int(badge["number"])}</text></a>')
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
        from app.services.highlight_priority import prioritize_svg_highlights
        page_overlays = prioritize_svg_highlights(''.join(overlays_by_page.get(page_index, [])))
        pages.append(
            f'<figure class="paper-page" id="paper-page-{page_index}"><figcaption>Page {page_index + 1}</figcaption>'
            f'<div class="page-container" style="aspect-ratio:{width:.3f}/{height:.3f}" data-page-index="{page_index}"><svg class="page-surface" data-page-index="{page_index}" viewBox="0 0 {width:.3f} {height:.3f}" '
            f'role="img" aria-label="Submitted paper page {page_index + 1}">'
            f'<image href="{escape(href, quote=True)}" width="{width:.3f}" height="{height:.3f}" />'
            f'{page_overlays}</svg><div class="paper-text-layer">{text_layer}</div></div></figure>'
        )
    return f'<div class="paper-pages">{"".join(pages)}</div>', placed


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
        '<details class="unplaced"><summary>Citations Without Exact Page Geometry</summary>'
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
        "limited_evidence": "abstract available",
        "partial_evidence": "limited text available",
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
        # Blue states that inspectable evidence is available. A member holding
        # no passage has none to show, whatever the source offers, so it never
        # carries the available tone. An unconfirmed identity suppresses
        # relevance assessment entirely, which is "not assessed" rather than a
        # search that finished without a match.
        if member.get("best_evidence"):
            return "evidence_available"
        return ("not_assessed" if member.get("identity_status") == "possible_match"
                else "retrieved_no_connection")
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
        '<button type="button" data-choose-source>Upload Source</button>'
        '<span class="upload-status" aria-live="polite"></span></form>'
    )


def _render_search_again(action: dict) -> str:
    if not action.get("enabled") or not action.get("href"):
        return ""
    # Owner-approved label (2026-09-29).
    return (
        f'<form class="search-again" method="post" action="{escape(action["href"], quote=True)}">'
        '<button type="submit">Search Again</button>'
        '<span class="upload-status" aria-live="polite"></span></form>'
    )


def _render_export_details(action: dict) -> str:
    if not action.get("manifest_href"):
        return ""
    return (
        '<details class="technical-export"><summary>Technical Export Record</summary>'
        '<p>For verification and support: identifies the report version, original paper and PDF fingerprint. '
        'These details help an owner or administrator establish which version was shared; they are not academic findings.</p>'
        f'<a href="{escape(action["manifest_href"].split("?")[0], quote=True)}">Download Verification Record (JSON)</a></details>'
    )


# One wording for availability in every window (owner decision 2026-09-30).
_COVERAGE_HEADINGS = {'full_text': 'Full Text Retrieved', 'partial_text': 'Limited Text Retrieved',
                      'abstract_only': 'Abstract Retrieved', 'unavailable': 'No Text Retrieved'}
# Owner-approved window label (2026-09-28): "<judgment> - <evidence kind>", the
# evidence kind in its colour. The judgment part of a judged source is filled in
# by the Judgment script when its result arrives.
_EVIDENCE_KIND_LABELS = {'full_text': 'Full Text Retrieved', 'partial_text': 'Limited Text Retrieved',
                         'abstract_only': 'Abstract Retrieved'}
_NO_TEXT_LABEL = 'No Text Retrieved'
EVIDENCE_LABEL_CSS = (
    '.citation-overlay.member-target .source-highlight:not(.unverified-highlight):not(.academic-highlight)'
    '{fill-opacity:0}'
    '.member-label,.member-label .evidence-kind{color:var(--ink)}'
    '.evidence-disclosure,.abstract-disclosure{margin:.5rem 0}'
    '.evidence-disclosure>summary,.abstract-disclosure>summary{cursor:pointer;font-weight:600}'
    '.evidence-sentences{list-style:none;padding-left:0;margin:.3rem 0}'
    '.evidence-sentences li{margin:.35rem 0}'
    '.evidence-sentences .ev-page{font-size:.8rem;color:var(--muted)}'
    '.abstract-disclosure blockquote{margin:.35rem 0;padding:.5rem .8rem;border-left:3px solid var(--line)}'
    '.check-line{margin:.3rem 0}'
    '.selected-citation strong.proposition{font-weight:700}'
    '.citation-overlay.hovered .selection-bg{fill:#b9dcff;fill-opacity:.22}'
)


def _member_label(item: dict, citation: dict | None = None) -> str:
    """The member heading: judgment part, then evidence kind; HTML, escaped."""
    if _member_is_media(item):
        return 'Media Reference - Cannot Retrieve'
    if item.get('status') == 'citation_not_assessed':
        return 'Not assessed'
    level = item.get('coverage_level') if item.get('coverage_level') in _EVIDENCE_KIND_LABELS else 'unavailable'
    kind = (f'<span class="evidence-kind kind-{level}">'
            f'{escape(_EVIDENCE_KIND_LABELS.get(level, _NO_TEXT_LABEL))}</span>')
    record = str(item.get('verification_report_id') or '')
    # The whole label is underlined in its judgment's style; the Judgment script
    # fills a judged source's part and state when its result arrives.
    if level == 'full_text' and record:
        heading = (f'<span class="judgment-label" data-judgment-label="{escape(record, quote=True)}">'
                   '<span class="judgment-part" hidden></span><span class="judgment-sep" hidden> - </span>'
                   f'{kind}</span>')
    else:
        heading = ('<span class="judgment-label jk state-not_judged"><span class="judgment-part">Not Judged</span>'
                   f'<span class="judgment-sep"> - </span>{kind}</span>')
    if item.get('unverified') and level == 'unavailable':
        heading += ' – <mark class="issue-heading unverified">Cannot be verified</mark>'
    elif citation is not None:
        from app.services.report_layers import topical_mismatch
        if topical_mismatch(item, citation):
            heading += ' – <mark class="issue-heading topical">Possible Topical Mismatch</mark>'
    return heading


def _member_coverage_heading(item: dict, citation: dict | None = None) -> str:
    """Availability heading for one source member; HTML, already escaped."""
    if item.get('unverified') and item.get('coverage_level') in {None, 'unavailable'}:
        return 'No Text Retrieved – <mark class="issue-heading unverified">Cannot be verified</mark>'
    if item.get('status') == 'citation_not_assessed':
        return 'Not assessed'
    if _member_is_media(item):
        return 'Media Reference - Cannot Retrieve'
    heading = _COVERAGE_HEADINGS.get(item.get('coverage_level'), 'No Text Retrieved')
    if citation is not None:
        from app.services.report_layers import topical_mismatch
        if topical_mismatch(item, citation):
            heading += ' – <mark class="issue-heading topical">Possible Topical Mismatch</mark>'
    return heading


def _render_panel_template(citation: dict, index: int, *, patchwriting: dict | None = None) -> str:
    upload = citation.get("upload_action") or {}

    members = ''.join(
        f'<section class="source-group {escape(str(item.get("coverage_level") or "unavailable"), quote=True)}" '
        f'data-source-member="{member_index}"'+ (' hidden' if member_index else '') + '>'
        f'<h3 class="member-label">{_member_label(item, citation)}</h3>'
        + _render_member({**item, 'upload_action':upload if _member_accepts_upload(item) else {},
                          '_citation_text': str(citation.get('display_student_text') or citation.get('student_text') or '')},
                         grouped=True,
                         patchwriting=(patchwriting or {}).get(('citation', index, member_index)))
        + '</section>' for member_index,item in enumerate(citation.get('members', [])))
    count = len(citation.get('members', []))
    unplaced = (len({t['member_index'] for t in citation.get('source_member_targets', [])}) < count)
    navigation = (
        '<nav class="member-navigation" aria-label="Sources in this citation">'
        '<button type="button" data-source-step="-1" aria-label="Previous source">←</button>'
        f'<span data-source-position aria-live="polite">Source 1 of {count}</span>'
        '<button type="button" data-source-step="1" aria-label="Next source">→</button></nav>'
        if count > 1 else ''
    )
    # The citation's own description already states why no source is shown
    # (owner decision 2026-09-30).
    practice = (
        _category_heading('academic', 'p') +
        f'<p class="practice-notice">{escape(_panel_statement(citation.get("boundary_reason") or ""))}</p>'
        if citation.get("missing_reference_members") else ""
    )
    heading = (f'<a href="#citation-location-{index}">Citation {index}</a>'
               if (citation.get('paper_location') or {}).get('rectangles') else f'Citation {index}')
    return (
        f'<template id="citation-panel-{index}"><h2>{heading}<span data-proposition-suffix></span></h2>'
        f'{_render_citation_texts(citation)}'
        f'{navigation}'
        + f'{practice}{members}</template>'
    )


def attach_propositions(citations: list[dict], layer: dict) -> None:
    """A citation whose statement holds several judged propositions gets one
    window per proposition, lettered A, B, ... (owner request 2026-09-28).

    Each proposition is one claim group of the Judgment marks; its wording is
    located in the window's citation text by the group's paper ranges.
    """
    groups: dict = {}
    shown: dict = {}
    for mark in layer.get('marks') or []:
        number, group = mark.get('citation'), str(mark.get('group') or '')
        if number is None or group.endswith(':whole') or group.endswith(':static'):
            continue
        groups.setdefault(number, {}).setdefault(group, mark.get('order') or 0)
        if mark.get('display_ranges'):
            shown[group] = mark['display_ranges']
    for index, citation in enumerate(citations, 1):
        found = groups.get(citation.get('citation_number', index)) or {}
        if len(found) < 2:
            continue
        student = str(citation.get('student_text') or '')
        display = str(citation.get('display_student_text') or student)
        start = citation.get('paper_character_start')
        propositions = []
        for position, (group, _order) in enumerate(sorted(found.items(), key=lambda kv: kv[1])):
            local = []
            # Bold the words the proposition adds when it contains another one.
            parts = ([f"{a}-{b}" for a, b in shown[group]] if group in shown
                     else group.split(':', 1)[1].split(','))
            for part in parts:
                try:
                    a, b = (int(v) for v in part.split('-'))
                except ValueError:
                    continue
                if not isinstance(start, int):
                    continue
                a, b = a - start, b - start
                if not (0 <= a < b <= len(student)):
                    continue
                if display == student:
                    local.append((a, b))
                else:
                    at = display.find(student[a:b])
                    if at >= 0:
                        local.append((at, at + b - a))
            propositions.append({'group': group, 'letter': chr(ord('A') + position), 'local_ranges': local})
        citation['propositions'] = propositions


def _render_citation_texts(citation: dict) -> str:
    """The citation text; with several judged propositions, one copy per
    proposition with its wording in bold (owner request 2026-09-28)."""
    propositions = citation.get('propositions') or []
    if len(propositions) < 2:
        return f'<p class="selected-citation">{_render_citation_text(citation)}</p>'
    return ''.join(
        f'<p class="selected-citation" data-proposition="{escape(item["group"], quote=True)}"'
        f' data-proposition-letter="{escape(item["letter"], quote=True)}"{" hidden" if position else ""}>'
        f'{_render_citation_text(citation, bold=item["local_ranges"])}</p>'
        for position, item in enumerate(propositions))


def _render_citation_text(citation: dict, *, bold: list | None = None) -> str:
    text = str(citation.get("display_student_text") or citation.get("student_text") or "")
    if bold:
        # Proposition wording in bold; quotation marks are not combined with it.
        output, cursor = [], 0
        for start, end in sorted(bold):
            if not (cursor <= start < end <= len(text)):
                continue
            output.append(escape(text[cursor:start]))
            output.append(f'<strong class="proposition">{escape(text[start:end])}</strong>')
            cursor = end
        output.append(escape(text[cursor:]))
        return "".join(output)
    # Quotation differences are shown below, beside the source wording, not
    # highlighted in the citation text (owner request 2026-09-30).
    return escape(text)


def _panel_statement(text: str) -> str:
    """Remove generated revision instructions, never source or student excerpts."""
    text = str(text or '').replace(
        'This is a media reference. Automated source-text checks are not available.',
        'This is a media reference. Automated source retrieval and checks are not available.',
    )
    # No coaching text in windows (owner decision 2026-09-30).
    instruction_starts = ('Add ', 'Apply ', 'Correct ', 'Make ', 'Identify ', 'Compare ',
                          'Check ', 'Open ', 'Use ', 'Obtain ', 'Distinguish ', 'Cite ', 'Give ')
    statements = []
    for part in re.split(r'(?<=[.!?])\s+', str(text or '')):
        if part == 'Source completeness is uncertain.':
            continue  # The limited-text heading already conveys this boundary.
        if part.startswith(instruction_starts):
            # Preserve an explanatory limitation following a request to review.
            if '; ' in part:
                tail = part.split('; ', 1)[1]
                if tail.startswith(('this ', 'their ', 'the ', 'its ')):
                    statements.append(tail[0].upper()+tail[1:])
            continue
        statements.append(part)
    return ' '.join(statements)


def _render_citation_information(citation: dict, index: int) -> str:
    """Citation notes; the PDF export lists them (the window no longer does)."""
    notes = []
    if citation.get("boundary_reason"):
        notes.append(str(citation["boundary_reason"]))
    for member in citation.get("members") or []:
        notes.extend(str(item) for item in member.get("limitations") or [] if item)
        version = str(member.get("source_version") or "").strip()
        if version:
            notes.append(f"Source version: {_source_version_label(version)}")
    notes = list(dict.fromkeys(filter(None, (_panel_statement(note) for note in notes))))
    detail_items = notes
    if not detail_items:
        return ''
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


_CATEGORY_LABELS = {'evidence': 'Evidence', 'formatting': 'Citation and Reference Formatting',
                    'academic': 'Academic Practice'}
_CATEGORY_ORDER = ('evidence', 'formatting', 'academic')


def _category_heading(category: str, tag: str = 'h2') -> str:
    # The app no longer uses the term "Evidence" for a category of issues
    # (owner decision 2026-09-30); those findings carry no heading.
    if category == 'evidence':
        return ''
    label = _CATEGORY_LABELS.get(category, _CATEGORY_LABELS['formatting'])
    return f'<{tag}><strong class="issue-heading {category}">{label}</strong></{tag}>'


_LEADING_HEADING = re.compile(
    r'^\s*(?:<h[23][^>]*>.*?</h[23]>|<p><strong[^>]*>[^<]*</strong></p>)', re.S)


def _finding_body(finding: dict, index: int, *, combined: bool = False) -> str:
    """One finding's content without its heading or its own copy of the reference.

    Findings are grouped under one heading per category (owner decision
    2026-09-24), so the per-finding heading is dropped here.
    """
    inner = _render_reference_panel_template(finding, index, combined=combined)
    inner = inner.split('>', 1)[1].rsplit('</template>', 1)[0]
    inner = _LEADING_HEADING.sub('', inner, count=1)
    inner = re.sub(r'<h3 class="own-reference-heading">Submitted Reference</h3>', '', inner)
    inner = re.sub(r'<p class="full-reference own-reference">.*?</p>', '', inner, flags=re.S)
    if finding.get('finding_type') == 'source_topical_mismatch':
        inner = '<p><mark class="issue-heading topical">Possible topical mismatch.</mark></p>' + inner
    return inner.replace('<h2', '<h3').replace('</h2>', '</h3>')


def _grouped_findings(items: list[tuple[dict, int]], *, combined: bool = False, tag: str = 'h3',
                      citation_format: str = '', guidance: bool = True, extra: dict | None = None) -> str:
    """Findings under one heading per category: Evidence, formatting, Academic Practice.

    Each category ends with its style-guide links, when the findings have any.
    """
    from app.services.report_style_guidance import guidance_links
    by_category: dict[str, list[str]] = {}
    kinds: dict[str, list[str]] = {}
    for finding, index in items:
        body = _finding_body(finding, index, combined=combined)
        category = finding_category(finding.get('finding_type'))
        by_category.setdefault(category, []).append(
            f'<div class="finding-item" data-finding-index="{index}">{body}</div>')
        kinds.setdefault(category, []).append(finding.get('finding_type'))
    for category, html in (extra or {}).items():
        by_category.setdefault(category, []).append(html)
        kinds.setdefault(category, [])
    return ''.join(
        f'<section class="reference-finding" data-category="{category}">'
        + _category_heading(category, tag) + ''.join(by_category[category])
        + (guidance_links(kinds[category], citation_format) if guidance else '') + '</section>'
        for category in _CATEGORY_ORDER if category in by_category)


def _citation_guidance_kinds(citation: dict) -> list[str]:
    """Finding types a citation window explains directly, for its style links."""
    kinds = []
    if citation.get('missing_reference_members'):
        kinds.append('missing_reference_entry')
    for member in citation.get('members') or []:
        if member.get('show_quotation_check') and (member.get('quotation_check') or {}).get('attention'):
            kinds.append('quotation_difference')
        if member.get('secondary_citation'):
            kinds.append('indirect_source')
    return kinds


def _render_reference_panel_template(finding: dict, index: int, *, combined: bool = False) -> str:
    kind = finding.get('finding_type')
    original = str(finding.get('finding') or '')
    if kind in {'reference_title_style','body_title_style'} and original.startswith(('Italicize','Use regular')):
        subject = 'The reference title' if kind == 'reference_title_style' else 'The title in the paper'
        finding = {**finding, 'finding':subject + (' lacks required italics.' if original.startswith('Italicize') else ' is incorrectly italicized.')}
    content = _render_reference_panel_content(finding, index, combined=combined)
    if kind in SUBMITTED_LINK_FINDINGS | {'source_topical_mismatch'}:
        return content
    category = finding_category(kind)
    if combined:
        return content.replace('<p><strong>Citation and Reference Formatting</strong></p>', _category_heading(category,'p'), 1)
    return re.sub(r'<h2>.*?</h2>', _category_heading(category), content, count=1)


def _render_reference_panel_content(finding: dict, index: int, *, combined: bool = False) -> str:
    finding = {**finding, 'finding': _panel_statement(finding.get('finding') or '')}
    if finding.get('finding_type') in {'potentially_fabricated_reference', 'reference_identifier_conflict',
                                       'unverified_reference'}:
        from app.services.reference_credibility import credibility_records_html, credibility_finding_html
        # Carry the finding's candidate records with the source so an
        # identifier shown to reach a different work is not offered as a route
        # to the cited one. These records stay audit-only.
        source = {**finding['source'], 'reference_findings': [finding]}
        reference = '' if combined else f'<p class="full-reference own-reference">{_render_formatted_reference(source, source.get("raw_reference", ""))}</p>'
        heading = '<p><strong>Citation and Reference Formatting</strong></p>' if combined else '<h2>Reference Information</h2>'
        if finding.get('finding_type') in SUBMITTED_LINK_FINDINGS:
            heading = '<h2>Submitted-Link Issue</h2>'
        records = credibility_records_html(finding)
        if finding.get('finding_type') == 'reference_identifier_conflict':
            # The identified record completes "The submitted DOI identifies:".
            return (f'<template id="reference-panel-{index}">{heading}<p>{credibility_finding_html(finding)}</p>'
                    + records + reference + '</template>')
        return (f'<template id="reference-panel-{index}">{heading}<p>{credibility_finding_html(finding)}</p>'
                + reference + records + '</template>')
    if finding.get('finding_type') == 'duplicate_reference_entry':
        rows = ''.join(f'<p class="full-reference">{_render_formatted_reference(peer, peer.get("raw_reference", ""))}</p>'
                       for peer in finding.get('related_references') or [])
        return (f'<template id="reference-panel-{index}"><h2>Academic Practice</h2>'
                f'<p>{escape(finding["finding"])}</p>{rows}</template>')
    if finding.get('finding_type') == 'required_quotation_locator_missing':
        finding['finding'] = ('This passage lacks a page number or other numbered location.'
                              if finding.get('citation_style') == 'mla' else
                              'This quotation lacks a page or paragraph locator.')
    source = finding["source"]
    if combined:
        # The citation panel already supplies the student text and source.
        return (f'<template id="reference-panel-{index}"><p><strong>Citation and Reference Formatting</strong></p>'
                f'<p class="reference-issue">{escape(finding["finding"])}</p></template>')
    if finding.get('finding_type') == 'source_topical_mismatch':
        return (f'<template id="reference-panel-{index}"><h2>Abstract Retrieved – <mark class="issue-heading topical">Possible Topical Mismatch</mark></h2>'
                f'<p class="full-reference own-reference">{_render_formatted_reference(source, source["raw_reference"])}</p>'
                f'<p>{escape(finding["finding"])}</p><h3>Selected Citation</h3>'
                f'<blockquote>{escape(finding["citation_text"])}</blockquote><h3>Abstract</h3>'
                f'<blockquote>{escape(finding["abstract_text"])}</blockquote></template>')
    if finding.get('finding_type') == 'submitted_link_issue':
        return (f'<template id="reference-panel-{index}"><h2>Submitted-Link Issue</h2>'
                f'<p>{escape(finding["finding"])}</p><p class="full-reference own-reference">{_render_formatted_reference(source, source.get("raw_reference", ""))}</p>'
                + '</template>')
    if finding.get('finding_type') == 'required_quotation_locator_missing':
        mla = finding.get('citation_style') == 'mla'
        heading = 'Passage location missing' if mla else 'Quotation location missing'
        return (f'<template id="reference-panel-{index}"><h2>{heading}</h2>'
                f'<p>{escape(finding["finding"])}</p><blockquote>{escape(finding["quote_text"])}</blockquote>'
                f'<p class="full-reference own-reference">{_render_formatted_reference(source, source["raw_reference"])}</p>'
                '</template>')
    if finding.get('finding_type') in {'reference_title_style', 'reference_order', 'body_title_style'}:
        heading = ('Body title formatting' if finding['finding_type']=='body_title_style' else
                   'Reference title formatting' if finding['finding_type']=='reference_title_style' else 'Reference order')
        return (f'<template id="reference-panel-{index}"><h2>{heading}</h2>'
                f'<p>{escape(finding["finding"])}</p>'
                f'<p class="full-reference own-reference">{_render_formatted_reference(source, source["raw_reference"])}</p>'
                '</template>')
    if finding.get('finding_type') == 'assessment_link_missing':
        return (f'<template id="reference-panel-{index}"><h2>Assessment-Required Link Missing</h2>'
                f'<p>{escape(finding["finding"])}</p><p>This is an assessment requirement, not a universal citation-style rule. '
                'A DOI, ordinary URL, or library permalink satisfies the presence check; public full-text access is not required.</p>'
                f'<p class="full-reference own-reference">{escape(source["raw_reference"])}</p></template>')
    if finding.get('finding_type') == 'required_author_missing':
        return (f'<template id="reference-panel-{index}"><h2>Required Author Missing</h2>'
                f'<p>{escape(finding["finding"])}</p>'
                '<p>The author is verified from bibliographic registration metadata, not citation evidence.</p>'
                f'<p class="full-reference own-reference">{escape(source["raw_reference"])}</p></template>')
    if finding.get('finding_type') == 'required_doi_missing':
        from urllib.parse import quote
        doi = str(finding['verified_doi'])
        return (f'<template id="reference-panel-{index}"><h2>Required DOI Missing</h2>'
                f'<p>{escape(finding["finding"])}</p>'
                f'<p>Verified DOI: <a href="https://doi.org/{escape(quote(doi, safe="/"), quote=True)}" target="_blank" rel="noopener noreferrer">{escape(doi)}</a></p>'
                f'<p class="full-reference own-reference">{escape(source["raw_reference"])}</p></template>')
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
        # The finding names the issue and the located record follows in full;
        # the field list, located value and provider line were removed
        # (owner decision 2026-09-30).
        return (
            f'<template id="reference-panel-{index}"><h2>Reference Information</h2>'
            f'<p>{escape(finding["finding"])}</p>'
            + (f'<p><strong>In your reference:</strong> {escape(str(difference.get("submitted_value") or "Not retained"))}</p>'
               if difference else '') +
            '<h3 class="own-reference-heading">Submitted Reference</h3>'
            f'<p class="full-reference own-reference">{_render_formatted_reference(source, fallback)}</p>'
            '<h3>Located Record</h3>'
            f'<p class="full-reference">{escape(located_text or "No displayable located-record fields were retained.")}</p>'
            '</template>'
        )
    related = "".join(
        f'<p class="full-reference">{_render_formatted_reference(peer, peer["raw_reference"])}</p>'
        for peer in finding.get("related_references") or []
    )
    return (
        f'<template id="reference-panel-{index}"><h2>Reference Practice</h2>'
        f'<p>{escape(finding["finding"])}</p>'
        + (related or f'<p class="full-reference own-reference">{_render_formatted_reference(source, fallback)}</p>') + '</template>'
    )


def _media_named_in_paper(source: dict, paper_key: str) -> bool:
    """A film or programme the paper names by its title (owner decision 2026-09-30).

    Film essays name the work they discuss rather than citing (Director, year);
    that is a reference to the work, so its entry is not called uncited.
    """
    if not paper_key or not _member_is_media({'source': source}):
        return False
    title = re.sub(r'\s*\[[^\]]*\]\s*', ' ', str(source.get('title') or ''))
    key = _fold_for_titles(title)
    # The reference entry itself contains the title once; a body mention is another.
    return len(key) >= 8 and paper_key.count(key) >= 2


def _fold_for_titles(text: str) -> str:
    """Letters and digits only, for title lookup."""
    return ''.join(ch for ch in unicodedata.normalize('NFKC', text).casefold() if ch.isalnum())


def _paper_key(surface: dict) -> str:
    """The paper's words folded for title lookup (letters and digits only)."""
    words = [str(word[4]) for page in (surface.get('selectable_words') or {}).values() for word in page
             if isinstance(word, (list, tuple)) and len(word) > 4]
    return _fold_for_titles(' '.join(words))


def _render_reference_window_template(entry: dict, findings: list[dict], citations: list[dict],
                                      *, citation_format: str = '', patchwriting: list[dict] | None = None,
                                      paper_key: str = '') -> str:
    """One window per reference: the entry, its availability, its citations
    and every located reference-list finding on it, grouped by category.

    The window shows the complete reference once; each finding section drops
    its own copy of that reference but keeps peers and located records.
    """
    number = int(entry['number'])
    source = entry.get('source') or {}
    raw = str(source.get('raw_reference') or source.get('title') or 'Reference')
    heading = (f'<a href="#reference-location-{number}">Reference {number}</a>'
               if entry.get('location') else f'Reference {number}')
    parts = [f'<template id="{escape(entry["template_id"], quote=True)}"><h2>{heading}</h2>',
             f'<p class="full-reference reference-window-entry">{_render_formatted_reference(source, raw)}</p>']
    member = entry.get('member')
    if member:
        # Owner wording 2026-09-29 for a media work's reference window.
        availability = ('Media Reference - Cannot Assess' if _member_is_media(member)
                        else _member_coverage_heading(member))
        parts.append(f'<h3 class="reference-availability">{availability}</h3>')
        first = entry.get('first_citation')
        if first and _member_accepts_upload(member):
            parts.append(_render_upload(citations[first[0] - 1].get('upload_action') or {}))
        if not member.get('unverified'):  # owner decision 2026-09-30
            parts.append(_render_search_again(member.get('search_again_action') or {}))
    cited = entry.get('citation_numbers') or []
    if cited:
        parts.append('<p class="reference-citations">Cited in ' + ' '.join(
            f'<button type="button" data-go-to="citation-panel-{i}">Citation {i}</button>' for i in cited) + '</p>')
    elif not _media_named_in_paper(source, paper_key):
        parts.append('<p class="muted">No in-text citation in this report is linked to this reference.</p>')
    grouped = _grouped_findings([(findings[j - 1], j) for j in entry.get('finding_indexes') or []],
                                citation_format=citation_format,
                                extra={'academic': render_patchwriting(patchwriting)} if patchwriting else None)
    if member and 'issue-heading unverified' in _member_coverage_heading(member):
        # The heading already says "Cannot be verified"; the finding keeps
        # only its explanation (owner decision 2026-09-30).
        grouped = grouped.replace('<mark class="issue-heading unverified">Cannot be verified.</mark> ', '')
        grouped = grouped.replace('<mark class="issue-heading unverified">Cannot be verified.</mark>', '')
    parts.append(grouped)
    parts.append('</template>')
    return ''.join(parts)


def _member_evidence_extracts(member: dict) -> list[dict]:
    """Multiple compact extracts require the projection's validated joint decision."""
    if (member.get('display_selection') or {}).get('joint_selection_status') == 'applied':
        return list(member.get('evidence_extracts') or [])[:3]
    return [member['best_evidence']] if member.get('best_evidence') else []


def _member_evidence_contexts(member: dict) -> list[dict]:
    """One bounded context area, omitting text already shown as a short extract."""
    extracts = _member_evidence_extracts(member)
    normalize = lambda value: ' '.join(str(value or '').split())
    visible = [normalize(p.get('display_text', p.get('text'))) for p in extracts]
    joint = (member.get('display_selection') or {}).get('joint_selection_status') == 'applied'
    candidates = extracts if joint else [member.get('best_evidence') or {}, *(member.get('additional_evidence') or [])[:2]]
    contexts, seen = [], set()
    for item in candidates:
        # Legacy short contexts remain readable, but missing/empty context must
        # never bring a multi-page parent back into the panel.
        fallback = item.get('display_text', item.get('text'))
        if 'context_text' not in item and len(str(item.get('text') or '')) <= 1400:
            fallback = item.get('text')
        text = item.get('context_text') or (
            fallback
            if item is not member.get('best_evidence') or joint else '')
        normalized = normalize(text)
        short = normalize(item.get('display_text', item.get('text')))
        if not normalized or normalized in seen or normalized in visible:
            continue
        if joint and (len(normalized) <= len(short) or short not in normalized):
            continue
        seen.add(normalized)
        contexts.append({**item, 'context_text': text})
    return contexts[:3]


def _render_member(member: dict, *, grouped: bool = False, patchwriting: list[dict] | None = None) -> str:
    member = {**member, 'availability': _panel_statement(member.get('availability') or '')}
    for key in ('best_evidence',):
        if member.get(key):
            member[key] = {**member[key], 'evidence_note': _display_evidence_note(member, member[key])}
    member['additional_evidence'] = [{**item, 'evidence_note': _display_evidence_note(member,item)}
                                     for item in member.get('additional_evidence') or []]
    source = member["source"]
    source_label = ". ".join(
        value for value in (source["author"], source["year"], source["title"]) if value
    ) or source["raw_reference"]
    # One evidence source per window (owner decisions 2026-09-28): the GLM-selected
    # sentences, collapsed under "Evidence"; an abstract collapsed under "Abstract".
    # The relevance gate's excerpts are no longer shown.
    evidence = ""
    if member.get("evidence_sentences"):
        evidence = ('<details class="evidence-disclosure"><summary>Evidence</summary>'
                    '<ul class="evidence-sentences" data-evidence-list>' + "".join(
            f'<li data-evidence-key="{escape(item["key"], quote=True)}">'
            + (f'<span class="ev-page">p. {escape(str(item["page"]))}</span> ' if item.get("page") else "")
            + f'<q>{escape(item["text"])}</q></li>' for item in member["evidence_sentences"]) + '</ul></details>')
    # An abstract-only source's evidence is its abstract.
    abstract = next((item for item in _member_evidence_extracts(member)
                     if item.get("evidence_kind") in {"abstract", None} and item.get("text")), None)
    if abstract is not None and member.get("coverage_level") == "abstract_only":
        evidence += ('<details class="abstract-disclosure"><summary>Abstract</summary>'
                     f'<blockquote>{escape(str(abstract["text"]))}</blockquote></details>')
    additional = ""
    more_context = ""  # Candidate reserves stay in the package, not the reader's window.
    check_items = []
    if member.get("show_quotation_check") and member.get("coverage_level") != "unavailable":
        check_items.append(("Quotation", member.get("quotation_check") or {}, False))
    if member.get("show_locator_check"):
        check_items.append(("Locator", member.get("locator_check") or {}, False))
    checks = ""
    if check_items:
        rows = "".join(
            f'<p class="check-line">{escape(_panel_statement(_check_sentence(label, check)))}</p>'
            + (render_quotation_comparisons(check, str(member.get("_citation_text") or ""))
               if label == "Quotation" else "")
            for label, check, _identity_item in check_items
        )
        heading = _category_heading('academic','p') if any(check.get('attention') for _, check, _ in check_items) else ''
        checks = heading + rows
    secondary = secondary_citation_line(member.get("secondary_citation"))
    if secondary:
        checks += ('' if 'issue-heading academic' in checks else _category_heading('academic', 'p')) \
            + f'<p class="check-line">{escape(secondary)}</p>'
    if patchwriting:
        # Patchwriting sits with the window's other Academic Practice issues.
        checks += ('' if 'issue-heading academic' in checks else _category_heading('academic', 'p')) \
            + render_patchwriting(patchwriting)
    if any(f.get("finding_type") == "duplicate_citation_key" for f in member.get("reference_findings") or []):
        checks += _category_heading('formatting','p') + '<p class="check-line">Two or more references share this author and year.</p>'
    # Year findings belong to the bound reference-field layer. An unresolved
    # comparison against an arbitrary search candidate is not a reader finding.
    reference_html = _member_reference_html(source, source_label)
    action_html = _member_action_html(member, reference_html)
    reference_html = (
        f'<p class="full-reference">{reference_html}</p>'
    )
    from app.services.submitted_link_display import render_link_checks
    link_checks = ''  # Submitted-link diagnostics belong only to reference flags.
    edition_notice = _render_alternate_edition(member.get("alternate_edition"))
    scope_notice = ""
    if member.get("abstract_scope_attention"):
        scope = (member.get("abstract_relevance") or {}).get("scope_assessment") or {}
        # The source's label already says "possible topical mismatch"; the
        # window gives only the reason (owner decision 2026-09-30).
        scope_notice = f'<p class="attention">{escape(str(scope.get("rationale") or ""))}</p>'

    elif (member.get("scope_disagreement") or {}).get("note") and member.get("coverage_level") != "full_text":
        # A full-text source has its Judgment; the topic comparison read only its opening.
        # One flag, not a gradient: a mark takes the attention style, and
        # a judgment that stopped short of one is shown here as the
        # comparison itself, visibly weaker, for the reader to weigh.
        scope_notice = (
            '<p class="muted">'
            f'{escape(_panel_statement(member["scope_disagreement"]["note"]))}</p>'
        )
    availability = (
        '<p class="source-unavailable"><strong>Source Not Retrieved</strong></p>'
        if member.get("availability") == "Source Not Retrieved" and not grouped
        else f'<p>{escape(member["availability"])}</p>'
        if member.get("availability") and member.get("availability") != "Source Not Retrieved"
        else ""
    )
    if _member_is_media(member):
        availability = ''
    return (
        f'<section class="member {member_tone(member)}" data-reference-id="{escape(str(member.get("reference_id") or ""), quote=True)}">'
        f'{edition_notice}{availability}{scope_notice}{_judgment_slot(member)}{evidence}{additional}{more_context}{reference_html}{action_html}{checks}{link_checks}{_render_upload(member.get("upload_action") or {})}{"" if member.get("unverified") else _render_search_again(member.get("search_again_action") or {})}</section>'
    )


def _member_reference_html(source: dict, label: str) -> str:
    """The source's full reference, with its submitted link when not already shown."""
    reference_html = _render_formatted_reference(source, label)
    submitted_href = str(source.get('url') or '')
    if submitted_href and _safe_reference_href(submitted_href) and escape(submitted_href,quote=True) not in reference_html:
        reference_html += f' <a href="{escape(submitted_href,quote=True)}" target="_blank" rel="noopener">{escape(submitted_href)}</a>'
    return reference_html


def _member_action_html(member: dict, reference_html: str) -> str:
    """The member's source button. A DOI or cited link is already in the
    reference; only a route to retrieved text gets a button."""
    action = member.get("source_action") or {}
    if (member.get("coverage_level") != "abstract_only" and action.get("enabled")
            and action.get("status") not in {"canonical_landing_available", "cited_https_route_available"}
            and action.get("href") and escape(action["href"], quote=True) not in reference_html):
        return (
            f'<p class="actions"><a class="compact-action" href="{escape(action["href"], quote=True)}" '
            f'target="_blank" rel="noopener">{escape(action.get("label", "Open source"))}</a></p>'
        )
    return ""


def secondary_citation_line(flag) -> str:
    """Owner wording 2026-09-29 (option A, with the attributed authors)."""
    if not isinstance(flag, dict):
        return ''
    names = [str(n) for n in flag.get('attributed_to') or [] if str(n).strip()]
    return ("This citation relies on source passages that report another author's work"
            + (f" ({'; '.join(names)})." if names else "."))


# Owner wording 2026-09-29, one line per matched source.
PASSAGE_WORDING = {
    'close_paraphrase': 'This passage closely follows the wording and structure of the source without quotation marks.',
    'unquoted_verbatim': "This passage uses the source's exact wording without quotation marks.",
    'source_quoted': 'This passage uses wording that the source quotes from another author, without quotation marks.',
}


def passage_source_label(page) -> str:
    """"Source (p. N):", the page shown as for evidence sentences."""
    return f'Source (p. {page}):' if page not in (None, '') else 'Source:'


def _patchwriting_by_window(passages: list[dict], citations: list[dict]) -> dict[tuple, list[dict]]:
    """Each passage's matched sources, keyed by the window that shows them."""
    from app.services.patchwriting_report import passage_windows
    windows: dict[tuple, list[dict]] = {}
    for passage in passages:
        for window in passage_windows(passage, citations):
            key = (('citation', window['citation'], window['member_index']) if window['citation']
                   else ('reference', window['reference_id']))
            windows.setdefault(key, []).append(window['item'])
    return windows


def render_quotation_comparisons(check: dict, citation_text: str, *, portable: bool = False) -> str:
    """The student's quotation, differing words in bold, above the source's wording.

    The same layout as patchwriting (owner request 2026-09-30).
    """
    from app.services.text_quality import readable_text
    open_q, close_q = ('“', '”') if portable else ('<q>', '</q>')
    parts = []
    for difference in (check or {}).get('differences') or []:
        excerpt = difference.get('source_excerpt') or {}
        start, end = difference.get('quote_start'), difference.get('quote_end')
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(citation_text)
                and excerpt.get('text')):
            continue
        spans = sorted((int(s['local_start']), int(s['local_end'])) for s in difference.get('spans') or []
                       if isinstance(s.get('local_start'), int) and isinstance(s.get('local_end'), int)
                       and start <= s['local_start'] < s['local_end'] <= end
                       and citation_text[s['local_start']:s['local_end']] == s.get('paper_text'))
        student, cursor = [], start
        for low, high in spans:
            if low < cursor:
                continue
            student.append(escape(citation_text[cursor:low]))
            student.append(f'<strong>{escape(citation_text[low:high])}</strong>')
            cursor = high
        student.append(escape(citation_text[cursor:end]))
        parts.append(f'<p class="patchwriting-student">{open_q}{"".join(student)}{close_q}</p>'
                     f'<p class="passage-source-label">{escape(passage_source_label(excerpt.get("page")))}</p>'
                     f'<p class="patchwriting-source">{open_q}{escape(readable_text(str(excerpt["text"])))}{close_q}</p>')
    return ''.join(parts)


def render_patchwriting(items: list[dict], *, portable: bool = False) -> str:
    """Per matched source: its approved line, then for each finding the student's
    words with the copied words in bold above the source clause they follow."""
    # The PDF engine does not draw <q> quotation marks.
    open_q, close_q = ('“', '”') if portable else ('<q>', '</q>')
    blocks = []
    for item in items or []:
        line = PASSAGE_WORDING.get(item.get('wording'))
        if not line:
            continue
        parts = [f'<p class="check-line">{escape(line)}</p>']
        for comparison in item.get('comparisons') or []:
            student = ''.join(f'<strong>{escape(g["text"])}</strong>' if g.get('copied') else escape(g['text'])
                              for g in comparison.get('student') or [] if g.get('text'))
            if student:
                parts.append(f'<p class="patchwriting-student">{open_q}{student}{close_q}</p>')
            for excerpt in comparison.get('excerpts') or []:
                parts.append(f'<p class="passage-source-label">{escape(passage_source_label(excerpt.get("page")))}</p>'
                             f'<p class="patchwriting-source">{open_q}{escape(str(excerpt.get("text") or ""))}{close_q}</p>')
        blocks.append(('<div>' if portable else '<div class="patchwriting-finding">') + ''.join(parts) + '</div>')
    return ''.join(blocks)


def _judgment_slot(member: dict) -> str:
    record = str(member.get("verification_report_id") or "")
    if member.get("coverage_level") != "full_text" or not record:
        return ""
    return f'<div class="judgment-slot" data-judgment-record="{escape(record, quote=True)}"></div>'


def _display_evidence_note(member: dict, evidence: dict) -> str:
    note = str(evidence.get('evidence_note') or '')
    for text in ('The abstract addresses only part of the attributed statement.',
                 "This passage states the source author's synthesis or conclusion.",
                 "This passage reports another work rather than this source's own finding. Check whether the original work should also be cited.",
                 'This passage appears to provide methods or background rather than a direct finding.',
                 "This information helps identify the cited study's topic, date, population, or design; one source alone may not establish a broader claim about the literature.",
                 'Explanatory source note.',
                 'This passage addresses only part of the citation; it does not establish every material detail.'):
        note = note.replace(text,'').strip()
    return _panel_statement(note)


def _render_alternate_edition(value) -> str:
    """Show reviewed relationship without deriving task outcomes or issue colors.

    Records must originate from the trusted, authorized review workflow.
    Hash-bound attestations detect staleness; they are not authentication.
    """
    if value is None:
        return ""
    from app.services.alternate_edition import AlternateEditionRecord
    try:
        record = AlternateEditionRecord.model_validate(value)
    except (ValueError, TypeError):
        text = "The edition information could not be validated. Compare the source manually."
    else:
        text = (
            "Human review identified the cited edition with a later printing. The printing date alone "
            "does not establish a reference-year error. Text and page-number checks remain separate."
            if record.human_review and record.human_review.decision == 'same_edition_later_printing' else
            "Human review confirmed that this is an alternate edition or reissue of the cited work, "
            "not an exact edition match. Quotation and page-number checks remain separate."
            if record.human_verified else
            "Human review did not confirm the alternate-edition relationship. "
            "Do not assume that this copy corresponds to the cited edition."
            if record.human_review and record.human_review.decision == 'rejected' else
            "An alternate edition is recorded for this source. Its relationship to the cited "
            "edition still needs evidence review; matching text or a page number in this copy "
            "does not establish correspondence across editions."
        )
    return '<p class="muted alternate-edition">' + escape(text) + '</p>'


def _render_more_source_context(items: list[dict]) -> str:
    if not items:
        return ""
    blocks = "".join(
        '<blockquote class="source-excerpt">'
        + escape(item.get("display_text", item.get("text")) or "")
        + _inline_locator(item) + '</blockquote>'
        + ('<p class="muted">This retained excerpt is bounded; open the source for surrounding text.</p>'
           if item.get("excerpt_truncated") else "")
        for item in items
    )
    return (
        '<details class="more-source-context"><summary>More source context '
        f'({len(items)})</summary>'
        '<p class="muted">Other retained passages for manual comparison. Their inclusion '
        'does not establish that they address this statement.</p>'
        + blocks + '</details>'
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


def _safe_reference_href(href: str) -> bool:
    import ipaddress
    try:
        parsed = urlsplit(href)
        host = parsed.hostname or ''
        if (parsed.scheme not in {'http','https'} or not host or parsed.username or parsed.password
                or host == 'localhost' or host.endswith(('.localhost','.local','.internal'))):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except ValueError:
        return False


def _render_formatted_reference(source: dict, fallback: str) -> str:
    # An identifier already shown to reach a different work must not be offered
    # as a route to the cited source. Those records stay audit-only.
    suppressed = _conflicting_reference_dois(source)
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
    # Replace only a uniquely bound trailing link label after the actual
    # bibliographic title/year. Never delete a title or author that is linked.
    observations = sorted(source.get('submitted_hyperlink_labels') or [], key=lambda o:o.get('start',-1))
    groups = []
    for observation in observations:
        start,end=observation.get('start',-1),observation.get('end',-1)
        href=observation.get('href','')
        if not (isinstance(start,int) and isinstance(end,int) and 0<=start<end<=len(text)
                and text[start:end]==observation.get('label') and _safe_reference_href(href)):
            continue
        if groups and groups[-1][2]==href and not text[groups[-1][1]:start].strip(' \n\t|–—-') and start>=groups[-1][1]:
            groups[-1]=(groups[-1][0],end,href)
        else:
            groups.append((start,end,href))
    for start,end,href in reversed(groups):
        prefix=text[:start]
        # Native Word hyperlinks may bind only the suffix of an already
        # written URL. Replacing that suffix with the whole URL duplicates it.
        preceding_url = re.search(r'https?://[^\s<>"]*$', prefix)
        if preceding_url and text[preceding_url.start():end] == href:
            continue
        title=_field_characters(source.get('title') or '')[0]
        if (title and title in _field_characters(prefix)[0] and str(source.get('year') or '') in prefix
                and not text[end:].strip(' .\n\t') and text[start:end]!=href):
            text=text[:start]+href+text[end:]
            flags=flags[:start]+[(False,False)]*len(href)+flags[end:]
    # Font-run boundaries must not split one URL into separately linked pieces.
    for match in re.finditer(r'https?://[^\s<>"]+',text):
        flags[match.start():match.end()] = [(False,False)] * len(match.group())
    chunks = []
    cursor = 0
    while cursor < len(text):
        italic, bold = flags[cursor]
        end = cursor + 1
        while end < len(text) and flags[end] == (italic, bold):
            end += 1
        value = _link_reference_text(text[cursor:end], suppressed_dois=suppressed)
        if italic:
            value = f"<em>{value}</em>"
        if bold:
            value = f"<strong>{value}</strong>"
        chunks.append(value)
        cursor = end
    links = [href for href in dict.fromkeys(source.get('submitted_hyperlinks') or [])
             if _safe_reference_href(href) and href not in text]
    return "".join(chunks) + ''.join(
        f'<br><a class="reference-url" href="{escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(href)}</a>'
        for href in links)


# A reference may carry its identifier as a bare `doi:` string rather than a
# doi.org URL. That is a resolvable address the reader should be able to open,
# so it is linked to its canonical resolver while the original wording is
# displayed unchanged. Only the explicit `doi:` form is matched; a bare
# `10.x/...` is left alone because it cannot be told from ordinary text safely.
_REFERENCE_ADDRESS_RE = re.compile(
    r"(?P<url>https?://[^\s<>\"\u201c\u201d]+)"
    r"|(?P<doi>\bdoi:\s*10\.\d{4,9}/[^\s<>\"\u201c\u201d]+)",
    re.IGNORECASE,
)


def _conflicting_reference_dois(source: dict) -> frozenset[str]:
    """Identifiers this reference is known to share with a different work."""
    values = set()
    for finding in source.get("reference_findings") or []:
        for record in (finding.get("records") or []):
            observed = (record.get("observed") or {}) if isinstance(record, dict) else {}
            doi = str(observed.get("doi") or "").strip().casefold()
            if doi:
                values.add(doi)
    difference = source.get("field_difference") or {}
    if str(difference.get("field_name") or "") == "doi":
        for key in ("submitted_value", "located_value"):
            value = str(difference.get(key) or "").strip().casefold()
            if value:
                values.add(value)
    return frozenset(values)


def _link_reference_text(text: str, *, suppressed_dois: frozenset[str] = frozenset()) -> str:
    chunks, cursor = [], 0
    for match in _REFERENCE_ADDRESS_RE.finditer(text):
        if match.start() < cursor:
            continue
        label = match.group().rstrip(".,;)")
        if match.group("doi"):
            identifier = label.split(":", 1)[1].strip()
            if identifier.casefold() in suppressed_dois:
                continue
            href = "https://doi.org/" + identifier
        else:
            href = label
        chunks.append(escape(text[cursor:match.start()]))
        chunks.append(
            f'<a class="reference-url" href="{escape(href, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
        )
        cursor = match.start() + len(label)
    chunks.append(escape(text[cursor:]))
    return "".join(chunks)
