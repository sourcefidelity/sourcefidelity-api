"""Regenerate a derived report projection from immutable paper/source evidence."""

from __future__ import annotations

import uuid
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import fitz
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.report import Report, ReportPaperArtifactRecord, VerificationReportRecord
from app.services.evidence_report import (
    attach_quotation_difference_geometry,
    build_evidence_report_view,
)
from app.services.paper_extraction import (
    PaperExtractionArtifact,
    _body_before_reference_section,
    _word_count,
    extract_paper_evidence,
    report_anchor_citations,
)
from app.services.presentation_anchors import bind_citations_to_pdf
from app.services.text_extractor import extract_text_from_bytes
from app.services.parsers.apa_parser import ApaParser
from app.services.parsers.mla_parser import MlaParser
from app.services.passage_relevance import assess_abstract_relevance
from app.services.reference_parser import extract_reference_section
from app.services.report_paper_artifact import attach_report_paper_artifact
from app.services.storage.backend import StorageBackend


def refresh_evidence_report_projection(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    assess_abstracts: bool = False,
    reextract_citations: bool = False,
    submitted_link_updates: list[dict] | None = None,
    verification_replacements: dict[str, str] | None = None,
    reassess_reference_formatting: bool = False,
) -> dict:
    """Refresh only the report-facing projection; Evidence Packages stay immutable."""
    parsed_id = report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
    report = session.get(Report, parsed_id)
    if report is None:
        raise ValueError("Report not found")
    job = session.scalar(select(Job).where(Job.id == report.job_id).with_for_update())
    if job is None or str(job.status) != "completed":
        raise ValueError("Report job must be completed before projection correction")
    latest_report = session.scalar(
        select(Report)
        .where(Report.job_id == report.job_id)
        .order_by(Report.report_version.desc(), Report.created_at.desc())
        .limit(1)
        .with_for_update()
    )
    if latest_report is None:
        raise ValueError("Report lineage is unavailable")
    # Projection refreshes extend the current linear lineage even if a caller
    # followed an older report URL. They never fork or overwrite history.
    base_report = latest_report
    paper_artifact = session.scalar(
        select(ReportPaperArtifactRecord).where(
            ReportPaperArtifactRecord.job_id == report.job_id
        )
    )
    if job is None or paper_artifact is None:
        raise ValueError("Report dependencies are unavailable")
    extraction = PaperExtractionArtifact.model_validate(job.extraction_payload or {})
    if not extraction.total_word_count:
        extraction = _backfill_word_counts(extraction, paper_artifact, backend)

    aggregate = deepcopy(base_report.report_json or {})
    if verification_replacements:
        from app.services.paper_workflow import _report_member
        if reextract_citations:
            raise ValueError('Evidence replacement cannot also re-extract citations')
        known = set(aggregate.get('report_ids') or [])
        if len(set(verification_replacements.values())) != len(verification_replacements):
            raise ValueError('Replacement members must remain distinct')
        for old_id, new_id in verification_replacements.items():
            if old_id not in known or new_id in known:
                raise ValueError('Replacement is not a new version of a retained member')
            old = session.get(VerificationReportRecord, uuid.UUID(old_id))
            new = session.get(VerificationReportRecord, uuid.UUID(new_id))
            _validate_verification_replacement(old, new, job)
            replacement = _report_member(new)
            matches = 0
            for group in aggregate.get('citation_groups') or []:
                for member in group.get('members') or []:
                    if member.get('report_id') == old_id:
                        member.update(replacement)
                        matches += 1
            if matches != 1:
                raise ValueError('Replacement must identify exactly one aggregate member')
        aggregate['report_ids'] = [verification_replacements.get(r,r) for r in aggregate['report_ids']]
        aggregate['evidence_reassessment'] = {'base_report_id':str(base_report.id),
                                            'replacements':dict(verification_replacements)}
    if submitted_link_updates:
        from app.services.submitted_links import SubmittedLink, initial_observations
        expected = {(r.reference_id, row.kind): row for r in extraction.references for row in initial_observations(r)}
        updates = {}
        for value in submitted_link_updates:
            row = SubmittedLink.model_validate(value)
            key = (row.reference_id, row.kind)
            original = expected.get(key)
            if (original is None or row.reference_snapshot_sha256 != original.reference_snapshot_sha256
                    or row.submitted_sha256 != original.submitted_sha256 or row.request_sha256 != original.request_sha256
                    or row.state != 'observed'):
                raise ValueError('Submitted-link update is not bound to this reference')
            updates[key] = row.model_dump(mode='json')
        aggregate['submitted_link_observations'] = [r for r in aggregate.get('submitted_link_observations', [])
            if (r.get('reference_id'), r.get('kind')) not in updates] + list(updates.values())
    citation_groups = list(aggregate.get("citation_groups") or [])
    abstract_assessments = 0
    if assess_abstracts:
        claims = {claim.claim_id: claim for claim in extraction.citation_claims}
        for group in citation_groups:
            claim = claims.get(group.get("claim_id"))
            if claim is None:
                continue
            for member in group.get("members") or []:
                abstract = member.get("abstract_evidence") or {}
                if not abstract.get("text"):
                    continue
                member["abstract_relevance"] = assess_abstract_relevance(
                    claim, str(abstract["text"]), source_title=next((r.title for r in extraction.references if r.reference_id == member.get('reference_id')), "")
                )
                abstract_assessments += 1
        aggregate["citation_groups"] = citation_groups

    report_ids = [uuid.UUID(value) for value in aggregate.get("report_ids") or []]
    verification_records = (
        list(
            session.scalars(
                select(VerificationReportRecord).where(
                    VerificationReportRecord.id.in_(report_ids)
                )
            )
        )
        if report_ids
        else []
    )
    if len(verification_records) != len(set(report_ids)):
        raise ValueError("An immutable Evidence Package is unavailable")
    projected_artifact = paper_artifact
    if reextract_citations:
        if assess_abstracts:
            raise ValueError("Citation correction cannot also request new semantic judgments")
        extraction, projected_artifact, correction = _correct_citation_projection(
            backend, job, paper_artifact, extraction, verification_records
        )
        from app.services.paper_workflow import _citation_group_index, _report_member
        aggregate["citation_groups"] = _citation_group_index(
            extraction.citation_claims,
            [_report_member(record) for record in verification_records],
            aggregate.get("source_failures") or [],
        )
        aggregate["reference_consistency"] = extraction.reference_consistency.model_dump(mode="json")
        aggregate["citation_unit_count"] = len(extraction.citation_claims)
        aggregate["rejected_citation_count"] = len(extraction.rejected_citations)
        aggregate["extraction_snapshot"] = extraction.model_dump(mode="json")
        anchor_correction = {
            "extraction_sha256": correction["corrected_extraction_sha256"],
            "anchors": projected_artifact.presentation_evidence["citation_anchors"],
        }
        aggregate["citation_anchor_correction"] = anchor_correction
        aggregate["projection_correction"] = {
            **correction, "base_report_id": str(base_report.id),
            "retained_report_ids": [str(value) for value in report_ids],
        }
        # Future member uploads must use the same corrected extraction as the
        # new report. Historical report projections and paper anchors stay put.
        job.extraction_payload = extraction.model_dump(mode="json")
        job.upload_evidence = {**deepcopy(job.upload_evidence or {}),
                               "citation_anchor_correction": deepcopy(anchor_correction)}
        job.verification_summary = {
            **deepcopy(job.verification_summary or {}),
            "citation_groups": deepcopy(aggregate["citation_groups"]),
        }
    if reassess_reference_formatting:
        from app.services.reference_formatting import assess_reference_formatting
        from app.services.source_type import classify_reference_source_kind
        layout = extraction.reference_layout
        # DOCX style observations bind to the submitted DOCX, not the retained
        # converted PDF used for page geometry.
        if layout is None or layout.content_sha256 != job.input_sha256:
            raise ValueError('Reference style observations are not bound to the retained input')
        before = _json_digest(extraction.model_dump(mode='json'))
        extraction = extraction.model_copy(deep=True)
        for reference in extraction.references:
            kind = classify_reference_source_kind(reference.raw_ref, title=reference.title, url=reference.url)
            if reference.source_kind == 'webpage' and kind.kind == 'monograph' and kind.confidence == 'high':
                reference.source_kind = kind.kind
                reference.source_kind_confidence = kind.confidence
                reference.source_kind_evidence = list(kind.evidence)
        extraction.reference_formatting = assess_reference_formatting(
            layout, references=extraction.references,
        )
        # Preserve independently assessed reference order; no new bibliography
        # segmentation or citation extraction occurs in a style-only correction.
        previous_formatting = PaperExtractionArtifact.model_validate(job.extraction_payload).reference_formatting
        if previous_formatting is not None:
            extraction.reference_formatting.order_result = previous_formatting.order_result
            order = previous_formatting.order_result
            if order is not None and order.status != 'not_assessed':
                assessment = extraction.reference_formatting
                assessment.assessed_rule_ids = list(dict.fromkeys([
                    *assessment.assessed_rule_ids, order.rule_id,
                ]))
                assessment.status = 'partial'
            extraction.reference_formatting = type(extraction.reference_formatting).model_validate(
                extraction.reference_formatting.model_dump(mode='json')
            )
        after = _json_digest(extraction.model_dump(mode='json'))
        aggregate['reference_formatting_correction'] = {
            'version': 'reference-formatting-reassessment-v1', 'base_report_id': str(base_report.id),
            'previous_extraction_sha256': before, 'corrected_extraction_sha256': after,
            'source_sha256': job.input_sha256,
            'presentation_sha256': paper_artifact.presentation_sha256,
        }
        aggregate['extraction_snapshot'] = extraction.model_dump(mode='json')
        aggregate['reference_formatting'] = extraction.reference_formatting.model_dump(mode='json')
        # Citation geometry is unchanged; only renew its full-snapshot binding.
        if aggregate.get('citation_anchor_correction'):
            aggregate['citation_anchor_correction']['extraction_sha256'] = after
            job.upload_evidence = {**deepcopy(job.upload_evidence or {}),
                'citation_anchor_correction': deepcopy(aggregate['citation_anchor_correction'])}
        job.extraction_payload = extraction.model_dump(mode='json')
    successor = Report(
        job_id=report.job_id,
        total_references=base_report.total_references,
        verified_references=base_report.verified_references,
        summary=base_report.summary,
        report_markdown=base_report.report_markdown,
        report_json=aggregate,
        report_version=latest_report.report_version + 1,
        previous_report_id=latest_report.id,
        amendment_reason=(
            "bounded_evidence_reassessment" if verification_replacements else
            "reference_formatting_reassessment" if reassess_reference_formatting else
            "citation_projection_correction" if reextract_citations else
            "abstract_relevance_assessment" if assess_abstracts else "submitted_link_recheck" if submitted_link_updates else "projection_refresh"
        ),
    )
    session.add(successor)
    session.flush()
    attach_report_paper_artifact(session, job=job, report=successor)
    view = build_evidence_report_view(
        report=successor,
        job=job,
        extraction=extraction,
        verification_records=verification_records,
        paper_artifact=projected_artifact,
    )
    if paper_artifact.presentation_storage_key:
        view = attach_quotation_difference_geometry(
            view, backend.download(paper_artifact.presentation_storage_key)
        )
    successor.report_json = {**aggregate, "evidence_report": view}
    if verification_replacements:
        job.verification_summary = {**deepcopy(job.verification_summary or {}),
            'report_ids':list(aggregate['report_ids']),
            'citation_groups':deepcopy(aggregate['citation_groups'])}
    session.commit()
    return {
        "report_id": str(successor.id),
        "previous_report_id": str(latest_report.id),
        "report_version": successor.report_version,
        "abstract_assessments": abstract_assessments,
        "citation_extraction_corrected": reextract_citations,
        "citation_count": len(view.get("citations") or []),
        "reference_finding_count": len(view.get("reference_practice") or []),
        "word_counts": dict(view.get("word_counts") or {}),
    }


def _validate_verification_replacement(old, new, job):
    """No new paper, member, source or extracted generation through re-ranking."""
    from app.services.verification_report import _payload_digest
    if old is None or new is None:
        raise ValueError('Replacement evidence is unavailable')
    for record in (old,new):
        if (record.scope_type != job.scope_type or record.scope_id != job.scope_id
                or record.paper_version_id != job.paper_version_id
                or record.evidence_sha256 != _payload_digest(record.report_payload)):
            raise ValueError('Replacement evidence scope or fingerprint differs')
    a,b=old.report_payload,new.report_payload
    if a.get('claim') != b.get('claim') or a.get('source_binding') != b.get('source_binding'):
        raise ValueError('Replacement claim or member differs')
    for key in ('representation_id','content_sha256','canonical_work_id'):
        if (a.get('source_identity') or {}).get(key) != (b.get('source_identity') or {}).get(key):
            raise ValueError('Replacement source differs')
    for key in ('extracted_text_sha256','extraction_version'):
        if (a.get('authoritative_evidence_package') or {}).get(key) != (b.get('authoritative_evidence_package') or {}).get(key):
            raise ValueError('Replacement extraction differs')


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _correct_citation_projection(backend, job, paper_artifact, previous, records):
    """Re-extract retained input without models and reject changed evidence targets."""
    if not paper_artifact.storage_key or not paper_artifact.presentation_storage_key:
        raise ValueError("Retained source and presentation are required")
    source = backend.download(paper_artifact.storage_key)
    paper = backend.download(paper_artifact.presentation_storage_key)
    if (hashlib.sha256(source).hexdigest() != paper_artifact.content_sha256
            or hashlib.sha256(paper).hexdigest() != paper_artifact.presentation_sha256):
        raise ValueError("Retained paper hash mismatch")
    corrected = extract_paper_evidence(
        extract_text_from_bytes(source, job.filename),
        paper_version_id=previous.paper_version_id, format_hint=previous.citation_format,
        use_llm_boundaries=False, use_llm_atomizer=False, use_llm_reference_fallback=False,
    )
    if [r.model_dump(mode="json") for r in previous.references] != [r.model_dump(mode="json") for r in corrected.references]:
        raise ValueError("Reference changes require source reanalysis, not projection correction")
    _validate_retained_claims(previous, corrected, records)
    corrected = corrected.model_copy(update={
        "reference_layout": previous.reference_layout,
        "reference_formatting": previous.reference_formatting,
    })
    anchors = bind_citations_to_pdf(paper, citations=report_anchor_citations(corrected))
    if anchors.matched_citation_count != anchors.citation_count:
        raise ValueError("Corrected citation anchors need manual localization review")
    # Never update the one shared ORM artifact's presentation_evidence.
    projected = SimpleNamespace(**{
        column.name: deepcopy(getattr(paper_artifact, column.name))
        for column in paper_artifact.__table__.columns
    })
    projected.presentation_evidence = {
        **(projected.presentation_evidence or {}), "citation_anchors": anchors.model_dump(mode="json"),
    }
    return corrected, projected, {
        "version": "citation-projection-correction-v1",
        "previous_extraction_sha256": _json_digest(previous.model_dump(mode="json")),
        "corrected_extraction_sha256": _json_digest(corrected.model_dump(mode="json")),
        "source_sha256": paper_artifact.content_sha256,
        "presentation_sha256": paper_artifact.presentation_sha256,
        "anchors_sha256": _json_digest(anchors.model_dump(mode="json")),
        "retained_evidence_targets_unchanged": True,
        "llm_calls": 0, "source_search_calls": 0,
    }


def _validate_retained_claims(previous, corrected, records):
    old = {c.claim_id: c.model_dump(mode="json") for c in previous.citation_claims}
    new = {c.claim_id: c.model_dump(mode="json") for c in corrected.citation_claims}
    for record in records:
        package = record.report_payload.get("authoritative_evidence_package") or {}
        claim_id = package.get("claim_id")
        if claim_id not in old or claim_id not in new or old[claim_id] != new[claim_id]:
            raise ValueError("A verified claim changed; fresh evidence analysis is required")


def _backfill_word_counts(
    extraction: PaperExtractionArtifact,
    paper_artifact: ReportPaperArtifactRecord,
    backend: StorageBackend,
) -> PaperExtractionArtifact:
    if not paper_artifact.presentation_storage_key:
        return extraction
    content = backend.download(paper_artifact.presentation_storage_key)
    document = fitz.open(stream=content, filetype="pdf")
    try:
        paper_text = "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()
    reference_text = extract_reference_section(paper_text, extraction.citation_format)
    parser = ApaParser if extraction.citation_format == "apa" else MlaParser
    body_text = (
        _body_before_reference_section(paper_text, reference_text, parser)
        if reference_text
        else paper_text
    )
    return extraction.model_copy(
        update={
            "total_word_count": _word_count(paper_text),
            "body_word_count": _word_count(body_text),
            "reference_word_count": _word_count(reference_text or ""),
        }
    )
