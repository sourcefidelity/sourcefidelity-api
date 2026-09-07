"""Checkpointed paper extraction, retrieval, shadow verification, and summary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job, JobStage, JobStatus
from app.services.paper_dispatch import prepare_dispatch
from app.models.report import (
    Report,
    ReportPaperArtifactRecord,
    VerificationReportRecord,
)
from app.models.source_repository import SourceRepresentationRecord
from app.models.verification_run import VerificationRunRecord
from app.services.evidence_package import _package_payload_sha256
from app.services.candidate_relationship_judgment import (
    attach_verification_candidates,
)
from app.services.citation_use_router import attach_citation_use_routes
from app.services.decisive_label_critic import apply_decisive_label_critic
from app.services.facet_evidence_judgment import (
    apply_facet_evidence_judgment,
    attach_facet_evidence_foundation,
)
from app.services.file_safety import (
    FileSafetyUnavailable,
    SafetyVerdict,
    inspect_uploaded_pdf,
)
from app.services.paper_extraction import (
    PaperExtractionArtifact,
    extract_paper_evidence,
    report_anchor_citations,
)
from app.services.parsers import detect_format
from app.services.paper_dispatch import DISPATCH_KEY, MAX_STAGE_STARTS
from app.services.reference_parser import (
    extract_and_parse_references,
    extract_reference_section,
)
from app.services.evidence_report import (
    attach_quotation_difference_geometry,
    build_evidence_report_view,
)
from app.services.evidence_obligations import attach_evidence_obligations
from app.services.llm_service import chat_completion_json
from app.services.reference_layout import extract_reference_layout_from_bytes
from app.services.reference_formatting import assess_reference_formatting
from app.services.paper_upload import (
    PaperUploadError,
    cleanup_paper_job_input,
    load_paper_job_input,
)
from app.services.paper_retention import (
    PaperRetentionPolicyError,
    resolve_paper_retention_policy,
)
from app.services.report_paper_artifact import (
    ReportPaperArtifactError,
    attach_report_paper_artifact,
    cleanup_report_paper_artifact,
    ensure_report_paper_artifact,
)
from app.services.passage_relevance import (
    apply_passage_relevance_gate,
    assess_abstract_relevance,
)
from app.services.student_statement_interpretation import (
    attach_source_blind_interpretations,
)
from app.services.retrieval.base import RepresentationKind
from app.services.source_resolver import SourceResolutionError, SourceResolver
from app.services.storage.backend import StorageBackend
from app.services.text_extractor import (
    QualifiedTextExtraction,
    TextCandidate,
    TextExtractionError,
    extract_qualified_text_from_bytes,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    VerificationEvidenceArtifact,
    attach_candidate_passage_retrieval,
    attach_local_semantic_retrieval_rescue,
    authorize_representation,
    build_passage_evidence,
)
from app.services.verification_report import persist_verification_report, _payload_digest
from app.services.verification_run import (
    VerificationRunError,
    VerificationRunCleanupPending,
    VerificationRunRequest,
    begin_verification_run,
    complete_active_verification_run,
    renew_verification_run_lease,
)


PAPER_WORKFLOW_VERSION = "paper-workflow-v7"
TARGETED_SOURCE_REFRESH_KEY = "targeted_source_refresh"
MAX_REPORT_ABSTRACT_CHARACTERS = 6_000
_STAGE_ORDER = {
    JobStage.UPLOADED: 0,
    JobStage.EXTRACTING: 1,
    JobStage.EXTRACTED: 2,
    JobStage.RETRIEVING: 3,
    JobStage.RETRIEVED: 4,
    JobStage.VERIFYING: 5,
    JobStage.VERIFIED: 6,
    JobStage.FINALIZING: 7,
    JobStage.COMPLETED: 8,
}


class PaperWorkflowError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def extract_paper_job(
    session: Session,
    backend: StorageBackend,
    job_id,
    *,
    llm_enabled: bool | None = None,
) -> dict:
    job = _job(session, job_id)
    if job.extraction_payload and _at_or_after(job, JobStage.EXTRACTED):
        artifact = _extraction(job)
        return _extraction_counts(artifact)
    _enter(job, JobStage.EXTRACTING)
    session.commit()
    try:
        _, content = load_paper_job_input(session, backend, job.id)
        qualified_text = extract_qualified_text_from_bytes(content, job.filename)
        text = qualified_text.selected.text
        if len(text) > settings.MAX_PAPER_TEXT_LENGTH_CHARS:
            raise PaperWorkflowError(
                "paper_text_limit_exceeded",
                "Extracted paper text exceeds the configured processing limit",
            )
        allow_llm = settings.PAPER_LLM_PROCESSING_ENABLED if llm_enabled is None else llm_enabled
        detected_parser = detect_format(text)
        format_hint = "mla" if detected_parser.__name__.lower().startswith("mla") else "apa"
        reference_candidate, reference_counts = _select_reference_candidate(
            qualified_text,
            citation_format=format_hint,
        )
        artifact = extract_paper_evidence(
            text,
            paper_version_id=job.paper_version_id,
            reference_text=reference_candidate.text,
            format_hint=format_hint,
            use_llm_boundaries=allow_llm,
            use_llm_atomizer=False,
            use_llm_reference_fallback=allow_llm,
        )
        reference_layout = extract_reference_layout_from_bytes(
            content,
            job.filename,
            references=artifact.references,
            citation_format=artifact.citation_format,
        )
        reference_formatting = assess_reference_formatting(reference_layout)
        reference_consistency = artifact.reference_consistency
        if reference_consistency is not None and reference_layout.status in {
            "complete",
            "partial",
        }:
            reference_consistency = reference_consistency.model_copy(
                update={
                    "formatting_reason_codes": [
                        "layout_evidence_ready_style_rules_not_accepted"
                    ]
                }
            )
        artifact = artifact.model_copy(
            update={
                "reference_layout": reference_layout,
                "reference_formatting": reference_formatting,
                "reference_consistency": reference_consistency,
                "text_extraction": {
                    **qualified_text.evidence(),
                    "reference_backend": reference_candidate.backend,
                    "reference_text_sha256": reference_candidate.sha256,
                    "deterministic_reference_counts": reference_counts,
                },
            }
        )
        ensure_report_paper_artifact(
            session,
            backend,
            job=job,
            content=content,
            citations=report_anchor_citations(artifact),
        )
    except (
        PaperUploadError,
        ReportPaperArtifactError,
        TextExtractionError,
        ValueError,
    ) as exc:
        raise PaperWorkflowError(
            getattr(exc, "code", "paper_extraction_failed"), str(exc)
        ) from exc
    job.extraction_payload = artifact.model_dump(mode="json")
    job.stage = JobStage.EXTRACTED
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
    # Apply the stable upload-time policy only after the bounded extraction
    # checkpoint exists and before any source retrieval or model work.
    retention_mode = (job.upload_evidence or {}).get(
        "paper_retention_mode", "temporary"
    )
    try:
        retention_policy = resolve_paper_retention_policy(retention_mode)
    except PaperRetentionPolicyError as exc:
        raise PaperWorkflowError(exc.code, str(exc)) from exc
    retention = retention_policy.after_extraction(
        lambda: cleanup_paper_job_input(session, backend, job.id)
    )
    if not retention.completed:
        job.input_expires_at = datetime.now(timezone.utc)
        session.commit()
    return _extraction_counts(artifact)


def retrieve_paper_sources(
    session: Session,
    backend: StorageBackend,
    job_id,
    *,
    resolver: SourceResolver | None = None,
) -> dict:
    job = _job(session, job_id)
    if job.source_results is not None and _at_or_after(job, JobStage.RETRIEVED):
        return _source_counts(list(job.source_results))
    artifact = _extraction(job)
    _enter(job, JobStage.RETRIEVING)
    session.commit()
    cited_reference_ids = {
        reference_id
        for claim in artifact.citation_claims
        for reference_id in claim.reference_ids
    }
    active_resolver = resolver or SourceResolver()
    cited_references = [
        reference
        for reference in artifact.references
        if reference.reference_id in cited_reference_ids
    ]
    # Batch-capable title providers and deferred DOI providers are part of the
    # ordinary paper workflow.  Leaving them unused made single-reference
    # fallback behavior look like completed discovery.
    prefetch_titles = getattr(active_resolver, "prefetch_title_candidates", None)
    if prefetch_titles is not None:
        prefetch_titles(
            [
                (reference.title, reference.author or None)
                for reference in cited_references
                if reference.title
            ]
        )
    prefetch_dois = getattr(active_resolver, "prefetch_deferred_dois", None)
    if prefetch_dois is not None:
        prefetch_dois(
            [reference.doi for reference in cited_references if reference.doi]
        )
    results: list[dict] = list(job.source_results or [])
    completed_reference_ids = {item["reference_id"] for item in results}

    def checkpoint(result_record: dict) -> None:
        results.append(result_record)
        completed_reference_ids.add(result_record["reference_id"])
        # Retrieval is intentionally serialized and may outlive the transient
        # verification lease. Keep every already-admitted source alive while
        # later references are still being discovered; otherwise the earliest
        # clean source can expire before verification begins.
        for pending in results:
            if pending.get("status") != "transient_authorized":
                continue
            try:
                renew_verification_run_lease(
                    session,
                    pending["verification_run_id"],
                    scope_type=job.scope_type,
                    scope_id=job.scope_id,
                    lease_seconds=settings.PAPER_SOURCE_RETRIEVAL_LEASE_SECONDS,
                )
            except VerificationRunError as exc:
                pending.update(
                    status="unavailable",
                    reason_code="transient_verification_run_unavailable",
                    detail=type(exc).__name__,
                )
        job.source_results = list(results)
        job.updated_at = datetime.now(timezone.utc)
        session.commit()

    for reference in artifact.references:
        if (
            reference.reference_id not in cited_reference_ids
            or reference.reference_id in completed_reference_ids
        ):
            continue
        result_record = {"reference_id": reference.reference_id}
        try:
            result = active_resolver.resolve_reference(reference)
        except SourceResolutionError as exc:
            result_record.update(
                status="unavailable",
                reason_code="source_not_found",
                reference_discovery_trace=exc.reference_discovery_trace,
                reference_discovery=exc.reference_discovery,
            )
            dependencies = _retryable_provider_dependencies(
                exc.reference_discovery_trace,
                exc.reference_discovery,
            )
            if dependencies:
                result_record["retryable_provider_dependencies"] = dependencies
            checkpoint(result_record)
            continue
        metadata = result.metadata or {}
        result_record["reference_discovery_trace"] = metadata.get(
            "reference_discovery_trace"
        )
        result_record["reference_discovery"] = metadata.get("reference_discovery")
        representation_id = metadata.get("repository_representation_id")
        admission = metadata.get("durable_admission") or {}
        if admission.get("state") == "accepted":
            representation_id = admission.get("representation_id") or representation_id
        if representation_id:
            try:
                authorize_representation(
                    session,
                    backend,
                    representation_id=representation_id,
                    scope_type=job.scope_type,
                    scope_id=job.scope_id,
                )
            except Exception:
                result_record.update(
                    status="unavailable",
                    reason_code="durable_representation_not_authorized",
                )
            else:
                result_record.update(
                    status="durable_authorized",
                    representation_id=str(representation_id),
                    source_name=result.source_name,
                )
            checkpoint(result_record)
            continue
        representation = result.representation
        if representation is None or not representation.content:
            abstract_evidence = _bounded_abstract_evidence(
                result.abstract,
                source_name=result.source_name,
            )
            result_record.update(
                status="abstract_only" if abstract_evidence else "unavailable",
                reason_code="full_text_unavailable",
                abstract_available=bool(abstract_evidence),
            )
            if abstract_evidence:
                result_record["abstract_evidence"] = abstract_evidence
            checkpoint(result_record)
            continue
        try:
            cleanliness = _transient_cleanliness(representation)
        except PaperWorkflowError as exc:
            result_record.update(
                status="unavailable",
                reason_code=exc.code,
            )
            checkpoint(result_record)
            continue
        confidence = metadata.get("identity_confidence")
        if confidence != "high":
            result_record.update(status="unavailable", reason_code="identity_not_verified")
            checkpoint(result_record)
            continue
        completeness = (
            representation.completeness
            or metadata.get("completeness")
            or "not_assessed"
        )
        try:
            run = begin_verification_run(
                session,
                backend,
                VerificationRunRequest(
                    paper_version_id=job.paper_version_id,
                    scope_type=job.scope_type,
                    scope_id=job.scope_id,
                    canonical_work_id=_canonical_work_id(reference),
                    representation=representation,
                    acquisition_route=result.source_name,
                    identity_verdict="verified",
                    identity_confidence=1.0,
                    completeness_verdict=completeness,
                    cleanliness_verdict=cleanliness,
                    text_quality=metadata.get("text_quality") or "not_assessed",
                    edition_or_version=next(
                        (location.version for location in result.locations if location.version),
                        metadata.get("edition_or_version"),
                    ),
                ),
                lease_seconds=settings.PAPER_SOURCE_RETRIEVAL_LEASE_SECONDS,
            )
        except Exception as exc:
            result_record.update(
                status="unavailable",
                reason_code="transient_source_admission_failed",
                detail=type(exc).__name__,
            )
        else:
            result_record.update(
                status="transient_authorized",
                verification_run_id=str(run.id),
                source_name=result.source_name,
            )
        checkpoint(result_record)
    job.source_results = results
    job.stage = JobStage.RETRIEVED
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
    return _source_counts(results)


def verify_paper_sources(
    session_factory,
    backend: StorageBackend,
    job_id,
    *,
    llm_enabled: bool | None = None,
) -> dict:
    with session_factory() as session:
        job = _job(session, job_id)
        if job.verification_summary and _at_or_after(job, JobStage.VERIFIED):
            return dict(job.verification_summary)
        artifact = _extraction(job)
        _enter(job, JobStage.VERIFYING)
        session.commit()
        source_results = list(job.source_results or [])
        scope_type, scope_id = job.scope_type, job.scope_id
        refresh_reference_ids = set(_refresh_reference_ids(job))
        previous_summary = dict(job.verification_summary or {})
        # The task wrapper persists each start under the job execution lock.
        # Direct service calls remain bounded by the existing source lease.
        starts = (job.upload_evidence or {}).get(DISPATCH_KEY, {}).get("stage_starts", {}).get("verify", 0)
        retain_for_retry = starts < MAX_STAGE_STARTS
        references_by_id = {
            reference.reference_id: reference for reference in artifact.references
        }
    allow_llm = settings.PAPER_LLM_PROCESSING_ENABLED if llm_enabled is None else llm_enabled
    claims_by_reference = _claims_by_reference(artifact.citation_claims)

    report_ids: list[str] = []
    report_members: list[dict] = []
    failures: list[dict] = []
    if refresh_reference_ids and previous_summary:
        previous_report_ids = [
            uuid.UUID(value) for value in previous_summary.get("report_ids", [])
        ]
        with session_factory() as session:
            previous_records = (
                list(
                    session.scalars(
                        select(VerificationReportRecord).where(
                            VerificationReportRecord.id.in_(previous_report_ids)
                        )
                    )
                )
                if previous_report_ids
                else []
            )
        for record in previous_records:
            member = _report_member(record)
            if member["reference_id"] in refresh_reference_ids:
                continue
            report_ids.append(str(record.id))
            report_members.append(member)
        failures.extend(
            failure
            for failure in previous_summary.get("source_failures", [])
            if failure.get("reference_id") not in refresh_reference_ids
        )
    for source_result in source_results:
        reference_id = source_result["reference_id"]
        if refresh_reference_ids and reference_id not in refresh_reference_ids:
            continue
        reference = references_by_id.get(reference_id)
        claims = claims_by_reference.get(reference_id, [])
        if reference is None:
            failures.append(
                {"reference_id": reference_id, "reason_code": "active_reference_missing"}
            )
            continue
        if not claims or source_result["status"] not in {
            "durable_authorized", "transient_authorized"
        }:
            if claims:
                failure = {
                    "reference_id": reference_id,
                    "reason_code": _report_source_failure_reason(source_result),
                }
                if source_result.get("abstract_available"):
                    failure["abstract_available"] = True
                if source_result.get("abstract_evidence"):
                    failure["abstract_evidence"] = dict(
                        source_result["abstract_evidence"]
                    )
                    if allow_llm:
                        abstract_text = str(
                            source_result["abstract_evidence"].get("text") or ""
                        )
                        failure["abstract_relevance_by_claim"] = {
                            claim.claim_id: assess_abstract_relevance(
                                claim, abstract_text
                            )
                            for claim in claims
                        }
                failures.append(failure)
            continue
        if source_result["status"] == "durable_authorized":
            with session_factory() as session:
                source = authorize_representation(
                    session,
                    backend,
                    representation_id=source_result["representation_id"],
                    scope_type=scope_type,
                    scope_id=scope_id,
                )
                artifacts = [
                    _shadow_artifact(
                        source,
                        claim,
                        allow_llm,
                        active_reference_id=reference_id,
                        cited_author_label=reference.author,
                    )
                    for claim in claims
                ]
                for evidence in artifacts:
                    record = persist_verification_report(
                        session,
                        evidence,
                        scope_type=scope_type,
                        scope_id=scope_id,
                    )
                    report_ids.append(str(record.id))
                    report_members.append(_report_member(record))
                session.commit()
        else:
            with session_factory() as session:
                retained = _retained_run_reports(
                    session, source_result["verification_run_id"],
                    scope_type=scope_type, scope_id=scope_id,
                    paper_version_id=artifact.paper_version_id,
                    reference_id=reference_id, claims=claims,
                )
                if retained is not None:
                    report_ids.extend(str(record.id) for record in retained)
                    report_members.extend(_report_member(record) for record in retained)
                    continue

            def processor(_session, source, _run_id):
                return [
                    _shadow_artifact(
                        source,
                        claim,
                        allow_llm,
                        active_reference_id=reference_id,
                        cited_author_label=reference.author,
                    )
                    for claim in claims
                ]

            try:
                with session_factory() as session:
                    renew_verification_run_lease(
                        session,
                        source_result["verification_run_id"],
                        scope_type=scope_type,
                        scope_id=scope_id,
                        lease_seconds=settings.PAPER_SOURCE_RETRIEVAL_LEASE_SECONDS,
                    )
                completed = complete_active_verification_run(
                    session_factory,
                    backend,
                    source_result["verification_run_id"],
                    scope_type=scope_type,
                    scope_id=scope_id,
                    processor=processor,
                    retain_on_retryable_failure=retain_for_retry,
                )
            except VerificationRunCleanupPending:
                # The evidence transaction succeeded. Cleanup is separately
                # leased/retried and must not erase evidence availability.
                with session_factory() as session:
                    retained = _retained_run_reports(
                        session, source_result["verification_run_id"],
                        scope_type=scope_type, scope_id=scope_id,
                        paper_version_id=artifact.paper_version_id,
                        reference_id=reference_id, claims=claims,
                    )
                    if retained is None:
                        raise PaperWorkflowError("verification_report_missing", "Committed run evidence is unavailable") from None
                    report_ids.extend(str(record.id) for record in retained)
                    report_members.extend(_report_member(record) for record in retained)
                continue
            except VerificationRunError as exc:
                # An acquired source whose run can no longer be checked is an
                # interrupted workflow, not a successful source-absence result.
                raise PaperWorkflowError(
                    "transient_verification_run_unavailable",
                    "The acquired source could not complete verification",
                ) from exc
            completed_ids = [report_id for report_id, _version in completed.reports]
            report_ids.extend(str(report_id) for report_id in completed_ids)
            with session_factory() as session:
                for report_id in completed_ids:
                    record = session.get(VerificationReportRecord, report_id)
                    if record is None:
                        raise PaperWorkflowError(
                            "verification_report_missing",
                            "A completed verification run did not persist its report",
                        )
                    report_members.append(_report_member(record))

    summary = {
        "workflow_version": PAPER_WORKFLOW_VERSION,
        "model_processing_enabled": allow_llm,
        "relationship_mode": "shadow",
        "reports_persisted": len(report_ids),
        "report_ids": report_ids,
        "source_failures": failures,
        "citation_groups": _citation_group_index(
            artifact.citation_claims,
            report_members,
            failures,
        ),
        "decision_applied": False,
    }
    with session_factory() as session:
        job = _job(session, job_id)
        job.verification_summary = summary
        job.stage = JobStage.VERIFIED
        job.updated_at = datetime.now(timezone.utc)
        session.commit()
    return summary


def _retained_run_reports(
    session, run_id, *, scope_type, scope_id, paper_version_id, reference_id, claims,
):
    """Recover an already-committed source batch without reopening deleted bytes.

    Reuse requires the exact scope, run, paper, member, complete claim set and
    immutable content hashes. Never reconstruct a successful batch from a
    partial set or borrow another source's reports.
    """
    run = session.get(VerificationRunRecord, uuid.UUID(str(run_id)))
    if run is None:
        return None
    if (run.scope_type, run.scope_id, run.paper_version_id) != (scope_type, scope_id, paper_version_id):
        raise PaperWorkflowError("verification_run_scope_mismatch", "Saved evidence is outside this paper scope")
    if run.terminal_outcome != "success" or run.status not in {"report_persisted", "cleanup_pending", "cleaned"}:
        return None
    records = list(session.scalars(select(VerificationReportRecord).where(
        VerificationReportRecord.verification_run_id == run.id,
    )))
    expected = {claim.claim_id: claim for claim in claims}
    retained = {}
    for record in records:
        payload = record.report_payload or {}
        package = payload.get("authoritative_evidence_package") or {}
        binding = package.get("source_binding") or {}
        identity = package.get("source_identity") or {}
        claim = expected.get(package.get("claim_id"))
        if (
            claim is None or claim.claim_id in retained
            or (record.scope_type, record.scope_id, record.paper_version_id) != (scope_type, scope_id, paper_version_id)
            or record.evidence_sha256 != _payload_digest(payload)
            or package.get("package_sha256") != _package_payload_sha256({key: value for key, value in package.items() if key != "package_sha256"})
            or package.get("paper_version_id") != paper_version_id
            or package.get("student_text") != claim.text
            or package.get("paper_character_start") != claim.passage_start
            or package.get("paper_character_end") != claim.passage_end
            or binding.get("reference_id") != reference_id
            or identity.get("content_sha256") != run.content_sha256
            or identity.get("canonical_work_id") != run.canonical_work_id
            or (identity.get("authorization_scope_type"), identity.get("authorization_scope_id")) != (scope_type, scope_id)
            or identity.get("verification_run_id") != str(run.id)
        ):
            raise PaperWorkflowError("verification_report_binding_invalid", "Saved evidence does not match the active paper and source")
        retained[claim.claim_id] = record
    if set(retained) != set(expected):
        raise PaperWorkflowError("verification_report_missing", "The committed source batch is incomplete")
    return [retained[claim.claim_id] for claim in claims]


def finalize_paper_job(session: Session, backend: StorageBackend, job_id) -> dict:
    job = _job(session, job_id)
    if job.status == JobStatus.COMPLETED:
        report = session.scalar(
            select(Report)
            .where(Report.job_id == job.id)
            .order_by(Report.report_version.desc(), Report.created_at.desc())
            .limit(1)
            .with_for_update()
        )
        if report is None:
            raise PaperWorkflowError("aggregate_report_missing", "Completed job has no aggregate report")
        return {"job_id": str(job.id), "report_id": str(report.id), "input_cleaned": job.input_deleted_at is not None}
    if job.status == JobStatus.FAILED:
        raise PaperWorkflowError("job_terminal", "Paper job is already terminal")
    if job.store_only and job.stage in {JobStage.EXTRACTED, JobStage.FINALIZING}:
        summary = {
            "workflow_version": PAPER_WORKFLOW_VERSION,
            "store_only": True,
            "reports_persisted": 0,
            "decision_applied": False,
        }
    elif job.stage in {JobStage.VERIFIED, JobStage.FINALIZING} and job.verification_summary is not None:
        summary = dict(job.verification_summary or {})
    else:
        raise PaperWorkflowError("job_not_ready", "Paper job is not ready to finalize")
    job.stage = JobStage.FINALIZING
    session.commit()
    extraction = _extraction(job)
    aggregate_payload = {
        **summary,
        "paper_version_id": job.paper_version_id,
        "reference_count": len(extraction.references),
        "citation_unit_count": len(extraction.citation_claims),
        "rejected_citation_count": len(extraction.rejected_citations),
        "reference_consistency": (
            extraction.reference_consistency.model_dump(mode="json")
            if extraction.reference_consistency is not None
            else None
        ),
        "reference_layout": (
            extraction.reference_layout.model_dump(mode="json")
            if extraction.reference_layout is not None
            else None
        ),
        "reference_formatting": (
            extraction.reference_formatting.model_dump(mode="json")
            if extraction.reference_formatting is not None
            else None
        ),
    }
    refresh_reference_ids = _refresh_reference_ids(job)
    refresh = dict((job.upload_evidence or {}).get(TARGETED_SOURCE_REFRESH_KEY) or {})
    previous_report = None
    if refresh_reference_ids:
        previous_report = session.scalar(
            select(Report)
            .where(Report.job_id == job.id)
            .order_by(Report.report_version.desc(), Report.created_at.desc())
            .limit(1)
            .with_for_update()
        )
    report = Report(
        job_id=job.id,
        total_references=str(len(extraction.references)),
        verified_references=str(summary.get("reports_persisted", 0)),
        summary="Inspectable evidence report; no relationship decision was applied.",
        report_json=aggregate_payload,
        report_version=(previous_report.report_version + 1 if previous_report else 1),
        previous_report_id=(previous_report.id if previous_report else None),
        amendment_reason=(
            str(refresh.get("reason") or "provider_recovery")
            if previous_report
            else None
        ),
    )
    session.add(report)
    session.flush()
    paper_artifact = attach_report_paper_artifact(
        session, job=job, report=report
    )
    verification_records = []
    report_ids = [uuid.UUID(value) for value in summary.get("report_ids", [])]
    if report_ids:
        verification_records = list(
            session.scalars(
                select(VerificationReportRecord).where(
                    VerificationReportRecord.id.in_(report_ids)
                )
            )
        )
        if len(verification_records) != len(set(report_ids)):
            raise PaperWorkflowError("verification_report_missing", "An immutable Evidence Package is unavailable")
    evidence_view = build_evidence_report_view(
        report=report,
        job=job,
        extraction=extraction,
        verification_records=verification_records,
        paper_artifact=paper_artifact,
    )
    if paper_artifact is not None and paper_artifact.presentation_storage_key:
        evidence_view = attach_quotation_difference_geometry(
            evidence_view,
            backend.download(paper_artifact.presentation_storage_key),
        )
    if previous_report and refresh:
        aggregate_payload["amendment"] = _targeted_amendment_manifest(
            refresh,
            summary,
            previous_report=previous_report,
        )
    report.report_json = {**aggregate_payload, "evidence_report": evidence_view}
    job.status = JobStatus.COMPLETED
    job.stage = JobStage.COMPLETED
    if refresh_reference_ids:
        upload_evidence = dict(job.upload_evidence or {})
        upload_evidence.pop("provider_refresh_reference_ids", None)
        upload_evidence.pop("provider_refresh_provider", None)
        upload_evidence.pop(TARGETED_SOURCE_REFRESH_KEY, None)
        job.upload_evidence = upload_evidence
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
    cleaned = cleanup_paper_job_input(session, backend, job.id)
    return {"job_id": str(job.id), "report_id": str(report.id), "input_cleaned": cleaned}


def prepare_provider_recovery_refresh(
    session: Session,
    job_id,
    *,
    provider: str,
    commit: bool = True,
) -> list[str]:
    """Reset only unresolved reference members dependent on a recovered provider."""
    job = _job(session, job_id)
    normalized_provider = provider.strip().casefold()
    if (
        not normalized_provider
        or job.status != JobStatus.COMPLETED
        or job.store_only
        or not job.extraction_payload
    ):
        return []
    source_results = list(job.source_results or [])
    affected = sorted(
        {
            str(item.get("reference_id"))
            for item in source_results
            if normalized_provider
            in {
                str(value).strip().casefold()
                for value in item.get("retryable_provider_dependencies", [])
            }
            and item.get("reference_id")
        }
    )
    if not affected:
        return []
    job.source_results = [
        item for item in source_results if item.get("reference_id") not in affected
    ]
    upload_evidence = dict(job.upload_evidence or {})
    upload_evidence["provider_refresh_provider"] = normalized_provider
    upload_evidence["provider_refresh_reference_ids"] = affected
    upload_evidence[TARGETED_SOURCE_REFRESH_KEY] = {
        "attempt_id": str(uuid.uuid4()),
        "reason": "provider_recovery",
        "reference_ids": affected,
        "previous_source_results": source_results,
        "previous_verification_summary": dict(job.verification_summary or {}),
    }
    job.upload_evidence = upload_evidence
    job.status = JobStatus.RUNNING
    job.stage = JobStage.EXTRACTED
    job.error_message = None
    job.updated_at = datetime.now(timezone.utc)
    prepare_dispatch(job, attempt_id=upload_evidence[TARGETED_SOURCE_REFRESH_KEY]["attempt_id"])
    if commit:
        session.commit()
    else:
        session.flush()
    return affected


def prepare_uploaded_source_refresh(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    reference_id: str,
    representation_id: str | uuid.UUID,
    commit: bool = True,
) -> dict:
    """Bind one admitted upload to its exact paper member and targeted refresh.

    The report-local member is only a hint.  This function independently checks
    the admitted representation, authorization scope, canonical work identity,
    and current report lineage before changing the job checkpoint.
    """
    parsed_report_id = (
        report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
    )
    requested_report = session.get(Report, parsed_report_id)
    if requested_report is None:
        raise PaperWorkflowError("report_missing", "Report does not exist")
    job = session.scalar(
        select(Job).where(Job.id == requested_report.job_id).with_for_update()
    )
    if job is None or job.store_only or not job.extraction_payload:
        raise PaperWorkflowError(
            "report_dependencies_unavailable",
            "Report dependencies are unavailable for source reanalysis",
        )
    latest_report = session.scalar(
        select(Report)
        .where(Report.job_id == job.id)
        .order_by(Report.report_version.desc(), Report.created_at.desc())
        .limit(1)
        .with_for_update()
    )
    if latest_report is None:
        raise PaperWorkflowError("report_lineage_unavailable", "Report lineage is unavailable")

    parsed_representation_id = (
        representation_id
        if isinstance(representation_id, uuid.UUID)
        else uuid.UUID(str(representation_id))
    )
    representation = session.get(SourceRepresentationRecord, parsed_representation_id)
    if representation is None:
        raise PaperWorkflowError("source_representation_missing", "Uploaded source is unavailable")
    if (
        representation.scope_type != job.scope_type
        or representation.scope_id != job.scope_id
        or representation.admission_state != "accepted"
    ):
        raise PaperWorkflowError(
            "source_representation_not_authorized",
            "Uploaded source is not admitted in this report scope",
        )
    authorized = authorize_representation(
        session,
        backend,
        representation_id=representation.id,
        scope_type=job.scope_type,
        scope_id=job.scope_id,
    )

    artifact = _extraction(job)
    reference = next(
        (item for item in artifact.references if item.reference_id == reference_id),
        None,
    )
    if reference is None or not any(
        reference_id in claim.reference_ids for claim in artifact.citation_claims
    ):
        raise PaperWorkflowError(
            "reference_not_cited", "Uploaded source target is not an active cited reference"
        )
    if not _canonical_work_matches_reference(representation, reference):
        raise PaperWorkflowError(
            "uploaded_source_identity_mismatch",
            "Uploaded source identity does not match the selected reference",
        )

    active_refresh = dict(
        (job.upload_evidence or {}).get(TARGETED_SOURCE_REFRESH_KEY) or {}
    )
    if job.status != JobStatus.COMPLETED:
        if (
            active_refresh
            and active_refresh.get("reference_ids") == [reference_id]
            and active_refresh.get("representation_id") == str(representation.id)
        ):
            return {
                "status": "already_scheduled",
                "scheduled": False,
                "job_id": str(job.id),
                "report_id": str(latest_report.id),
                "reference_id": reference_id,
                "representation_id": str(representation.id),
                "content_sha256": authorized.content_sha256,
                "attempt_id": active_refresh.get("attempt_id"),
            }
        raise PaperWorkflowError(
            "report_reanalysis_busy", "Another report reanalysis is already in progress"
        )

    source_results = list(job.source_results or [])
    current = next(
        (item for item in source_results if item.get("reference_id") == reference_id),
        None,
    )
    if (
        current
        and current.get("status") == "durable_authorized"
        and current.get("representation_id") == str(representation.id)
    ):
        return {
            "status": "already_current",
            "scheduled": False,
            "job_id": str(job.id),
            "report_id": str(latest_report.id),
            "reference_id": reference_id,
            "representation_id": str(representation.id),
            "content_sha256": authorized.content_sha256,
        }

    replacement = {
        "reference_id": reference_id,
        "status": "durable_authorized",
        "representation_id": str(representation.id),
        "source_name": "instructor_upload",
    }
    updated_results = [
        item for item in source_results if item.get("reference_id") != reference_id
    ]
    updated_results.append(replacement)
    attempt_id = str(uuid.uuid4())
    upload_evidence = dict(job.upload_evidence or {})
    upload_evidence[TARGETED_SOURCE_REFRESH_KEY] = {
        "attempt_id": attempt_id,
        "reason": "user_source_upload",
        "requested_report_id": str(requested_report.id),
        "base_report_id": str(latest_report.id),
        "base_report_version": latest_report.report_version,
        "reference_ids": [reference_id],
        "representation_id": str(representation.id),
        "source_content_sha256": authorized.content_sha256,
        "previous_source_results": source_results,
        "previous_verification_summary": dict(job.verification_summary or {}),
    }
    job.source_results = updated_results
    job.upload_evidence = upload_evidence
    job.status = JobStatus.RUNNING
    job.stage = JobStage.RETRIEVED
    job.error_message = None
    job.updated_at = datetime.now(timezone.utc)
    prepare_dispatch(job, attempt_id=attempt_id)
    if commit:
        session.commit()
    else:
        session.flush()
    return {
        "status": "scheduled",
        "scheduled": True,
        "job_id": str(job.id),
        "report_id": str(latest_report.id),
        "reference_id": reference_id,
        "representation_id": str(representation.id),
        "content_sha256": authorized.content_sha256,
        "attempt_id": attempt_id,
    }


def rollback_targeted_source_refresh(
    session: Session,
    job_id: str | uuid.UUID,
    *,
    error_code: str,
    commit: bool = True,
) -> bool:
    """Restore the completed checkpoint after a failed targeted refresh."""
    parsed_job_id = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(str(job_id))
    job = session.scalar(select(Job).where(Job.id == parsed_job_id).with_for_update())
    if job is None:
        return False
    upload_evidence = dict(job.upload_evidence or {})
    refresh = dict(upload_evidence.get(TARGETED_SOURCE_REFRESH_KEY) or {})
    if not refresh:
        return False
    previous_summary = dict(refresh.get("previous_verification_summary") or {})
    previous_ids = set(previous_summary.get("report_ids") or [])
    current_ids = set((job.verification_summary or {}).get("report_ids") or [])
    orphan_ids = current_ids - previous_ids
    linked_ids = {
        str(value)
        for report in session.scalars(select(Report).where(Report.job_id == job.id))
        for value in (report.report_json or {}).get("report_ids", [])
    }
    for value in orphan_ids:
        if str(value) in linked_ids:
            continue
        try:
            record_id = uuid.UUID(str(value))
        except ValueError:
            continue
        record = session.get(VerificationReportRecord, record_id)
        if record is not None:
            session.delete(record)
    job.source_results = list(refresh.get("previous_source_results") or [])
    job.verification_summary = previous_summary
    job.status = JobStatus.COMPLETED
    job.stage = JobStage.COMPLETED
    job.error_message = None
    upload_evidence.pop(TARGETED_SOURCE_REFRESH_KEY, None)
    upload_evidence["last_targeted_source_refresh_failure"] = {
        "attempt_id": refresh.get("attempt_id"),
        "error_code": str(error_code)[:100],
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    job.upload_evidence = upload_evidence
    job.updated_at = datetime.now(timezone.utc)
    if commit:
        session.commit()
    else:
        session.flush()
    return True


def fail_paper_job(session: Session, backend: StorageBackend, job_id, exc: BaseException) -> None:
    job = _job(session, job_id)
    if (job.upload_evidence or {}).get(TARGETED_SOURCE_REFRESH_KEY):
        rollback_targeted_source_refresh(
            session,
            job.id,
            error_code=str(getattr(exc, "code", type(exc).__name__)),
        )
        return
    if job.status == JobStatus.COMPLETED:
        return
    job.status = JobStatus.FAILED
    job.stage = JobStage.FAILED
    job.error_message = getattr(exc, "code", type(exc).__name__)
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
    artifact = session.scalar(
        select(ReportPaperArtifactRecord).where(
            ReportPaperArtifactRecord.job_id == job.id,
            ReportPaperArtifactRecord.report_id.is_(None),
        )
    )
    if artifact is not None:
        cleanup_report_paper_artifact(session, backend, artifact)
    cleanup_paper_job_input(session, backend, job.id)


def _shadow_artifact(
    source: AuthorizedRepresentation,
    claim,
    llm_enabled: bool,
    *,
    active_reference_id: str,
    cited_author_label: str,
) -> VerificationEvidenceArtifact:
    evidence = build_passage_evidence(
        source,
        claim=claim,
        top_k=3,
        active_reference_id=active_reference_id,
        cited_author_label=cited_author_label,
    )
    evidence = attach_verification_candidates(evidence)
    evidence = attach_citation_use_routes(evidence)
    if llm_enabled:
        evidence = attach_source_blind_interpretations(
            evidence,
            response_provider=_student_interpretation_response,
        )
    evidence = attach_evidence_obligations(evidence)
    evidence = attach_candidate_passage_retrieval(source, evidence)
    if llm_enabled:
        evidence = apply_passage_relevance_gate(evidence)
        if (
            settings.EVIDENCE_RETRIEVAL_SEMANTIC_BACKEND == "deberta_nli"
            and evidence.passage_relevance.outcome
            == "no_relevant_candidate_passage"
        ):
            evidence = attach_local_semantic_retrieval_rescue(source, evidence)
            if (
                evidence.candidate_passage_retrieval.semantic_rescue_status
                == "complete"
                and evidence.candidate_passage_retrieval.semantic_addition_count > 0
            ):
                evidence = apply_passage_relevance_gate(evidence)
    evidence = attach_facet_evidence_foundation(evidence)
    if llm_enabled:
        evidence = apply_facet_evidence_judgment(evidence)
        evidence = apply_decisive_label_critic(evidence)
    return evidence


def _student_interpretation_response(system_prompt: str, user_prompt: str) -> dict:
    """Use the configured provider for a bounded source-blind interpretation."""

    return chat_completion_json(
        system_prompt,
        user_prompt,
        model=settings.LLM_MODEL,
        temperature=0.0,
        max_tokens=settings.VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS,
        max_retries=2,
        disable_thinking=True,
    )


def _transient_cleanliness(representation) -> str:
    if representation.kind is not RepresentationKind.PDF:
        return "clean"
    try:
        report = inspect_uploaded_pdf(representation.content)
    except FileSafetyUnavailable as exc:
        raise PaperWorkflowError(
            "source_safety_unavailable", "Source safety scanner is unavailable"
        ) from exc
    if report.verdict is not SafetyVerdict.CLEAN:
        raise PaperWorkflowError("source_safety_not_clean", "Retrieved source did not pass the cleanliness gate")
    return "clean"


def _bounded_abstract_evidence(value, *, source_name: str) -> dict | None:
    """Retain bounded abstract evidence without treating it as full text."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return None
    bounded = text[:MAX_REPORT_ABSTRACT_CHARACTERS]
    return {
        "text": bounded,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "truncated": len(bounded) < len(text),
        "source_name": str(source_name or "scholarly metadata service")[:200],
    }


def _canonical_work_id(reference) -> str:
    identity = reference.doi or reference.title or reference.raw_ref
    digest = hashlib.sha256(identity.strip().casefold().encode("utf-8")).hexdigest()
    return f"work:{digest}"


def _refresh_reference_ids(job: Job) -> list[str]:
    evidence = dict(job.upload_evidence or {})
    targeted = dict(evidence.get(TARGETED_SOURCE_REFRESH_KEY) or {})
    values = targeted.get("reference_ids")
    if values is None:
        values = evidence.get("provider_refresh_reference_ids") or []
    return sorted({str(value) for value in values if str(value)})


def _canonical_work_matches_reference(record: SourceRepresentationRecord, reference) -> bool:
    work = record.canonical_work

    def normalize_doi(value) -> str:
        return re.sub(
            r"^https?://(?:dx\.)?doi\.org/",
            "",
            str(value or "").strip().casefold(),
        )

    def normalize_title(value) -> str:
        return " ".join(str(value or "").casefold().split())

    expected_doi = normalize_doi(reference.doi)
    actual_doi = normalize_doi(work.doi)
    if expected_doi:
        return bool(actual_doi and actual_doi == expected_doi)
    expected_title = normalize_title(reference.title)
    actual_title = normalize_title(work.display_title)
    return bool(expected_title and actual_title == expected_title)


def _targeted_amendment_manifest(
    refresh: dict,
    summary: dict,
    *,
    previous_report: Report,
) -> dict:
    """Describe only the immutable dependencies changed by one refresh."""

    def members(payload: dict) -> dict[tuple[str, str], dict]:
        indexed: dict[tuple[str, str], dict] = {}
        for group in payload.get("citation_groups") or []:
            claim_id = str(group.get("claim_id") or "")
            for member in group.get("members") or []:
                reference_id = str(member.get("reference_id") or "")
                if claim_id and reference_id:
                    indexed[(claim_id, reference_id)] = member
        return indexed

    previous = members(dict(refresh.get("previous_verification_summary") or {}))
    current = members(summary)
    affected = set(str(value) for value in refresh.get("reference_ids") or [])
    changed = []
    for key in sorted(set(previous) | set(current)):
        claim_id, reference_id = key
        if reference_id not in affected:
            continue
        old = previous.get(key) or {}
        new = current.get(key) or {}
        changed.append(
            {
                "claim_id": claim_id,
                "reference_id": reference_id,
                "old_report_id": old.get("report_id"),
                "new_report_id": new.get("report_id"),
                "old_package_id": old.get("package_id"),
                "new_package_id": new.get("package_id"),
                "old_package_sha256": old.get("package_sha256"),
                "new_package_sha256": new.get("package_sha256"),
                "old_status": old.get("status"),
                "new_status": new.get("status"),
            }
        )
    return {
        "attempt_id": refresh.get("attempt_id"),
        "reason": refresh.get("reason"),
        "requested_report_id": refresh.get("requested_report_id"),
        "base_report_id": str(previous_report.id),
        "base_report_version": previous_report.report_version,
        "reference_ids": sorted(affected),
        "representation_id": refresh.get("representation_id"),
        "source_content_sha256": refresh.get("source_content_sha256"),
        "changed_dependencies": changed,
    }


def _source_counts(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


def _retryable_provider_dependencies(
    discovery_trace: dict | None,
    discovery_record: dict | None,
) -> list[str]:
    """Identify providers that made an unresolved reference operationally incomplete.

    Only a typed ``search_incomplete`` record is retryable.  A completed
    no-match, bibliographic conflict, or insufficient citation metadata must
    never be silently converted into provider-recovery work.
    """
    if (discovery_record or {}).get("outcome") != "search_incomplete":
        return []
    incomplete = {
        "timeout",
        "captcha",
        "operational_failure",
        "access_restricted",
        "rate_limited",
        "response_invalid",
        "budget_skipped",
        "cooldown_skipped",
        "recovery_probe_in_progress",
    }
    providers = {
        str(query.get("execution_provider") or "").strip().casefold()
        for query in (discovery_trace or {}).get("queries", [])
        if query.get("execution_outcome") in incomplete
    }
    return sorted(provider for provider in providers if provider)


def _report_source_failure_reason(source_result: dict) -> str:
    """Preserve a failed cited-webpage route without claiming why it failed."""
    reason_code = source_result.get("reason_code", "source_unavailable")
    if reason_code != "source_not_found":
        return reason_code
    trace = source_result.get("reference_discovery_trace") or {}
    cited_url_attempts = [
        attempt
        for attempt in trace.get("attempts", [])
        if attempt.get("route_category") == "student_url"
    ]
    if not any(
        attempt.get("outcome") in {"operational_failure", "access_restricted"}
        for attempt in cited_url_attempts
    ):
        return reason_code
    other_candidate_found = any(
        attempt.get("route_category") != "student_url"
        and attempt.get("outcome") == "candidate_found"
        for attempt in trace.get("attempts", [])
    )
    return (
        "cited_webpage_unavailable_metadata_only"
        if other_candidate_found
        else "cited_webpage_unavailable"
    )


def _claims_by_reference(claims: list) -> dict[str, list]:
    """Fan exact collective claims out to every linked source retrieval."""
    grouped: dict[str, list] = {}
    for claim in claims:
        for reference_id in claim.reference_ids:
            grouped.setdefault(reference_id, []).append(claim)
    return grouped


def _report_member(record: VerificationReportRecord) -> dict:
    """Return a text-free immutable link from a paper aggregate to one package."""
    package = record.report_payload.get("authoritative_evidence_package") or {}
    binding = package.get("source_binding") or {}
    source_identity = package.get("source_identity") or {}
    required = {
        "package_id": package.get("package_id"),
        "package_sha256": package.get("package_sha256"),
        "claim_id": package.get("claim_id"),
        "reference_id": binding.get("reference_id"),
        "source_content_sha256": source_identity.get("content_sha256"),
    }
    if not all(required.values()):
        raise PaperWorkflowError(
            "evidence_package_binding_incomplete",
            "A persisted Evidence Package lacks an immutable claim/source binding",
        )
    marker_text = binding.get("marker_text") or ""
    return {
        "status": "evidence_package_persisted",
        "report_id": str(record.id),
        "report_version": record.report_version,
        "verification_id": record.verification_id,
        **required,
        "binding_status": binding.get("status"),
        "marker_local_start": binding.get("marker_local_start"),
        "marker_local_end": binding.get("marker_local_end"),
        "marker_text_sha256": hashlib.sha256(marker_text.encode()).hexdigest(),
    }


def _citation_group_index(claims, report_members: list[dict], failures: list[dict]) -> list[dict]:
    """Index every expected compound member without borrowing another member's result."""
    member_by_binding = {
        (member["claim_id"], member["reference_id"]): member
        for member in report_members
    }
    failure_by_reference = {failure["reference_id"]: failure for failure in failures}
    groups = []
    for claim in claims:
        members = []
        for reference_id in dict.fromkeys(claim.reference_ids):
            persisted = member_by_binding.get((claim.claim_id, reference_id))
            if persisted is not None:
                members.append(persisted)
            else:
                failure = failure_by_reference.get(reference_id, {})
                members.append(
                    {
                        "status": "source_unavailable",
                        "claim_id": claim.claim_id,
                        "reference_id": reference_id,
                        "reason_code": failure.get("reason_code", "source_result_missing"),
                        "abstract_available": bool(failure.get("abstract_available")),
                        "abstract_evidence": failure.get("abstract_evidence"),
                        "abstract_relevance": (
                            failure.get("abstract_relevance_by_claim") or {}
                        ).get(claim.claim_id),
                    }
                )
        persisted_count = sum(
            member["status"] == "evidence_package_persisted" for member in members
        )
        groups.append(
            {
                "claim_id": claim.claim_id,
                "expected_member_count": len(members),
                "persisted_member_count": persisted_count,
                "coverage_status": (
                    "complete"
                    if persisted_count == len(members)
                    else "partial"
                    if persisted_count
                    else "unavailable"
                ),
                "members": members,
            }
        )
    return groups


def _extraction_counts(artifact: PaperExtractionArtifact) -> dict:
    counts = {
        "references": len(artifact.references),
        "citations": len(artifact.citations),
        "citation_units": len(artifact.citation_claims),
        "rejected_citations": len(artifact.rejected_citations),
    }
    if artifact.reference_consistency is not None:
        counts["reference_consistency_findings"] = len(
            artifact.reference_consistency.findings
        )
        counts["reference_consistency_attention"] = sum(
            finding.level == "attention"
            for finding in artifact.reference_consistency.findings
        )
        counts["reference_consistency_review"] = sum(
            finding.level == "review"
            for finding in artifact.reference_consistency.findings
        )
        counts["reference_consistency_neutral"] = sum(
            finding.level == "neutral"
            for finding in artifact.reference_consistency.findings
        )
    if artifact.reference_layout is not None:
        counts["reference_layout_status"] = artifact.reference_layout.status
        counts["reference_layout_matched"] = artifact.reference_layout.matched_reference_count
        counts["reference_layout_total"] = artifact.reference_layout.reference_count
    if artifact.reference_formatting is not None:
        counts["reference_formatting_status"] = artifact.reference_formatting.status
        counts["reference_formatting_differences"] = artifact.reference_formatting.result_counts.get(
            "difference", 0
        )
    return counts


def _at_or_after(job: Job, stage: str) -> bool:
    return _STAGE_ORDER.get(job.stage, -1) >= _STAGE_ORDER[stage]


def _extraction(job: Job) -> PaperExtractionArtifact:
    if not job.extraction_payload:
        raise PaperWorkflowError("extraction_missing", "Paper extraction checkpoint is missing")
    return PaperExtractionArtifact.model_validate(job.extraction_payload)


def _select_reference_candidate(
    qualified: QualifiedTextExtraction,
    *,
    citation_format: str,
) -> tuple[TextCandidate, dict[str, int]]:
    """Choose reference text independently from the semantic body extraction.

    PDF backends have different strengths. A cleaner body must not cost valid
    bibliography entries, so deterministic parsing counts choose the reference
    representation while the selected semantic candidate owns body offsets.
    Ties retain the semantic candidate.
    """
    counts: dict[str, int] = {}
    for candidate in qualified.candidates:
        section = extract_reference_section(candidate.text, citation_format)
        if not section:
            counts[candidate.backend] = 0
            continue
        counts[candidate.backend] = len(
            extract_and_parse_references(
                section,
                format_hint=citation_format,
                use_regex_first=True,
                use_llm_fallback=False,
            )
        )
    selected_count = counts.get(qualified.selected.backend, 0)
    chosen = qualified.selected
    for candidate in qualified.candidates:
        if counts.get(candidate.backend, 0) > selected_count:
            chosen = candidate
            selected_count = counts[candidate.backend]
    return chosen, counts


def _enter(job: Job, stage: str) -> None:
    if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
        raise PaperWorkflowError("job_terminal", "Paper job is already terminal")
    job.status = JobStatus.RUNNING
    job.stage = stage
    job.updated_at = datetime.now(timezone.utc)


def _job(session: Session, value) -> Job:
    try:
        job_id = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise PaperWorkflowError("job_id_invalid", "Invalid paper job ID") from exc
    job = session.get(Job, job_id)
    if job is None:
        raise PaperWorkflowError("job_not_found", "Paper job does not exist")
    return job
