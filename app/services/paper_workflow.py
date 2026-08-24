"""Checkpointed paper extraction, retrieval, shadow verification, and summary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report
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
from app.services.paper_extraction import PaperExtractionArtifact, extract_paper_evidence
from app.services.paper_upload import (
    PaperUploadError,
    cleanup_paper_job_input,
    load_paper_job_input,
)
from app.services.paper_retention import (
    PaperRetentionPolicyError,
    resolve_paper_retention_policy,
)
from app.services.passage_relevance import apply_passage_relevance_gate
from app.services.retrieval.base import RepresentationKind
from app.services.source_resolver import SourceResolutionError, SourceResolver
from app.services.storage.backend import StorageBackend
from app.services.text_extractor import TextExtractionError, extract_text_from_bytes
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    VerificationEvidenceArtifact,
    attach_candidate_passage_retrieval,
    authorize_representation,
    build_passage_evidence,
)
from app.services.verification_report import persist_verification_report
from app.services.verification_run import (
    VerificationRunRequest,
    begin_verification_run,
    complete_active_verification_run,
)


PAPER_WORKFLOW_VERSION = "paper-workflow-v3"
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
        text = extract_text_from_bytes(content, job.filename)
        if len(text) > settings.MAX_PAPER_TEXT_LENGTH_CHARS:
            raise PaperWorkflowError(
                "paper_text_limit_exceeded",
                "Extracted paper text exceeds the configured processing limit",
            )
        allow_llm = settings.PAPER_LLM_PROCESSING_ENABLED if llm_enabled is None else llm_enabled
        artifact = extract_paper_evidence(
            text,
            paper_version_id=job.paper_version_id,
            use_llm_boundaries=allow_llm,
            use_llm_atomizer=False,
            use_llm_reference_fallback=allow_llm,
        )
    except (PaperUploadError, TextExtractionError, ValueError) as exc:
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
    results: list[dict] = []
    for reference in artifact.references:
        if reference.reference_id not in cited_reference_ids:
            continue
        result_record = {"reference_id": reference.reference_id}
        try:
            result = active_resolver.resolve_reference(reference)
        except SourceResolutionError:
            result_record.update(status="unavailable", reason_code="source_not_found")
            results.append(result_record)
            continue
        metadata = result.metadata or {}
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
            results.append(result_record)
            continue
        representation = result.representation
        if representation is None or not representation.content:
            result_record.update(
                status="abstract_only" if result.abstract else "unavailable",
                reason_code="full_text_unavailable",
                abstract_available=bool(result.abstract),
            )
            results.append(result_record)
            continue
        cleanliness = _transient_cleanliness(representation)
        confidence = metadata.get("identity_confidence")
        if confidence != "high":
            result_record.update(status="unavailable", reason_code="identity_not_verified")
            results.append(result_record)
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
                        None,
                    ),
                ),
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
        results.append(result_record)
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
        references_by_id = {
            reference.reference_id: reference for reference in artifact.references
        }
    allow_llm = settings.PAPER_LLM_PROCESSING_ENABLED if llm_enabled is None else llm_enabled
    claims_by_reference = _claims_by_reference(artifact.citation_claims)

    report_ids: list[str] = []
    failures: list[dict] = []
    for source_result in source_results:
        reference_id = source_result["reference_id"]
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
                failures.append(
                    {"reference_id": reference_id, "reason_code": source_result.get("reason_code", "source_unavailable")}
                )
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
                session.commit()
        else:
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

            completed = complete_active_verification_run(
                session_factory,
                backend,
                source_result["verification_run_id"],
                scope_type=scope_type,
                scope_id=scope_id,
                processor=processor,
            )
            report_ids.extend(str(report_id) for report_id, _version in completed.reports)

    summary = {
        "workflow_version": PAPER_WORKFLOW_VERSION,
        "model_processing_enabled": allow_llm,
        "relationship_mode": "shadow",
        "reports_persisted": len(report_ids),
        "report_ids": report_ids,
        "source_failures": failures,
        "decision_applied": False,
    }
    with session_factory() as session:
        job = _job(session, job_id)
        job.verification_summary = summary
        job.stage = JobStage.VERIFIED
        job.updated_at = datetime.now(timezone.utc)
        session.commit()
    return summary


def finalize_paper_job(session: Session, backend: StorageBackend, job_id) -> dict:
    job = _job(session, job_id)
    if job.status == JobStatus.COMPLETED:
        report = session.scalar(
            select(Report)
            .where(Report.job_id == job.id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
        if report is None:
            raise PaperWorkflowError("aggregate_report_missing", "Completed job has no aggregate report")
        return {"job_id": str(job.id), "report_id": str(report.id), "input_cleaned": job.input_deleted_at is not None}
    if job.store_only and job.stage == JobStage.EXTRACTED:
        summary = {
            "workflow_version": PAPER_WORKFLOW_VERSION,
            "store_only": True,
            "reports_persisted": 0,
            "decision_applied": False,
        }
    elif job.stage == JobStage.VERIFIED:
        summary = dict(job.verification_summary or {})
    else:
        raise PaperWorkflowError("job_not_ready", "Paper job is not ready to finalize")
    job.stage = JobStage.FINALIZING
    session.commit()
    extraction = _extraction(job)
    report = Report(
        job_id=job.id,
        total_references=str(len(extraction.references)),
        verified_references=str(summary.get("reports_persisted", 0)),
        summary="Inspectable shadow verification; no decision was applied.",
        report_json={
            **summary,
            "paper_version_id": job.paper_version_id,
            "reference_count": len(extraction.references),
            "citation_unit_count": len(extraction.citation_claims),
            "rejected_citation_count": len(extraction.rejected_citations),
        },
    )
    session.add(report)
    job.status = JobStatus.COMPLETED
    job.stage = JobStage.COMPLETED
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
    cleaned = cleanup_paper_job_input(session, backend, job.id)
    return {"job_id": str(job.id), "report_id": str(report.id), "input_cleaned": cleaned}


def fail_paper_job(session: Session, backend: StorageBackend, job_id, exc: BaseException) -> None:
    job = _job(session, job_id)
    if job.status == JobStatus.COMPLETED:
        return
    job.status = JobStatus.FAILED
    job.stage = JobStage.FAILED
    job.error_message = getattr(exc, "code", type(exc).__name__)
    job.updated_at = datetime.now(timezone.utc)
    session.commit()
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
    evidence = attach_candidate_passage_retrieval(source, evidence)
    evidence = attach_facet_evidence_foundation(evidence)
    if llm_enabled:
        evidence = apply_passage_relevance_gate(evidence)
        evidence = apply_facet_evidence_judgment(evidence)
        evidence = apply_decisive_label_critic(evidence)
    return evidence


def _transient_cleanliness(representation) -> str:
    if representation.kind is not RepresentationKind.PDF:
        return "clean"
    try:
        report = inspect_uploaded_pdf(representation.content)
    except FileSafetyUnavailable as exc:
        raise PaperWorkflowError("source_safety_unavailable", str(exc)) from exc
    if report.verdict is not SafetyVerdict.CLEAN:
        raise PaperWorkflowError("source_safety_not_clean", "Retrieved source did not pass the cleanliness gate")
    return "clean"


def _canonical_work_id(reference) -> str:
    identity = reference.doi or reference.title or reference.raw_ref
    digest = hashlib.sha256(identity.strip().casefold().encode("utf-8")).hexdigest()
    return f"work:{digest}"


def _source_counts(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


def _claims_by_reference(claims: list) -> dict[str, list]:
    """Fan exact collective claims out to every linked source retrieval."""
    grouped: dict[str, list] = {}
    for claim in claims:
        for reference_id in claim.reference_ids:
            grouped.setdefault(reference_id, []).append(claim)
    return grouped


def _extraction_counts(artifact: PaperExtractionArtifact) -> dict:
    return {
        "references": len(artifact.references),
        "citations": len(artifact.citations),
        "citation_units": len(artifact.citation_claims),
        "rejected_citations": len(artifact.rejected_citations),
    }


def _at_or_after(job: Job, stage: str) -> bool:
    return _STAGE_ORDER.get(job.stage, -1) >= _STAGE_ORDER[stage]


def _extraction(job: Job) -> PaperExtractionArtifact:
    if not job.extraction_payload:
        raise PaperWorkflowError("extraction_missing", "Paper extraction checkpoint is missing")
    return PaperExtractionArtifact.model_validate(job.extraction_payload)


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
