"""Checkpointed paper extraction, retrieval, shadow verification, and summary."""

from __future__ import annotations
from copy import deepcopy

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
from app.services.assessment_configuration import AssessmentConfiguration
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
    assess_retrieved_text_scope,
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
    SourceScopeAssessmentEvidence,
    VerificationEvidenceArtifact,
    attach_candidate_passage_retrieval,
    attach_local_semantic_retrieval_rescue,
    SourceTextUnusable,
    authorize_representation,
    build_passage_evidence,
    _SCOPE_EVIDENCE_HEADING,
    claim_terms_present_in_source,
    leading_source_excerpt,
    scope_evidence_block,
)
from app.services.verification_report import persist_verification_report, _payload_digest
from app.services.verification_run import (
    VerificationRunError,
    TransientSourceTextUnusable,
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
            docx_content=content if job.filename.lower().endswith('.docx') else None,
            link_targets=_pdf_link_targets(content) if job.filename.lower().endswith('.pdf') else frozenset(),
            paper_version_id=job.paper_version_id,
            reference_text=reference_candidate.text,
            reference_layout_text=reference_candidate.layout_text,
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
        reference_formatting = assess_reference_formatting(
            reference_layout, references=artifact.references,
            reference_section=extract_reference_section(reference_candidate.layout_text or reference_candidate.text, artifact.citation_format),
        )
        from app.services.submitted_locator_inventory import inventory_submitted_locators, bind_submitted_hyperlinks
        try:
            locator_inventory = inventory_submitted_locators(
                content, references=artifact.references, layout=reference_layout
            )
            artifact = artifact.model_copy(update={'references': bind_submitted_hyperlinks(
                content, references=artifact.references, layout=reference_layout
            )})
        except Exception:
            # Optional neutral inventory must not abort evidence processing or
            # turn an inspection failure into an absence count.
            locator_inventory = None
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
                "assessment_configuration": AssessmentConfiguration.model_validate((job.upload_evidence or {}).get('assessment_configuration') or {}),
                "reference_layout": reference_layout,
                "submitted_locator_inventory": locator_inventory,
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
            references=(artifact.references if locator_inventory is not None
                        and (locator_inventory.counts['not_observed'] > 0
                             or any(not r.author and r.extraction_method == 'authorless_journal_regex'
                                    for r in artifact.references))
                        and (artifact.citation_format == 'apa' or artifact.assessment_configuration.require_reference_links) else None),
            citation_format=artifact.citation_format,
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
    from app.services.bibliography_identity import eligible_identity_only, POLICY as IDENTITY_ONLY_POLICY
    # New bibliography-only work must not spend the shared resolver allowance
    # ahead of the pre-existing cited-source acquisition workload.
    ordered_references = sorted(artifact.references,
        key=lambda reference: reference.reference_id not in cited_reference_ids)
    discovery_references = [
        reference
        for reference in ordered_references
        if reference.reference_id in cited_reference_ids or eligible_identity_only(reference)
    ]
    # Batch-capable title providers and deferred DOI providers are part of the
    # ordinary paper workflow.  Leaving them unused made single-reference
    # fallback behavior look like completed discovery.
    prefetch_titles = getattr(active_resolver, "prefetch_title_candidates", None)
    if prefetch_titles is not None:
        prefetch_titles(
            [
                (reference.title, reference.author or None)
                for reference in discovery_references
                if reference.title
            ]
        )
    prefetch_dois = getattr(active_resolver, "prefetch_deferred_dois", None)
    if prefetch_dois is not None:
        prefetch_dois(
            [reference.doi for reference in discovery_references if reference.doi]
        )
    results: list[dict] = list(job.source_results or [])
    completed_reference_ids = {item["reference_id"] for item in results}
    # Completed-search memos are read and written within this job's own
    # authorization scope only (search-reuse-memo-v1). A targeted refresh
    # requested with force_search repeats the paid search regardless.
    from app.services.search.search_memo import SearchMemoContext, SearchMemoStore, search_memo_scope
    targeted_refresh = dict((getattr(job, "upload_evidence", None) or {}).get(TARGETED_SOURCE_REFRESH_KEY) or {})
    memo_context = SearchMemoContext(
        scope_type=getattr(job, "scope_type", None) or "", scope_id=getattr(job, "scope_id", None) or "",
        store=SearchMemoStore(session), force_search=bool(targeted_refresh.get("force_search")),
    )
    # An unchanged reference already searched by an earlier run of this paper
    # reuses that result instead of paying again (paper-search-reuse-v1). A
    # targeted refresh never reuses: it exists to search again.
    from app.services.search.rerun_reuse import prior_results_index, reused_result
    rerun_index = {} if targeted_refresh else prior_results_index(session, job)

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

    from app.services.required_reference_author import eligible_author_lookup, discover_author_metadata
    # Bibliography-only inspection is independent of successful author linking.
    # Keep this new registration path bounded; never acquire an uncited source.
    author_lookup_slots = sum('author_metadata_lookup' in r for r in results)
    for reference in ordered_references:
        if (reference.reference_id not in cited_reference_ids
                and reference.reference_id not in completed_reference_ids
                and artifact.citation_format.lower() == 'apa'
                and eligible_author_lookup(reference, artifact.required_author_policy_version)):
            checkpoint(discover_author_metadata(reference,
                getattr(active_resolver, '_retrieval_sources', []),
                permitted=author_lookup_slots < 2))
            author_lookup_slots += 1
        if (
            (reference.reference_id not in cited_reference_ids and not eligible_identity_only(reference))
            or reference.reference_id in completed_reference_ids
        ):
            continue
        result_record = {"reference_id": reference.reference_id}
        identity_only = reference.reference_id not in cited_reference_ids
        if identity_only:
            result_record["identity_only_policy"] = IDENTITY_ONLY_POLICY
        reused = reused_result(rerun_index, reference, identity_only=identity_only)
        if reused is not None and reused.get("status") == "durable_authorized":
            try:
                authorize_representation(session, backend, representation_id=reused["representation_id"],
                                         scope_type=job.scope_type, scope_id=job.scope_id)
            except Exception:
                reused = None   # the stored source is no longer usable here: search again
        if reused is not None:
            checkpoint(reused)
            continue
        try:
            with search_memo_scope(memo_context):
                result = active_resolver.resolve_reference(reference, identity_only=True) if identity_only else active_resolver.resolve_reference(reference)
        except SourceResolutionError as exc:
            result_record.update(
                status="unavailable",
                reason_code="source_not_found",
                reference_discovery_trace=exc.reference_discovery_trace,
                reference_discovery=exc.reference_discovery,
                submitted_link_observations=getattr(exc, "submitted_link_observations", None),
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
        result_record["submitted_link_observations"] = metadata.get("submitted_link_observations")
        if identity_only:
            # Identity observations do not authorize a representation, abstract,
            # verification run or source admission, even from an injected adapter.
            result_record.update(status="metadata_only", reason_code="bibliography_identity_only")
            dependencies = _retryable_provider_dependencies(
                result_record["reference_discovery_trace"], result_record["reference_discovery"])
            if dependencies:
                result_record["retryable_provider_dependencies"] = dependencies
            checkpoint(result_record)
            continue
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
            except SourceTextUnusable:
                result_record.update(status="unavailable", reason_code="source_text_unusable")
            except Exception:
                result_record.update(
                    status="unavailable",
                    reason_code="durable_representation_not_authorized",
                )
            else:
                from app.services.submitted_links import bind_authorized_admission
                result_record['submitted_link_observations'] = bind_authorized_admission(
                    result_record.get('submitted_link_observations'), metadata, representation_id)
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
            # A required web search that was skipped (call ceiling, cooldown,
            # elapsed budget) or failed did not finish; "not retrieved" would
            # read as a completed search. The providers named are the ones a
            # targeted refresh can wait for (`prepare_provider_recovery_refresh`).
            incomplete = _full_text_search_incompleteness(result_record)
            result_record.update(
                status="abstract_only" if abstract_evidence else "unavailable",
                reason_code=(
                    "full_text_search_incomplete" if incomplete else "full_text_unavailable"
                ),
                abstract_available=bool(abstract_evidence),
            )
            if incomplete:
                result_record["full_text_search_incomplete_providers"] = incomplete["providers"]
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
        provisional = metadata.get('provisional_source') or {}
        possible_match = bool(confidence == 'medium'
            and provisional.get('policy_version') == 'submission-possible-match-v1'
            and provisional.get('content_sha256') == hashlib.sha256(representation.content).hexdigest())
        # A record page that named the cited work and advertised this exact file
        # as its full text corroborates it. Doster's thesis opens on a scanned
        # approval form and can never prove its own title, so acquisition
        # accepted it and this gate then discarded it — the reader saw nothing
        # either way. Bound to the validated bytes, not just to the flag.
        corroborated_landing = bool(
            confidence == 'medium'
            and metadata.get('identity_corroborated_by_landing_page')
            and metadata.get('accepted_representation_sha256')
            == hashlib.sha256(representation.content).hexdigest()
        )
        if confidence != "high" and not possible_match and not corroborated_landing:
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
                    identity_verdict="possible_match" if possible_match else "verified",
                    identity_confidence=None if possible_match else 1.0,
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
                provisional_source=provisional if possible_match else None,
            )
            # Independently verified acquisition provenance, not a retained
            # search result or a promise to retain transient source bytes.
            from urllib.parse import urlsplit
            # A repository's reader page (Europe PMC) rather than its XML API.
            source_url = (representation.metadata or {}).get('reader_url') or representation.source_url or ''
            try:
                parsed_url = urlsplit(source_url)
            except ValueError:
                parsed_url = urlsplit('')
            if (confidence == 'high' and parsed_url.scheme == 'https'
                    and parsed_url.hostname and not parsed_url.username
                    and not parsed_url.query):
                result_record['public_source_access'] = dict(
                    version='verified-public-source-access-v1', href=source_url,
                    content_sha256=hashlib.sha256(representation.content).hexdigest())
            # A page accepted as complete under rule A is recorded for review.
            stated = (representation.metadata or {}).get('stated_page_completeness')
            if isinstance(stated, dict):
                result_record['stated_page_completeness'] = dict(stated)
        checkpoint(result_record)
    # Uncited references are not searched, but their own links are visited so
    # a dead or wrong link is reported (owner decision 2026-10-02).
    check_links = getattr(active_resolver, "check_submitted_links", None)
    searched = {reference.reference_id for reference in discovery_references}
    for reference in artifact.references:
        if (check_links is None or reference.reference_id in searched
                or reference.reference_id in completed_reference_ids
                or not (getattr(reference, "url", None) or getattr(reference, "doi", None))):
            continue
        record = {"reference_id": reference.reference_id, "status": "link_check_only",
                  "reason_code": "uncited_reference"}
        try:
            checked = check_links(reference)
            record["submitted_link_observations"] = (checked.metadata or {}).get("submitted_link_observations")
        except Exception as exc:  # noqa: BLE001 - a failed visit is recorded, never fatal
            record["submitted_link_observations"] = getattr(exc, "submitted_link_observations", None)
        checkpoint(record)
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
    # Patchwriting while each source's text is authorized (owner decision
    # 2026-09-29); stored under the summary, never shown until approved.
    from app.services.patchwriting_at_check import SUMMARY_KEY as PATCHWRITING_KEY, PatchwritingAtCheck
    patchwriting = PatchwritingAtCheck(session_factory, backend, job_id, artifact,
                                       previous_summary=previous_summary,
                                       refresh_reference_ids=refresh_reference_ids)
    if any(item.get("status") in {"durable_authorized", "transient_authorized"} for item in source_results):
        # The body is read with its own session here, never inside a source's session.
        patchwriting.prepare()

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
                                claim, abstract_text, source_title=reference.title if reference else ""
                            )
                            for claim in claims
                        }
                failures.append(failure)
            continue
        if source_result["status"] == "durable_authorized":
            with session_factory() as session:
                try:
                    source = authorize_representation(
                        session,
                        backend,
                        representation_id=source_result["representation_id"],
                        scope_type=scope_type,
                        scope_id=scope_id,
                    )
                except SourceTextUnusable:
                    # Damaged pages that OCR could not repair: the text is not
                    # used, and only this reference loses its source.
                    failures.append({"reference_id": reference_id, "reason_code": "source_text_unusable"})
                    continue
                # Extracted once for the source, not once for each claim.
                scope_excerpt = leading_source_excerpt(source) if allow_llm else ""
                patchwriting.run(source, reference_id)
                artifacts = [
                    _shadow_artifact(
                        source,
                        claim,
                        allow_llm,
                        active_reference_id=reference_id,
                        cited_author_label=reference.author,
                        source_title=reference.title or "",
                        scope_excerpt=scope_excerpt,
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
                scope_excerpt = leading_source_excerpt(source) if allow_llm else ""
                patchwriting.run(source, reference_id)
                return [
                    _shadow_artifact(
                        source,
                        claim,
                        allow_llm,
                        active_reference_id=reference_id,
                        cited_author_label=reference.author,
                        source_title=reference.title or "",
                        identity_reason=(source_result.get('provisional_source') or {}).get('reason'),
                        scope_excerpt=scope_excerpt,
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
            except TransientSourceTextUnusable:
                failures.append({"reference_id": reference_id, "reason_code": "source_text_unusable"})
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

    from app.services.submitted_links import project_observations

    summary = {
        "workflow_version": PAPER_WORKFLOW_VERSION,
        "model_processing_enabled": allow_llm,
        "relationship_mode": "shadow",
        "reports_persisted": len(report_ids),
        "report_ids": report_ids,
        "source_failures": failures,
        "submitted_link_observations": project_observations(artifact.references, source_results,
            cited_reference_ids={ref_id for claim in artifact.citation_claims for ref_id in claim.reference_ids}),
        "citation_groups": _citation_group_index(
            artifact.citation_claims,
            report_members,
            failures,
        ),
        "decision_applied": False,
    }
    patchwriting_block = patchwriting.summary_block()
    if patchwriting_block is not None:
        summary[PATCHWRITING_KEY] = patchwriting_block
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
    from app.services.judgment_runs import schedule_run_at_check
    schedule_run_at_check(session, report, job)
    cleaned = cleanup_paper_job_input(session, backend, job.id)
    return {"job_id": str(job.id), "report_id": str(report.id), "input_cleaned": cleaned}


def prepare_provider_recovery_refresh(
    session: Session,
    job_id,
    *,
    provider: str,
    commit: bool = True,
    force_search: bool = False,
) -> list[str]:
    """Reset only unresolved reference members dependent on a recovered provider.

    ``force_search`` repeats the paid web search for the affected references
    even where this scope holds a completed-search memo (search-reuse-memo-v1).
    """
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
    # Recomputed from each stored discovery record rather than read from the
    # stored `retryable_provider_dependencies`, which on jobs recorded before
    # 2026-09-24 name every queried provider. Trusting that list would keep
    # re-running old papers whenever an optional provider recovered.
    affected = sorted(
        {
            str(item.get("reference_id"))
            for item in source_results
            if item.get("reference_id")
            and (
                normalized_provider
                in _blocking_providers_from_record(item.get("reference_discovery"))
                # Identity settled but a required full-text search was
                # skipped or failed (`full_text_search_incomplete`).
                or normalized_provider in _full_text_blocking_providers(item)
            )
        }
    )
    if not affected:
        return []
    upload_evidence = dict(job.upload_evidence or {})
    upload_evidence["provider_refresh_provider"] = normalized_provider
    upload_evidence["provider_refresh_reference_ids"] = affected
    job.upload_evidence = upload_evidence
    begin_reference_search_refresh(
        job, affected, reason="provider_recovery", force_search=force_search
    )
    if commit:
        session.commit()
    else:
        session.flush()
    return affected


def begin_reference_search_refresh(
    job: Job,
    reference_ids: list[str],
    *,
    reason: str,
    force_search: bool = False,
    extra: dict | None = None,
) -> str:
    """Reset the named references of a completed job for a new search.

    Their source results are removed so retrieval resolves only them again;
    every other reference keeps its checkpoint. The caller commits. Returns
    the dispatch attempt id.
    """
    source_results = list(job.source_results or [])
    affected = set(reference_ids)
    job.source_results = [
        item for item in source_results if item.get("reference_id") not in affected
    ]
    attempt_id = str(uuid.uuid4())
    upload_evidence = dict(job.upload_evidence or {})
    upload_evidence[TARGETED_SOURCE_REFRESH_KEY] = {
        **(extra or {}),
        "attempt_id": attempt_id,
        "reason": reason,
        "force_search": bool(force_search),
        "reference_ids": sorted(affected),
        "previous_source_results": source_results,
        "previous_verification_summary": dict(job.verification_summary or {}),
    }
    job.upload_evidence = upload_evidence
    job.status = JobStatus.RUNNING
    job.stage = JobStage.EXTRACTED
    job.error_message = None
    job.updated_at = datetime.now(timezone.utc)
    prepare_dispatch(job, attempt_id=attempt_id)
    return attempt_id


def prepare_uploaded_source_refresh(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    reference_id: str,
    representation_id: str | uuid.UUID,
    commit: bool = True,
    refresh_reason: str = "user_source_upload",
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
    if refresh_reason not in {'user_source_upload', 'authorized_source_reuse'}:
        raise ValueError('Unsupported source refresh reason')
    matches = _canonical_work_matches_reference(representation, reference)
    if not matches and refresh_reason == 'authorized_source_reuse':
        from app.services.source_resolver import _accepted_copy_has_edition_label, _normalize_cache_author
        base_title = re.sub(r'\s*\(\d+(?:st|nd|rd|th)\s+ed\.?\)\s*$', '', reference.title or '', flags=re.I)
        work = representation.canonical_work
        matches = bool(not reference.doi and base_title != reference.title
            and ' '.join(base_title.casefold().split()) == work.normalized_title
            and reference.year and reference.year == work.year
            and reference.author and _normalize_cache_author(reference.author) == _normalize_cache_author(work.author)
            and _accepted_copy_has_edition_label(authorized.content, reference.title))
    if not matches:
        raise PaperWorkflowError(
            "uploaded_source_identity_mismatch",
            "Uploaded source identity does not match the selected reference",
        )
    # A source now exists for this reference in this scope; a completed-search
    # memo must not hold back a later search for it (search-reuse-memo-v1).
    from app.services.search.search_memo import (
        SearchMemoContext, SearchMemoStore, clear_memo, reference_key,
    )
    memo_key = reference_key(reference.doi, reference.title, reference.author, reference.year)
    if memo_key:
        clear_memo(SearchMemoContext(job.scope_type, job.scope_id, SearchMemoStore(session)), memo_key)

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
        "source_name": "local_cache" if refresh_reason == 'authorized_source_reuse' else "instructor_upload",
        # A replacement source does not change what happened at the submitted URL.
        "submitted_link_observations": deepcopy((current or {}).get('submitted_link_observations')),
        # Acquiring a copy does not erase earlier bibliographic observations.
        "reference_discovery": deepcopy((current or {}).get('reference_discovery')),
        "reference_discovery_trace": deepcopy((current or {}).get('reference_discovery_trace')),
    }
    updated_results = [
        item for item in source_results if item.get("reference_id") != reference_id
    ]
    updated_results.append(replacement)
    attempt_id = str(uuid.uuid4())
    upload_evidence = dict(job.upload_evidence or {})
    upload_evidence[TARGETED_SOURCE_REFRESH_KEY] = {
        "attempt_id": attempt_id,
        "reason": refresh_reason,
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


def _failure_record(exc: BaseException) -> str:
    """One bounded description for every failure path a job can take.

    The detail (role names, field paths, retrieval methods) is validated by
    the exception itself or by `safe_exception_detail` against a fixed shape,
    so nothing here can carry student, source or network text. Computed once
    so the targeted-refresh rollback and the terminal failure record the same
    thing; the rollback path previously kept only the code, which left a
    failed targeted rerun exactly as undiagnosable as before.
    """
    from app.log_safety import safe_exception_detail

    code = getattr(exc, "code", type(exc).__name__)
    detail = safe_exception_detail(exc)
    return f"{code} ({detail})" if detail else str(code)


def fail_paper_job(session: Session, backend: StorageBackend, job_id, exc: BaseException) -> None:
    job = _job(session, job_id)
    record = _failure_record(exc)
    if (job.upload_evidence or {}).get(TARGETED_SOURCE_REFRESH_KEY):
        rollback_targeted_source_refresh(session, job.id, error_code=record)
        return
    if job.status == JobStatus.COMPLETED:
        return
    job.status = JobStatus.FAILED
    job.stage = JobStage.FAILED
    job.error_message = record
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
    identity_reason: str | None = None,
    source_title: str = "",
    scope_excerpt: str = "",
) -> VerificationEvidenceArtifact:
    evidence = build_passage_evidence(
        source,
        claim=claim,
        top_k=3,
        active_reference_id=active_reference_id,
        cited_author_label=cited_author_label,
    )
    if source.identity_verdict == 'possible_match' and identity_reason:
        evidence.source_identity.limitations.append(identity_reason[:1000])
    evidence = attach_verification_candidates(evidence)
    evidence = attach_citation_use_routes(evidence)
    # Judged before the evidence work, not after it. A source whose own
    # subject excludes the statement cannot supply evidence for it, so
    # selecting and ranking passages from it spends two model calls to
    # produce material the report then withholds. Retrieval itself cannot
    # be avoided: this judgment reads the document.
    if llm_enabled:
        evidence = _attach_source_scope(evidence, claim, scope_excerpt,
                                        source_title=source_title, source=source)
    topically_mismatched = _scope_mismatch_established(evidence)
    if llm_enabled and not topically_mismatched:
        evidence = attach_source_blind_interpretations(
            evidence,
            response_provider=_student_interpretation_response,
        )
    evidence = attach_evidence_obligations(evidence)
    if not topically_mismatched:
        evidence = attach_candidate_passage_retrieval(source, evidence)
    if llm_enabled and not topically_mismatched:
        evidence = apply_passage_relevance_gate(evidence, source_title=source_title)
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
                evidence = apply_passage_relevance_gate(evidence, source_title=source_title)
        if settings.PAPER_EXPERIMENTAL_JOINT_SELECTION_ENABLED:
            from app.services.joint_evidence_selection import select_joint_evidence
            evidence = evidence.model_copy(update={
                "joint_evidence_selection": select_joint_evidence(evidence, source_title=source_title),
            })
    if llm_enabled and not topically_mismatched:
        # The citation's one evidence list (owner decision 2026-09-28): GLM picks
        # the numbered source sentences; without it the gate's display is used.
        from app.services.evidence_sentence_selection import attach_sentence_evidence
        evidence = attach_sentence_evidence(evidence)
    evidence = attach_facet_evidence_foundation(evidence)
    if (llm_enabled and not topically_mismatched
            and settings.PAPER_EXPERIMENTAL_RELATIONSHIP_JUDGMENTS_ENABLED
            and source.identity_verdict != 'possible_match'):
        evidence = apply_facet_evidence_judgment(evidence)
        evidence = apply_decisive_label_critic(evidence)
    from app.services.judgment_reserve import build_judgment_reserve, reserve_enabled
    if reserve_enabled() and not topically_mismatched:
        # Wider-search sentences for the Judgment layout, built while the
        # source is still available; never part of the Evidence Package.
        evidence._judgment_reserve = build_judgment_reserve(source, evidence)
    return evidence


# Headroom kept between the composed scope text and the input budget.
_SCOPE_COMPOSITION_MARGIN = 250


def _scope_mismatch_established(evidence) -> bool:
    """Has the scope comparison already ruled this source out for this claim?"""
    from app.services.report_layers import scope_mark_qualifies
    record = getattr(evidence, "source_scope_assessment", None)
    if record is None or record.status != "complete":
        return False
    scope = dict((record.assessment or {}).get("scope_assessment") or {})
    if not scope:
        return False
    scope.setdefault("claim_terms_present", record.claim_terms_present)
    scope.setdefault("claim_terms_total", record.claim_terms_total)
    return scope_mark_qualifies(scope, record.coverage)


def _attach_source_scope(evidence, claim, excerpt: str, *, source_title: str, source=None):
    """Compare the retrieved work's own scope with what the citation claims.

    Runs only for coverage the contract accepts and only on text that was
    actually extracted. An unavailable excerpt leaves the default `not_run`,
    which the report reads as no comparison rather than as agreement.
    """
    # `coverage.level` is a CoverageLevel enum, whose str() is its member name,
    # not its value. Read the value so the comparison means what it says.
    level = getattr(evidence.coverage, "level", None)
    coverage = str(getattr(level, "value", level) or "")
    if coverage not in {"full_text", "partial_text"} or not excerpt:
        return evidence
    # The opening states the work's own scope; the passages show what it
    # actually discusses. Composed into one block so the existing binding,
    # hashing and span checks apply unchanged to exactly what was sent.
    # Sized from the live budget, not a constant: the prompt and the source
    # text share one input allowance, and a composed block that overruns it
    # is silently truncated, which sets `abstract_truncated` and withdraws
    # every mark. Leaving a margin keeps a later prompt edit from doing that.
    from app.services.passage_relevance import (
        FULL_TEXT_SCOPE_POLICY_VERSION, fit_scope_text,
    )
    evidence_block = scope_evidence_block(getattr(evidence, 'passages', None))
    composed = fit_scope_text(claim, excerpt, evidence_block,
                              FULL_TEXT_SCOPE_POLICY_VERSION,
                              source_title=source_title)
    assessment = assess_retrieved_text_scope(
        claim, composed, source_title=source_title, coverage=coverage,
    )
    # Counted over the COMPLETE document, not the excerpt the model saw.
    present, total = claim_terms_present_in_source(source, getattr(claim, 'text', ''))
    return evidence.model_copy(update={
        "source_scope_assessment": SourceScopeAssessmentEvidence(
            status=assessment.get("status", "not_assessed"),
            coverage=coverage,
            excerpt=composed,
            excerpt_sha256=hashlib.sha256(composed.encode()).hexdigest(),
            assessment=assessment,
            claim_terms_present=present,
            claim_terms_total=total,
        )
    })


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
    # Catalog records sometimes expose a book's table of contents in the field
    # a metadata service calls the abstract. Presenting chapter headings as
    # retrieved evidence tells the reader nothing about the cited content, so
    # treat it as no abstract at all rather than as a summary of the work.
    from app.services.abstract_shape import looks_like_contents_listing
    if looks_like_contents_listing(text):
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
    """Providers whose recovery could let this unresolved reference complete.

    Only a typed ``search_incomplete`` record is retryable. A completed
    no-match, bibliographic conflict, or insufficient citation metadata must
    never be silently converted into provider-recovery work. Within that, only
    providers the completion gate itself counts as blocking are named -- see
    `search_blocking_providers`. ``discovery_trace`` is no longer read; the
    trace lists every provider that was queried, required or not, which is how
    an optional provider came to trigger re-runs it could never complete.
    """
    return _blocking_providers_from_record(discovery_record)


def _full_text_search_incompleteness(source_result: dict) -> dict | None:
    """Recompute, from the stored trace, whether the full-text search finished.

    The trace holds every recorded route and query; the record is the fallback
    when no trace was stored.
    """
    from app.services.reference_discovery import full_text_search_incompleteness
    return full_text_search_incompleteness(
        source_result.get("reference_discovery_trace")
        or source_result.get("reference_discovery"))


def _full_text_blocking_providers(source_result: dict) -> list[str]:
    """Providers whose recovery could let an unfinished full-text search finish.

    Only a `full_text_search_incomplete` result qualifies, and the providers
    are recomputed from the stored trace rather than read from the stored
    list, for the same reason as `_blocking_providers_from_record`.
    """
    if source_result.get("reason_code") != "full_text_search_incomplete":
        return []
    incomplete = _full_text_search_incompleteness(source_result)
    return list(incomplete["providers"]) if incomplete else []


def _blocking_providers_from_record(discovery_record: dict | None) -> list[str]:
    """Recompute blocking providers from a stored discovery record.

    A record that is missing, not `search_incomplete`, or fails validation names
    no provider: declining to re-run leaves the reference honestly incomplete,
    which is the safe outcome, whereas guessing would re-run it on a trigger
    nobody can justify.
    """
    from app.services.reference_discovery import (
        ReferenceDiscoveryRecord, search_blocking_providers)
    if (discovery_record or {}).get("outcome") != "search_incomplete":
        return []
    try:
        record = ReferenceDiscoveryRecord.model_validate(discovery_record)
    except ValueError:
        return []
    return search_blocking_providers(
        record.required_route_categories, record.attempts, record.queries,
        record.search_policy_version)


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
        if _unbound_continuation(claim):
            continue
        for reference_id in claim.reference_ids:
            grouped.setdefault(reference_id, []).append(claim)
    return grouped


def _unbound_continuation(claim) -> bool:
    """Inference alone cannot supply the exact marker required by a package."""
    return (
        claim.citation_marker == "implicit_continuation"
        and not claim.citation_markers
    )


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
            elif _unbound_continuation(claim):
                members.append({
                    "status": "citation_not_assessed",
                    "claim_id": claim.claim_id,
                    "reference_id": reference_id,
                    "reason_code": "continuation_source_marker_unresolved",
                })
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


def _pdf_link_targets(content: bytes) -> frozenset[str]:
    """The paper's own web link targets, which prove a wrapped URL's joined form."""
    try:
        import fitz
        with fitz.open(stream=content, filetype="pdf") as document:
            return frozenset(link["uri"] for page in document for link in page.get_links()
                             if str(link.get("uri") or "").lower().startswith(("http://", "https://")))
    except Exception:
        return frozenset()


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
