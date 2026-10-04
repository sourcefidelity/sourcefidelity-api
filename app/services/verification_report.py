"""Immutable, exact-scope persistence for inspectable verification reports."""

from __future__ import annotations

import logging

import hashlib

from app.log_safety import bounded_detail
import re
import json
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.report import VerificationReportRecord
from app.models.verification_run import VerificationRunRecord
from app.services.evidence_package import (
    build_evidence_package,
    validate_evidence_package,
)
from app.services.source_navigation import build_source_navigation_descriptor
from app.services.decisive_label_critic import (
    CHECK_TYPES,
    aggregate_critic_adjusted_citation,
)
from app.services.facet_evidence_judgment import (
    _source_voice_fields,
    _student_epistemic_commitment_cues,
    aggregate_candidate_facets,
    aggregate_citation_findings,
    attribution_support_is_invalid,
    inherited_scope_mapping_is_invalid,
    source_discourse_annotation_is_valid,
)
from app.services.citation_use_router import (
    citation_use_route_input_ids,
    routed_relationship_candidate_ids,
)
from app.services.student_claim_clarity import claim_clarity_explanation
from app.services.verification_evidence import (
    FacetEvidenceMapping,
    VerificationEvidenceArtifact,
    passage_role_from_text,
    document_metadata_member_admissible,
    _passage_boundary_status,
    passage_matches_page_locator,
    _is_explanatory_note,
)


REPORT_FORMAT_VERSION = "verification-report-v20"
MAX_REPORT_CLAIM_CHARACTERS = 5_000
MAX_REPORT_PASSAGE_CHARACTERS = 1_200


class ReportAuthorizationError(ValueError):
    """The requested report scope does not match its evidence provenance.

    Every message raised here is a fixed internal literal, never interpolated
    student, source or network text, so a slug of the message is safe to
    record. Recording only the class name made distinct causes indistinguishable
    in job failures and required an offline replay to diagnose.
    """

    def __init__(self, *args: object, detail: str | None = None) -> None:
        super().__init__(*args)
        message = str(args[0]) if args else ""
        slug = re.sub(r"[^a-z0-9]+", "_", message.casefold()).strip("_")[:80]
        self.code = slug or "report_authorization_failed"
        self.reason_code = self.code
        # One shape for every bounded detail in the application; a second
        # definition here once accepted less than `safe_exception_detail`
        # would, so the same failure was diagnosable or not depending on
        # which class raised it.
        self.detail = bounded_detail(detail)


def persist_verification_report(
    session: Session,
    artifact: VerificationEvidenceArtifact,
    *,
    scope_type: str,
    scope_id: str,
    verification_run_id: str | uuid.UUID | None = None,
    mark_run_persisted: bool = True,
) -> VerificationReportRecord:
    """Persist an immutable version, or reuse an identical latest snapshot."""
    normalized_type, normalized_id = _validated_scope(scope_type, scope_id)
    _validate_artifact_scope(artifact, normalized_type, normalized_id)
    run = _validate_verification_run(
        session,
        artifact,
        verification_run_id=verification_run_id,
        scope_type=normalized_type,
        scope_id=normalized_id,
    )
    payload = build_inspectable_report_payload(artifact)
    digest = _payload_digest(payload)
    _lock_report_stream(
        session,
        scope_type=normalized_type,
        scope_id=normalized_id,
        verification_id=artifact.verification_id,
    )

    latest = session.scalar(
        select(VerificationReportRecord)
        .where(
            VerificationReportRecord.scope_type == normalized_type,
            VerificationReportRecord.scope_id == normalized_id,
            VerificationReportRecord.verification_id == artifact.verification_id,
        )
        .order_by(VerificationReportRecord.report_version.desc())
        .limit(1)
        .with_for_update()
    )
    if (
        latest is not None
        and latest.evidence_sha256 == digest
        and latest.verification_run_id == (run.id if run is not None else None)
    ):
        _store_judgment_reserve(session, artifact, latest)
        if run is not None and mark_run_persisted:
            run.status = "report_persisted"
            run.terminal_outcome = "success"
            session.flush()
        return latest

    record = VerificationReportRecord(
        verification_id=artifact.verification_id,
        paper_version_id=artifact.claim.paper_version_id,
        scope_type=normalized_type,
        scope_id=normalized_id,
        report_version=1 if latest is None else latest.report_version + 1,
        previous_report_id=None if latest is None else latest.id,
        verification_run_id=None if run is None else run.id,
        artifact_version=artifact.artifact_version,
        verdict=artifact.verdict.value,
        evidence_sha256=digest,
        report_payload=payload,
    )
    session.add(record)
    session.flush()
    _store_judgment_reserve(session, artifact, record)
    if run is not None and mark_run_persisted:
        run.status = "report_persisted"
        run.terminal_outcome = "success"
        session.flush()
    return record


def _store_judgment_reserve(session: Session, artifact, record) -> None:
    """Save the optional Judgment reserve beside its report; never fail the report."""
    reserve = getattr(artifact, "_judgment_reserve", None)
    if not reserve:
        return
    from app.services.judgment_reserve import store_reserve
    try:
        with session.begin_nested():
            store_reserve(session, record, reserve)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Judgment reserve not stored (type=%s)", type(exc).__name__)


def build_inspectable_report_payload(
    artifact: VerificationEvidenceArtifact,
) -> dict:
    """Build a bounded evidence snapshot without source binaries/full text."""
    if artifact.source_binding is None:
        raise ReportAuthorizationError(
            "Legacy evidence artifact lacks exact source-specific binding; "
            "rebuild or explicitly migrate it before current report persistence"
        )
    evidence_package = build_evidence_package(artifact)
    from app.services.joint_evidence_selection import joint_selection_context
    validate_evidence_package(evidence_package)
    source_navigation = build_source_navigation_descriptor(evidence_package)
    claim_text = artifact.claim.text
    passages = []
    for passage in artifact.passages:
        excerpt = passage.text[:MAX_REPORT_PASSAGE_CHARACTERS]
        passages.append(
            {
                "passage_id": passage.passage_id,
                "representation_id": passage.representation_id,
                "content_sha256": passage.content_sha256,
                "authorization_scope_type": passage.authorization_scope_type,
                "authorization_scope_id": passage.authorization_scope_id,
                "page_index": passage.page_index,
                "page_label": passage.page_label,
                "character_start": passage.character_start,
                "character_end": passage.character_end,
                "excerpt": excerpt,
                "excerpt_truncated": len(passage.text) > len(excerpt),
                "passage_text_sha256": hashlib.sha256(
                    passage.text.encode("utf-8")
                ).hexdigest(),
                "retrieval_method": passage.retrieval_method,
                "retrieval_score": passage.retrieval_score,
                "passage_role": passage.passage_role,
                "retrieval_rule_version": passage.retrieval_rule_version,
            }
        )
    return {
        "report_format_version": REPORT_FORMAT_VERSION,
        "artifact_version": artifact.artifact_version,
        "verification_id": artifact.verification_id,
        "evidence_created_at": artifact.created_at.isoformat(),
        "paper_version_id": artifact.claim.paper_version_id,
        "verification_run_id": artifact.source_identity.verification_run_id,
        "authoritative_evidence_package": evidence_package.model_dump(mode="json"),
        "source_navigation": source_navigation.model_dump(mode="json"),
        "claim": {
            **artifact.claim.model_dump(mode="json", exclude={"text"}),
            "text": claim_text[:MAX_REPORT_CLAIM_CHARACTERS],
            "text_truncated": len(claim_text) > MAX_REPORT_CLAIM_CHARACTERS,
            "text_sha256": hashlib.sha256(claim_text.encode("utf-8")).hexdigest(),
        },
        "source_binding": artifact.source_binding.model_dump(mode="json"),
        "source_identity": artifact.source_identity.model_dump(mode="json"),
        "coverage": artifact.coverage.model_dump(mode="json"),
        "passages": passages,
        "relationship_signal": artifact.relationship.model_dump(mode="json"),
        "passage_relevance_gate": artifact.passage_relevance.model_dump(mode="json"),
        # The scope comparison travels with the excerpt it was made against,
        # bounded by the same limit as any other stored source text. Without
        # it the report cannot verify the judgment and will not display one.
        "source_scope_assessment": artifact.source_scope_assessment.model_dump(
            mode="json"
        ),
        "joint_evidence_selection": artifact.joint_evidence_selection.model_dump(mode="json"),
        "joint_selection_context": joint_selection_context(artifact),
        "sentence_evidence": artifact.sentence_evidence.model_dump(mode="json"),
        "structured_judgment": artifact.judgment.model_dump(mode="json"),
        "evidence_conditioned_unit_judgment": artifact.unit_judgment.model_dump(
            mode="json"
        ),
        "verification_candidates": artifact.verification_candidates.model_dump(
            mode="json"
        ),
        "citation_use_routing": artifact.citation_use_routing.model_dump(
            mode="json"
        ),
        "student_claim_clarity": artifact.student_claim_clarity.model_dump(
            mode="json"
        ),
        "candidate_passage_retrieval": artifact.candidate_passage_retrieval.model_dump(
            mode="json"
        ),
        "candidate_relationships": artifact.candidate_relationships.model_dump(
            mode="json"
        ),
        "facet_evidence_foundation": artifact.facet_evidence_foundation.model_dump(
            mode="json"
        ),
        "facet_evidence_ledger": artifact.facet_evidence_ledger.model_dump(
            mode="json"
        ),
        "pairwise_facet_evaluation": artifact.pairwise_facet_evaluation.model_dump(
            mode="json"
        ),
        "decisive_label_checklist": artifact.decisive_critic.model_dump(mode="json"),
        "verdict": artifact.verdict.value,
        "reason_codes": list(artifact.reason_codes),
        "retrieval_route": artifact.retrieval_route,
        "report_limits": [
            "This report presents inspectable evidence, not a determination of intent or misconduct.",
            "Passage excerpts are bounded; source binaries and unrestricted full text are not stored in the report.",
            "Inconclusive and not-assessed results identify evidence limits rather than source or student fault.",
        ],
    }


def get_verification_report(
    session: Session,
    report_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
) -> VerificationReportRecord:
    """Return a record only after an exact scope match."""
    normalized_type, normalized_id = _validated_scope(scope_type, scope_id)
    try:
        parsed_id = report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
    except (TypeError, ValueError) as exc:
        raise ReportAuthorizationError("Invalid report identifier") from exc
    record = session.get(VerificationReportRecord, parsed_id)
    if record is None:
        raise ReportAuthorizationError("Report does not exist in the authorized scope")
    if record.scope_type != normalized_type or record.scope_id != normalized_id:
        raise ReportAuthorizationError("Report does not exist in the authorized scope")
    return record


def _validate_artifact_scope(artifact, scope_type, scope_id):
    identity = artifact.source_identity
    _validate_claim_marker_membership(artifact.claim)
    _validate_source_binding(artifact)
    _validate_claim_antecedents(artifact.claim)
    _validate_claim_discourse(artifact.claim)
    if artifact.joint_evidence_selection.status == "complete":
        from app.services.joint_evidence_selection import validate_joint_selection
        if not validate_joint_selection(artifact, artifact.joint_evidence_selection):
            raise ReportAuthorizationError("Joint evidence selection is not bound to current evidence")
    if (
        identity.authorization_scope_type != scope_type
        or identity.authorization_scope_id != scope_id
    ):
        raise ReportAuthorizationError("Evidence identity does not match the report scope")
    for passage in artifact.passages:
        if (
            passage.authorization_scope_type != scope_type
            or passage.authorization_scope_id != scope_id
            or passage.representation_id != identity.representation_id
            or passage.content_sha256 != identity.content_sha256
            or passage.verification_run_id != identity.verification_run_id
        ):
            raise ReportAuthorizationError(
                "Passage evidence does not match the authorized representation"
            )
    supplied_ids = {passage.passage_id for passage in artifact.passages}
    _validate_evidence_obligations(artifact)
    relevance = artifact.passage_relevance
    assessment_ids = [
        assessment.passage_id for assessment in relevance.assessments
    ]
    if relevance.decision_applied:
        raise ReportAuthorizationError("Passage-relevance gate is shadow-only")
    if len(assessment_ids) != len(set(assessment_ids)):
        raise ReportAuthorizationError(
            "Passage-relevance gate contains duplicate passage assessments"
        )
    if any(passage_id not in supplied_ids for passage_id in assessment_ids):
        raise ReportAuthorizationError(
            "Passage-relevance gate refers to non-persisted passage evidence"
        )
    passage_by_id = {passage.passage_id: passage for passage in artifact.passages}
    for assessment in relevance.assessments:
        if assessment.display_observation is not None and assessment.assessed_text_sha256 is None:
            raise ReportAuthorizationError("Display observation lacks assessed-text binding")
        if assessment.assessed_text_sha256 is None:
            continue
        passage = passage_by_id[assessment.passage_id]
        end = assessment.assessed_text_offset_end
        if end is None or end > len(passage.text):
            raise ReportAuthorizationError(
                "Passage-relevance assessed-text bounds are invalid"
            )
        excerpt = passage.text[assessment.assessed_text_offset_start:end]
        _validate_display_observation(assessment, excerpt, artifact.claim.text)
        if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != (
            assessment.assessed_text_sha256
        ):
            raise ReportAuthorizationError(
                "Passage-relevance assessment is not bound to its exact input text"
            )
        if assessment.assessment_input_truncated != (len(excerpt) != len(passage.text)):
            raise ReportAuthorizationError(
                "Passage-relevance truncation provenance is inconsistent"
            )
    expected_relevant_ids = [
        assessment.passage_id
        for assessment in relevance.assessments
        if assessment.relevance in {"relevant", "partially_relevant"}
    ]
    if (
        len(relevance.relevant_passage_ids)
        != len(set(relevance.relevant_passage_ids))
        or set(relevance.relevant_passage_ids) != set(expected_relevant_ids)
    ):
        raise ReportAuthorizationError(
            "Passage-relevance selected IDs do not match its typed assessments"
        )
    _validate_obligation_relevance_findings(
        artifact,
        supplied_ids=supplied_ids,
        passage_by_id=passage_by_id,
    )
    for selected_id in [
        *artifact.relationship.passage_ids,
        *artifact.judgment.passage_ids,
        *relevance.relevant_passage_ids,
        *(
            passage_id
            for finding in artifact.unit_judgment.findings
            for passage_id in finding.passage_ids
        ),
        *(
            passage_id
            for finding in artifact.candidate_relationships.findings
            for passage_id in finding.passage_ids
        ),
    ]:
        if selected_id not in supplied_ids:
            raise ReportAuthorizationError(
                "A recorded relationship refers to non-persisted passage evidence"
            )
    relevant_ids = set(relevance.relevant_passage_ids)
    for finding in artifact.unit_judgment.findings:
        if any(passage_id not in relevant_ids for passage_id in finding.passage_ids):
            raise ReportAuthorizationError(
                "A unit finding refers to a passage not admitted by the relevance gate"
            )
    _validate_unit_judgment_segments(artifact)
    _validate_citation_use_routing(artifact)
    _validate_student_claim_clarity(artifact)
    _validate_candidate_passage_retrieval(artifact, supplied_ids)
    _validate_candidate_relationships(artifact)
    _validate_facet_evidence(artifact)
    _validate_pairwise_facet_evaluation(artifact)
    _validate_decisive_critic(artifact)


def _validate_evidence_obligations(artifact):
    obligations = artifact.evidence_obligations
    if obligations.status != "complete":
        return
    binding = artifact.source_binding
    if binding is None or binding.status != "exact":
        raise ReportAuthorizationError(
            "Complete evidence obligations require an exact source binding"
        )
    ids = [item.obligation_id for item in obligations.obligations]
    if not ids or len(ids) != len(set(ids)):
        raise ReportAuthorizationError(
            "Evidence obligations must be nonempty and uniquely identified"
        )
    interpretations = {
        item.interpretation_id: item
        for item in artifact.student_statement_interpretations
    }
    candidate_text = {
        item.candidate_id: item.text
        for item in artifact.verification_candidates.candidates
    }
    for interpretation in artifact.student_statement_interpretations:
        text = candidate_text.get(interpretation.candidate_id)
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != (
            interpretation.candidate_text_sha256
        ):
            raise ReportAuthorizationError(
                "Student interpretation is not bound to an exact candidate"
            )
        if interpretation.source_evidence_received is not False:
            raise ReportAuthorizationError(
                "Student interpretation exceeded the source-blind boundary"
            )
    for item in obligations.obligations:
        if item.reference_id != binding.reference_id:
            raise ReportAuthorizationError(
                "Evidence obligation does not match the active reference member"
            )
        if hashlib.sha256(item.target_text.encode("utf-8")).hexdigest() != (
            item.target_text_sha256
        ):
            raise ReportAuthorizationError(
                "Evidence obligation target hash does not match"
            )
        if item.obligation_type == "coverage_only_semantic_repair":
            if (
                item.accuracy_judgment_allowed
                or not item.coverage_judgment_allowed
                or not item.interpretation_id
            ):
                raise ReportAuthorizationError(
                    "Coverage-only evidence obligation exceeded its judgment boundary"
                )
            interpretation = interpretations.get(item.interpretation_id)
            if (
                interpretation is None
                or interpretation.status != "semantic_repair"
                or not interpretation.coverage_judgment_allowed
                or interpretation.interpreted_statement != item.target_text
            ):
                raise ReportAuthorizationError(
                    "Coverage-only obligation is not bound to its source-blind repair"
                )


def _validate_obligation_relevance_findings(
    artifact,
    *,
    supplied_ids,
    passage_by_id,
):
    relevance = artifact.passage_relevance
    findings = relevance.obligation_findings
    if not findings:
        return
    obligations = artifact.evidence_obligations.obligations
    obligation_by_id = {item.obligation_id: item for item in obligations}
    finding_ids = [item.obligation_id for item in findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReportAuthorizationError(
            "Passage relevance contains duplicate obligation findings"
        )
    if any(item_id not in obligation_by_id for item_id in finding_ids):
        raise ReportAuthorizationError(
            "Passage relevance refers to an unknown evidence obligation"
        )
    for finding in findings:
        obligation = obligation_by_id[finding.obligation_id]
        if finding.obligation_type != obligation.obligation_type:
            raise ReportAuthorizationError(
                "Passage relevance changed an evidence-obligation type"
            )
        ids = [item.passage_id for item in finding.assessments]
        if len(ids) != len(set(ids)) or any(item_id not in supplied_ids for item_id in ids):
            raise ReportAuthorizationError(
                "An obligation finding refers to invalid passage evidence"
            )
        expected = {
            item.passage_id
            for item in finding.assessments
            if item.relevance in {"relevant", "partially_relevant"}
        }
        if set(finding.relevant_passage_ids) != expected:
            raise ReportAuthorizationError(
                "An obligation finding's selected passages do not match its assessments"
            )
        for assessment in finding.assessments:
            if assessment.display_observation is not None and assessment.assessed_text_sha256 is None:
                raise ReportAuthorizationError("Display observation lacks assessed-text binding")
            if assessment.assessed_text_sha256 is None:
                continue
            passage = passage_by_id[assessment.passage_id]
            end = assessment.assessed_text_offset_end
            if end is None or end > len(passage.text):
                raise ReportAuthorizationError(
                    "An obligation finding has invalid assessed-text bounds"
                )
            excerpt = passage.text[assessment.assessed_text_offset_start:end]
            _validate_display_observation(assessment, excerpt, obligation.target_text)
            if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != (
                assessment.assessed_text_sha256
            ):
                raise ReportAuthorizationError(
                    "An obligation finding is not bound to its exact input text"
                )
    primary = next(
        (
            finding
            for finding in findings
            if obligation_by_id[finding.obligation_id].accuracy_judgment_allowed
            and finding.obligation_type != "coverage_only_semantic_repair"
        ),
        None,
    )
    if primary is None or (
        relevance.outcome != primary.outcome
        or relevance.relevant_passage_ids != primary.relevant_passage_ids
        or relevance.assessments != primary.assessments
    ):
        raise ReportAuthorizationError(
            "Top-level passage relevance is not the exact accuracy-eligible projection"
        )


def _validate_display_observation(assessment, excerpt, target):
    observation = assessment.display_observation
    if observation is not None and (observation.source_span not in excerpt
            or any(not span.strip() or span not in target for span in observation.claim_spans)):
        raise ReportAuthorizationError("Display observation is not bound to supplied source and claim spans")


def _validate_unit_judgment_segments(artifact):
    judgment = artifact.unit_judgment
    if judgment.decision_applied:
        raise ReportAuthorizationError(
            "Evidence-conditioned unit judgment is shadow-only"
        )
    if judgment.status == "complete" and judgment.unresolved_segments:
        raise ReportAuthorizationError(
            "A complete unit judgment cannot contain unresolved segments"
        )
    if (
        judgment.status == "complete"
        and not judgment.findings
        and judgment.outcome != "no_relevant_candidate_passage"
    ):
        raise ReportAuthorizationError(
            "A complete unit judgment requires at least one exact finding"
        )
    if judgment.outcome == "no_relevant_candidate_passage" and (
        judgment.status != "complete"
        or judgment.findings
        or judgment.unresolved_segments
        or artifact.passage_relevance.outcome != "no_relevant_candidate_passage"
    ):
        raise ReportAuthorizationError(
            "No-relevant-candidate unit outcome is inconsistent with its relevance gate"
        )
    if judgment.status == "not_assessed" and (
        judgment.findings or judgment.unresolved_segments
    ):
        raise ReportAuthorizationError(
            "A not-assessed unit judgment cannot contain segmented findings"
        )
    claim = artifact.claim
    all_segments = [
        segment
        for finding in judgment.findings
        for segment in finding.segments
    ] + list(judgment.unresolved_segments)
    for segment in all_segments:
        if (
            segment.local_end > len(claim.text)
            or claim.text[segment.local_start:segment.local_end] != segment.text
            or segment.paper_start != claim.passage_start + segment.local_start
            or segment.paper_end != claim.passage_start + segment.local_end
        ):
            raise ReportAuthorizationError(
                "Unit-judgment segment does not match the persisted citation unit"
            )
    for finding in judgment.findings:
        rendered = " ".join(
            segment.text.strip() for segment in finding.segments
        )
        if finding.text != rendered:
            raise ReportAuthorizationError(
                "Unit-judgment finding text is not derived from its exact segments"
            )
        if finding.attribution == "student" and (
            finding.relationship.value != "not_assessed" or finding.passage_ids
        ):
            raise ReportAuthorizationError(
                "Student analysis cannot carry a source relationship or passage ID"
            )
        if finding.attribution == "ambiguous" and finding.relationship.value not in {
            "insufficient_evidence",
            "not_assessed",
        }:
            raise ReportAuthorizationError(
                "Ambiguous attribution cannot carry a substantive relationship"
            )
        if finding.relationship.value in {
            "supports",
            "contradicts",
            "unrelated",
        } and not finding.passage_ids:
            raise ReportAuthorizationError(
                "A substantive unit finding requires persisted passage evidence"
            )


def _routed_eligible_ids(artifact) -> set[str]:
    return routed_relationship_candidate_ids(artifact)


def _validate_citation_use_routing(artifact):
    routing = artifact.citation_use_routing
    if routing.decision_applied:
        raise ReportAuthorizationError("Citation-use routing is shadow-only")
    candidates = {
        candidate.candidate_id: candidate
        for candidate in artifact.verification_candidates.candidates
    }
    relationship_candidate_ids = citation_use_route_input_ids(artifact)
    route_ids = [route.candidate_id for route in routing.routes]
    if len(route_ids) != len(set(route_ids)):
        raise ReportAuthorizationError("Citation-use routing contains duplicate candidates")
    if routing.status in {"complete", "incomplete"} and set(route_ids) != relationship_candidate_ids:
        raise ReportAuthorizationError(
            "Citation-use routing did not cover every exact relationship candidate"
        )
    guards = [
        candidate.candidate_id
        for candidate in candidates.values()
        if candidate.role == "whole_unit_guard"
    ]
    expected_guard = guards[0] if len(guards) == 1 else None
    if routing.status in {"complete", "incomplete"} and (
        routing.complete_citation_unit_candidate_id != expected_guard
    ):
        raise ReportAuthorizationError(
            "Citation-use routing does not preserve the complete citation-unit guard"
        )
    for route in routing.routes:
        candidate = candidates.get(route.candidate_id)
        if candidate is None or candidate.role != "relationship_candidate":
            raise ReportAuthorizationError(
                "Citation-use routing refers to an unknown or guard candidate"
            )
        expected_hash = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        if route.candidate_text_sha256 != expected_hash:
            raise ReportAuthorizationError(
                "Citation-use routing does not bind the exact candidate text"
            )
        allowed = (
            route.status == "ready"
            and route.evidence_procedure
            in {
                "bounded_passage_relationship",
                "bounded_source_proposition_relationship",
            }
            and candidate.relationship_eligible
        )
        if route.relationship_judgment_allowed != allowed:
            raise ReportAuthorizationError(
                "Citation-use route status, procedure, and candidate eligibility conflict"
            )
        if (
            candidate.verification_scope == "source_wide_coverage"
            and route.citation_use != "source_wide_coverage"
        ):
            raise ReportAuthorizationError(
                "Source-wide candidate was routed to an ordinary evidence procedure"
            )
        if (
            candidate.verification_scope == "not_source_verification"
            and route.citation_use != "student_analysis"
        ):
            raise ReportAuthorizationError(
                "Student analysis was routed to source verification"
            )
        if candidate.attribution == "ambiguous" and route.citation_use != "unresolved":
            raise ReportAuthorizationError(
                "Ambiguous student/source voice was assigned a substantive route"
            )


def _validate_student_claim_clarity(artifact):
    clarity = artifact.student_claim_clarity
    if clarity.decision_applied:
        raise ReportAuthorizationError("Student-claim clarity gate is shadow-only")
    candidates = {
        candidate.candidate_id: candidate
        for candidate in artifact.verification_candidates.candidates
    }
    routed_ids = _routed_eligible_ids(artifact)
    finding_ids = [finding.candidate_id for finding in clarity.findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReportAuthorizationError(
            "Student-claim clarity contains duplicate candidate findings"
        )
    if any(candidate_id not in routed_ids for candidate_id in finding_ids):
        raise ReportAuthorizationError(
            "Student-claim clarity refers to an unknown or unrouted candidate"
        )
    if clarity.status in {"complete", "incomplete"} and set(finding_ids) != routed_ids:
        raise ReportAuthorizationError(
            "Student-claim clarity did not assess every routed candidate"
        )
    if clarity.status == "complete" and any(
        finding.status == "uncertain" for finding in clarity.findings
    ):
        raise ReportAuthorizationError(
            "A complete student-claim clarity gate contains uncertainty"
        )
    expected_blocked = [
        finding.candidate_id
        for finding in clarity.findings
        if finding.status == "not_assessed"
    ]
    if clarity.blocked_candidate_ids != expected_blocked:
        raise ReportAuthorizationError(
            "Student-claim blocked IDs do not match typed findings"
        )
    claim = artifact.claim
    for finding in clarity.findings:
        candidate = candidates[finding.candidate_id]
        expected_hash = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        if finding.candidate_text_sha256 != expected_hash:
            raise ReportAuthorizationError(
                "Student-claim clarity does not bind the exact candidate text"
            )
        if finding.explanation != claim_clarity_explanation(finding.reason_code):
            raise ReportAuthorizationError(
                "Student-claim clarity explanation is not application-derived"
            )
        if finding.status == "clear" and (
            finding.reason_code != "interpretable_relationship"
            or finding.problem_segments
        ):
            raise ReportAuthorizationError(
                "A clear student claim cannot carry a wording problem"
            )
        if finding.status == "not_assessed" and (
            finding.reason_code
            not in {
                "unresolved_local_reference",
                "internally_underspecified_relationship",
                "semantically_uninterpretable_wording",
                "conflicting_internal_scope",
            }
            or not finding.problem_segments
        ):
            raise ReportAuthorizationError(
                "A clarity abstention requires a typed exact wording problem"
            )
        if finding.status == "uncertain" and (
            finding.reason_code
            not in {"clarity_uncertain", "clarity_assessment_unavailable"}
            or finding.problem_segments
        ):
            raise ReportAuthorizationError(
                "An uncertain clarity finding cannot claim an exact wording problem"
            )
        candidate_ranges = [
            (segment.local_start, segment.local_end)
            for segment in candidate.segments
        ]
        for segment in finding.problem_segments:
            _validate_candidate_segment(claim, segment)
            if not any(
                start <= segment.local_start
                and segment.local_end <= end
                for start, end in candidate_ranges
            ):
                raise ReportAuthorizationError(
                    "A clarity problem span is outside the exact candidate"
                )


def _validate_candidate_relationships(artifact):
    candidate_set = artifact.verification_candidates
    evaluation = artifact.candidate_relationships
    if evaluation.decision_applied:
        raise ReportAuthorizationError(
            "Candidate relationship evaluation is shadow-only"
        )
    candidates = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    if len(candidates) != len(candidate_set.candidates):
        raise ReportAuthorizationError("Verification candidate IDs must be unique")
    if candidate_set.status == "complete" and candidate_set.uncovered_segments:
        raise ReportAuthorizationError(
            "A complete verification-candidate set cannot contain uncovered text"
        )
    claim = artifact.claim
    route_input_ids = citation_use_route_input_ids(artifact)
    for candidate in candidate_set.candidates:
        expected_eligible = (
            candidate.role == "relationship_candidate"
            and candidate.attribution == "cited_source"
            and candidate.verification_scope == "bounded_passage_relationship"
            and candidate.candidate_id in route_input_ids
            and "candidate_integrity:materially_redundant_candidate"
            not in candidate.limitations
        )
        if candidate.relationship_eligible != expected_eligible:
            raise ReportAuthorizationError(
                "Verification candidate role, scope, attribution, and eligibility are inconsistent"
            )
        for segment in candidate.segments:
            _validate_candidate_segment(claim, segment)
        rendered = " ".join(segment.text.strip() for segment in candidate.segments)
        if candidate.text != rendered:
            raise ReportAuthorizationError(
                "Verification-candidate text is not derived from exact segments"
            )
    for segment in candidate_set.uncovered_segments:
        _validate_candidate_segment(claim, segment)

    finding_ids = [finding.candidate_id for finding in evaluation.findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReportAuthorizationError(
            "Candidate relationship evaluation contains duplicate candidates"
        )
    eligible_ids = _routed_eligible_ids(artifact)
    if any(candidate_id not in eligible_ids for candidate_id in finding_ids):
        raise ReportAuthorizationError(
            "A relationship finding refers to an ineligible or unknown candidate"
        )
    expected_unresolved = [
        finding.candidate_id
        for finding in evaluation.findings
        if finding.status in {"uncertain", "not_assessed"}
    ]
    if evaluation.unresolved_candidate_ids != expected_unresolved:
        raise ReportAuthorizationError(
            "Candidate unresolved IDs do not match their typed findings"
        )
    if evaluation.status in {"complete", "incomplete"} and set(finding_ids) != eligible_ids:
        raise ReportAuthorizationError(
            "Candidate relationship evaluation did not classify every eligible candidate"
        )
    if evaluation.status == "complete" and (
        evaluation.unresolved_candidate_ids or candidate_set.uncovered_segments
    ):
        raise ReportAuthorizationError(
            "A complete candidate relationship evaluation contains unresolved material"
        )
    if evaluation.status == "not_assessed" and evaluation.findings:
        raise ReportAuthorizationError(
            "A not-assessed candidate relationship evaluation cannot contain findings"
        )
    supplied_ids = {passage.passage_id for passage in artifact.passages}
    selected_by_candidate = {
        selection.candidate_id: {
            item.passage_id for item in selection.passages
        }
        for selection in artifact.candidate_passage_retrieval.selections
    }
    for finding in evaluation.findings:
        candidate = candidates[finding.candidate_id]
        if candidate.requires_antecedent_context:
            if (
                artifact.claim.context_dependency_status in {"ambiguous", "unresolved"}
                and finding.status == "assessed"
            ):
                raise ReportAuthorizationError(
                    "An assessed candidate bypassed an unresolved local antecedent"
                )
            if finding.status == "assessed" and finding.context_resolution != "resolved":
                raise ReportAuthorizationError(
                    "An assessed context-dependent candidate lacks a resolved antecedent"
                )
        elif finding.context_resolution != "not_required":
            raise ReportAuthorizationError(
                "A candidate recorded an unexpected antecedent resolution"
            )
        if any(passage_id not in supplied_ids for passage_id in finding.passage_ids):
            raise ReportAuthorizationError(
                "A candidate relationship refers to non-persisted passage evidence"
            )
        if (
            finding.candidate_id in selected_by_candidate
            and any(
                passage_id not in selected_by_candidate[finding.candidate_id]
                for passage_id in finding.passage_ids
            )
        ):
            raise ReportAuthorizationError(
                "A candidate relationship refers to evidence not selected for that candidate"
            )
        if finding.status in {"not_proposition", "uncertain", "not_assessed"} and (
            finding.relationship.value != "not_assessed"
            or finding.evidence_coverage != "not_assessed"
            or finding.passage_ids
        ):
            raise ReportAuthorizationError(
                "An abstained candidate cannot carry a source relationship"
            )
        if finding.attribution == "student" and (
            finding.relationship.value != "not_assessed" or finding.passage_ids
        ):
            raise ReportAuthorizationError(
                "Student analysis cannot carry a candidate source relationship"
            )
        if finding.attribution == "ambiguous" and finding.relationship.value not in {
            "insufficient_evidence", "not_assessed"
        }:
            raise ReportAuthorizationError(
                "Ambiguous candidate attribution cannot carry a substantive relationship"
            )
        if finding.relationship.value in {"supports", "contradicts"} and finding.evidence_coverage != "complete":
            raise ReportAuthorizationError(
                "A substantive candidate relationship admitted missing material details"
            )
        if finding.relationship.value == "insufficient_evidence" and finding.evidence_coverage not in {
            "partial", "absent", "uncertain"
        }:
            raise ReportAuthorizationError(
                "Candidate evidence coverage conflicts with insufficient evidence"
            )
        if finding.relationship.value in {"supports", "contradicts", "unrelated"} and not finding.passage_ids:
            raise ReportAuthorizationError(
                "A substantive candidate relationship requires passage evidence"
            )


def _validate_candidate_passage_retrieval(artifact, supplied_ids):
    retrieval = artifact.candidate_passage_retrieval
    candidate_ids = _routed_eligible_ids(artifact)
    selection_ids = [selection.candidate_id for selection in retrieval.selections]
    if len(selection_ids) != len(set(selection_ids)):
        raise ReportAuthorizationError(
            "Candidate passage retrieval contains duplicate candidate IDs"
        )
    if any(candidate_id not in candidate_ids for candidate_id in selection_ids):
        raise ReportAuthorizationError(
            "Candidate passage retrieval refers to an unknown or ineligible candidate"
        )
    if retrieval.status in {"complete", "incomplete"} and set(selection_ids) != candidate_ids:
        raise ReportAuthorizationError(
            "Candidate passage retrieval did not search every eligible candidate"
        )
    if retrieval.status == "complete" and any(
        not selection.passages for selection in retrieval.selections
    ):
        raise ReportAuthorizationError(
            "Complete candidate passage retrieval contains an empty selection"
        )
    for selection in retrieval.selections:
        ranks = [item.rank for item in selection.passages]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ReportAuthorizationError(
                "Candidate passage ranks must be unique and contiguous"
            )
        passage_ids = [item.passage_id for item in selection.passages]
        if len(passage_ids) != len(set(passage_ids)):
            raise ReportAuthorizationError(
                "Candidate passage retrieval contains duplicate passage IDs"
            )
        if any(passage_id not in supplied_ids for passage_id in passage_ids):
            raise ReportAuthorizationError(
                "Candidate passage retrieval refers to non-persisted passage evidence"
            )
    passages = {passage.passage_id: passage for passage in artifact.passages}
    excluded_roles = {"reference_list", "publication_metadata"}
    selected_channels_by_passage = {
        item.passage_id: set(item.channels)
        for selection in retrieval.selections
        for item in selection.passages
    }
    for passage_id in {
        item.passage_id
        for selection in retrieval.selections
        for item in selection.passages
    }:
        passage = passages[passage_id]
        text_role = passage_role_from_text(passage.text)
        channels = selected_channels_by_passage.get(passage_id, set())
        # A single extracted explanatory note need not repeat its source's
        # Notes heading. Recheck the existing producer's prose rule and typed
        # retrieval channel; never relabel it as ordinary body evidence.
        explanatory_note = bool(
            passage.passage_role == "citation_notes"
            and passage.retrieval_method == "explanatory_note_context"
            and "candidate_explanatory_note_context" in channels
            and text_role in {"body_prose", "unknown", "citation_notes"}
            and _is_explanatory_note(passage.text)
        )
        # Opening-page titles have a geometry-derived role: their bare words
        # cannot reproduce the title/byline layout when read as plain text.
        # Preserve the existing explicit aggregate-member channel, not a
        # general exemption for publication furniture or arbitrary role claims.
        document_title = bool(
            passage.passage_role == "document_metadata"
            and passage.retrieval_method == "document_level_member_evidence"
            and "candidate_document_metadata" in channels
            and len(set(artifact.claim.reference_ids)) > 1
            and document_metadata_member_admissible(
                page_index=passage.page_index, text=passage.text
            )
        )
        if passage.passage_role != text_role and not (explanatory_note or document_title):
            # Naming both roles and the producing method is the difference
            # between a diagnosable defect and an offline replay.
            raise ReportAuthorizationError(
                "Candidate passage role is not application-derived",
                detail=(
                    f"stored={passage.passage_role}; derived={text_role}; "
                    f"method={passage.retrieval_method}"
                ),
            )
        if passage.boundary_status != _passage_boundary_status(passage.text):
            raise ReportAuthorizationError(
                "Candidate passage boundary status is not application-derived",
                detail=(
                    f"stored={passage.boundary_status}; "
                    f"derived={_passage_boundary_status(passage.text)}; "
                    f"role={passage.passage_role}"
                ),
            )
        if passage.passage_role in excluded_roles:
            raise ReportAuthorizationError(
                "Candidate passage retrieval admitted a definite non-body block"
            )
        if passage.passage_role == "citation_notes" and not explanatory_note and not (
            selected_channels_by_passage.get(passage_id, set())
            & {"candidate_explicit_note", "candidate_note_fallback"}
        ):
            raise ReportAuthorizationError(
                "Citation-note evidence lacks an authorized note retrieval channel"
            )


def _validate_facet_evidence(artifact):
    foundation = artifact.facet_evidence_foundation
    ledger = artifact.facet_evidence_ledger
    if ledger.decision_applied:
        raise ReportAuthorizationError("Facet-evidence judgment is shadow-only")

    eligible_ids = _routed_eligible_ids(artifact)
    candidates = {
        candidate.candidate_id: candidate
        for candidate in artifact.verification_candidates.candidates
    }
    bundle_ids = [bundle.candidate_id for bundle in foundation.candidate_bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ReportAuthorizationError("Facet foundation contains duplicate candidate bundles")
    if any(candidate_id not in eligible_ids for candidate_id in bundle_ids):
        raise ReportAuthorizationError("Facet foundation refers to an unknown candidate")
    if foundation.status in {"complete", "incomplete"} and set(bundle_ids) != eligible_ids:
        raise ReportAuthorizationError("Facet foundation did not cover every eligible candidate")

    passages = {passage.passage_id: passage for passage in artifact.passages}
    sentences = {sentence.sentence_id: sentence for sentence in foundation.source_sentences}
    if len(sentences) != len(foundation.source_sentences):
        raise ReportAuthorizationError("Facet foundation contains duplicate evidence sentences")
    for sentence in foundation.source_sentences:
        passage = passages.get(sentence.passage_id)
        if (
            passage is None
            or sentence.passage_end > len(passage.text)
            or passage.text[sentence.passage_start:sentence.passage_end] != sentence.text
        ):
            raise ReportAuthorizationError("Facet evidence sentence does not match its persisted passage")
        expected_voice = _source_voice_fields(sentence.text)
        if (
            sentence.voice_role != expected_voice["voice_role"]
            or sentence.attributed_actor_texts
            != expected_voice["attributed_actor_texts"]
            or sentence.voice_cues != expected_voice["voice_cues"]
            or [
                relation.model_dump(mode="json")
                for relation in sentence.attribution_relations
            ]
            != expected_voice["attribution_relations"]
            or [
                cue.model_dump(mode="json")
                for cue in sentence.epistemic_commitment_cues
            ]
            != expected_voice["epistemic_commitment_cues"]
        ):
            raise ReportAuthorizationError(
                "Facet evidence source-voice annotations are not application-derived"
            )
        if not source_discourse_annotation_is_valid(sentence, sentences):
            raise ReportAuthorizationError(
                "Facet evidence source-discourse annotations are not application-derived"
            )

    selected_by_candidate = {
        selection.candidate_id: {item.passage_id for item in selection.passages}
        for selection in artifact.candidate_passage_retrieval.selections
    }
    facets_by_bundle = {}
    for bundle in foundation.candidate_bundles:
        candidate = candidates.get(bundle.candidate_id)
        if candidate is None:
            raise ReportAuthorizationError(
                "Facet bundle refers to an unknown verification candidate"
            )
        if bundle.student_epistemic_commitment_cues != (
            _student_epistemic_commitment_cues(artifact, candidate)
        ):
            raise ReportAuthorizationError(
                "Student epistemic cues are not application-derived"
            )
        facet_ids = [facet.facet_id for facet in bundle.facets]
        if len(facet_ids) != len(set(facet_ids)):
            raise ReportAuthorizationError("Facet bundle contains duplicate facet IDs")
        guards = [facet for facet in bundle.facets if facet.kind == "candidate_as_written"]
        if len(guards) != 1 or not guards[0].material_to_aggregate:
            raise ReportAuthorizationError("Facet bundle lacks one material controlling facet")
        for facet in bundle.facets:
            if facet.candidate_id != bundle.candidate_id:
                raise ReportAuthorizationError("Facet candidate ID does not match its bundle")
            if facet.kind == "exact_component" and facet.material_to_aggregate:
                raise ReportAuthorizationError("A diagnostic exact component cannot control aggregation")
            for segment in facet.segments:
                _validate_candidate_segment(artifact.claim, segment)
            for context_segment in facet.context_segments:
                matching_context = next(
                    (
                        context
                        for context in artifact.claim.antecedent_context
                        if context.context_index == context_segment.context_index
                    ),
                    None,
                )
                if matching_context != context_segment:
                    raise ReportAuthorizationError(
                        "Facet context does not match exact persisted student context"
                    )
            if facet.kind == "inherited_discourse_scope":
                if (
                    not facet.context_segments
                    or not facet.segments
                    or not facet.material_to_aggregate
                ):
                    raise ReportAuthorizationError(
                        "Inherited discourse scope lacks exact material context"
                    )
            elif facet.context_segments:
                raise ReportAuthorizationError(
                    "Only inherited discourse facets may carry student context"
                )
            rendered = " ".join(
                [
                    *(segment.text.strip() for segment in facet.context_segments),
                    *(segment.text.strip() for segment in facet.segments),
                ]
            )
            if facet.text != rendered:
                raise ReportAuthorizationError("Facet text is not derived from exact student spans")
        if len(bundle.evidence_sentence_ids) != len(set(bundle.evidence_sentence_ids)):
            raise ReportAuthorizationError("Facet bundle contains duplicate evidence sentence IDs")
        if len(bundle.source_discourse_sentence_ids) != len(
            set(bundle.source_discourse_sentence_ids)
        ):
            raise ReportAuthorizationError(
                "Facet bundle contains duplicate source-discourse sentence IDs"
            )
        for sentence_id in bundle.evidence_sentence_ids:
            sentence = sentences.get(sentence_id)
            if sentence is None:
                raise ReportAuthorizationError("Facet bundle refers to an unknown evidence sentence")
            if sentence.passage_id not in selected_by_candidate.get(bundle.candidate_id, set()):
                raise ReportAuthorizationError("Facet bundle uses evidence not selected for its candidate")
            if any(
                scope_id not in bundle.source_discourse_sentence_ids
                for scope_id in sentence.discourse_evidence_sentence_ids
            ):
                raise ReportAuthorizationError(
                    "Facet sentence uses source-discourse evidence outside its bundle"
                )
        for sentence_id in bundle.source_discourse_sentence_ids:
            if sentence_id not in sentences:
                raise ReportAuthorizationError(
                    "Facet bundle refers to unknown source-discourse evidence"
                )
        facets_by_bundle[bundle.candidate_id] = {facet.facet_id: facet for facet in bundle.facets}

    finding_ids = [finding.candidate_id for finding in ledger.findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReportAuthorizationError("Facet ledger contains duplicate candidate findings")
    if ledger.status in {"complete", "incomplete"} and set(finding_ids) != set(bundle_ids):
        raise ReportAuthorizationError("Facet ledger did not cover every candidate bundle")
    if ledger.status == "not_assessed" and ledger.findings:
        raise ReportAuthorizationError("A not-assessed facet ledger cannot contain findings")

    bundles = {bundle.candidate_id: bundle for bundle in foundation.candidate_bundles}
    for finding in ledger.findings:
        bundle = bundles.get(finding.candidate_id)
        if bundle is None:
            raise ReportAuthorizationError("Facet ledger refers to an unknown candidate bundle")
        candidate = candidates[finding.candidate_id]
        if candidate.requires_antecedent_context:
            if (
                artifact.claim.context_dependency_status in {"ambiguous", "unresolved"}
                and finding.status == "assessed"
            ):
                raise ReportAuthorizationError(
                    "An assessed facet candidate bypassed an unresolved local antecedent"
                )
            if finding.status == "assessed" and finding.context_resolution != "resolved":
                raise ReportAuthorizationError("An assessed facet candidate lacks a resolved antecedent")
            if finding.context_resolution == "not_required":
                raise ReportAuthorizationError("A context-dependent facet candidate omitted resolution")
        elif finding.context_resolution != "not_required":
            raise ReportAuthorizationError("A facet candidate recorded unexpected antecedent resolution")
        mapping_ids = [mapping.facet_id for mapping in finding.mappings]
        if finding.status == "not_assessed":
            if finding.mappings or finding.derived_outcome != "not_assessed":
                raise ReportAuthorizationError("An abstained facet finding carries a derived judgment")
            continue
        if len(mapping_ids) != len(set(mapping_ids)) or set(mapping_ids) != set(facets_by_bundle[finding.candidate_id]):
            raise ReportAuthorizationError("Facet finding does not map every fixed facet exactly once")
        for mapping in finding.mappings:
            facet = facets_by_bundle[finding.candidate_id][mapping.facet_id]
            allowed_sentence_ids = set(bundle.evidence_sentence_ids)
            if facet.kind == "source_attribution":
                allowed_sentence_ids.update(bundle.source_discourse_sentence_ids)
            if any(sentence_id not in allowed_sentence_ids for sentence_id in mapping.evidence_sentence_ids):
                raise ReportAuthorizationError("Facet mapping refers to unauthorized candidate evidence")
            if mapping.direction in {"supports", "contradicts", "qualifies", "mixed"} and not mapping.evidence_sentence_ids:
                raise ReportAuthorizationError("Evidentiary facet direction lacks evidence")
            if mapping.direction == "none" and mapping.evidence_sentence_ids:
                raise ReportAuthorizationError("A none facet direction carries evidence")
            if attribution_support_is_invalid(
                facet,
                mapping,
                sentences,
                artifact.source_binding.cited_author_label,
                bundle.evidence_sentence_ids,
            ):
                raise ReportAuthorizationError(
                    "Facet attribution support conflicts with application-owned source voice"
                )
            if inherited_scope_mapping_is_invalid(facet, mapping, sentences):
                raise ReportAuthorizationError(
                    "Facet relationship ignores its exact inherited discourse domain"
                )
        derived = aggregate_candidate_facets(
            bundle,
            finding.mappings,
            context_resolution=finding.context_resolution,
        )
        if (
            derived.status != finding.status
            or derived.derived_outcome != finding.derived_outcome
            or derived.evidence_coverage != finding.evidence_coverage
        ):
            raise ReportAuthorizationError("Facet candidate aggregate was not derived by application rules")
        expected_locator_status = _expected_locator_status(
            artifact,
            finding,
            sentences,
            passages,
            source_discourse_sentence_ids=set(
                bundle.source_discourse_sentence_ids
            ),
        )
        if finding.locator_status != expected_locator_status:
            raise ReportAuthorizationError(
                "Facet candidate locator status is not application-derived"
            )

    expected_citation = (
        "not_assessed"
        if foundation.status == "incomplete"
        else aggregate_citation_findings(ledger.findings)
    )
    if ledger.status in {"complete", "incomplete"} and ledger.derived_citation_outcome != expected_citation:
        raise ReportAuthorizationError("Citation aggregate was not derived by application rules")


def _validate_pairwise_facet_evaluation(artifact):
    evaluation = artifact.pairwise_facet_evaluation
    if evaluation.decision_applied:
        raise ReportAuthorizationError("Pairwise facet evaluation is shadow-only")
    if evaluation.status == "not_run":
        if evaluation.findings or evaluation.derived_citation_outcome != "not_run":
            raise ReportAuthorizationError(
                "A not-run pairwise evaluation carries findings"
            )
        return
    if evaluation.status == "not_assessed":
        if evaluation.findings or evaluation.derived_citation_outcome != "not_assessed":
            raise ReportAuthorizationError(
                "A not-assessed pairwise evaluation carries findings"
            )
        return
    if evaluation.status not in {"complete", "incomplete"}:
        raise ReportAuthorizationError("Unknown pairwise evaluation status")

    foundation = artifact.facet_evidence_foundation
    bundles = {
        bundle.candidate_id: bundle
        for bundle in foundation.candidate_bundles
    }
    sentences = {
        sentence.sentence_id: sentence
        for sentence in foundation.source_sentences
    }
    passages = {passage.passage_id: passage for passage in artifact.passages}
    finding_ids = [finding.candidate_id for finding in evaluation.findings]
    if (
        len(finding_ids) != len(set(finding_ids))
        or set(finding_ids) != set(bundles)
    ):
        raise ReportAuthorizationError(
            "Pairwise evaluation did not cover each fixed candidate exactly once"
        )

    derived_findings = []
    for finding in evaluation.findings:
        bundle = bundles[finding.candidate_id]
        facets = {facet.facet_id: facet for facet in bundle.facets}
        selection_ids = [item.facet_id for item in finding.evidence_selections]
        decision_ids = [item.facet_id for item in finding.facet_decisions]
        if (
            len(selection_ids) != len(set(selection_ids))
            or set(selection_ids) != set(facets)
            or len(decision_ids) != len(set(decision_ids))
            or set(decision_ids) != set(facets)
        ):
            raise ReportAuthorizationError(
                "Pairwise candidate did not cover every fixed facet exactly once"
            )
        selections = {
            item.facet_id: item for item in finding.evidence_selections
        }
        allowed_ordinary = set(bundle.evidence_sentence_ids)
        allowed_discourse = set(bundle.source_discourse_sentence_ids)
        for facet_id, selection in selections.items():
            facet = facets[facet_id]
            allowed = set(allowed_ordinary)
            if facet.kind == "source_attribution":
                allowed.update(allowed_discourse)
            if any(
                sentence_id not in allowed
                for sentence_id in selection.evidence_sentence_ids
            ):
                raise ReportAuthorizationError(
                    "Pairwise selection refers to unauthorized evidence"
                )
            if selection.status == "evidence_selected" and not selection.evidence_sentence_ids:
                raise ReportAuthorizationError(
                    "Pairwise selected-evidence status lacks evidence"
                )
            if selection.status != "evidence_selected" and selection.evidence_sentence_ids:
                raise ReportAuthorizationError(
                    "Pairwise non-selection status carries evidence"
                )
            if not facet.material_to_aggregate and selection.status != "not_assessed":
                raise ReportAuthorizationError(
                    "Pairwise path routed a nonmaterial diagnostic facet"
                )

        mappings = []
        for decision in finding.facet_decisions:
            facet = facets[decision.facet_id]
            selection = selections[decision.facet_id]
            allowed = set(selection.evidence_sentence_ids)
            if facet.kind == "source_attribution":
                allowed.update(allowed_discourse)
            if any(
                sentence_id not in allowed
                for sentence_id in decision.evidence_sentence_ids
            ):
                raise ReportAuthorizationError(
                    "Pairwise direction refers to evidence outside its selection"
                )
            if decision.direction in {
                "supports",
                "contradicts",
                "qualifies",
                "mixed",
            } and not decision.evidence_sentence_ids:
                raise ReportAuthorizationError(
                    "Pairwise evidentiary direction lacks evidence"
                )
            if decision.direction == "none" and decision.evidence_sentence_ids:
                raise ReportAuthorizationError(
                    "Pairwise none direction carries evidence"
                )
            if not facet.material_to_aggregate and (
                decision.status != "not_assessed"
                or decision.direction != "uncertain"
            ):
                raise ReportAuthorizationError(
                    "Pairwise path judged a nonmaterial diagnostic facet"
                )
            mapping = {
                "facet_id": decision.facet_id,
                "direction": decision.direction,
                "confidence": decision.confidence,
                "evidence_sentence_ids": decision.evidence_sentence_ids,
                "rationale": decision.rationale,
                "limitations": decision.limitations,
            }
            mappings.append(mapping)

        holder = finding.proposition_holder
        attribution_facets = [
            facet for facet in facets.values() if facet.kind == "source_attribution"
        ]
        if attribution_facets:
            attribution_facet = attribution_facets[0]
            if (
                holder is None
                or holder.candidate_id != finding.candidate_id
                or holder.facet_id != attribution_facet.facet_id
                or any(
                    sentence_id not in allowed_ordinary | allowed_discourse
                    for sentence_id in holder.evidence_sentence_ids
                )
            ):
                raise ReportAuthorizationError(
                    "Pairwise proposition-holder assessment is not authorized"
                )
            holder_selection = selections[attribution_facet.facet_id]
            holder_decision = next(
                decision
                for decision in finding.facet_decisions
                if decision.facet_id == attribution_facet.facet_id
            )
            if holder.status == "assessed":
                if (
                    holder.holder_relation
                    not in {"document_author", "different_actor", "mixed"}
                    or not holder.evidence_sentence_ids
                    or holder_selection.status != "evidence_selected"
                    or holder_selection.evidence_sentence_ids
                    != holder.evidence_sentence_ids
                ):
                    raise ReportAuthorizationError(
                        "Pairwise assessed proposition holder lacks its exact evidence selection"
                    )
            elif (
                holder.holder_relation not in {"uncertain", "not_assessed"}
                or holder_selection.status == "evidence_selected"
            ):
                raise ReportAuthorizationError(
                    "Pairwise unresolved proposition holder carries a decisive selection"
                )
            required_direction = {
                "different_actor": "contradicts",
                "mixed": "qualifies",
                "document_author": "supports",
                "uncertain": "uncertain",
                "not_assessed": "uncertain",
            }[holder.holder_relation]
            if holder.mapped_direction != required_direction:
                raise ReportAuthorizationError(
                    "Pairwise proposition-holder relation changed its mapped direction"
                )
            allowed_final_directions = {
                "contradicts": {"contradicts"},
                "qualifies": {"qualifies", "contradicts"},
                "supports": {
                    "supports",
                    "qualifies",
                    "contradicts",
                    "uncertain",
                },
                "uncertain": {"uncertain"},
            }[holder.mapped_direction]
            if holder_decision.direction not in allowed_final_directions:
                raise ReportAuthorizationError(
                    "Pairwise proposition-holder relation does not match its attribution direction"
                )
        elif holder is not None:
            raise ReportAuthorizationError(
                "Pairwise proposition-holder result lacks an attribution facet"
            )

        typed_mappings = [FacetEvidenceMapping(**mapping) for mapping in mappings]
        derived = finding.derived_finding
        if derived.candidate_id != finding.candidate_id:
            raise ReportAuthorizationError(
                "Pairwise derived finding changed candidate identity"
            )
        if [mapping.model_dump() for mapping in derived.mappings] != [
            mapping.model_dump() for mapping in typed_mappings
        ]:
            raise ReportAuthorizationError(
                "Pairwise derived finding does not preserve its facet decisions"
            )
        recomputed = aggregate_candidate_facets(
            bundle,
            typed_mappings,
            context_resolution=derived.context_resolution,
        )
        if (
            recomputed.status != derived.status
            or recomputed.derived_outcome != derived.derived_outcome
            or recomputed.evidence_coverage != derived.evidence_coverage
        ):
            raise ReportAuthorizationError(
                "Pairwise candidate outcome is not application-derived"
            )
        expected_locator = _expected_locator_status(
            artifact,
            derived,
            sentences,
            passages,
            source_discourse_sentence_ids=allowed_discourse,
        )
        if derived.locator_status != expected_locator:
            raise ReportAuthorizationError(
                "Pairwise locator status is not application-derived"
            )
        derived_findings.append(derived)

    expected_citation = (
        "not_assessed"
        if foundation.status == "incomplete"
        else aggregate_citation_findings(derived_findings)
    )
    if evaluation.derived_citation_outcome != expected_citation:
        raise ReportAuthorizationError(
            "Pairwise citation outcome is not application-derived"
        )
    expected_status = (
        "incomplete"
        if foundation.status == "incomplete"
        or any(finding.status != "assessed" for finding in derived_findings)
        else "complete"
    )
    if evaluation.status != expected_status:
        raise ReportAuthorizationError(
            "Pairwise evaluation completeness is not application-derived"
        )


def _validate_decisive_critic(artifact):
    critic = artifact.decisive_critic
    ledger = artifact.facet_evidence_ledger
    if critic.decision_applied:
        raise ReportAuthorizationError("Decisive-label checklist is shadow-only")

    targets = {
        finding.candidate_id: finding
        for finding in ledger.findings
        if finding.status == "assessed"
        and finding.derived_outcome in {"supports", "contradicts"}
    }
    if ledger.status in {"complete", "incomplete"} and critic.status == "not_run":
        raise ReportAuthorizationError(
            "Artifact v10 facet judgments require an explicit decisive-checklist stage"
        )
    finding_ids = [finding.candidate_id for finding in critic.findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReportAuthorizationError("Decisive checklist contains duplicate candidates")
    if any(candidate_id not in targets for candidate_id in finding_ids):
        raise ReportAuthorizationError("Decisive checklist reviewed a non-decisive candidate")
    if critic.status in {"complete", "incomplete"} and set(finding_ids) != set(targets):
        raise ReportAuthorizationError("Decisive checklist did not cover every proposed decisive label")
    if critic.status == "not_run" and (
        critic.findings
        or critic.challenged_candidate_ids
        or critic.derived_citation_outcome != "not_run"
    ):
        raise ReportAuthorizationError("A not-run critic carries derived decisions")
    if critic.status == "not_assessed" and (
        critic.findings
        or critic.challenged_candidate_ids
        or critic.derived_citation_outcome != "not_assessed"
    ):
        raise ReportAuthorizationError("A not-assessed critic carries inconsistent decisions")

    bundles = {
        bundle.candidate_id: bundle
        for bundle in artifact.facet_evidence_foundation.candidate_bundles
    }
    challenged_ids = []
    for finding in critic.findings:
        proposed = targets[finding.candidate_id]
        bundle = bundles.get(finding.candidate_id)
        if bundle is None:
            raise ReportAuthorizationError("Decisive checklist lacks its fixed facet bundle")
        if finding.proposed_outcome != proposed.derived_outcome:
            raise ReportAuthorizationError("Decisive checklist changed its proposed label")
        material_ids = {
            facet.facet_id for facet in bundle.facets if facet.material_to_aggregate
        }
        evidence_ids = set(bundle.evidence_sentence_ids) | set(
            bundle.source_discourse_sentence_ids
        )
        if finding.status == "not_assessed":
            if (
                finding.failure_code == "none"
                or finding.checks
                or finding.challenge_types
                or finding.rationale
                or finding.reviewed_facet_ids
                or finding.challenged_facet_ids
                or finding.evidence_sentence_ids
                or finding.effective_outcome != "not_assessed"
            ):
                raise ReportAuthorizationError(
                    "Failed decisive checklist carries decisions or lacks a reason code"
                )
            continue

        if finding.failure_code != "none":
            raise ReportAuthorizationError(
                "Completed decisive checklist carries a failure reason code"
            )
        if (
            len(finding.reviewed_facet_ids)
            != len(set(finding.reviewed_facet_ids))
            or set(finding.reviewed_facet_ids) != material_ids
        ):
            raise ReportAuthorizationError(
                "Decisive checklist did not review every material facet exactly once"
            )
        check_types = [check.check_type for check in finding.checks]
        if len(check_types) != len(CHECK_TYPES) or set(check_types) != set(CHECK_TYPES):
            raise ReportAuthorizationError(
                "Decisive checklist type coverage is inconsistent"
            )
        by_type = {check.check_type: check for check in finding.checks}
        mapping_by_facet = {
            mapping.facet_id: mapping
            for mapping in proposed.mappings
            if mapping.facet_id in material_ids
        }
        normalized_checks = []
        for check_type in CHECK_TYPES:
            check = by_type[check_type]
            if (
                len(check.facet_ids) != len(set(check.facet_ids))
                or any(facet_id not in material_ids for facet_id in check.facet_ids)
            ):
                raise ReportAuthorizationError(
                    "Decisive checklist carries a nonmaterial or duplicate facet"
                )
            if (
                len(check.evidence_sentence_ids)
                != len(set(check.evidence_sentence_ids))
                or any(
                    sentence_id not in evidence_ids
                    for sentence_id in check.evidence_sentence_ids
                )
            ):
                raise ReportAuthorizationError(
                    "Decisive checklist carries unauthorized or duplicate evidence"
                )
            if check.result == "no_defect":
                expected_reconciliation = (
                    "mapping_supports_proposed_label"
                    if check_type == "missing_material_detail"
                    else "not_required"
                )
                if (
                    check.facet_ids
                    or check.evidence_sentence_ids
                    or check.mapping_reconciliation != expected_reconciliation
                ):
                    raise ReportAuthorizationError(
                        "No-defect checklist entry carries decision evidence"
                    )
            elif check.result == "defect":
                if not check.facet_ids or not check.evidence_sentence_ids:
                    raise ReportAuthorizationError(
                        "Defect checklist entry lacks a facet or evidence"
                    )
                if check_type == "missing_material_detail":
                    if check.mapping_reconciliation != "mapping_does_not_establish_facet":
                        raise ReportAuthorizationError(
                            "Missing-detail defect did not reconcile the original mapping"
                        )
                    for facet_id in check.facet_ids:
                        mapping = mapping_by_facet.get(facet_id)
                        if mapping is None or not (
                            set(mapping.evidence_sentence_ids)
                            & set(check.evidence_sentence_ids)
                        ):
                            raise ReportAuthorizationError(
                                "Missing-detail defect does not cite its mapped evidence"
                            )
                elif check.mapping_reconciliation != "not_required":
                    raise ReportAuthorizationError(
                        "Non-detail checklist entry altered mapping reconciliation"
                    )
            else:
                if check_type == "missing_material_detail":
                    if check.mapping_reconciliation != "uncertain":
                        raise ReportAuthorizationError(
                            "Uncertain missing-detail check has inconsistent reconciliation"
                        )
                elif check.mapping_reconciliation not in {"not_required", "uncertain"}:
                    raise ReportAuthorizationError(
                        "Uncertain checklist entry has inconsistent reconciliation"
                    )
            normalized_checks.append(check)

        defects = [check for check in normalized_checks if check.result == "defect"]
        uncertainties = [
            check for check in normalized_checks if check.result == "uncertain"
        ]
        expected_status = (
            "challenged" if defects else "uncertain" if uncertainties else "upheld"
        )
        expected_challenge_types = [check.check_type for check in defects]
        expected_challenged_facets = list(
            dict.fromkeys(
                facet_id for check in defects for facet_id in check.facet_ids
            )
        )
        controlling = defects if defects else uncertainties
        if expected_status == "upheld":
            expected_selected_evidence = list(
                dict.fromkeys(
                    sentence_id
                    for mapping in proposed.mappings
                    if mapping.facet_id in material_ids
                    for sentence_id in mapping.evidence_sentence_ids
                )
            )
            expected_effective = finding.proposed_outcome
        else:
            expected_selected_evidence = list(
                dict.fromkeys(
                    sentence_id
                    for check in controlling
                    for sentence_id in check.evidence_sentence_ids
                )
            )
            expected_effective = "not_assessed"
        if (
            finding.status != expected_status
            or finding.challenge_types != expected_challenge_types
            or finding.challenged_facet_ids != expected_challenged_facets
            or finding.evidence_sentence_ids != expected_selected_evidence
            or finding.effective_outcome != expected_effective
        ):
            raise ReportAuthorizationError(
                "Decisive checklist outcome was not application-derived"
            )
        if finding.status == "challenged":
            challenged_ids.append(finding.candidate_id)

    if critic.challenged_candidate_ids != challenged_ids:
        raise ReportAuthorizationError("Decisive checklist challenged IDs are inconsistent")
    expected_status = (
        "incomplete"
        if any(
            finding.status in {"uncertain", "not_assessed"}
            for finding in critic.findings
        )
        else "complete"
    )
    if critic.status in {"complete", "incomplete"} and critic.status != expected_status:
        raise ReportAuthorizationError("Decisive checklist aggregate status is inconsistent")
    expected_citation = aggregate_critic_adjusted_citation(
        ledger.findings,
        critic.findings,
    )
    if (
        critic.status in {"complete", "incomplete"}
        and critic.derived_citation_outcome != expected_citation
    ):
        raise ReportAuthorizationError("Critic-adjusted citation outcome is not application-derived")


def _validate_claim_marker_membership(claim):
    """Revalidate exact grouped-marker source membership before persistence."""
    if not claim.citation_markers:
        return
    assigned_reference_ids: set[str] = set()
    previous_end = -1
    for marker in claim.citation_markers:
        if (
            marker.local_start < previous_end
            or marker.local_end > len(claim.text)
            or claim.text[marker.local_start:marker.local_end] != marker.text
            or not marker.reference_ids
            or not set(marker.reference_ids).issubset(claim.reference_ids)
        ):
            raise ReportAuthorizationError(
                "Claim citation-marker membership is not exact"
            )
        assigned_reference_ids.update(marker.reference_ids)
        previous_end = marker.local_end
    if assigned_reference_ids != set(claim.reference_ids):
        raise ReportAuthorizationError(
            "Claim citation-marker membership does not cover every reference"
        )


def _validate_source_binding(artifact):
    """Require one exact reference/marker/author binding per source artifact."""
    binding = artifact.source_binding
    claim = artifact.claim
    if binding is None:
        raise ReportAuthorizationError(
            "Evidence artifact lacks a source-specific citation binding"
        )
    if binding.status != "exact":
        raise ReportAuthorizationError(
            "Evidence artifact source binding is unresolved"
        )
    if binding.reference_id not in claim.reference_ids:
        raise ReportAuthorizationError(
            "Evidence source binding is not a claim reference member"
        )
    if (
        binding.marker_local_end > len(claim.text)
        or claim.text[binding.marker_local_start:binding.marker_local_end]
        != binding.marker_text
    ):
        raise ReportAuthorizationError(
            "Evidence source binding does not match the exact citation marker"
        )
    matching_members = [
        marker
        for marker in claim.citation_markers
        if (
            binding.reference_id in marker.reference_ids
            and marker.text == binding.marker_text
            and marker.local_start == binding.marker_local_start
            and marker.local_end == binding.marker_local_end
        )
    ]
    if claim.citation_markers and len(matching_members) != 1:
        raise ReportAuthorizationError(
            "Evidence source binding does not identify one exact marker member"
        )
    if not binding.cited_author_label.strip():
        raise ReportAuthorizationError(
            "Evidence source binding lacks a cited-author identity"
        )


def _validate_claim_antecedents(claim):
    dependencies = claim.antecedent_dependencies
    if dependencies:
        expected_status = (
            "ambiguous"
            if any(item.resolution_status == "ambiguous" for item in dependencies)
            else "unresolved"
            if any(item.resolution_status == "unresolved" for item in dependencies)
            else "resolved"
        )
        if claim.context_dependency_status != expected_status:
            raise ReportAuthorizationError("Claim antecedent aggregate state is inconsistent")
    elif claim.context_dependency_status != "not_required":
        raise ReportAuthorizationError("Claim records antecedent state without dependencies")

    contexts = {context.context_index: context for context in claim.antecedent_context}
    for dependency in dependencies:
        if (
            dependency.mention_local_end > len(claim.text)
            or claim.text[dependency.mention_local_start:dependency.mention_local_end]
            != dependency.mention_text
            or dependency.mention_paper_start
            != claim.passage_start + dependency.mention_local_start
            or dependency.mention_paper_end
            != claim.passage_start + dependency.mention_local_end
        ):
            raise ReportAuthorizationError("Antecedent mention does not match the citation unit")
        if dependency.method.startswith("previous-sentence-antecedent-v1"):
            # The referent is the exact previous sentence (owner decision 2026-10-03).
            context = contexts.get(dependency.antecedent_context_index)
            if (context is None or context.distance_before != 1
                    or dependency.antecedent_text != context.text
                    or dependency.antecedent_paper_start != context.paper_start
                    or dependency.antecedent_paper_end != context.paper_end):
                raise ReportAuthorizationError("Previous-sentence antecedent does not match its context")
            continue
        if not dependency.method.startswith(
            ("local-document-antecedent-rescue-v1", "local-document-antecedent-rescue-v2",
             "local-document-antecedent-rescue-v3")
        ):
            continue
        candidate_ids = [candidate.candidate_id for candidate in dependency.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ReportAuthorizationError("Local antecedent candidates contain duplicate IDs")
        candidates = {
            candidate.candidate_id: candidate for candidate in dependency.candidates
        }
        if any(
            candidate.search_tier != dependency.search_tier
            for candidate in dependency.candidates
        ):
            raise ReportAuthorizationError("Local antecedent candidate tier is inconsistent")
        if any(
            candidate_id not in candidates
            for candidate_id in dependency.selected_candidate_ids
        ):
            raise ReportAuthorizationError("Local antecedent selected an unknown candidate")
        if dependency.resolution_status == "resolved":
            if len(dependency.selected_candidate_ids) != 1:
                raise ReportAuthorizationError("Resolved local antecedent lacks one selected candidate")
            selected = candidates[dependency.selected_candidate_ids[0]]
            if (
                dependency.antecedent_text != selected.text
                or dependency.antecedent_paper_start != selected.paper_start
                or dependency.antecedent_paper_end != selected.paper_end
            ):
                raise ReportAuthorizationError("Resolved local antecedent does not match its selected evidence")
            if dependency.search_tier == "immediate_context":
                context = contexts.get(dependency.antecedent_context_index)
                if (
                    context is None
                    or not (
                        context.paper_start <= selected.paper_start
                        and selected.paper_end <= context.paper_end
                    )
                    or context.text[
                        selected.paper_start - context.paper_start:
                        selected.paper_end - context.paper_start
                    ]
                    != selected.text
                ):
                    raise ReportAuthorizationError("Immediate antecedent evidence does not match its context")
        elif (
            dependency.selected_candidate_ids
            or dependency.antecedent_text is not None
            or dependency.antecedent_paper_start is not None
            or dependency.antecedent_paper_end is not None
        ):
            raise ReportAuthorizationError("Unresolved local antecedent carries selected evidence")


def _validate_claim_discourse(claim):
    contexts = {context.context_index: context for context in claim.antecedent_context}
    seen = set()
    for dependency in claim.discourse_dependencies:
        if dependency.context_index in seen:
            raise ReportAuthorizationError("Claim contains duplicate discourse dependencies")
        seen.add(dependency.context_index)
        context = contexts.get(dependency.context_index)
        if (
            context is None
            or dependency.context_text != context.text
            or dependency.context_paper_start != context.paper_start
            or dependency.context_paper_end != context.paper_end
        ):
            raise ReportAuthorizationError(
                "Claim discourse dependency does not match exact student context"
            )
        if (
            dependency.relation != "answers_preceding_question"
            or dependency.method != "local-exact-discourse-dependency-v1"
            or not dependency.context_text.rstrip().endswith("?")
        ):
            raise ReportAuthorizationError(
                "Claim discourse dependency is not a recognized exact relation"
            )


def _expected_locator_status(
    artifact,
    finding,
    sentences,
    passages,
    *,
    source_discourse_sentence_ids=frozenset(),
):
    if not artifact.claim.page_locator:
        return "not_provided"
    if finding.status != "assessed":
        return "unresolved"
    sentence_ids = {
        sentence_id
        for mapping in finding.mappings
        for sentence_id in mapping.evidence_sentence_ids
        if sentence_id not in source_discourse_sentence_ids
    }
    if not sentence_ids:
        return "no_evidence"
    matches = []
    for sentence_id in sentence_ids:
        sentence = sentences.get(sentence_id)
        passage = passages.get(sentence.passage_id) if sentence is not None else None
        if passage is not None:
            matches.append(
                passage_matches_page_locator(passage, artifact.claim.page_locator)
            )
    return "evidence_at_locator" if True in matches else "evidence_only_elsewhere"


def _validate_candidate_segment(claim, segment):
    if (
        segment.local_end > len(claim.text)
        or claim.text[segment.local_start:segment.local_end] != segment.text
        or segment.paper_start != claim.passage_start + segment.local_start
        or segment.paper_end != claim.passage_start + segment.local_end
    ):
        raise ReportAuthorizationError(
            "Verification-candidate segment does not match the persisted citation unit"
        )


def _validate_verification_run(
    session: Session,
    artifact: VerificationEvidenceArtifact,
    *,
    verification_run_id: str | uuid.UUID | None,
    scope_type: str,
    scope_id: str,
) -> VerificationRunRecord | None:
    artifact_run_id = artifact.source_identity.verification_run_id
    if verification_run_id is None:
        if artifact_run_id is not None:
            raise ReportAuthorizationError(
                "Transient evidence requires an explicit verification-run linkage"
            )
        return None
    try:
        parsed_id = (
            verification_run_id
            if isinstance(verification_run_id, uuid.UUID)
            else uuid.UUID(str(verification_run_id))
        )
    except (TypeError, ValueError) as exc:
        raise ReportAuthorizationError("Invalid verification-run identifier") from exc
    run = session.scalar(
        select(VerificationRunRecord)
        .where(VerificationRunRecord.id == parsed_id)
        .with_for_update()
    )
    if (
        run is None
        or run.scope_type != scope_type
        or run.scope_id != scope_id
        or run.status != "active"
        or artifact_run_id != str(run.id)
        or artifact.claim.paper_version_id != run.paper_version_id
        or artifact.source_identity.content_sha256 != run.content_sha256
        or artifact.source_identity.canonical_work_id != run.canonical_work_id
    ):
        raise ReportAuthorizationError(
            "Transient evidence does not match the active verification run"
        )
    return run


def _validated_scope(scope_type: str, scope_id: str) -> tuple[str, str]:
    normalized_type = scope_type.strip().casefold()
    normalized_id = scope_id.strip()
    if not normalized_type or not normalized_id:
        raise ReportAuthorizationError("Report authorization scope is required")
    return normalized_type, normalized_id


def _payload_digest(payload: dict) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _lock_report_stream(
    session: Session,
    *,
    scope_type: str,
    scope_id: str,
    verification_id: str,
) -> None:
    """Serialize report-version allocation on PostgreSQL, including version one."""
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    material = f"{scope_type}\0{scope_id}\0{verification_id}".encode("utf-8")
    lock_id = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)
    session.execute(select(func.pg_advisory_xact_lock(lock_id))).scalar_one()
