"""Assessment-level "marks released" setting (owner decision 2026-09-29).

Once marks are released for an assessment, further searching can no longer
affect it: the automatic retries of incomplete searches and "Search Again"
stop for every paper of that assessment, and a Judgment reserve kept
``until_grades_released`` is purged.

A paper belongs to an assessment only when it was submitted with an
``assessment_id`` (``Job.assessment_id``) in the same authorization scope as
the record. A paper without one is never affected.

Two callers set the same record:

* an instructor or administrator, through ``POST /assessments/{id}/marks-released``
  (``source="manual"``);
* the planned LMS grade-release signal, which calls
  ``set_marks_released(..., source="lms")`` with the LMS scope and assessment
  identifier the paper was submitted under.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from app.models.assessment_marks import AssessmentMarksRelease
from app.models.job import Job

ReleaseSource = Literal["manual", "lms"]
_SOURCES = frozenset({"manual", "lms"})
UNTIL_GRADES_RELEASED = "until_grades_released"
_MAX_ID = 255


class AssessmentMarksError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def normalize_assessment_id(value) -> str | None:
    """A stripped assessment identifier, None when absent; invalid input raises."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssessmentMarksError("assessment_id_invalid", "Assessment identifier is invalid")
    text = value.strip()
    if not text:
        return None
    if len(text) > _MAX_ID or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AssessmentMarksError("assessment_id_invalid", "Assessment identifier is invalid")
    return text


def _require_id(value) -> str:
    text = normalize_assessment_id(value)
    if text is None:
        raise AssessmentMarksError("assessment_id_invalid", "Assessment identifier is invalid")
    return text


def _require_source(source: str) -> str:
    if source not in _SOURCES:
        raise AssessmentMarksError("release_source_invalid", "Release source is invalid")
    return source


def _record(session: Session, scope_type: str, scope_id: str, assessment_id: str,
            *, lock: bool = False) -> AssessmentMarksRelease | None:
    query = select(AssessmentMarksRelease).where(
        AssessmentMarksRelease.scope_type == scope_type,
        AssessmentMarksRelease.scope_id == scope_id,
        AssessmentMarksRelease.assessment_id == assessment_id,
    )
    return session.scalar(query.with_for_update() if lock else query)


def marks_release_record(session: Session, *, scope_type: str, scope_id: str,
                         assessment_id: str) -> AssessmentMarksRelease | None:
    return _record(session, scope_type, scope_id, _require_id(assessment_id))


def set_marks_released(
    session: Session,
    *,
    scope_type: str,
    scope_id: str,
    assessment_id: str,
    released_by: str,
    source: ReleaseSource = "manual",
    now: datetime | None = None,
    commit: bool = True,
) -> tuple[AssessmentMarksRelease, int]:
    """Record that marks are released; idempotent. Returns (record, reserves purged).

    An already released assessment keeps its original release time and actor.
    """
    assessment_id = _require_id(assessment_id)
    source = _require_source(source)
    current = now or datetime.now(timezone.utc)
    record = _record(session, scope_type, scope_id, assessment_id, lock=True)
    if record is None:
        record = AssessmentMarksRelease(
            scope_type=scope_type, scope_id=scope_id, assessment_id=assessment_id,
            created_at=current, updated_at=current)
        session.add(record)
    if record.marks_released_at is None:
        record.marks_released_at = current
        record.released_by = str(released_by)[:_MAX_ID]
        record.release_source = source
        record.updated_at = current
    session.flush()
    purged = purge_released_judgment_reserves(
        session, scope_type=scope_type, scope_id=scope_id, assessment_id=assessment_id)
    if commit:
        session.commit()
    else:
        session.flush()
    return record, purged


def clear_marks_released(
    session: Session,
    *,
    scope_type: str,
    scope_id: str,
    assessment_id: str,
    cleared_by: str,
    source: ReleaseSource = "manual",
    now: datetime | None = None,
    commit: bool = True,
) -> AssessmentMarksRelease | None:
    """Clear a release; the row is kept with who cleared it. A purged reserve stays purged."""
    assessment_id = _require_id(assessment_id)
    source = _require_source(source)
    current = now or datetime.now(timezone.utc)
    record = _record(session, scope_type, scope_id, assessment_id, lock=True)
    if record is not None and record.marks_released_at is not None:
        record.marks_released_at = None
        record.cleared_at = current
        record.cleared_by = str(cleared_by)[:_MAX_ID]
        record.clear_source = source
        record.updated_at = current
    if commit:
        session.commit()
    else:
        session.flush()
    return record


def marks_released_clause():
    """SQL: the ``Job`` row's assessment has marks released in the job's scope."""
    return exists().where(
        AssessmentMarksRelease.scope_type == Job.scope_type,
        AssessmentMarksRelease.scope_id == Job.scope_id,
        AssessmentMarksRelease.assessment_id == Job.assessment_id,
        AssessmentMarksRelease.marks_released_at.is_not(None),
    )


def institutional_deployment() -> bool:
    """Marks release exists only in an Institutional deployment (owner decision
    2026-09-29): there it is Moodle's release of marks. A Personal user works on
    the report or batch at hand; Search again stays available to them."""
    from app.config import settings
    return settings.REPORT_AUTH_MODE == "institutional_adapter"


def job_marks_released(session: Session, job: Job) -> bool:
    """Whether marks are released for ``job``'s assessment; False without one,
    and always False in a Personal deployment."""
    if not institutional_deployment():
        return False
    assessment_id = getattr(job, "assessment_id", None)
    if not assessment_id:
        return False
    record = _record(session, job.scope_type, job.scope_id, assessment_id)
    return record is not None and record.marks_released_at is not None


def paper_version_marks_released(session: Session, *, paper_version_id: str,
                                 scope_type: str, scope_id: str) -> bool:
    job = session.scalar(select(Job).where(
        Job.paper_version_id == paper_version_id,
        Job.scope_type == scope_type, Job.scope_id == scope_id).limit(1))
    return job is not None and job_marks_released(session, job)


def purge_released_judgment_reserves(session: Session, *, scope_type: str, scope_id: str,
                                     assessment_id: str) -> int:
    """Delete ``until_grades_released`` reserves of the assessment's papers (if released)."""
    from app.models.judgment import JudgmentSourceReserve
    from app.models.report import VerificationReportRecord

    record = _record(session, scope_type, scope_id, assessment_id)
    if record is None or record.marks_released_at is None:
        return 0
    report_ids = select(VerificationReportRecord.id).join(
        Job,
        (Job.paper_version_id == VerificationReportRecord.paper_version_id)
        & (Job.scope_type == VerificationReportRecord.scope_type)
        & (Job.scope_id == VerificationReportRecord.scope_id),
    ).where(Job.scope_type == scope_type, Job.scope_id == scope_id,
            Job.assessment_id == assessment_id)
    result = session.execute(
        delete(JudgmentSourceReserve)
        .where(
            JudgmentSourceReserve.retention_policy == UNTIL_GRADES_RELEASED,
            JudgmentSourceReserve.scope_type == scope_type,
            JudgmentSourceReserve.scope_id == scope_id,
            JudgmentSourceReserve.verification_report_id.in_(report_ids),
        )
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount or 0)
