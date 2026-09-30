"""Append-only authored annotations over an immutable report paper surface."""

from __future__ import annotations

import hashlib
import json
import uuid
import fitz

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.report import PaperAnnotationRecord, Report, ReportPaperArtifactRecord


class PaperAnnotationError(ValueError):
    """Raised when an annotation cannot be safely created or revised."""


_ANNOTATION_TYPES = {"comment", "highlight"}
_VISIBILITIES = {"private", "released"}


def personal_annotation_export_enabled() -> bool:
    from app.config import settings
    return settings.REPORT_AUTH_MODE in {'personal_local', 'personal_bearer'}
_STATES = {"active", "deleted"}


def text_selection_anchor(content: bytes, supplied: dict) -> tuple[dict, dict]:
    """Derive word/line geometry from the retained PDF, never from client boxes."""
    selections = supplied.get("selected_words") or []
    if not isinstance(selections, list) or not 1 <= len(selections) <= 20:
        raise PaperAnnotationError("Select text on at most 20 pages")
    rectangles, text_parts, dimensions = [], [], {}
    previous_page = -1
    with fitz.open(stream=content, filetype="pdf") as document:
        for selection in selections:
            try:
                page_index, start, end = (int(selection[k]) for k in ("page_index", "start", "end"))
            except (KeyError, TypeError, ValueError) as exc:
                raise PaperAnnotationError("Text selection is invalid") from exc
            if not previous_page < page_index < document.page_count:
                raise PaperAnnotationError("Selected paper page is invalid")
            previous_page = page_index
            page = document[page_index]
            words = page.get_text("words", sort=True)
            if not 0 <= start <= end < len(words) or end - start > 3000:
                raise PaperAnnotationError("Selected paper words are invalid")
            dimensions[page_index] = (page.rect.width, page.rect.height)
            lines = {}
            for word in words[start:end + 1]:
                key = (word[5], word[6])
                box = fitz.Rect(word[:4]) & page.rect
                lines[key] = (lines[key] | box) if key in lines else box
            for box in lines.values():
                rectangles.append({"page_index": page_index, "x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1})
            text_parts.append(" ".join(word[4] for word in words[start:end + 1]))
    selected_text = "\n".join(text_parts)
    return {
        "anchor_version": "text-selection-anchor-v1", "anchor_kind": "text_selection",
        "localization_level": "exact_rectangle", "rectangles": rectangles,
        "selected_text_sha256": hashlib.sha256(selected_text.encode()).hexdigest(),
    }, dimensions


def list_current_paper_annotations(
    session: Session,
    *,
    report_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
    visibility: str | None = None,
) -> list[dict]:
    """Return only the latest active revision of each exact-scope annotation."""
    parsed_report_id = _uuid(report_id, "report_id")
    if visibility is not None and visibility not in _VISIBILITIES:
        raise PaperAnnotationError("Annotation visibility is invalid")
    ranked = (
        select(
            PaperAnnotationRecord.id.label("revision_id"),
            func.row_number()
            .over(
                partition_by=PaperAnnotationRecord.annotation_id,
                order_by=PaperAnnotationRecord.revision.desc(),
            )
            .label("revision_rank"),
        )
        .where(
            PaperAnnotationRecord.report_id == parsed_report_id,
            PaperAnnotationRecord.scope_type == scope_type,
            PaperAnnotationRecord.scope_id == scope_id,
        )
        .subquery()
    )
    query = (
        select(PaperAnnotationRecord)
        .join(ranked, PaperAnnotationRecord.id == ranked.c.revision_id)
        .where(
            ranked.c.revision_rank == 1,
            PaperAnnotationRecord.state == "active",
        )
        .order_by(PaperAnnotationRecord.created_at, PaperAnnotationRecord.annotation_id)
    )
    if visibility is not None:
        query = query.where(PaperAnnotationRecord.visibility == visibility)
    records = list(
        session.scalars(query)
    )
    return [_view(record) for record in records]


def create_paper_annotation(
    session: Session,
    *,
    artifact: ReportPaperArtifactRecord,
    report_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
    author_provider: str,
    author_subject: str,
    annotation_type: str,
    anchor: dict,
    content: str | None = None,
    user_label: str | None = None,
    visibility: str = "private",
    page_dimensions: dict[int, tuple[float, float]] | None = None,
) -> dict:
    """Create revision one after binding the annotation to the exact paper bytes."""
    parsed_report_id = _uuid(report_id, "report_id")
    _validate_artifact(session, artifact, parsed_report_id, scope_type, scope_id)
    normalized_type, normalized_content, normalized_label, normalized_visibility = (
        _validate_values(annotation_type, content, user_label, visibility, "active")
    )
    paper_content_sha256 = artifact.presentation_sha256 or artifact.content_sha256
    anchor_payload = _validated_anchor(
        anchor,
        paper_content_sha256=paper_content_sha256,
        page_dimensions=page_dimensions,
    )
    record = PaperAnnotationRecord(
        annotation_id=uuid.uuid4(),
        report_id=parsed_report_id,
        paper_artifact_id=artifact.id,
        paper_version_id=artifact.paper_version_id,
        paper_content_sha256=paper_content_sha256,
        scope_type=scope_type,
        scope_id=scope_id,
        revision=1,
        previous_revision_id=None,
        annotation_type=normalized_type,
        anchor_kind=anchor_payload["anchor_kind"],
        anchor_sha256=_digest(anchor_payload),
        anchor_payload=anchor_payload,
        content=normalized_content,
        user_label=normalized_label,
        author_provider=_bounded(author_provider, 60, "author_provider"),
        author_subject=_bounded(author_subject, 255, "author_subject"),
        visibility=normalized_visibility,
        state="active",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _view(record)


def revise_paper_annotation(
    session: Session,
    *,
    report_id: str | uuid.UUID,
    annotation_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
    author_provider: str,
    author_subject: str,
    expected_revision: int,
    content: str | None = None,
    user_label: str | None = None,
    visibility: str | None = None,
    state: str = "active",
) -> dict:
    """Append a revision; never rewrite an earlier authored record."""
    parsed_report_id = _uuid(report_id, "report_id")
    parsed_annotation_id = _uuid(annotation_id, "annotation_id")
    current = session.scalar(
        select(PaperAnnotationRecord)
        .where(
            PaperAnnotationRecord.report_id == parsed_report_id,
            PaperAnnotationRecord.annotation_id == parsed_annotation_id,
            PaperAnnotationRecord.scope_type == scope_type,
            PaperAnnotationRecord.scope_id == scope_id,
        )
        .order_by(PaperAnnotationRecord.revision.desc())
        .limit(1)
        .with_for_update()
    )
    if current is None:
        raise PaperAnnotationError("Annotation not found")
    if current.revision != expected_revision:
        raise PaperAnnotationError("Annotation changed; reload before editing")
    next_visibility = current.visibility if visibility is None else visibility
    next_content = current.content if content is None else content
    next_label = current.user_label if user_label is None else user_label
    normalized_type, normalized_content, normalized_label, normalized_visibility = (
        _validate_values(
            current.annotation_type,
            next_content,
            next_label,
            next_visibility,
            state,
        )
    )
    revision = PaperAnnotationRecord(
        annotation_id=current.annotation_id,
        report_id=current.report_id,
        paper_artifact_id=current.paper_artifact_id,
        paper_version_id=current.paper_version_id,
        paper_content_sha256=current.paper_content_sha256,
        scope_type=current.scope_type,
        scope_id=current.scope_id,
        revision=current.revision + 1,
        previous_revision_id=current.id,
        annotation_type=normalized_type,
        anchor_kind=current.anchor_kind,
        anchor_sha256=current.anchor_sha256,
        anchor_payload=dict(current.anchor_payload),
        content=normalized_content,
        user_label=normalized_label,
        author_provider=_bounded(author_provider, 60, "author_provider"),
        author_subject=_bounded(author_subject, 255, "author_subject"),
        visibility=normalized_visibility,
        state=state,
    )
    session.add(revision)
    session.commit()
    session.refresh(revision)
    return _view(revision)


def _validate_artifact(
    session: Session,
    artifact: ReportPaperArtifactRecord,
    report_id: uuid.UUID,
    scope_type: str,
    scope_id: str,
) -> None:
    report = session.get(Report, report_id)
    if (
        report is None
        or report.job_id != artifact.job_id
        or artifact.scope_type != scope_type
        or artifact.scope_id != scope_id
        or artifact.deleted_at is not None
        or not (artifact.presentation_sha256 or artifact.content_sha256)
    ):
        raise PaperAnnotationError("Paper annotation surface is unavailable")


def _validated_anchor(
    anchor: dict,
    *,
    paper_content_sha256: str,
    page_dimensions: dict[int, tuple[float, float]] | None,
) -> dict:
    anchor_version = str(anchor.get("anchor_version") or "citation-anchor-v1")
    anchor_kind = str(anchor.get("anchor_kind") or "")
    if not anchor_kind:
        anchor_kind = (
            "page_region"
            if anchor_version == "page-region-anchor-v1"
            else "citation_anchor"
        )
    if anchor_kind not in {"citation_anchor", "page_region", "text_selection"}:
        raise PaperAnnotationError("Annotation anchor type is invalid")
    if anchor_kind == "citation_anchor" and anchor_version != "citation-anchor-v1":
        raise PaperAnnotationError("Annotation anchor version is invalid")
    if anchor_kind == "page_region" and anchor_version != "page-region-anchor-v1":
        raise PaperAnnotationError("Annotation anchor version is invalid")
    if anchor_kind == "text_selection" and anchor_version != "text-selection-anchor-v1":
        raise PaperAnnotationError("Annotation anchor version is invalid")
    if anchor.get("localization_level") != "exact_rectangle":
        raise PaperAnnotationError("Annotation requires exact paper geometry")
    rectangles = []
    for item in anchor.get("rectangles") or []:
        try:
            page_index = int(item["page_index"])
            x0, y0, x1, y1 = (
                round(float(item[key]), 3) for key in ("x0", "y0", "x1", "y1")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperAnnotationError("Annotation geometry is invalid") from exc
        if page_index < 0 or not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise PaperAnnotationError("Annotation geometry is invalid")
        if anchor_kind in {"page_region", "text_selection"}:
            if page_dimensions is None or page_index not in page_dimensions:
                raise PaperAnnotationError("Annotation page is unavailable")
            width, height = page_dimensions[page_index]
            if x1 > width or y1 > height:
                raise PaperAnnotationError("Annotation geometry is outside the paper")
        rectangles.append(
            {"page_index": page_index, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        )
    if not rectangles:
        raise PaperAnnotationError("Annotation requires exact paper geometry")
    if anchor_kind == "page_region" and (
        len(rectangles) != 1
        or (rectangles[0]["x1"] - rectangles[0]["x0"]) < 2
        or (rectangles[0]["y1"] - rectangles[0]["y0"]) < 2
    ):
        raise PaperAnnotationError("A page-region annotation requires one visible rectangle")
    page_indexes = sorted({item["page_index"] for item in rectangles})
    if anchor_kind in {"page_region", "text_selection"}:
        identity_payload = {
            "anchor_version": anchor_version,
            "anchor_kind": anchor_kind,
            "paper_content_sha256": paper_content_sha256,
            "localization_level": "exact_rectangle",
            "page_indexes": page_indexes,
            "rectangles": rectangles,
        }
        anchor_id = _digest(identity_payload)
        supplied_anchor_id = str(anchor.get("anchor_id") or "")
        if supplied_anchor_id and supplied_anchor_id != anchor_id:
            raise PaperAnnotationError("Annotation anchor does not match the paper region")
    else:
        anchor_id = str(anchor.get("anchor_id") or "")
        if len(anchor_id) != 64 or any(
            value not in "0123456789abcdef" for value in anchor_id
        ):
            raise PaperAnnotationError("Annotation anchor is invalid")
    return {
        "anchor_version": anchor_version,
        "anchor_kind": anchor_kind,
        "anchor_id": anchor_id,
        "localization_level": "exact_rectangle",
        "page_indexes": page_indexes,
        "rectangles": rectangles,
    }


def _validate_values(
    annotation_type: str,
    content: str | None,
    user_label: str | None,
    visibility: str,
    state: str,
) -> tuple[str, str | None, str | None, str]:
    normalized_type = annotation_type.strip().casefold()
    if normalized_type not in _ANNOTATION_TYPES:
        raise PaperAnnotationError("Annotation type is not supported")
    normalized_visibility = visibility.strip().casefold()
    if normalized_visibility not in _VISIBILITIES:
        raise PaperAnnotationError("Annotation visibility is invalid")
    if state not in _STATES:
        raise PaperAnnotationError("Annotation state is invalid")
    normalized_content = (content or "").strip() or None
    if normalized_type == "comment" and state == "active" and not normalized_content:
        raise PaperAnnotationError("A comment requires text")
    if normalized_content and len(normalized_content) > 4000:
        raise PaperAnnotationError("Annotation text is too long")
    normalized_label = (user_label or "").strip() or None
    if normalized_label and len(normalized_label) > 100:
        raise PaperAnnotationError("Annotation label is too long")
    return normalized_type, normalized_content, normalized_label, normalized_visibility


def _bounded(value: str, maximum: int, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > maximum:
        raise PaperAnnotationError(f"{field} is invalid")
    return normalized


def _uuid(value: str | uuid.UUID, field: str) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except ValueError as exc:
        raise PaperAnnotationError(f"{field} is invalid") from exc


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _view(record: PaperAnnotationRecord) -> dict:
    return {
        "annotation_id": str(record.annotation_id),
        "revision": record.revision,
        "paper_artifact_id": str(record.paper_artifact_id),
        "paper_version_id": record.paper_version_id,
        "paper_content_sha256": record.paper_content_sha256,
        "annotation_type": record.annotation_type,
        "anchor_kind": record.anchor_kind,
        "anchor_sha256": record.anchor_sha256,
        "anchor": dict(record.anchor_payload),
        "content": record.content,
        "user_label": record.user_label,
        "author": {
            "provider": record.author_provider,
            "subject": record.author_subject,
        },
        "visibility": record.visibility,
        "state": record.state,
        "created_at": record.created_at.isoformat(),
    }
