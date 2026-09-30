"""Isolated Personal workflow checks; synthetic decisions are not attestations."""
import uuid
from datetime import datetime, timedelta, timezone

import fitz
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import get_db
from app.main import app
from app.models import Base
from app.models.edition_review import EditionReviewDecision, EditionReviewSnapshot
from app.models.source_repository import CanonicalWorkRecord, ContentObjectRecord, SourceRepresentationRecord
from app.security import AuthenticatedPrincipal, EDITION_REVIEW_CAPABILITY, REPORT_SOURCE_CAPABILITY, get_report_browser_principal, get_report_principal
from app.services import edition_review_entry as entry
from app.services.alternate_edition import AlternateEditionRecord
from app.services.file_safety import FileSafetyReport, SafetyVerdict, FileSafetyUnavailable
from app.services.storage.backend import get_storage_backend
from app.services.verification_evidence import authorize_representation, EvidenceAuthorizationError
from tests.unit.test_source_repository import MemoryStorage


@pytest.fixture
def review_env(monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_local")
    clean = FileSafetyReport(SafetyVerdict.CLEAN, SafetyVerdict.CLEAN, SafetyVerdict.CLEAN)
    monkeypatch.setattr(entry, "inspect_uploaded_pdf", lambda content: clean)
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    event.listen(engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    session = Session(engine)
    storage = MemoryStorage()
    with fitz.open() as doc:
        for line in ("Synthetic Work - Synthetic Author", "First edition 2004; reissue 2012. Synthetic test only."):
            doc.new_page().insert_text((72, 72), line)
        content = doc.tobytes(no_new_id=True)
    work = CanonicalWorkRecord(display_title="Synthetic Work", normalized_title="synthetic work", work_type="book")
    obj = ContentObjectRecord(content_sha256=entry.sha(content), storage_key="test-source", media_type="application/pdf",
                              byte_size=len(content), license_class="user_upload", deletion_pending=False)
    source = SourceRepresentationRecord(canonical_work=work, content_object=obj, representation_kind="pdf",
        provenance="user_upload", admission_state="needs_review", scope_type="personal_owner", scope_id="one",
        identity_verdict="uncertain", completeness_verdict="uncertain", cleanliness_verdict="clean", validation_evidence={})
    session.add(source); session.commit(); storage.upload(content, "test-source")
    principal = AuthenticatedPrincipal(provider="test", subject="owner", scope_type="personal_owner", scope_id="one",
        capabilities={EDITION_REVIEW_CAPABILITY, REPORT_SOURCE_CAPABILITY})
    yield session, storage, principal, source
    session.close(); engine.dispose()


def prepare(env):
    session, storage, principal, source = env
    snapshot = entry.prepare_review(session, storage, principal, source.id, "Synthetic Author (2004). Synthetic Work.")
    snapshot.payload = dict(snapshot.payload, version="personal-edition-review-v2")
    snapshot.snapshot_sha256 = entry.payload_hash(snapshot.payload)
    session.commit()
    return snapshot


def submission(snapshot, decision="confirmed", **updates):
    value = dict(snapshot_sha256=snapshot.snapshot_sha256, decision=decision,
                 acknowledged=True, notes="Synthetic observation, not a real human decision.")
    if decision in {"confirmed", "same_edition_later_printing"}:
        value.update(work_page=1, edition_page=2)
    return entry.ReviewDecisionInput(**(value | updates))


@pytest.mark.parametrize("decision", ["confirmed", "same_edition_later_printing", "uncertain", "rejected"])
def test_roundtrip_append_only_and_no_admission(review_env, decision):
    session, storage, principal, source = review_env
    snapshot = prepare(review_env)
    with pytest.raises(EvidenceAuthorizationError):
        authorize_representation(session, storage, representation_id=source.id, scope_type=principal.scope_type, scope_id=principal.scope_id)
    result = entry.save_review(session, storage, principal, snapshot.id, submission(snapshot, decision))
    session.commit(); session.expire_all()
    saved = session.get(EditionReviewDecision, result.id)
    record = AlternateEditionRecord.model_validate(saved.payload["alternate_edition"])
    assert record.human_verified == (decision == "confirmed")
    assert not record.quotation_usable and not record.paraphrase_usable
    if decision == "same_edition_later_printing":
        assert record.contract_version == "alternate-edition-v2"
        assert record.relationship == "same_edition_later_printing"
        assert not record.exact_edition_match
        from app.services.evidence_report import _render_alternate_edition
        assert "printing date alone" in _render_alternate_edition(record.model_dump(mode="json"))
    assert saved.reviewer_provider == "test" and record.human_review.reviewer_id == "owner"
    assert source.admission_state == "needs_review" and source.identity_verdict == "uncertain"
    with pytest.raises(entry.EditionReviewError, match="already"):
        entry.save_review(session, storage, principal, snapshot.id, submission(snapshot, decision))
    assert session.scalar(select(func.count()).select_from(EditionReviewDecision)) == 1
    newer = prepare(review_env)
    assert newer.id != snapshot.id


def test_later_printing_requires_new_snapshot_and_evidence(review_env):
    from app.services.edition_review_html import review_html
    session, storage, principal, _ = review_env
    snapshot = prepare(review_env)
    assert "Same cited edition, later printing" in review_html(snapshot)
    with pytest.raises(ValueError):
        submission(snapshot, "same_edition_later_printing", edition_page=None)
    snapshot.payload = dict(snapshot.payload, version="personal-edition-review-v1")
    snapshot.snapshot_sha256 = entry.payload_hash(snapshot.payload)
    session.commit()
    assert "Same cited edition, later printing" not in review_html(snapshot)
    with pytest.raises(entry.EditionReviewError, match="new review snapshot"):
        entry.save_review(session, storage, principal, snapshot.id,
                          submission(snapshot, "same_edition_later_printing"))
    # An older pending review retains its original options and can still save.
    entry.save_review(session, storage, principal, snapshot.id, submission(snapshot, "uncertain"))


def test_later_printing_cannot_use_old_contract_or_grant_tasks():
    binding = dict(evidence_id="work", representation_sha256="a" * 64,
                   passage_sha256="b" * 64, purpose="work_identity")
    value = dict(submitted_reference_sha256="c" * 64, retrieved_representation_sha256="a" * 64,
                 relationship="same_edition_later_printing", work_identity="verified",
                 evidence=[binding, dict(binding, evidence_id="edition", purpose="edition_relationship")])
    with pytest.raises(ValueError, match="requires_v2"):
        AlternateEditionRecord(**value)
    value['contract_version'] = 'alternate-edition-v2'
    record = AlternateEditionRecord(**value)
    assert not record.human_verified
    for flag in ('quotation_usable', 'paraphrase_usable'):
        with pytest.raises(ValueError):
            AlternateEditionRecord(**value, **{flag: True})


@pytest.mark.parametrize("change", ["scope", "expired", "deleted", "rejected", "unsafe", "bytes", "missing", "derivative", "audit"])
def test_source_revocation_blocks_all_review_resolution(review_env, change):
    session, storage, principal, source = review_env
    snapshot = prepare(review_env)
    if change == "scope": source.scope_id = "other"
    if change == "expired": source.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    if change == "deleted": source.content_object.deletion_pending = True
    if change == "rejected": source.admission_state = "rejected"
    if change == "unsafe": source.cleanliness_verdict = "unknown"
    if change == "bytes": storage.objects["test-source"] = b"changed"
    if change == "missing": storage.objects.clear()
    if change == "derivative": source.validation_evidence = {"ocr_derivative": {"parent": "not-resolved"}}
    if change == "audit": source.validation_evidence = {"authorization_scope": {"type": "personal_owner", "id": "other"}}
    session.commit()
    with pytest.raises(entry.EditionReviewError):
        entry.resolve_review(session, storage, principal, snapshot.id)


def test_institutional_and_missing_capability_fail_before_bytes(review_env, monkeypatch):
    session, storage, principal, source = review_env
    for mode, p in [("institutional_adapter", principal), ("personal_local", principal.model_copy(update={"capabilities": frozenset()}))]:
        monkeypatch.setattr(settings, "REPORT_AUTH_MODE", mode)
        with pytest.raises(HTTPException) as error:
            entry.load_review_source(session, storage, p, source.id)
        assert error.value.status_code == 403


def test_safety_unavailable_fails_before_render(review_env, monkeypatch):
    def unavailable(_): raise FileSafetyUnavailable("private scanner detail")
    monkeypatch.setattr(entry, "inspect_uploaded_pdf", unavailable)
    monkeypatch.setattr(entry, "render_review_pages", lambda _: pytest.fail("rendered before safety"))
    with pytest.raises(FileSafetyUnavailable): prepare(review_env)


def test_changed_snapshot_and_missing_confirmation_rejected(review_env):
    session, storage, principal, _ = review_env
    snapshot = prepare(review_env)
    with pytest.raises(ValueError): submission(snapshot, acknowledged=False)
    with pytest.raises(ValueError): submission(snapshot, work_page=None)
    with pytest.raises(ValueError): submission(snapshot, "uncertain", work_page=1)
    with pytest.raises(entry.EditionReviewError):
        entry.save_review(session, storage, principal, snapshot.id, submission(snapshot, edition_page=3))
    with pytest.raises(entry.EditionReviewError):
        entry.save_review(session, storage, principal, snapshot.id, submission(snapshot, snapshot_sha256="0" * 64))
    snapshot.payload = dict(snapshot.payload, reference_text="changed")
    session.commit()
    with pytest.raises(entry.EditionReviewError): entry.resolve_review(session, storage, principal, snapshot.id)


def test_source_deletion_cascades_review_text(review_env):
    session, storage, principal, source = review_env
    snapshot = prepare(review_env)
    entry.save_review(session, storage, principal, snapshot.id, submission(snapshot))
    session.commit(); session.delete(source); session.commit()
    assert session.scalar(select(func.count()).select_from(EditionReviewSnapshot)) == 0
    assert session.scalar(select(func.count()).select_from(EditionReviewDecision)) == 0


def test_routes_save_reload_export_csrf_and_escape(review_env):
    session, storage, principal, source = review_env
    overrides = dict(app.dependency_overrides)
    app.dependency_overrides.update({get_db: lambda: session, get_storage_backend: lambda: storage,
        get_report_principal: lambda: principal, get_report_browser_principal: lambda: principal})
    try:
        with TestClient(app) as client:
            new = client.get("/edition-reviews/new")
            assert new.status_code == 200 and new.headers["cache-control"] == "no-store"
            created = client.post("/edition-reviews", data={"representation_id": str(source.id), "reference_text": "<script>untrusted</script>"}, follow_redirects=False)
            assert created.status_code == 303
            url = created.headers["location"]
            view = client.get(url)
            assert view.status_code == 200 and "&lt;script&gt;untrusted" in view.text
            assert "<script>untrusted" not in view.text
            assert all(label in view.text for label in ("Option", "Select when", "Yes", "No"))
            pending = client.get(url + "/export").json()
            assert pending["status"] == "pending" and pending["decision"] is None
            assert client.get(url + "/pages/1").headers["content-type"] == "image/png"
            assert client.get(url + "/pages/2").headers["content-type"] == "image/png"
            assert client.get(url + "/pages/3").status_code == 409
            data = dict(snapshot_sha256=pending["snapshot_sha256"], answer="no")
            assert client.post(url + "/answer", data=data, headers={"Origin": "https://other.example"}).status_code == 403
            assert client.post(url + "/answer", data=data, follow_redirects=False).status_code == 303
            assert "Saved answer: No" in client.get(url).text
            exported = client.get(url + "/export").json()
            assert exported["status"] == "complete"
            assert exported["decision"]["outcome"] == "unverified"
            assert client.post(url + "/answer", data=data).status_code == 409
            source.content_object.deletion_pending = True; session.commit()
            assert client.get(url + "/export").status_code == 409
    finally:
        app.dependency_overrides.clear(); app.dependency_overrides.update(overrides)


def test_single_page_render_matches_frozen_manifest(review_env):
    session, storage, principal, _ = review_env
    snapshot = prepare(review_env)
    for number in (1, 2):
        _, _, pages = entry.resolve_review(session, storage, principal, snapshot.id, page_number=number)
        assert len(pages.images) == 1
        assert entry.sha(pages.images[0]) == snapshot.payload["page_manifest"]["pages"][number-1]["render_sha256"]


def test_non_book_and_unassessed_safety_rejected(review_env, monkeypatch):
    session, storage, principal, source = review_env
    source.canonical_work.work_type = "article"
    with pytest.raises(entry.EditionReviewError): prepare(review_env)
    source.canonical_work.work_type = "book"
    monkeypatch.setattr(entry, "inspect_uploaded_pdf", lambda _: FileSafetyReport(
        SafetyVerdict.NOT_ASSESSED, SafetyVerdict.CLEAN, SafetyVerdict.NOT_ASSESSED))
    with pytest.raises(entry.EditionReviewError): prepare(review_env)


def test_oversize_page_budget_checked_before_rasterization(monkeypatch):
    class Document:
        page_count = 1
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, index): return self
        rect = type("Rect", (), {"width": 10000, "height": 10000})()
        def get_pixmap(self, **kwargs): pytest.fail("allocated oversized page")
    monkeypatch.setattr(entry.fitz, "open", lambda **kwargs: Document())
    with pytest.raises(entry.EditionReviewError, match="budget"):
        entry.render_review_pages(b"synthetic")


def test_migration_roundtrip_isolated_database():
    from importlib import import_module
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect
    migration = import_module("app.alembic.versions.a1b7c3d8e425_personal_edition_reviews")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[CanonicalWorkRecord.__table__, ContentObjectRecord.__table__, SourceRepresentationRecord.__table__])
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            inspector = inspect(connection)
            for model in (EditionReviewSnapshot, EditionReviewDecision):
                assert {c["name"] for c in inspector.get_columns(model.__tablename__)} == set(model.__table__.columns.keys())
                assert all(f["options"]["ondelete"] == "CASCADE" for f in inspector.get_foreign_keys(model.__tablename__))
            assert any(c["column_names"] == ["snapshot_id"] for c in inspector.get_unique_constraints("edition_review_decisions"))
            migration.downgrade()
            assert "edition_review_snapshots" not in inspect(connection).get_table_names()
    engine.dispose()
