"""What a scope keeps and for how long (owner decision 2026-10-09)."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Base
from app.models.storage_policy import StoragePolicyRecord
from app.services.retrieval.base import (
    AcquisitionLocation, RepresentationKind, RetrievalResult, SourceRepresentation)
from app.services.source_resolver import SourceResolver
from app.services.storage_policy import (
    StoragePolicy, deployment_defaults, policy_for, profile_for, set_policy)


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def test_the_defaults_reproduce_the_deployment_settings():
    policy = deployment_defaults()
    assert policy.sources.keep_public_copies == (
        "all" if settings.PUBLIC_RETRIEVAL_RETENTION_POLICY == "store_scoped" else "open_access_only")
    assert policy.sources.public_copy_days == 0 and policy.reports.report_days == 0
    assert policy.sources.keep_licensed_copies is bool(settings.CACHE_PAYWALLED_PDFS)
    assert policy.papers.paper_copy_days == max(1, settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS)
    assert profile_for("personal_owner") == "personal" and profile_for("course") == "institutional"


def test_a_scope_keeps_only_its_differences_and_others_keep_the_defaults(session):
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}}, updated_by="owner")
    set_policy(session, "personal_owner", "owner-1", {"papers": {"paper_copy_days": 14}})
    mine = policy_for(session, "personal_owner", "owner-1")
    assert mine.reports.report_days == 30 and mine.papers.paper_copy_days == 14
    assert mine.sources == deployment_defaults().sources
    assert session.query(StoragePolicyRecord).one().settings == {
        "reports": {"report_days": 30}, "papers": {"paper_copy_days": 14}}
    assert policy_for(session, "personal_owner", "owner-2") == deployment_defaults()


@pytest.mark.parametrize("changes", [
    {"papers": {"paper_copy_days": 0}},
    {"sources": {"keep_public_copies": "some"}},
    {"reports": {"report_days": 3}, "papers": {"paper_copy_days": 7}},
    {"unknown": True},
])
def test_an_invalid_choice_is_refused(session, changes):
    with pytest.raises(ValidationError):
        set_policy(session, "personal_owner", "owner-1", changes)


def test_a_damaged_row_fails_closed_to_the_defaults(session):
    session.add(StoragePolicyRecord(scope_type="personal_owner", scope_id="owner-1",
                                    settings={"reports": {"report_days": -4}},
                                    created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
    session.flush()
    assert policy_for(session, "personal_owner", "owner-1") == deployment_defaults()


def _public(oa=False, publisher=False):
    result = RetrievalResult(
        source_name="web_search", success=True, metadata={},
        representation=SourceRepresentation(kind=RepresentationKind.PDF, media_type="application/pdf",
                                             content=b"%PDF", source_url="https://ports.example/ames.pdf"),
        locations=[AcquisitionLocation(url="https://ports.example/ames.pdf", provider="web_search",
                                       access_type="open_access" if oa else None)])
    return result, publisher


@pytest.mark.parametrize("keep,oa,expected", [
    ("all", False, "rights_unclassified"),
    ("all", True, "open_access"),
    ("open_access_only", True, "open_access"),
    ("open_access_only", False, None),
    ("none", True, None),
])
def test_what_public_copies_are_kept_follows_the_setting(keep, oa, expected):
    policy = StoragePolicy.model_validate({"sources": {"keep_public_copies": keep}})
    result, publisher = _public(oa=oa)
    decided, _ = SourceResolver._retrieval_retention_decision(result, downloaded_via_publisher=publisher, policy=policy)
    assert decided == expected


@pytest.mark.parametrize("location,keep,grants,expected", [
    ("off_campus", True, "{}", "paywalled_db_retrieved"),
    ("off_campus", False, "{}", None),
    ("off_campus", True, '{"publisher_pdf": {"retention": false}}', None),
    ("on_campus", True, "{}", None),
    ("on_campus", True, '{"publisher_pdf": {"retention": true}}', "paywalled_db_retrieved"),
    ("on_campus", True, '{"doi_resolver": {"retention": true}}', None),
])
def test_licensed_copies_are_kept_only_when_chosen(monkeypatch, location, keep, grants, expected):
    """Kept only when the storage settings keep licensed copies and the
    acquiring service's retention switch allows it; on a campus network the
    publisher route's switch starts off."""
    monkeypatch.setattr(settings, "LIBRARY_PERMISSIONS", grants)
    monkeypatch.setattr(settings, "NETWORK_LOCATION", location)
    policy = StoragePolicy.model_validate({"sources": {"keep_licensed_copies": keep}})
    result, _ = _public()
    decided, _ = SourceResolver._retrieval_retention_decision(result, downloaded_via_publisher=True, policy=policy)
    assert decided == expected


@pytest.mark.parametrize("days,licensed,expect_expiry", [(0, False, False), (30, False, True), (0, True, True)])
def test_a_kept_copy_expires_after_the_chosen_days(monkeypatch, days, licensed, expect_expiry):
    from types import SimpleNamespace
    monkeypatch.setattr(settings, "LIBRARY_PERMISSIONS", '{"publisher_pdf": {"retention": true}}')
    from unittest.mock import MagicMock, Mock
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = Mock()
    resolver._repository_session_factory = MagicMock()
    resolver.storage_policy = StoragePolicy.model_validate(
        {"sources": {"public_copy_days": days, "keep_licensed_copies": True, "licensed_copy_days": 60}})
    admitted = []
    monkeypatch.setattr("app.services.source_resolver.admit_representation",
                        lambda session, backend, request: admitted.append(request) or SimpleNamespace(
                            id="r1", admission_state="accepted"))
    monkeypatch.setattr("app.services.source_resolver.commit_source_admissions", lambda _: None)
    result, _ = _public()
    result.representation = SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain",
                                                 content=b"Harbour cranes", source_url="https://ports.example/ames")
    resolver._persist_retrieved_representation(
        result, ref_doi=None, ref_title="Harbour cranes", ref_author=None, ref_year=None, identity_confidence="high",
        identity_reason="fixture", downloaded_via_publisher=licensed, safety_report=None)
    assert (admitted[0].expires_at is not None) is expect_expiry
    assert admitted[0].validation_evidence["storage_policy"]["sources"]["public_copy_days"] == days
