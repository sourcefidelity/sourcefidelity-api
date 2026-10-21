"""What the user's deployment keeps and for how long (owner decisions 2026-10-09 and 2026-10-19)."""
from datetime import datetime, timedelta, timezone

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
    StoragePolicy, copy_expiry, deployment_defaults, policy_for, set_policy, upload_expiry)


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def test_the_defaults_keep_all_copies_permanently():
    policy = deployment_defaults()
    assert policy.version == "storage-policy-v2"
    assert policy.sources.keep_copies == "all" and policy.sources.keep_months == 0
    assert policy.reports.report_days == 0 and policy.papers.keep_submitted_file is True
    assert policy.papers.paper_copy_days == max(1, settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS)
    assert policy.searches.finish_incomplete_automatically is False


def test_the_row_keeps_only_its_differences(session):
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}}, updated_by="owner")
    set_policy(session, "personal_owner", "owner-1", {"papers": {"paper_copy_days": 14}})
    mine = policy_for(session, "personal_owner", "owner-1")
    assert mine.reports.report_days == 30 and mine.papers.paper_copy_days == 14
    assert mine.sources == deployment_defaults().sources
    assert session.query(StoragePolicyRecord).one().settings == {
        "reports": {"report_days": 30}, "papers": {"paper_copy_days": 14}}


@pytest.mark.parametrize("changes", [
    {"papers": {"paper_copy_days": 0}},
    {"sources": {"keep_copies": "some"}},
    {"sources": {"keep_months": 4}},
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


@pytest.mark.parametrize("v1,keep,months", [
    ({"keep_public_copies": "open_access_only"}, "open_access_only", 0),
    ({"keep_public_copies": "all", "public_copy_days": 90}, "all", 3),
    ({"public_copy_days": 200}, "all", 9),
    ({"public_copy_days": 500}, "all", 0),
    ({"keep_licensed_copies": True, "licensed_copy_days": 30}, "all", 0),
])
def test_a_v1_row_is_read_in_the_new_shape(session, v1, keep, months):
    session.add(StoragePolicyRecord(scope_type="personal_owner", scope_id="owner-1",
                                    settings={"sources": v1, "reports": {"report_days": 0}},
                                    created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
    session.flush()
    sources = policy_for(session, "personal_owner", "owner-1").sources
    assert (sources.keep_copies, sources.keep_months) == (keep, months)


def _public(oa=False):
    return RetrievalResult(
        source_name="web_search", success=True, metadata={},
        representation=SourceRepresentation(kind=RepresentationKind.PDF, media_type="application/pdf",
                                             content=b"%PDF", source_url="https://ports.example/ames.pdf"),
        locations=[AcquisitionLocation(url="https://ports.example/ames.pdf", provider="web_search",
                                       access_type="open_access" if oa else None)])


@pytest.mark.parametrize("keep,oa,publisher,expected", [
    ("all", False, False, "rights_unclassified"),
    ("all", True, False, "open_access"),
    ("all", False, True, "paywalled_db_retrieved"),
    ("open_access_only", True, False, "open_access"),
    ("open_access_only", False, False, None),
    ("open_access_only", False, True, None),
    ("none", True, False, None),
])
def test_what_is_kept_follows_the_one_setting(keep, oa, publisher, expected):
    policy = StoragePolicy.model_validate({"sources": {"keep_copies": keep}})
    decided, _ = SourceResolver._retrieval_retention_decision(
        _public(oa=oa), downloaded_via_publisher=publisher, policy=policy)
    assert decided == expected


@pytest.mark.parametrize("months,days", [(0, None), (3, 91), (12, 365)])
def test_a_kept_copy_lasts_the_chosen_months(months, days):
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    policy = StoragePolicy.model_validate({"sources": {"keep_months": months}})
    expiry = copy_expiry(policy, now)
    assert (expiry is None) if days is None else (expiry - now).days == days


def test_an_upload_follows_the_setting():
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    permanent = StoragePolicy.model_validate({})
    assert upload_expiry(permanent, now) is None
    months = StoragePolicy.model_validate({"sources": {"keep_months": 6}})
    assert upload_expiry(months, now) == copy_expiry(months, now)
    oa_only = StoragePolicy.model_validate({"sources": {"keep_copies": "open_access_only"},
                                            "papers": {"paper_copy_days": 7}})
    assert upload_expiry(oa_only, now) == now + timedelta(days=7)


def test_using_a_kept_copy_renews_it_and_never_shortens_it():
    from types import SimpleNamespace
    from unittest.mock import Mock
    resolver = SourceResolver.__new__(SourceResolver)
    resolver.storage_policy = StoragePolicy.model_validate({"sources": {"keep_months": 3}})
    session = Mock()
    soon = datetime.now(timezone.utc) + timedelta(days=5)
    record = SimpleNamespace(expires_at=soon)
    resolver._record_copy_use(session, record)
    assert record.expires_at > soon + timedelta(days=80)
    later = datetime.now(timezone.utc) + timedelta(days=400)
    record = SimpleNamespace(expires_at=later)
    resolver._record_copy_use(session, record)
    assert record.expires_at == later
    record = SimpleNamespace(expires_at=None)
    resolver._record_copy_use(session, record)
    assert record.expires_at is None


def test_a_persisted_copy_gets_the_months_expiry(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = Mock()
    resolver._repository_session_factory = MagicMock()
    resolver.storage_policy = StoragePolicy.model_validate({"sources": {"keep_months": 6}})
    admitted = []
    monkeypatch.setattr("app.services.source_resolver.admit_representation",
                        lambda session, backend, request: admitted.append(request) or SimpleNamespace(
                            id="r1", admission_state="accepted"))
    monkeypatch.setattr("app.services.source_resolver.commit_source_admissions", lambda _: None)
    result = _public()
    result.representation = SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain",
                                                 content=b"Harbour cranes", source_url="https://ports.example/ames")
    resolver._persist_retrieved_representation(
        result, ref_doi=None, ref_title="Harbour cranes", ref_author=None, ref_year=None, identity_confidence="high",
        identity_reason="fixture", downloaded_via_publisher=False, safety_report=None)
    assert admitted[0].expires_at is not None
    assert admitted[0].validation_evidence["storage_policy"]["sources"]["keep_months"] == 6
