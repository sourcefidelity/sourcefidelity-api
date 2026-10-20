"""Storage settings page (owner decisions 2026-10-13, 2026-10-19 and 2026-10-20: approved wording; who may change it)."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base
from app.models.storage_policy import StoragePolicyRecord
from app.security import (
    PAPER_CHECK_CAPABILITY,
    STORAGE_SETTINGS_CAPABILITY,
    AuthenticatedPrincipal,
    _personal_principal,
    get_authenticated_principal,
)
from app.services.storage_policy import policy_for

FORM = {"keep_copies": "all", "keep_months": "6", "paper_copy_days": "14",
        "report_days": "365", "keep_submitted_file": "on", "submitted_file_days": "30"}


@pytest.fixture
def database():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    def session():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = session
    yield factory
    app.dependency_overrides.clear()


def _client(principal):
    app.dependency_overrides[get_authenticated_principal] = lambda: principal
    return TestClient(app)


def test_the_page_shows_only_the_approved_wording_and_current_values(database):
    page = _client(_personal_principal()).get("/settings/storage")
    assert page.status_code == 200
    for gone in ("library-licensed", "Keep downloaded sources"):
        assert gone not in page.text
    for text in ("<title>Storage</title>", "<h1>Storage</h1>", "<legend>Sources</legend>",
                 "<legend>Keep sources</legend>", "<legend>Keep each source after its last use for</legend>",
                 "3 months", "6 months", "9 months", "12 months", "Permanently",
                 "<legend>Finish incomplete searches automatically</legend>",
                 "<legend>Papers</legend>", "Keep the paper beside its report for", "<legend>Reports</legend>",
                 "Keep reports for", "Until deleted", "Open access only", ">Save</button>",
                 "<legend>Keep the submitted file</legend>", "Keep the submitted file for",
                 "<legend>Keep only the latest report</legend>"):
        assert text in page.text
    # Defaults: all sources permanently, reports until deleted, the paper copy for 7 days.
    assert 'name="keep_copies" value="all" checked' in page.text
    assert 'name="keep_months" value="0" checked' in page.text
    assert 'name="report_days_until_deleted" value="true" checked' in page.text
    assert 'name="paper_copy_days" min="1" step="1" value="7"' in page.text
    assert 'name="keep_latest_only" value="on" checked' in page.text


def test_a_save_stores_the_owners_settings(database):
    client = _client(_personal_principal())
    saved = client.post("/settings/storage", data=FORM, follow_redirects=False)
    assert saved.status_code == 303
    with database() as db:
        policy = policy_for(db, "personal_owner", _personal_principal().scope_id)
    assert policy.sources.keep_copies == "all" and policy.sources.keep_months == 6
    assert policy.papers.paper_copy_days == 14 and policy.reports.report_days == 365
    assert policy.reports.keep_latest_only        # absent from the form: the stored choice stays


def test_keep_only_the_latest_report_can_be_turned_off(database):
    client = _client(_personal_principal())
    assert client.post("/settings/storage", data={**FORM, "keep_latest_only": "off"},
                       follow_redirects=False).status_code == 303
    with database() as db:
        assert not policy_for(db, "personal_owner", _personal_principal().scope_id).reports.keep_latest_only
    assert 'name="keep_latest_only" value="off" checked' in client.get("/settings/storage").text


def test_a_paper_kept_longer_than_its_report_is_refused_with_the_approved_message(database):
    response = _client(_personal_principal()).post("/settings/storage", data={**FORM, "paper_copy_days": "400"})
    assert response.status_code == 400
    assert "The paper cannot be kept longer than its report." in response.text
    with database() as db:
        assert db.query(StoragePolicyRecord).count() == 0


def test_an_instructor_cannot_open_or_change_the_settings(database):
    instructor = AuthenticatedPrincipal(provider="lti", subject="instructor-1", scope_type="course",
                                        scope_id="course-1", capabilities=frozenset({PAPER_CHECK_CAPABILITY}))
    client = _client(instructor)
    assert client.get("/settings/storage").status_code == 403
    assert client.post("/settings/storage", data=FORM).status_code == 403


def test_a_cross_site_save_is_refused(database):
    response = _client(_personal_principal()).post("/settings/storage", data=FORM,
                                                   headers={"sec-fetch-site": "cross-site"})
    assert response.status_code == 403


def test_an_empty_days_box_without_until_deleted_asks_for_one(database):
    response = _client(_personal_principal()).post(
        "/settings/storage", data={**FORM, "report_days": ""})
    assert response.status_code == 400
    assert "Enter a number of days or choose Until deleted." in response.text
    with database() as db:
        assert db.query(StoragePolicyRecord).count() == 0


def test_a_submitted_file_kept_longer_than_its_report_is_refused(database):
    response = _client(_personal_principal()).post(
        "/settings/storage", data={**FORM, "submitted_file_days": "", "submitted_file_days_until_deleted": "true"})
    assert response.status_code == 400
    assert "The submitted file cannot be kept longer than its report." in response.text
    with database() as db:
        assert db.query(StoragePolicyRecord).count() == 0


def test_the_submitted_file_defaults_and_an_empty_box_when_not_kept(database):
    page = _client(_personal_principal()).get("/settings/storage")
    # Personal keeps the submitted file 30 days by default (owner decision 2026-10-13).
    assert 'name="keep_submitted_file" value="on" checked' in page.text
    assert 'name="submitted_file_days" min="1" step="1" value="30"' in page.text
    saved = _client(_personal_principal()).post(
        "/settings/storage", data={**FORM, "keep_submitted_file": "off", "submitted_file_days": ""},
        follow_redirects=False)
    assert saved.status_code == 303
    with database() as db:
        papers = policy_for(db, "personal_owner", _personal_principal().scope_id).papers
    assert papers.keep_submitted_file is False and papers.submitted_file_days == 30



def test_tightening_shows_the_count_and_saves_only_after_confirmation(database, monkeypatch):
    import app.routers.storage_settings as page
    applied = []
    monkeypatch.setattr(page, "copies_to_delete", lambda session, scope_type, scope_id, policy: 2)
    monkeypatch.setattr(page, "apply_source_setting",
                        lambda session, scope_type, scope_id, policy: applied.append(policy.sources.keep_copies))
    client = _client(_personal_principal())
    held = client.post("/settings/storage", data={**FORM, "keep_copies": "open_access_only"},
                       follow_redirects=False)
    assert held.status_code == 200
    assert "2 kept sources will be deleted." in held.text
    assert ">Delete and save</button>" in held.text and ">Cancel</a>" in held.text
    assert 'name="confirm" value="delete"' in held.text
    with database() as db:
        assert db.query(StoragePolicyRecord).count() == 0
    assert applied == []
    saved = client.post("/settings/storage", data={**FORM, "keep_copies": "open_access_only", "confirm": "delete"},
                        follow_redirects=False)
    assert saved.status_code == 303 and applied == ["open_access_only"]


def test_one_copy_reads_in_the_singular(database, monkeypatch):
    import app.routers.storage_settings as page
    monkeypatch.setattr(page, "copies_to_delete", lambda session, scope_type, scope_id, policy: 1)
    held = _client(_personal_principal()).post("/settings/storage", data={**FORM, "keep_copies": "none"},
                                               follow_redirects=False)
    assert "1 kept source will be deleted." in held.text


@pytest.mark.parametrize("keep,license_class,kept", [
    ("all", "commercial_user_upload", True),
    ("open_access_only", "open_access", True),
    ("open_access_only", "public_domain", True),
    ("open_access_only", "rights_unclassified", False),
    ("open_access_only", "commercial_user_upload", False),
    ("none", "open_access", False),
])
def test_which_kept_copies_the_setting_keeps(keep, license_class, kept):
    from app.services.source_retention import kept_by
    from app.services.storage_policy import StoragePolicy
    assert kept_by(StoragePolicy.model_validate({"sources": {"keep_copies": keep}}), license_class) is kept
