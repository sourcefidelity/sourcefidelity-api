"""Storage settings page (owner decision 2026-10-13: approved wording; who may change it)."""
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
from app.services.storage_policy import INSTITUTIONAL_SCOPE, policy_for

FORM = {"keep_public_copies": "open_access_only", "public_copy_days": "30", "keep_licensed_copies": "on",
        "licensed_copy_days": "", "licensed_copy_days_until_deleted": "true", "paper_copy_days": "14",
        "report_days": "365"}


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
    for text in ("<title>Storage</title>", "<h1>Storage</h1>", "<legend>Sources</legend>",
                 "<legend>Keep downloaded sources</legend>", "Keep downloaded sources for",
                 "<legend>Keep library-licensed sources</legend>", "Keep library-licensed sources for",
                 "<legend>Papers</legend>", "Keep the paper beside its report for", "<legend>Reports</legend>",
                 "Keep reports for", "Until deleted", "Open access only", ">Save</button>"):
        assert text in page.text
    # Defaults: sources and reports until deleted, the paper copy for 7 days.
    assert 'name="public_copy_days_until_deleted" value="true" checked' in page.text
    assert 'name="report_days_until_deleted" value="true" checked' in page.text
    assert 'name="paper_copy_days" min="1" step="1" value="7"' in page.text


def test_a_save_stores_the_owners_settings(database):
    client = _client(_personal_principal())
    saved = client.post("/settings/storage", data=FORM, follow_redirects=False)
    assert saved.status_code == 303
    with database() as db:
        policy = policy_for(db, "personal_owner", _personal_principal().scope_id)
    assert policy.sources.keep_public_copies == "open_access_only" and policy.sources.public_copy_days == 30
    assert policy.sources.keep_licensed_copies is True and policy.sources.licensed_copy_days == 0
    assert policy.papers.paper_copy_days == 14 and policy.reports.report_days == 365


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


def test_an_institutional_administrator_sets_the_whole_deployment(database):
    administrator = AuthenticatedPrincipal(provider="institutional", subject="it-admin", scope_type="course",
                                           scope_id="course-1",
                                           capabilities=frozenset({STORAGE_SETTINGS_CAPABILITY}))
    assert _client(administrator).post("/settings/storage", data=FORM, follow_redirects=False).status_code == 303
    with database() as db:
        rows = [(row.scope_type, row.scope_id) for row in db.query(StoragePolicyRecord)]
        assert rows == [INSTITUTIONAL_SCOPE]
        # Every course and assessment inherits it.
        assert policy_for(db, "assessment", "assessment-9").reports.report_days == 365


def test_an_empty_days_box_without_until_deleted_asks_for_one(database):
    response = _client(_personal_principal()).post(
        "/settings/storage", data={**FORM, "report_days": "", "public_copy_days_until_deleted": "true"})
    assert response.status_code == 400
    assert "Enter a number of days or choose Until deleted." in response.text
    with database() as db:
        assert db.query(StoragePolicyRecord).count() == 0
