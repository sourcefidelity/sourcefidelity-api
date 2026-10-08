"""Authentication and transport checks for protected report delivery."""

from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import fitz
from pydantic import SecretStr
import pytest

from app.config import settings
from app.database import get_db
from app.main import app
from app.security import create_report_session, verify_report_session
from app.security import authenticate_personal_bearer
from app.services.evidence_report import enable_authenticated_paper_actions
from app.services.storage.backend import get_storage_backend



@pytest.fixture(autouse=True)
def _no_run_details(monkeypatch):
    """These tests stand in for the database; judgment results and job metrics are not under test."""
    monkeypatch.setattr("app.routers.report._run_details", lambda *args, **kwargs: {})

def _view():
    return {
        "title": "Bound report",
        "citation_format": "APA",
        "paper_surface": {"message": "A page-faithful paper is available."},
        "overview": {
            "citations_analyzed": 0,
            "verified_full_text_sources": 0,
            "limited_or_unavailable_sources": 0,
            "reference_identity_attention": 0,
            "quotation_differences_attention": 0,
            "locator_differences_attention": 0,
        },
        "citations": [],
        "limits": ["Evidence remains inspectable."],
    }


def _anchor_view():
    anchor_id = "d" * 64
    view = _view()
    view["paper_surface"].update(
        {
            "citation_anchors": [
                {
                    "anchor_id": anchor_id,
                    "localization_level": "exact_rectangle",
                    "page_indexes": [0],
                    "rectangles": [
                        {
                            "page_index": 0,
                            "x0": 70.0,
                            "y0": 60.0,
                            "x1": 220.0,
                            "y1": 85.0,
                        }
                    ],
                }
            ]
        }
    )
    return view, anchor_id


def _pdf_bytes():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Bound citation text")
    content = document.tobytes(no_new_id=True)
    document.close()
    return content


def test_report_routes_fail_closed_then_deliver_html_and_pdf(monkeypatch):
    token = "a" * 48
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            _view(),
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        unauthenticated = client.get("/report/report-1", follow_redirects=False)
        assert unauthenticated.status_code == 303
        assert unauthenticated.headers["location"] == (
            "/report/login?next=/report/report-1"
        )
        assert (
            client.get(
                "/report/report-1", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )

        headers = {"Authorization": f"Bearer {token}"}
        report = client.get("/report/report-1", headers=headers)
        assert report.status_code == 200
        assert report.headers["cache-control"] == "no-store, private"
        assert "default-src 'none'" in report.headers["content-security-policy"]
        assert "connect-src 'self'" in report.headers["content-security-policy"]
        assert "unsafe-inline" not in report.headers["content-security-policy"]
        assert "Bound report" in report.text

        paper = client.get("/report/report-1/paper", headers=headers)
        assert paper.status_code == 200
        assert paper.headers["content-type"] == "application/pdf"
        assert paper.headers["content-security-policy"] == "sandbox"
        assert paper.content == _pdf_bytes()
    finally:
        app.dependency_overrides.clear()


def test_disabled_report_authentication_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "disabled")
    client = TestClient(app)
    response = client.get("/report/report-1")
    assert response.status_code == 503
    assert client.get("/report/login").status_code == 503


def test_personal_login_exchanges_bearer_for_bounded_http_only_session(monkeypatch):
    token = "b" * 48
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "REPORT_SESSION_COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "REPORT_SESSION_TTL_SECONDS", 900)
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            _view(),
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        login = client.get(
            "/report/login?next=https://attacker.invalid/report/stolen"
        )
        assert login.status_code == 200
        assert 'name="next" value="/"' in login.text

        exchange = client.post(
            "/report/session",
            data={"token": token, "next": "/report/report-1"},
            follow_redirects=False,
        )
        assert exchange.status_code == 303
        assert exchange.headers["location"] == "/report/report-1"
        cookie = exchange.headers["set-cookie"]
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
        assert "Max-Age=900" in cookie

        cross_site = client.post(
            "/report/session",
            data={"token": token, "next": "/report/report-1"},
            headers={
                "Origin": "https://attacker.invalid",
                "Sec-Fetch-Site": "cross-site",
            },
            follow_redirects=False,
        )
        assert cross_site.status_code == 403

        same_site_exact_origin = client.post(
            "/report/session",
            data={"token": "wrong" * 8, "next": "/report/report-1"},
            headers={
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-site",
            },
            follow_redirects=False,
        )
        assert same_site_exact_origin.status_code == 401

        same_site_rewritten_origin = client.post(
            "/report/session",
            data={"token": "wrong" * 8, "next": "/report/report-1"},
            headers={
                "Origin": "http://browser-proxy.invalid",
                "Sec-Fetch-Site": "same-site",
            },
            follow_redirects=False,
        )
        assert same_site_rewritten_origin.status_code == 403

        report = client.get("/report/report-1")
        assert report.status_code == 200
        assert '<strong>Paper View</strong>' not in report.text
        paper = client.get("/report/report-1/paper")
        assert paper.status_code == 200
        assert paper.content.startswith(b"%PDF")
    finally:
        app.dependency_overrides.clear()


def test_report_source_upload_is_member_bound_and_same_origin(monkeypatch):
    token = "u" * 48
    view = _view()
    view["citations"] = [
        {
            "members": [
                {
                    "reference_id": "ref-1",
                    "source": {
                        "title": "Bound source",
                        "author": "Example Author",
                        "year": "2024",
                        "doi": "10.1000/example",
                        "source_kind": "journal_article",
                    },
                }
            ]
        }
    ]
    captured = {}

    def fake_admit_uploaded_source(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "review_status": "accepted",
            "documents": [{"id": str(uuid.uuid4()), "admission_state": "accepted"}],
        }

    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            view,
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    monkeypatch.setattr(
        "app.routers.report.admit_uploaded_source", fake_admit_uploaded_source
    )
    monkeypatch.setattr(
        "app.routers.report.prepare_uploaded_source_refresh",
        lambda *args, **kwargs: {
            "status": "scheduled",
            "scheduled": True,
            "job_id": str(uuid.uuid4()),
            "report_id": "report-1",
            "attempt_id": "attempt-1",
        },
    )
    monkeypatch.setattr(
        "app.routers.report.schedule_uploaded_source_reanalysis",
        lambda _job_id, _attempt: "task-1",
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "http://testserver",
            "Sec-Fetch-Site": "same-origin",
        }
        response = client.post(
            "/report/report-1/source/ref-1/upload",
            headers=headers,
            files={"file": ("source.pdf", b"%PDF-1.7 test", "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json()["review_status"] == "accepted"
        assert response.json()["reanalysis_status"] == "scheduled"
        assert response.json()["publication_pending"] is False
        assert response.json()["task_id"] == "task-1"
        assert captured["title"] == "Bound source"
        assert captured["doi"] == "10.1000/example"
        assert captured["source_kind"] == "journal_article"

        rejected = client.post(
            "/report/report-1/source/ref-1/upload",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "https://attacker.invalid",
                "Sec-Fetch-Site": "cross-site",
            },
            files={"file": ("source.pdf", b"%PDF-1.7 test", "application/pdf")},
        )
        assert rejected.status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize('route', ['citation/claim-1/source/upload', 'source/upload'])
@pytest.mark.parametrize('duplicate', ['none', 'same', 'conflict', 'split'])
def test_citation_source_upload_identifies_exactly_one_missing_member(monkeypatch, route, duplicate):
    token = "v" * 48
    view = _view()
    view["citations"] = [
        {
            "claim_id": "claim-1",
            "members": [
                {
                    "reference_id": "ref-a",
                    "coverage_level": "unavailable",
                    "source": {"title": "First work", "author": "Alpha", "year": "2020", "doi": "", "source_kind": "journal_article"},
                },
                {
                    "reference_id": "ref-b",
                    "coverage_level": "abstract_only",
                    "source": {"title": "Second work", "author": "Beta", "year": "2021", "doi": "10.1000/second", "source_kind": "journal_article"},
                },
            ],
        }
    ]
    if duplicate == 'split':
        # One sentence split into clause citations keeps one claim id; the first
        # clause's source is already complete (2026-10-07, the owner's article).
        from copy import deepcopy
        first, second = deepcopy(view['citations'][0]), deepcopy(view['citations'][0])
        first['members'] = [{**first['members'][0], 'coverage_level': 'full_text'}]
        second['members'] = second['members'][1:]
        view['citations'] = [first, second]
    elif duplicate != 'none':
        from copy import deepcopy
        second = deepcopy(view['citations'][0])
        second['claim_id'] = 'claim-2'
        if duplicate == 'conflict':
            second['members'][1]['source']['year'] = '2022'
        view['citations'].append(second)
    captured = {}

    def fake_admit_uploaded_source(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "review_status": "accepted",
            "documents": [{"id": str(uuid.uuid4()), "admission_state": "accepted"}],
        }

    def fake_verify(_content, *, provided_doi, provided_title, provided_author, web_page=False):
        return provided_title == "Second work", []

    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), b"%PDF"),
    )
    monkeypatch.setattr("app.routers.report.verify_instructor_upload", fake_verify)
    monkeypatch.setattr("app.routers.report.admit_uploaded_source", fake_admit_uploaded_source)
    monkeypatch.setattr(
        "app.routers.report.prepare_uploaded_source_refresh",
        lambda *args, **kwargs: {
            "status": "scheduled",
            "scheduled": True,
            "job_id": str(uuid.uuid4()),
            "report_id": "report-1",
            "attempt_id": "attempt-2",
        },
    )
    monkeypatch.setattr(
        "app.routers.report.schedule_uploaded_source_reanalysis",
        lambda _job_id, _attempt: "task-2",
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        response = TestClient(app).post(
            f"/report/report-1/{route}",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-origin",
            },
            files={"file": ("source.pdf", b"%PDF-1.7 test", "application/pdf")},
        )
        if duplicate == 'conflict' and route == 'source/upload':
            assert response.status_code == 409
            assert not captured
            return
        assert response.status_code == 200
        assert response.json()["reference_id"] == "ref-b"
        assert captured["title"] == "Second work"
        assert captured["doi"] == "10.1000/second"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("match_count", [0, 2])
@pytest.mark.parametrize('route', ['citation/claim-1/source/upload', 'source/upload'])
def test_citation_source_upload_rejects_wrong_or_ambiguous_identity(
    monkeypatch, match_count, route
):
    token = "x" * 48
    view = _view()
    view["citations"] = [
        {
            "claim_id": "claim-1",
            "members": [
                {
                    "reference_id": "ref-a",
                    "coverage_level": "unavailable",
                    "source": {"title": "First work", "author": "Alpha"},
                },
                {
                    "reference_id": "ref-b",
                    "coverage_level": "abstract_only",
                    "source": {"title": "Second work", "author": "Beta"},
                },
            ],
        }
    ]
    admitted = []

    def fake_verify(_content, **kwargs):
        if match_count == 2:
            return True, []
        return False, []

    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), b"%PDF"),
    )
    monkeypatch.setattr("app.routers.report.verify_instructor_upload", fake_verify)
    monkeypatch.setattr(
        "app.routers.report.admit_uploaded_source",
        lambda **kwargs: admitted.append(kwargs),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        response = TestClient(app).post(
            f"/report/report-1/{route}",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-origin",
            },
            files={"file": ("source.pdf", b"%PDF-1.7 test", "application/pdf")},
        )
        assert response.status_code == 422
        assert admitted == []
    finally:
        app.dependency_overrides.clear()


def test_source_upload_queue_failure_keeps_reanalysis_for_recovery(monkeypatch):
    token = "w" * 48
    view = _view()
    view["citations"] = [
        {
            "members": [
                {
                    "reference_id": "ref-1",
                    "source": {
                        "title": "Bound source",
                        "author": "Example Author",
                        "year": "2024",
                        "doi": "10.1000/example",
                        "source_kind": "journal_article",
                    },
                }
            ]
        }
    ]
    job_id = str(uuid.uuid4())
    rolled_back = {}
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), b"%PDF"),
    )
    monkeypatch.setattr(
        "app.routers.report.admit_uploaded_source",
        lambda **kwargs: {
            "status": "ok",
            "review_status": "accepted",
            "documents": [{"id": str(uuid.uuid4()), "admission_state": "accepted"}],
        },
    )
    monkeypatch.setattr(
        "app.routers.report.prepare_uploaded_source_refresh",
        lambda *args, **kwargs: {
            "status": "scheduled",
            "scheduled": True,
            "job_id": job_id,
            "report_id": "report-1",
            "attempt_id": "attempt-pending",
        },
    )
    monkeypatch.setattr(
        "app.routers.report.schedule_uploaded_source_reanalysis",
        lambda _job_id, _attempt: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
    )
    monkeypatch.setattr(
        "app.services.paper_workflow.rollback_targeted_source_refresh",
        lambda _session, value, **kwargs: rolled_back.update(
            {"job_id": value, **kwargs}
        ),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        response = TestClient(app).post(
            "/report/report-1/source/ref-1/upload",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-origin",
            },
            files={"file": ("source.pdf", b"%PDF-1.7 test", "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json()["publication_pending"] is True
        assert response.json()["task_id"] is None
        assert rolled_back == {}
    finally:
        app.dependency_overrides.clear()


def test_annotation_write_routes_no_longer_exist(monkeypatch):
    """Comment, highlight and pen tools were removed (owner decision 2026-09-25)."""
    token = "n" * 48
    report_id = str(uuid.uuid4())
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {token}", "Origin": "http://testserver",
                   "Sec-Fetch-Site": "same-origin"}
        created = client.post(f"/report/{report_id}/annotations", headers=headers,
                              json={"annotation_type": "comment", "anchor_id": "a" * 64})
        revised = client.patch(f"/report/{report_id}/annotations/{uuid.uuid4()}", headers=headers,
                               json={"expected_revision": 1})
        assert created.status_code in {404, 405} and revised.status_code in {404, 405}
        from app import security
        assert not hasattr(security, "REPORT_ANNOTATION_CAPABILITY")
    finally:
        app.dependency_overrides.clear()


def test_get_report_session_redirects_to_login_instead_of_loading_report(monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr("d" * 48))
    response = TestClient(app).get("/report/session", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/report/login"


def test_signed_report_session_rejects_tampering_and_expiry(monkeypatch):
    token = "c" * 48
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "REPORT_SESSION_TTL_SECONDS", 120)
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    principal = authenticate_personal_bearer(token)
    session_token = create_report_session(principal, now=1_000)

    verified = verify_report_session(session_token, now=1_119)
    assert verified.scope_id == principal.scope_id
    assert verified.capabilities < principal.capabilities
    assert "paper:check" not in verified.capabilities
    with pytest.raises(Exception) as expired:
        verify_report_session(session_token, now=1_120)
    assert getattr(expired.value, "status_code", None) == 401
    with pytest.raises(Exception) as tampered:
        verify_report_session(session_token[:-1] + "x", now=1_010)
    assert getattr(tampered.value, "status_code", None) == 401


def test_authenticated_anchor_view_renders_only_bound_page_and_geometry(monkeypatch):
    token = "e" * 48
    view, anchor_id = _anchor_view()
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            view,
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {token}"}
        anchor = client.get(
            f"/report/report-1/paper/anchor/{anchor_id}", headers=headers
        )
        assert anchor.status_code == 200
        assert "The exact citation span is highlighted." in anchor.text
        assert '<rect x="70.000" y="60.000"' in anchor.text

        page = client.get(
            f"/report/report-1/paper/anchor/{anchor_id}/page/0", headers=headers
        )
        assert page.status_code == 200
        assert page.headers["content-type"] == "image/png"
        assert page.content.startswith(b"\x89PNG")

        continuous_page = client.get(
            "/report/report-1/paper/page/0", headers=headers
        )
        assert continuous_page.status_code == 200
        assert continuous_page.headers["content-type"] == "image/png"
        assert continuous_page.content.startswith(b"\x89PNG")

        missing = client.get(
            f"/report/report-1/paper/anchor/{'f' * 64}", headers=headers
        )
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_authenticated_source_route_requires_report_membership_and_exact_scope(monkeypatch):
    token = "f" * 48
    record_id = uuid.uuid4()
    view = _view()
    view["paper_version_id"] = "paper-v1"
    view["citations"] = [
        {"members": [{"verification_report_id": str(record_id)}]}
    ]
    descriptor = {
        "descriptor_sha256": "a" * 64,
        "status": "ready",
        "representation_id": "representation-1",
        "content_sha256": "b" * 64,
        "authorization_scope_type": "personal_owner",
        "authorization_scope_id": "owner-1",
        "representation_kind": "pdf",
        "media_type": "application/pdf",
        "pages_total": 1,
        "anchors": [
            {
                "passage_id": "passage-1",
                "target_kind": "pdf_page",
                "page_index": 0,
                "character_start": 0,
                "character_end": 10,
                "passage_text_sha256": "c" * 64,
            }
        ],
    }
    record = SimpleNamespace(
        id=record_id,
        scope_type="personal_owner",
        scope_id="owner-1",
        paper_version_id="paper-v1",
        report_payload={"source_navigation": descriptor},
    )
    session = SimpleNamespace(get=lambda _model, key: record if key == record_id else None)
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            view,
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    monkeypatch.setattr(
        "app.routers.report.authorize_source_navigation_document",
        lambda *args, **kwargs: SimpleNamespace(
            representation=SimpleNamespace(
                media_type="application/pdf", content=_pdf_bytes()
            )
        ),
    )
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {token}"}
        source = client.get(
            f"/report/report-1/source/{record_id}", headers=headers
        )
        assert source.status_code == 200
        assert source.headers["content-type"] == "application/pdf"
        assert source.headers["content-security-policy"] == "sandbox"
        assert source.content.startswith(b"%PDF")

        missing = client.get(
            f"/report/report-1/source/{uuid.uuid4()}", headers=headers
        )
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_only_page_warranted_citations_receive_authenticated_location_actions():
    view = _view()
    view["citations"] = [
        {
            "paper_location": {
                "anchor_id": "a" * 64,
                "localization_level": "exact_rectangle",
                "action": {"label": "View in paper", "enabled": False},
            }
        },
        {
            "paper_location": {
                "localization_level": "structural_only",
                "action": None,
            }
        },
    ]
    enabled = enable_authenticated_paper_actions(view, report_id="report-1")

    assert enabled["citations"][0]["paper_location"]["action"]["enabled"] is True
    assert enabled["citations"][0]["paper_location"]["action"]["href"].endswith(
        "/paper/anchor/" + "a" * 64
    )
    assert enabled["citations"][1]["paper_location"]["action"] is None
    assert view["citations"][0]["paper_location"]["action"]["enabled"] is False


def test_limited_and_unavailable_members_receive_one_citation_bound_upload_action():
    view = _view()
    view["citations"] = [
        {
            "claim_id": "claim-1",
            "paper_location": {"localization_level": "semantic_only", "action": None},
            "members": [
                {"reference_id": "ref full", "coverage_level": "full_text"},
                {"reference_id": "ref abstract", "coverage_level": "abstract_only"},
                {"reference_id": "ref unavailable", "coverage_level": "unavailable"},
            ],
        }
    ]

    enabled = enable_authenticated_paper_actions(view, report_id="report-1")
    members = enabled["citations"][0]["members"]

    assert all("upload_action" not in member for member in members)
    assert enabled["citations"][0]["upload_action"]["href"].endswith(
        "/citation/claim-1/source/upload"
    )


def test_personal_local_report_delivery_needs_no_password(monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_local")
    monkeypatch.setattr(settings, "API_BIND_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (
            _view(),
            SimpleNamespace(id="artifact-1"),
            _pdf_bytes(),
        ),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        assert client.get("/report/report-1").status_code == 200
        login = client.get("/report/login?next=/report/report-1", follow_redirects=False)
        assert login.status_code == 303
        assert login.headers["location"] == "/report/report-1"
    finally:
        app.dependency_overrides.clear()
