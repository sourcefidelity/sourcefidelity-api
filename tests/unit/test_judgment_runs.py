"""Judgment runs and endpoints (Phase B) on an in-memory database, fake judges only."""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import get_db
from app.main import app
from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.judgment import JudgmentArmResult, JudgmentCandidateResult, JudgmentRun
from app.models.report import Report, VerificationReportRecord
from app.services import judgment_runs as runs
from app.services.judge_arms import JudgePanelUnavailable
from app.services.llm_service import LLMCallFailure
from test_judgment_panel import ROUTES, _artifact, _fake_call, _payload

SCOPE = ("personal_owner", "owner-1")
PRINCIPAL = type("P", (), dict(provider="personal", subject="owner", scope_type=SCOPE[0], scope_id=SCOPE[1]))()
SUPPORTS = {"deepseek": "supports", "glm": "supports", "qwen": "supports"}


@pytest.fixture
def store(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        job = Job(filename="p.pdf", status=JobStatus.COMPLETED, stage=JobStage.COMPLETED,
                  paper_version_id="paper-v1", scope_type=SCOPE[0], scope_id=SCOPE[1],
                  input_sha256="0" * 64, input_media_type="application/pdf", input_byte_size=1,
                  input_expires_at=datetime.now(timezone.utc) + timedelta(days=1))
        session.add(job)
        session.flush()
        records = []
        for payload in (_payload(_artifact()),
                        _payload(_artifact(), coverage={"level": "abstract_only"})):
            record = VerificationReportRecord(
                verification_id=str(uuid.uuid4()), paper_version_id="paper-v1", scope_type=SCOPE[0],
                scope_id=SCOPE[1], report_version=1, artifact_version="v", verdict="supported",
                evidence_sha256=hashlib.sha256(str(len(records)).encode()).hexdigest(), report_payload=payload)
            session.add(record)
            session.flush()
            records.append(record)
        report = Report(job_id=job.id, report_json={}, report_version=1)
        session.add(report)
        session.flush()
        report.report_json = {"evidence_report": {
            "report_id": str(report.id), "paper_version_id": "paper-v1",
            "citations": [{"citation_number": 1, "members": [{"verification_report_id": str(records[0].id)}]},
                          {"citation_number": 2, "members": [{"verification_report_id": str(records[1].id)}]},
                          {"citation_number": 3, "members": [{"coverage_level": "unavailable"}]}]}}
        session.commit()
        yield session, report, records


def _run(session, report, **kwargs):
    run = runs.create_run(session, report.id, PRINCIPAL, force_new=kwargs.pop("force_new", False))
    session.commit()
    return runs.execute_run(session, run.id, routes_factory=lambda: (ROUTES, {"arms": "fake"}), **kwargs)


def _rows(session, run):
    return list(session.scalars(select(JudgmentCandidateResult).where(
        JudgmentCandidateResult.run_id == run.id).order_by(JudgmentCandidateResult.seq)))


def test_run_judges_eligible_candidates_in_order_and_explains_the_rest(store):
    session, report, records = store
    run = _run(session, report, call=_fake_call(SUPPORTS))
    assert run.status == "completed" and run.reason_code is None
    rows = _rows(session, run)
    assert [r.seq for r in rows] == list(range(1, len(rows) + 1))
    judged = [r for r in rows if r.citation_index == 1]
    assert judged and all(r.display_state == "supported" for r in judged)
    assert all(len(r.arm_result_ids) == 3 for r in judged)
    other = [r for r in rows if r.citation_index == 2]
    assert [(r.candidate_id, r.display_state, r.reason_code) for r in other] == [
        (runs.WHOLE_CITATION, "not_judged", "not_complete_full_text")]
    assert run.spend_usd == pytest.approx(0.003 * len(judged))
    assert session.scalar(select(JudgmentArmResult).limit(1)).scope_id == SCOPE[1]


def test_retry_reuses_cached_arms_and_spends_nothing(store):
    session, report, _ = store
    _run(session, report, call=_fake_call(SUPPORTS))
    boom = LLMCallFailure("rate_limited")
    again = _run(session, report, force_new=True,
                 call=_fake_call({"deepseek": boom, "glm": boom, "qwen": boom}))
    judged = [r for r in _rows(session, again) if r.citation_index == 1]
    assert judged and all(r.display_state == "supported" for r in judged)
    assert again.spend_usd == 0


def test_a_refused_judge_makes_no_call(store, monkeypatch):
    session, report, _ = store
    def forbidden(*a, **k):
        raise AssertionError("provider called")
    run = runs.create_run(session, report.id, PRINCIPAL, force_new=True)
    session.commit()
    def refuse():
        raise JudgePanelUnavailable("qwen", "retention not acknowledged")
    run = runs.execute_run(session, run.id, routes_factory=refuse, call=forbidden)
    assert (run.status, run.reason_code) == ("unavailable", "panel_unavailable:qwen")


def test_agreed_no_evidence_is_never_red_without_a_wider_search(store):
    session, report, _ = store
    run = _run(session, report, call=_fake_call({"deepseek": "none", "glm": "none", "qwen": "none"}))
    judged = [r for r in _rows(session, run) if r.citation_index == 1]
    assert judged and all((r.display_state, r.reason_code) == ("not_judged", "wider_search_unavailable")
                          for r in judged)
    assert all(r.wider_search == {"status": "unavailable"} for r in judged)


def test_agreed_no_evidence_is_red_only_when_the_wider_search_agrees(store):
    session, report, _ = store
    def wider(context, item, routes, run, cache, call):
        return {"display_state": "insufficient", "reason_code": "agreed_no_evidence_after_wider_search",
                "record": {"status": "completed", "new_sentences": 12}, "spend_usd": 0.001}
    run = _run(session, report, call=_fake_call({"deepseek": "none", "glm": "none", "qwen": "none"}),
               wider_search_factory=lambda *_: wider)
    judged = [r for r in _rows(session, run) if r.citation_index == 1]
    assert all(r.display_state == "insufficient" for r in judged)


def test_spend_limit_stops_before_the_next_call(store, monkeypatch):
    session, report, _ = store
    monkeypatch.setattr(settings, "JUDGMENT_MAX_USD_PER_REPORT", 0.0)
    def forbidden(*a, **k):
        raise AssertionError("provider called past the limit")
    run = _run(session, report, call=forbidden)
    assert (run.status, run.reason_code) == ("completed", "report_spend_limit")


def test_a_run_is_executed_once(store):
    session, report, _ = store
    run = _run(session, report, call=_fake_call(SUPPORTS))
    count = len(_rows(session, run))
    def forbidden(*a, **k):
        raise AssertionError("second execution")
    runs.execute_run(session, run.id, routes_factory=lambda: (ROUTES, {}), call=forbidden)
    assert len(_rows(session, run)) == count


# ---- endpoints ------------------------------------------------------------------------

@pytest.fixture
def client(store, monkeypatch):
    session, report, _ = store
    token = "j" * 48
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", SCOPE[1])
    dispatched = []
    monkeypatch.setattr("app.tasks.judgment.dispatch_report_judgment", dispatched.append)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield TestClient(app), {"Authorization": f"Bearer {token}"}, report, dispatched
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_there_is_no_notice_and_opening_starts_a_missing_run_once(client):
    """Owner decision 2026-09-28: no notice at all; results are simply polled."""
    http, headers, report, dispatched = client
    base = f"/report/{report.id}/judgment"
    status = http.get(f"{base}/status", headers=headers).json()
    assert status["run"] is None and "notice" not in status and "acknowledged" not in status
    assert http.get(f"{base}/results", headers=headers).status_code == 200
    started = http.post(f"{base}/start", headers=headers).json()
    # A report checked before judging at check time gets its run when first opened.
    assert started["run"]["status"] == "queued" and len(dispatched) == 1
    http.post(f"{base}/start", headers=headers)       # opening again starts nothing new
    assert len(dispatched) == 1
    assert http.post(f"{base}/acknowledge", headers=headers).status_code in {404, 405}
    assert http.get(f"{base}/results", headers=headers).json()["items"] == []


def test_endpoints_are_scoped_and_same_origin(client, monkeypatch):
    http, headers, report, _ = client
    assert http.get(f"/report/{uuid.uuid4()}/judgment/status", headers=headers).status_code == 404
    assert http.get(f"/report/{report.id}/judgment/status").status_code == 401
    cross = http.post(f"/report/{report.id}/judgment/start",
                      headers={**headers, "Origin": "https://attacker.example"})
    assert cross.status_code == 403


def test_results_stream_after_a_sequence_number(client, store):
    http, headers, report, _ = client
    session = store[0]
    http.post(f"/report/{report.id}/judgment/start", headers=headers)
    run = runs.latest_run(session, report.id, PRINCIPAL)
    runs.execute_run(session, run.id, routes_factory=lambda: (ROUTES, {}), call=_fake_call(SUPPORTS))
    body = http.get(f"/report/{report.id}/judgment/results?after=0", headers=headers).json()
    assert body["run"]["status"] == "completed" and body["items"]
    last = body["items"][-1]["seq"]
    assert http.get(f"/report/{report.id}/judgment/results?after={last}", headers=headers).json()["items"] == []
    assert body["items"][0]["labels"] == {"deepseek": "supports", "glm": "supports", "qwen": "supports"}


def test_amber_claims_get_a_checked_coaching_note_and_supported_ones_none(store):
    session, report, _ = store
    from app.services.judgment_fake_panel import fake_judge_arms
    from app.services.llm_service import chat_completion_json
    run = runs.create_run(session, report.id, PRINCIPAL)
    session.commit()
    run = runs.execute_run(session, run.id, routes_factory=fake_judge_arms, call=chat_completion_json)
    rows = [r for r in _rows(session, run) if r.citation_index == 1]
    coached = [r for r in rows if r.display_state in {"qualified", "contradicts"}]
    for row in rows:
        if row.display_state in {"qualified", "contradicts", "insufficient"}:
            assert row.coaching["status"] == "model" and "Stand-in coaching note" in row.coaching["note"]
        else:
            assert row.coaching is None
    assert run.spend_usd == 0


def test_a_coaching_failure_falls_back_without_failing_the_claim(store):
    session, report, _ = store
    run = _run(session, report, call=_fake_call({"deepseek": "supports", "glm": "qualifies", "qwen": "supports"}))
    judged = [r for r in _rows(session, run) if r.citation_index == 1]
    assert judged and all(r.display_state == "qualified" for r in judged)
    assert all(r.coaching["status"] == "template" for r in judged)


def test_a_checked_paper_is_judged_straight_away_and_scheduling_never_fails_the_check(store, monkeypatch):
    session, report, _ = store
    from app.models.job import Job
    job = session.get(Job, report.job_id)
    dispatched = []
    monkeypatch.setattr("app.tasks.judgment.dispatch_report_judgment", dispatched.append)
    run = runs.schedule_run_at_check(session, report, job)
    assert run is not None and run.requested_by == "paper_check" and dispatched == [run.id]
    monkeypatch.setattr("app.tasks.judgment.dispatch_report_judgment",
                        lambda _id: (_ for _ in ()).throw(RuntimeError("broker down")))
    assert runs.schedule_run_at_check(session, report, job) is None      # logged, not raised


def test_a_judge_that_keeps_failing_is_not_called_for_every_claim(store, monkeypatch):
    session, report, _ = store
    monkeypatch.setattr("app.services.judgment_panel._RETRY_PAUSE_SECONDS", 0)
    monkeypatch.setattr("app.services.judgment_panel.settings.JUDGMENT_SAMPLES", 1)
    calls = []
    def down(*a, **k):
        calls.append(1)
        raise LLMCallFailure("connection_failed")
    run = runs.create_run(session, report.id, PRINCIPAL)
    session.commit()
    run = runs.execute_run(session, run.id, routes_factory=lambda: (ROUTES[:1], {"arms": "one"}), call=down)
    rows = [r for r in _rows(session, run) if r.citation_index == 1]
    failed = [r for r in rows if r.reason_code == "judge_failed"]
    skipped = [r for r in rows if r.reason_code == "judge_unavailable"]
    assert len(failed) <= 3 and len(calls) == 2 * len(failed)     # one retry each, then stop
    assert len(failed) + len(skipped) == len([r for r in rows if r.candidate_id != runs.WHOLE_CITATION
                                              and r.reason_code not in {"antecedent_unresolved", "prompt_over_budget"}])


def test_an_uncertain_reading_gets_an_undecided_note_and_the_result_is_unchanged(store):
    """Owner decision 2026-09-30: DeepSeek explains an undecided judge whose readings name the part."""
    session, report, _ = store
    judge = _fake_call({"deepseek": "uncertain", "glm": "uncertain", "qwen": "uncertain"})
    systems = []

    def call(system_prompt, user_prompt, **kwargs):
        if '"coaching_request"' in user_prompt:
            systems.append(system_prompt)
            return {"note": "It is unclear whether the source addresses this.", "facet_ids": [], "sentence_ids": []}
        return judge(system_prompt, user_prompt, **kwargs)
    run = _run(session, report, call=call)
    judged = [r for r in _rows(session, run) if r.citation_index == 1]
    assert judged and all((r.display_state, r.reason_code) == ("not_judged", "judges_undecided") for r in judged)
    assert all(r.coaching["status"] == "model" and r.coaching["version"] == "judgment-undecided-note-v2"
               for r in judged)
    assert systems and all("could not decide" in s for s in systems)
    arm = judged[0].panel["arms"][0]
    assert arm["context_resolution"] == "not_required"
