"""Identical deterministic model requests in a paper stage reuse their answer (owner decision 2026-10-09)."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database
from app.models import Base
from app.models.llm_cache import LLMCallCacheRecord
from app.services import llm_cache, llm_service
from app.services.llm_cache import llm_cache_scope


@pytest.fixture
def calls(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(app.database, "SessionLocal", factory)
    class Sent(list):
        pass
    sent = Sent()

    def fake(**kwargs):
        sent.append(kwargs["user_prompt"])
        return '{"answer": "%s", "n": %d}' % (kwargs["user_prompt"], len(sent))
    monkeypatch.setattr(llm_service, "chat_completion", fake)
    monkeypatch.setattr(llm_service, "get_provider_config", lambda model=None: type("C", (), {"json_mode": True})())
    sent.factory = factory
    return sent


def _ask(prompt="cranes", **kwargs):
    return llm_service.chat_completion_json("system", prompt, **kwargs)


def test_an_identical_request_in_the_same_scope_is_answered_once(calls):
    with llm_cache_scope("personal_owner", "owner-1"):
        first = _ask()
        again = _ask()
        other = _ask("ports")
    assert first == again == {"answer": "cranes", "n": 1}
    assert other["n"] == 2 and calls == ["cranes", "ports"]
    with calls.factory() as session:
        row = session.query(LLMCallCacheRecord).filter_by(cache_key=llm_cache.request_key(
            model=None, system_prompt="system", user_prompt="cranes", temperature=0.0, max_tokens=4000,
            response_format={"type": "json_object"}, disable_thinking=False, reasoning_effort=None)).one()
        assert row.caller == __name__ and "cranes" not in row.caller


def test_other_scopes_outside_a_stage_and_non_deterministic_requests_ask_again(calls):
    with llm_cache_scope("personal_owner", "owner-1"):
        _ask()
    with llm_cache_scope("personal_owner", "owner-2"):
        _ask()
    _ask()
    with llm_cache_scope("personal_owner", "owner-1"):
        _ask(temperature=0.7)
        _ask(max_tokens=10)
        _ask(allow_partial=True)
    assert len(calls) == 6


def test_an_answer_older_than_the_reuse_window_is_not_used_and_is_expired(calls):
    with llm_cache_scope("personal_owner", "owner-1"):
        _ask()
    with calls.factory() as session:
        session.query(LLMCallCacheRecord).update({"created_at": datetime.now(timezone.utc) - timedelta(days=31)})
        session.commit()
    with llm_cache_scope("personal_owner", "owner-1"):
        _ask()
    assert len(calls) == 2
    with calls.factory() as session:
        session.query(LLMCallCacheRecord).update({"created_at": datetime.now(timezone.utc) - timedelta(days=31)})
        session.commit()
        assert llm_cache.expire(session) == 1


def test_a_routed_call_reuses_only_when_its_caller_asks_and_costs_nothing_on_reuse(calls, monkeypatch):
    from types import SimpleNamespace
    route = SimpleNamespace(model="glm", endpoint_host="glm.example", provider_config=SimpleNamespace(json_mode=True))
    with llm_cache_scope("personal_owner", "owner-1"):
        _ask(route=route)
        _ask(route=route)
        assert len(calls) == 2          # Judgment samples stay separate calls
        _ask(route=route, reuse_answer=True)
        receipt = {}
        _ask(route=route, reuse_answer=True, receipt=receipt)
    assert len(calls) == 3 and receipt["cache_hit"] is True and receipt["reported_cost_usd"] == 0.0
