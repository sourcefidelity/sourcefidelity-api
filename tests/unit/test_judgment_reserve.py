"""Wider-search sentence reserve (Phase B): new sentences only, never a paper-run failure."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.models import Base
from app.models.judgment import JudgmentSourceReserve
from app.models.report import VerificationReportRecord
from app.services import judgment_reserve as reserve_module
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.facet_evidence_judgment import (
    CandidateFacetFinding,
    attach_facet_evidence_foundation,
    prepare_candidate_prompts,
)
from app.services.judgment_panel import judge_candidate
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    attach_candidate_passage_retrieval,
    build_passage_evidence,
)
from app.services.verification_report import _store_judgment_reserve
from test_judgment_panel import ROUTES, _fake_call

CLAIM = "Teachers reported that feedback improved student motivation (Rivera, 2019)."


def _long_source_artifact(paragraphs=40):
    text = "\n\n".join(
        f"Section {i}. Teachers in cohort {i} reported that written feedback changed how students "
        f"approached revision, and motivation measures in cohort {i} rose by {i % 7} points. "
        f"Other observations in cohort {i} concerned attendance and scheduling."
        for i in range(paragraphs))
    content = text.encode()
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="reserve-rep-1", canonical_work_id="reserve-work-1", content_object_id="reserve-obj-1",
        content_sha256=hashlib.sha256(content).hexdigest(), content=content, representation_kind="plain_text",
        media_type="text/plain", provenance="authorized_upload", scope_type="personal_owner",
        scope_id="owner-1", identity_verdict="verified", identity_confidence=0.99,
        completeness_verdict="complete", text_quality="digital", edition_or_version=None,
        created_at=now, admitted_at=now)
    claim = ClaimEvidence(
        claim_id="reserve-unit-1", paper_version_id="paper-v1", text=CLAIM, granularity="citation_unit",
        reference_ids=["reference-1"], citation_marker="(Rivera, 2019)",
        citation_marker_type="parenthetical", extraction_confidence="high",
        passage_start=0, passage_end=len(CLAIM))
    artifact = attach_verification_candidates(build_passage_evidence(source, claim=claim, top_k=3))
    artifact = attach_candidate_passage_retrieval(source, artifact)
    return source, attach_facet_evidence_foundation(artifact)


def test_reserve_holds_only_sentences_the_first_selection_did_not_use():
    source, artifact = _long_source_artifact()
    reserve = reserve_module.build_judgment_reserve(source, artifact)
    assert reserve is not None and reserve["reserve_version"] == reserve_module.RESERVE_VERSION
    first = {" ".join(s.text.split()) for s in artifact.facet_evidence_foundation.source_sentences}
    new = [s["text"] for s in reserve["foundation"]["source_sentences"]]
    assert new and not ({" ".join(t.split()) for t in new} & first)
    from app.services.facet_evidence_judgment import MAX_EVIDENCE_SENTENCES_PER_CANDIDATE
    assert all(len(b["evidence_sentence_ids"]) <= MAX_EVIDENCE_SENTENCES_PER_CANDIDATE
               for b in reserve["foundation"]["candidate_bundles"])
    json.dumps(reserve)       # storable as JSON


def test_reserve_is_absent_for_a_short_source_or_a_non_full_text():
    source, artifact = _long_source_artifact(paragraphs=2)
    assert reserve_module.build_judgment_reserve(source, artifact) is None
    source, artifact = _long_source_artifact()
    partial = artifact.model_copy(update={"coverage": artifact.coverage.model_copy(update={"level": "partial_text"})})
    assert reserve_module.build_judgment_reserve(source, partial) is None


def test_reserve_errors_degrade_to_none(monkeypatch):
    source, artifact = _long_source_artifact()
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(reserve_module, "attach_candidate_passage_retrieval", boom)
    assert reserve_module.build_judgment_reserve(source, artifact) is None


def test_reserve_is_private_and_never_serialized():
    source, artifact = _long_source_artifact()
    artifact._judgment_reserve = reserve_module.build_judgment_reserve(source, artifact)
    assert "_judgment_reserve" not in artifact.model_dump() and "judgment_reserve" not in artifact.model_dump_json()
    assert artifact.model_copy()._judgment_reserve is not None


@pytest.fixture
def db():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        record = VerificationReportRecord(
            verification_id="v", paper_version_id="paper-v1", scope_type="personal_owner", scope_id="owner-1",
            report_version=1, artifact_version="v", verdict="supported", evidence_sha256="0" * 64,
            report_payload={})
        session.add(record)
        session.commit()
        yield session, record


def test_reserve_is_stored_once_beside_its_report_and_failures_do_not_propagate(db, monkeypatch):
    session, record = db
    source, artifact = _long_source_artifact()
    reserve = reserve_module.build_judgment_reserve(source, artifact)
    holder = SimpleNamespace(_judgment_reserve=reserve)
    _store_judgment_reserve(session, holder, record)
    _store_judgment_reserve(session, holder, record)
    rows = list(session.scalars(select(JudgmentSourceReserve)))
    assert len(rows) == 1 and rows[0].scope_id == "owner-1"
    assert reserve_module.load_reserve(session, record.id, "personal_owner", "owner-1") == reserve
    assert reserve_module.load_reserve(session, record.id, "personal_owner", "someone-else") is None
    monkeypatch.setattr(reserve_module, "store_reserve", lambda *a: (_ for _ in ()).throw(ValueError("x")))
    _store_judgment_reserve(session, holder, record)       # logged, not raised


def _wider(session, record, directions):
    source, artifact = _long_source_artifact()
    reserve = reserve_module.build_judgment_reserve(source, artifact)
    reserve_module.store_reserve(session, record, reserve)
    session.commit()
    prepared = prepare_candidate_prompts(artifact, max_input_tokens=100_000)
    item = next(i for i in prepared.items if not isinstance(i, CandidateFacetFinding))
    run = SimpleNamespace(scope_type="personal_owner", scope_id="owner-1", policy_hash="p")
    cache = SimpleNamespace(lookup=lambda key: None, store=lambda arm: None)
    wider = reserve_module.make_wider_search(session, record.id, run)
    seen = []
    return wider(artifact, item, ROUTES, run, cache, _fake_call(directions, seen=seen)), seen, item


def test_wider_search_red_only_when_the_new_sentences_also_show_nothing(db, monkeypatch):
    monkeypatch.setattr(settings, "JUDGMENT_MAX_INPUT_TOKENS", 100_000)
    session, record = db
    outcome, seen, item = _wider(session, record, {"deepseek": "none", "glm": "none", "qwen": "none"})
    assert (outcome["display_state"], outcome["reason_code"]) == ("insufficient",
                                                                  "agreed_no_evidence_after_wider_search")
    assert outcome["record"]["new_sentences"] > 0
    assert seen and all(user != item.user_prompt for _, _, user in seen)     # different sentences sent


def test_wider_search_can_find_support_the_first_selection_missed(db, monkeypatch):
    monkeypatch.setattr(settings, "JUDGMENT_MAX_INPUT_TOKENS", 100_000)
    session, record = db
    outcome, _, _ = _wider(session, record, {"deepseek": "supports", "glm": "supports", "qwen": "supports"})
    assert (outcome["display_state"], outcome["reason_code"]) == ("supported", "found_in_wider_search")


def test_no_reserve_means_no_wider_search(db):
    session, record = db
    run = SimpleNamespace(scope_type="personal_owner", scope_id="owner-1", policy_hash="p")
    wider = reserve_module.make_wider_search(session, uuid.uuid4(), run)
    assert wider(None, None, ROUTES, run, None, None) is None
