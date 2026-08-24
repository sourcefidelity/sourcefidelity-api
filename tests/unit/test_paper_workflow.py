"""End-to-end local checkpoint test for the shadow paper workflow."""

import io

from docx import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.models.job import Job
from app.models.report import Report, VerificationReportRecord
from app.models.verification_run import VerificationRunRecord
from app.services import paper_upload
from app.services.file_safety import SafetyVerdict
from app.services.paper_upload import DOCX_MEDIA_TYPE, create_paper_job
from app.services.paper_workflow import (
    _claims_by_reference,
    extract_paper_job,
    finalize_paper_job,
    retrieve_paper_sources,
    verify_paper_sources,
)
from app.services.verification_evidence import ClaimEvidence
from app.services.retrieval.base import (
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.storage.backend import StorageBackend


class MemoryStorage(StorageBackend):
    def __init__(self):
        self.objects = {}

    def upload(self, file_bytes, key):
        self.objects[key] = file_bytes
        return key

    def download(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key):
        self.objects.pop(key, None)
        return True

    def exists(self, key):
        return key in self.objects

    def list_keys(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


class Resolver:
    def __init__(self):
        self.calls = 0

    def resolve_reference(self, reference):
        self.calls += 1
        return RetrievalResult(
            source_name="test_source",
            success=True,
            representation=SourceRepresentation(
                kind=RepresentationKind.PLAIN_TEXT,
                media_type="text/plain",
                content=(
                    b"Careful verification improves accuracy by checking every source. "
                    b"The process records inspectable evidence."
                ),
                completeness="complete",
            ),
            title=reference.title,
            metadata={"identity_confidence": "high", "text_quality": "digital"},
        )


def _paper_bytes():
    document = Document()
    document.add_paragraph("Careful verification improves accuracy (Smith, 2020).")
    document.add_paragraph("Inspectable evidence improves review (Smith, 2020).")
    document.add_paragraph("References")
    document.add_paragraph("Smith, J. (2020). A useful title. Example Press.")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def test_collective_claim_is_verified_against_each_exact_source_membership():
    claim = ClaimEvidence(
        claim_id="collective-claim",
        paper_version_id="paper-v1",
        text="Smith and Jones show that the result is stable.",
        reference_ids=["ref-smith", "ref-jones"],
        passage_start=0,
        passage_end=47,
    )

    grouped = _claims_by_reference([claim])

    assert grouped == {"ref-smith": [claim], "ref-jones": [claim]}


def test_checkpointed_workflow_persists_shadow_report_and_cleans(monkeypatch):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    with factory() as session:
        job = create_paper_job(
            session,
            storage,
            content=_paper_bytes(),
            filename="paper.docx",
            media_type=DOCX_MEDIA_TYPE,
            scope_id="personal-default",
        )
        job_id = job.id
        extracted = extract_paper_job(session, storage, job_id, llm_enabled=False)
        assert extracted["citation_units"] == 2
        assert job.input_storage_key is None
        assert extract_paper_job(session, storage, job_id, llm_enabled=False) == extracted
        resolver = Resolver()
        retrieved = retrieve_paper_sources(
            session,
            storage,
            job_id,
            resolver=resolver,
        )
        assert retrieved == {"transient_authorized": 1}
        assert resolver.calls == 1
        assert retrieve_paper_sources(session, storage, job_id, resolver=resolver) == retrieved
        assert resolver.calls == 1

    summary = verify_paper_sources(factory, storage, job_id, llm_enabled=False)
    assert summary["reports_persisted"] == 2
    assert summary["relationship_mode"] == "shadow"
    assert summary["decision_applied"] is False
    assert verify_paper_sources(factory, storage, job_id, llm_enabled=False) == summary

    with factory() as session:
        finalized = finalize_paper_job(session, storage, job_id)
        job = session.get(Job, job_id)
        evidence = session.scalars(select(VerificationReportRecord)).all()
        aggregate = session.scalar(select(Report))
        run = session.scalar(select(VerificationRunRecord))
        assert finalized["input_cleaned"] is True
        assert job.status == "completed"
        assert job.stage == "completed"
        assert job.input_storage_key is None
        assert len(evidence) == 2
        assert all(item.verdict == "inconclusive" for item in evidence)
        assert all(
            item.report_payload["verification_candidates"]["status"] == "complete"
            for item in evidence
        )
        assert all(
            item.report_payload["candidate_relationships"]["status"] == "not_run"
            for item in evidence
        )
        assert all(
            item.report_payload["facet_evidence_foundation"]["status"] == "complete"
            for item in evidence
        )
        assert all(
            item.report_payload["facet_evidence_ledger"]["status"] == "not_run"
            for item in evidence
        )
        assert aggregate.report_json["decision_applied"] is False
        assert run.status == "cleaned"
        assert run.transient_objects == []
        assert finalize_paper_job(session, storage, job_id)["report_id"] == finalized["report_id"]
    assert storage.objects == {}


def test_store_only_job_stops_after_extraction_and_cleans(monkeypatch):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    with factory() as session:
        job = create_paper_job(
            session,
            storage,
            content=_paper_bytes(),
            filename="paper.docx",
            media_type=DOCX_MEDIA_TYPE,
            scope_id="personal-default",
            store_only=True,
        )
        job_id = job.id
        extract_paper_job(session, storage, job_id, llm_enabled=False)
        finalized = finalize_paper_job(session, storage, job_id)
        aggregate = session.scalar(select(Report))
        assert finalized["input_cleaned"] is True
        assert aggregate.report_json["store_only"] is True
        assert aggregate.report_json["reports_persisted"] == 0
        assert session.scalar(select(VerificationRunRecord)) is None
        assert session.scalar(select(VerificationReportRecord)) is None
    assert storage.objects == {}
