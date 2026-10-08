"""Run patchwriting-v5 during a paper check while each source's text is authorized.

Owner decision 2026-09-29. For every retrieved full-text source the paper check
authorizes, all body sentences of the paper are compared with that source
(citation statements of the cited source are labelled). The bounded result is
stored in the job's verification summary under ``patchwriting`` (and therefore
in the aggregate report JSON), keyed by reference: it is paper-level evidence
(body x source), not claim-level, so it is kept out of the per-claim immutable
Evidence Package and its authoritative fields. The report view projects it
into Passage windows in the owner's wording (``patchwriting_report``, owner
decision 2026-09-29); the measures, thresholds and keys stay here.

Stored per finding: exact student spans with paper offsets, the matched source
sentence(s) (key, page, offsets and only their text, bounded), measures, kind
and label; per source: coverage and policy version. Never the full source text.
Nothing here can fail a paper run: every entry point degrades and records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.services import patchwriting as pw

SUMMARY_KEY = "patchwriting"
MAX_FINDINGS_PER_SOURCE = 50
MAX_FINDINGS_PER_PAPER = 200
MAX_STUDENT_TEXT = 800
MAX_SOURCE_TEXT = 800
MAX_MATCHED_SPANS = 40
MAX_LIMITATIONS = 20
MAX_BLOCK_BYTES = 512_000     # hard bound on the serialized summary block


def _enabled() -> bool:
    try:
        from app.config import settings

        return bool(getattr(settings, "PATCHWRITING_AT_CHECK_ENABLED", False))
    except Exception:
        return False


@dataclass
class _Body:
    status: str
    text: str | None = None
    sha256: str | None = None
    sentences: int = 0
    statements: list[Any] = field(default_factory=list)     # (claim, CitationStatement)


def body_from_text(text: str, citation_format: str) -> str:
    """The paper body exactly as extraction defines it (text before the references)."""
    from app.services.paper_extraction import _body_before_reference_section
    from app.services.parsers.apa_parser import ApaParser
    from app.services.parsers.mla_parser import MlaParser
    from app.services.reference_parser import extract_reference_section

    parser = {"apa": ApaParser, "mla": MlaParser}[citation_format]
    section = extract_reference_section(text, citation_format)
    if not section:
        raise ValueError("no reference section")
    return _body_before_reference_section(text, section, parser)


def _locate_paper(session_factory, job_id) -> dict:
    """Where the paper's bytes can be read, found with the job's own session
    before any source session opens; no download happens here."""
    import uuid as _uuid
    from datetime import datetime, timezone
    from sqlalchemy import select
    from app.models.job import Job
    from app.models.report import ReportPaperArtifactRecord

    with session_factory() as session:
        job = session.get(Job, job_id if isinstance(job_id, _uuid.UUID) else _uuid.UUID(str(job_id)))
        if job is None:
            return {"status": "job_missing"}
        evidence = job.upload_evidence or {}
        if job.input_storage_key and job.input_deleted_at is None and not evidence.get("input_cleanup_started"):
            return {"status": "ready", "key": job.input_storage_key, "sha256": job.input_sha256,
                    "filename": job.filename}
        # The upload is deleted after extraction; the report keeps its own
        # hash-checked copy of the paper, made before that cleanup.
        copy = session.scalar(select(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id))
        expires = getattr(copy, "expires_at", None)
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if (copy is None or copy.deleted_at is not None or not copy.storage_key
                or (expires is not None and expires <= datetime.now(timezone.utc))):
            return {"status": "paper_copy_unavailable"}
        return {"status": "ready", "key": copy.storage_key, "sha256": copy.content_sha256, "filename": job.filename}


def _load_body(location: dict, backend, artifact) -> _Body:
    from app.services.text_extractor import extract_qualified_text_from_bytes

    if location.get("status") != "ready":
        return _Body(str(location.get("status") or "paper_unavailable"))
    content = backend.download(location["key"])
    if hashlib.sha256(content).hexdigest() != location["sha256"]:
        return _Body("paper_copy_hash_mismatch")
    text = extract_qualified_text_from_bytes(content, location["filename"]).selected.text
    body = body_from_text(text, artifact.citation_format)
    claims = list(artifact.citation_claims)
    # The body must reproduce every claim at its stored offsets, or spans
    # could not be bound to the paper; otherwise the comparison is not run.
    for claim in claims:
        if claim.passage_start < 0 or body[claim.passage_start:claim.passage_end] != claim.text:
            return _Body("body_offsets_unverified")
    statements = []
    for claim in claims:
        statement = pw.citation_statement_from_claim(claim)
        if statement is not None:
            statements.append((claim, statement))
    return _Body("ready", body, hashlib.sha256(body.encode("utf-8")).hexdigest(),
                 len(pw.sentence_spans(body)), statements)


class PatchwritingAtCheck:
    """Collects one bounded result per retrieved source during verification."""

    def __init__(self, session_factory, backend, job_id, artifact, *, previous_summary=None,
                 refresh_reference_ids=()):
        self.enabled = _enabled()
        self._session_factory = session_factory
        self._backend = backend
        self._job_id = job_id
        self._artifact = artifact
        self._body: _Body | None = None
        self._location: dict | None = None
        self.sources: dict[str, dict] = {}
        self.errors: list[str] = []
        if self.enabled and previous_summary and refresh_reference_ids:
            try:
                kept = ((previous_summary.get(SUMMARY_KEY) or {}).get("sources") or {})
                self.sources.update({ref: value for ref, value in kept.items()
                                     if ref not in set(refresh_reference_ids)})
            except Exception as exc:  # noqa: BLE001 - never fail the run
                self.errors.append(f"previous:{type(exc).__name__}")

    def prepare(self) -> None:
        """Find the paper's bytes with the job's own session, before any source
        session opens; nothing is downloaded until a source is compared."""
        if self.enabled and self._location is None:
            try:
                self._location = _locate_paper(self._session_factory, self._job_id)
            except Exception as exc:  # noqa: BLE001
                self._location = {"status": f"body_unavailable:{type(exc).__name__}"}

    def _ensure_body(self) -> _Body:
        if self._body is None:
            self.prepare()
            try:
                self._body = _load_body(self._location or {}, self._backend, self._artifact)
            except Exception as exc:  # noqa: BLE001
                self._body = _Body(f"body_unavailable:{type(exc).__name__}")
        return self._body

    def run(self, source, reference_id: str) -> None:
        """Compare the body with one authorized source; never raises."""
        if not self.enabled:
            return
        try:
            body = self._ensure_body()
            if body.status != "ready":
                self.sources[reference_id] = {"policy_version": pw.POLICY_VERSION,
                                              "status": "not_assessed", "reason": body.status}
                return
            from app.services.verification_evidence import _extract_pages, _extracted_pages_sha256

            pages, extraction_limits = _extract_pages(source)
            index = pw.build_source_index(
                pw.source_sentences_from_pages(pages),
                comparison_scope="full_text",
                representation_id=str(source.representation_id),
                content_sha256=source.content_sha256,
                extracted_text_sha256=_extracted_pages_sha256(pages) if pages else None,
            )
            statements = [
                pw.CitationStatement(
                    claim_id=statement.claim_id, paper_start=statement.paper_start,
                    paper_end=statement.paper_end, marker_spans=statement.marker_spans,
                    block_quotation=statement.block_quotation,
                    cited_representation_ids=((str(source.representation_id),)
                                              if reference_id in (claim.reference_ids or []) else ()),
                )
                for claim, statement in body.statements
            ]
            result = pw.detect_in_body(body.text, [index], statements=statements)
            result.limitations.extend(f"extraction:{item[:120]}" for item in extraction_limits)
            self.sources[reference_id] = bounded_result(result, reference_id=reference_id,
                                                        completeness=getattr(source, "completeness_verdict", None))
        except Exception as exc:  # noqa: BLE001 - degrade the one record
            self.sources[reference_id] = {"policy_version": pw.POLICY_VERSION, "status": "not_assessed",
                                          "reason": f"internal_error:{type(exc).__name__}"}

    def not_held(self, reference_id: str, reason: str) -> None:
        if self.enabled and reference_id not in self.sources:
            self.sources[reference_id] = {"policy_version": pw.POLICY_VERSION,
                                          "status": "not_assessed", "reason": reason}

    def summary_block(self) -> dict | None:
        """The bounded block stored under the verification summary, or None when off."""
        if not self.enabled:
            return None
        try:
            body = self._body
            block = {
                "policy_version": pw.POLICY_VERSION,
                "decision_applied": False,
                "body": ({"status": body.status, "body_sha256": body.sha256, "sentences": body.sentences}
                         if body is not None else {"status": "not_loaded"}),
                "sources": dict(self.sources),
                "limitations": self.errors[:MAX_LIMITATIONS],
            }
            return enforce_paper_bounds(block)
        except Exception as exc:  # noqa: BLE001
            return {"policy_version": pw.POLICY_VERSION, "decision_applied": False,
                    "status": "not_assessed", "reason": f"internal_error:{type(exc).__name__}"}


def _student(span: pw.StudentSpan) -> dict:
    return {"paper_start": span.paper_start, "paper_end": span.paper_end,
            "text": span.text[:MAX_STUDENT_TEXT], "text_truncated": len(span.text) > MAX_STUDENT_TEXT}


def _source_sentence(sentence: pw.MatchedSourceSentence) -> dict:
    row = {
        "sentence_key": sentence.sentence_key,
        "page_index": sentence.page_index,
        "page_label": sentence.page_label,
        "absolute_start": sentence.absolute_start,
        "absolute_end": sentence.absolute_end,
        "role": sentence.role,
        "text": sentence.text[:MAX_SOURCE_TEXT],
        "text_truncated": sentence.text_truncated or len(sentence.text) > MAX_SOURCE_TEXT,
        "matched_spans": [span.model_dump() for span in sentence.matched_spans[:MAX_MATCHED_SPANS]],
    }
    origin, stop = sentence.absolute_start, sentence.absolute_end
    if (len(sentence.text) > MAX_SOURCE_TEXT and not sentence.text_truncated and row["matched_spans"]
            and isinstance(origin, int) and isinstance(stop, int) and stop - origin == len(sentence.text)):
        # A long sentence keeps the stretch around its matched words, not its
        # first characters, so those words can be shown (2026-10-02).
        low = min(span["absolute_start"] for span in row["matched_spans"]) - origin
        high = max(span["absolute_end"] for span in row["matched_spans"]) - origin
        if 0 <= low < high <= len(sentence.text) and high - low <= MAX_SOURCE_TEXT:
            start = max(0, min(low - (MAX_SOURCE_TEXT - (high - low)) // 2, len(sentence.text) - MAX_SOURCE_TEXT))
            end = start + MAX_SOURCE_TEXT
            row.update(text=sentence.text[start:end], text_truncated=False,
                       absolute_start=origin + start, absolute_end=origin + end,
                       cut_before=start > 0, cut_after=end < len(sentence.text))
    return row


def bounded_result(result: pw.PatchwritingResult, *, reference_id: str, completeness=None) -> dict:
    findings = result.findings[:MAX_FINDINGS_PER_SOURCE]
    coverage = result.coverage
    source = coverage.sources[0] if coverage.sources else pw.SourceCoverage()
    return {
        "policy_version": result.policy_version,
        "status": result.status,
        "reason": result.reason,
        "reference_id": reference_id,
        "coverage": {
            "representation_id": source.representation_id,
            "content_sha256": source.content_sha256,
            "extracted_text_sha256": source.extracted_text_sha256,
            "comparison_scope": source.comparison_scope,
            "completeness_verdict": completeness,
            "source_sentences_indexed": source.source_sentences_indexed,
            "source_sentences_compared": source.source_sentences_compared,
            "student_sentences": coverage.student_sentences,
            "student_sentences_compared": coverage.student_sentences_compared,
            "student_words_total": coverage.student_words_total,
            "student_words_excluded_quotation": coverage.student_words_excluded_quotation,
            "student_words_excluded_marker": coverage.student_words_excluded_marker,
            "student_words_compared": coverage.student_words_compared,
        },
        "findings": [
            {
                "kind": f.kind,
                "label": f.label,
                "claim_ids": f.claim_ids[:8],
                "student_sentence": {"paper_start": f.student_sentence.paper_start,
                                     "paper_end": f.student_sentence.paper_end},
                "student_region": _student(f.student_region),
                "student_matched_spans": [_student(s) for s in f.student_matched_spans[:MAX_MATCHED_SPANS]],
                "source_sentences": [_source_sentence(s)
                                     for s in [f.source, *f.additional_source_sentences][:2]],
                "measures": f.measures.model_dump(),
            }
            for f in findings
        ],
        "findings_total": len(result.findings),
        "findings_truncated": max(0, len(result.findings) - len(findings)),
        "thresholds": dict(result.thresholds),
        "limitations": [item[:200] for item in result.limitations[:MAX_LIMITATIONS]],
        "decision_applied": False,
    }


def enforce_paper_bounds(block: dict) -> dict:
    """Cap findings per paper and the serialized size; record what was dropped."""
    sources = block.get("sources") or {}
    kept = 0
    for value in sources.values():
        findings = value.get("findings") or []
        room = max(0, MAX_FINDINGS_PER_PAPER - kept)
        if len(findings) > room:
            value["findings_truncated"] = value.get("findings_truncated", 0) + len(findings) - room
            value["findings"] = findings[:room]
        kept += len(value.get("findings") or [])
    while len(json.dumps(block, default=str).encode("utf-8")) > MAX_BLOCK_BYTES:
        largest = max(sources.values(), key=lambda v: len(v.get("findings") or []), default=None)
        if not largest or not largest.get("findings"):
            block["sources"] = {ref: {k: v for k, v in value.items() if k != "findings"}
                                for ref, value in sources.items()}
            block["size_truncated"] = True
            break
        largest["findings"].pop()
        largest["findings_truncated"] = largest.get("findings_truncated", 0) + 1
        block["size_truncated"] = True
    return block
