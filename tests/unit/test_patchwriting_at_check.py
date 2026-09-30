"""patchwriting-v3 at check time: bounded, degrading, never rendered."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import patchwriting_at_check as pac

SOURCE_TEXT = (
    "An unrelated opening sentence about archives and catalogues. "
    "Alice’s visit to Wonderland signifies a postmodern crisis pertaining to power, identity and "
    "the nature of reality. A closing sentence about Victorian publishing that nobody paraphrased."
)
STATEMENT = ("The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, "
             "and the nature of reality (Flegar & Wertag, 2015).")
PAPER = ("Title of the essay\n\nAn introduction written by the student in their own words. "
         + STATEMENT + " A conclusion follows.\n\nReferences\n\n"
         "Flegar, Z., & Wertag, I. (2015). Alice in Wonderland. Journal of Tests, 1(1), 1-10.\n")


def _claim(text=STATEMENT, reference_id="ref-1"):
    start = PAPER.index(text)
    marker = "Flegar & Wertag, 2015"
    local = text.index(marker)
    return SimpleNamespace(
        claim_id="claim-1", text=text, passage_start=start, passage_end=start + len(text),
        claim_type="paraphrase", reference_ids=[reference_id], source_segments=[],
        citation_marker=f"({marker})",
        citation_markers=[{"local_start": local, "local_end": local + len(marker)}],
    )


def _source(text=SOURCE_TEXT, representation_id="rep-1"):
    content = text.encode()
    return SimpleNamespace(
        representation_id=representation_id, representation_kind="plain_text", content=content,
        content_sha256=hashlib.sha256(content).hexdigest(), page_labels=None, derivation_method=None,
        completeness_verdict="complete",
    )


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(pac, "_enabled", lambda: True)
    import app.services.paper_upload as upload
    import app.services.text_extractor as extractor

    monkeypatch.setattr(pac, "_locate_paper", lambda session_factory, job_id: {
        "status": "ready", "key": "paper-key", "sha256": hashlib.sha256(b"bytes").hexdigest(),
        "filename": "paper.docx"})
    monkeypatch.setattr(extractor, "extract_qualified_text_from_bytes",
                        lambda content, filename: SimpleNamespace(selected=SimpleNamespace(text=PAPER)))

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def make(claims=None):
        artifact = SimpleNamespace(citation_format="apa", citation_claims=claims or [_claim()])
        return pac.PatchwritingAtCheck(_Session, SimpleNamespace(download=lambda key: b"bytes"), "job-1", artifact)

    return make


def test_result_is_persisted_with_bounded_fields_and_labels(checker):
    check = checker()
    check.run(_source(), "ref-1")
    block = check.summary_block()
    assert block["policy_version"] == "patchwriting-v3" and block["decision_applied"] is False
    assert block["body"]["status"] == "ready"
    entry = block["sources"]["ref-1"]
    assert entry["status"] == "compared"
    assert entry["coverage"]["representation_id"] == "rep-1"
    assert entry["coverage"]["content_sha256"] == _source().content_sha256
    finding = entry["findings"][0]
    assert finding["kind"] == "close_paraphrase"
    assert finding["label"] == "citation_statement_vs_cited_source"
    assert finding["claim_ids"] == ["claim-1"]
    region = finding["student_region"]
    assert PAPER[region["paper_start"]:region["paper_end"]] == region["text"]
    assert region["text"].startswith("The visit of Alice") and region["text"].endswith("reality")
    [sentence] = finding["source_sentences"]
    assert sentence["text"].startswith("Alice’s visit") and sentence["page_index"] is None
    assert sentence["sentence_key"] == f"None:{sentence['absolute_start']}:{sentence['absolute_end']}"
    serialized = json.dumps(block)
    assert "Victorian publishing" not in serialized and "archives and catalogues" not in serialized


def test_other_references_source_is_labelled_body_sentence(checker):
    check = checker()
    check.run(_source(representation_id="rep-2"), "ref-2")
    finding = check.summary_block()["sources"]["ref-2"]["findings"][0]
    assert finding["label"] == "body_sentence"


def test_failures_degrade_and_are_recorded(checker, monkeypatch):
    check = checker()
    check.run(SimpleNamespace(representation_kind="pdf", content=b"not a pdf", representation_id="x",
                              content_sha256="0" * 64), "ref-bad-pdf")
    check.run(object(), "ref-broken")
    check.not_held("ref-run", "source_not_held_at_check")
    block = check.summary_block()
    assert block["sources"]["ref-bad-pdf"]["status"] == "not_assessed"
    assert block["sources"]["ref-broken"]["reason"].startswith("internal_error:")
    assert block["sources"]["ref-run"]["reason"] == "source_not_held_at_check"

    import app.services.paper_upload as upload

    def unavailable(*_args):
        raise RuntimeError("input gone")

    monkeypatch.setattr(pac, "_locate_paper", unavailable)
    check = checker()
    check.run(_source(), "ref-1")
    block = check.summary_block()
    assert block["sources"]["ref-1"]["reason"] == "body_unavailable:RuntimeError"
    assert block["body"]["status"] == "body_unavailable:RuntimeError"


def test_body_offsets_must_reproduce_every_claim(checker):
    claim = _claim()
    claim.passage_start += 1
    check = checker([claim])
    check.run(_source(), "ref-1")
    assert check.summary_block()["sources"]["ref-1"]["reason"] == "body_offsets_unverified"


def test_disabled_setting_stores_nothing(monkeypatch):
    monkeypatch.setattr(pac, "_enabled", lambda: False)
    check = pac.PatchwritingAtCheck(None, None, "job", SimpleNamespace(citation_claims=[]))
    check.run(_source(), "ref-1")
    assert check.summary_block() is None and check.sources == {}


def test_refresh_keeps_previous_results_for_other_references(monkeypatch):
    monkeypatch.setattr(pac, "_enabled", lambda: True)
    previous = {pac.SUMMARY_KEY: {"sources": {"ref-a": {"status": "compared"}, "ref-b": {"status": "compared"}}}}
    check = pac.PatchwritingAtCheck(None, None, "job", SimpleNamespace(citation_claims=[]),
                                    previous_summary=previous, refresh_reference_ids={"ref-b"})
    assert set(check.sources) == {"ref-a"}


def test_block_size_and_finding_caps_are_enforced():
    finding = {"student_region": {"text": "x" * 800}, "source_sentences": [{"text": "y" * 800}]}
    block = {"sources": {f"ref-{i}": {"findings": [dict(finding) for _ in range(60)]} for i in range(6)}}
    bounded = pac.enforce_paper_bounds(block)
    kept = sum(len(v["findings"]) for v in bounded["sources"].values())
    assert kept <= pac.MAX_FINDINGS_PER_PAPER
    assert len(json.dumps(bounded).encode()) <= pac.MAX_BLOCK_BYTES
    assert sum(v.get("findings_truncated", 0) for v in bounded["sources"].values()) == 360 - kept


def test_only_the_report_projection_reads_the_result():
    # Owner decision 2026-09-29: the report shows passages through the
    # projection module; renderers never read the stored block or its internals.
    root = Path(__file__).resolve().parents[2] / "app"
    renderers = [
        "services/evidence_report.py", "services/report_export.py", "services/interactive_report_export.py",
        "services/report_interactions.js", "services/report_judgment.js", "services/judgment_report.py",
        "services/verification_report.py", "routers/report.py", "routers/status.py",
    ]
    import re
    reads = re.compile(r"""["']patchwriting["']|patchwriting_at_check|services\.patchwriting(?!_report)|"""
                       r"""import patchwriting\b|source_quoted_share|student_matched_spans|thresholds\[""")
    for name in renderers:
        path = root / name
        if path.exists():
            assert not reads.search(path.read_text(encoding="utf-8")), name
