"""Author identity comes from the opening byline (opening-byline-identity-v1).

A same-titled economic note whose byline named two other people was admitted as
the cited work because the cited surname appeared in a footnote. Fixtures are
invented: no real institution, person or student text.
"""
import hashlib
from unittest.mock import patch

import fitz
import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base
from app.services import source_validator as validator
from app.services.identity_recheck import recheck_admitted_representations
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.base import (
    AcquisitionLocation, RepresentationKind, RetrievalResult, SourceRepresentation,
)
from app.services.source_repository import (
    AdmissionRequest, WorkIdentity, admit_representation, commit_source_admissions,
)
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from app.services.source_type import SourceKindAssessment

TITLE = "Tariff Walls: A Drag on Coastal Growth"
CITED = "Halvard, S. J., & Brisk, J."
YEAR = "2017"
FOOTNOTE = "2 Halvard, S. J. (2016). Trade measures since the downturn. Working paper."
BODY = ("Trade restrictions raise input costs for coastal exporters and slow the "
        "reallocation of capital towards productive firms across the region. ") * 4


def note(byline=None, *, footnote=FOOTNOTE, header="Harbourline Bank Economic Note",
         lead=None):
    """An invented one-title economic note with a bottom-of-page footnote."""
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 40), header, fontsize=9)
        if lead:
            page.insert_text((50, 60), lead, fontsize=9)
        page.insert_text((50, 90), TITLE, fontsize=20)
        y = 120
        if byline:
            page.insert_text((50, y), byline, fontsize=11)
            y += 20
        page.insert_text((50, y), "17 February 2017", fontsize=10)
        y += 30
        for index in range(16):
            page.insert_text((50, y), BODY[index * 7:index * 7 + 90], fontsize=10)
            y += 14
        page.insert_text((50, 760), footnote, fontsize=7)
        page.insert_text((50, 780), "Copyright 2017 Harbourline Bank", fontsize=7)
        for _ in range(3):
            body = doc.new_page()
            line_y = 60
            for index in range(30):
                body.insert_text((50, line_y), BODY[index:index + 90], fontsize=10)
                line_y += 16
        return doc.tobytes()


CONFLICT_PDF = note("Avery Quint and Lena Moor")
CONTROL_PDF = note("Sven Halvard and Jonas Brisk",
                   footnote="2 Other, A. (2016). Trade measures since the downturn.")


def check(pdf, author=CITED, **kwargs):
    return validator.validate_retrieved_pdf(pdf, expected_title=TITLE, expected_author=author,
                                            expected_year=YEAR, skip_completeness=True, **kwargs)


# ── (a) validator rule ──────────────────────────────────────────────────

def test_surname_only_in_a_footnote_under_a_byline_naming_others_is_not_high():
    result = check(CONFLICT_PDF)
    assert result.identity_confidence == "medium" and not result.accept
    assert result.reason_code == "opening_byline_author_conflict"
    assert result.author_conflict["observed_authors"] == ["Avery Quint", "Lena Moor"]
    assert result.author_conflict["policy"] == validator.OPENING_BYLINE_POLICY
    assert result.observed_identity["title"] == TITLE
    assert result.observed_identity["authors"] == []


def test_cited_surname_in_the_byline_still_reaches_high():
    result = check(CONTROL_PDF)
    assert result.identity_confidence == "high" and result.author_conflict is None
    assert result.observed_identity["authors"] == [CITED]


def test_an_explicit_by_line_naming_someone_else_is_a_conflict():
    result = check(note("By Avery Quint"))
    assert result.identity_confidence == "medium"
    assert result.author_conflict["basis"] == "explicit_by_line"


def test_organisation_author_is_not_contradicted_by_a_personal_byline():
    for pdf in (note("Avery Quint and Lena Moor", footnote="2 Other (2016)."),
                note(None, footnote="2 Other (2016).")):
        result = check(pdf, author="Harbourline Bank")
        assert result.identity_confidence == "high"
        assert result.author_conflict is None


def test_organisation_named_only_in_the_footer_still_counts():
    pdf = note(None, header="Coastal Trade Briefing", footnote="2 Other (2016).")
    assert check(pdf, author="Harbourline Bank").identity_confidence == "high"


def test_work_without_a_byline_records_no_conflict():
    # A personal author named in the opening header keeps full credit.
    named = note(None, lead="Prepared by Sven Halvard", footnote="2 Other (2016).")
    assert check(named).identity_confidence == "high"
    # Named nowhere in the title block: no author credit, but no conflict either.
    unnamed = check(note(None))
    assert unnamed.identity_confidence == "medium" and unnamed.author_conflict is None


def test_ocr_text_uses_the_same_opening_rule():
    text = "\n".join([TITLE, "Avery Quint and Lena Moor", "17 February 2017",
                      *((BODY * 3)[index:index + 80] for index in range(0, 1600, 80)), FOOTNOTE])
    result = validator.validate_ocr_derivative_text(
        text, expected_title=TITLE, expected_author=CITED, expected_year=YEAR,
        completeness="complete", _layout_pages=[([{"text": TITLE, "size": 20.0,
                                                   "bbox": (50, 70, 400, 90)}], 800.0)])
    assert result.identity_confidence == "medium"
    assert result.author_conflict["observed_authors"] == ["Avery Quint", "Lena Moor"]


@pytest.mark.parametrize("line", [
    "Evidence From Coastal Regions",
    "Harbourline Bank Economic Note",
    "17 February 2017",
    "Avery Quint",
    "Tariff Walls and Coastal Growth",
])
def test_ordinary_lines_below_a_title_are_not_read_as_a_byline(line):
    assert validator._personal_name_line(line, {"tariff", "walls", "coastal", "growth"}) == []


# ── (b) a landing page's author conflict reaches the file it advertises ──

LANDING_URL = "https://notes.example/tariff-walls"
PDF_URL = "https://notes.example/tariff-walls.pdf"
LANDING_HTML = f'''<html><head><title>{TITLE}</title>
<meta name="citation_title" content="{TITLE}">
<meta name="citation_author" content="Quint, Avery">
<meta name="citation_author" content="Moor, Lena">
<meta name="citation_publication_date" content="2017-02-17">
<meta name="citation_pdf_url" content="{PDF_URL}">
</head><body><h1>{TITLE}</h1><p>Summary only.</p></body></html>'''


def _trace_token(expected=None):
    return _ACTIVE_DISCOVERY_TRACE.set({
        "reference_id": "invented-reference",
        "expected": expected or ExpectedBibliographicFields(title=TITLE, authors=[CITED], year=YEAR),
        "search_policy_version": "api-first-search-v2", "required_web_providers": ["exa"],
        "required": {"bounded_web"}, "queries": [], "attempts": [], "candidates": [],
        "limitations": []})


def _acquire(location, pdf):
    response = httpx.Response(200, text=LANDING_HTML, headers={"content-type": "text/html"},
                              request=httpx.Request("GET", LANDING_URL))
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    result = RetrievalResult(source_name="web_search", success=True, title=TITLE,
                             locations=[location], metadata={})
    token = _trace_token()
    try:
        with patch("app.services.source_resolver.safe_request", return_value=response), \
                patch.object(SourceResolver, "_safe_download", lambda self, url, **kw: pdf):
            accepted = resolver._acquire_from_locations(
                result, expected_title=TITLE, expected_author=CITED, expected_year=YEAR)
        result.metadata["retrieval_trace"] = [{
            "search_attempts": [{"provider": "exa", "query": TITLE, "outcome": "results",
                                 "result_count": 1, "provider_calls": 1}],
            "location_attempts": result.metadata["location_attempts"]}]
        resolver._record_discovery_attempt(category="bounded_web", provider="web_search",
                                           result=result, required=True)
        trace, record = resolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    return accepted, result, trace, record


def test_landing_page_author_conflict_is_carried_to_its_pdf():
    no_byline = note(None)
    landing = AcquisitionLocation(url=LANDING_URL, provider="web_search",
                                  representation_kind=RepresentationKind.HTML,
                                  metadata={"search_provider": "exa", "search_title": TITLE})
    accepted, result, trace, record = _acquire(landing, no_byline)
    assert not accepted and result.representation is None
    pdf_attempt = next(a for a in result.metadata["location_attempts"] if a["url"] == PDF_URL)
    assert pdf_attempt["outcome"] == "identity_unconfirmed"
    assert pdf_attempt["author_conflict_basis"] == "landing_page"
    assert record["outcome"] == "bibliographic_conflict"


def test_a_pdf_byline_conflict_becomes_an_inspectable_author_conflict():
    location = AcquisitionLocation(url=PDF_URL, provider="web_search",
                                   representation_kind=RepresentationKind.PDF,
                                   metadata={"search_provider": "exa", "search_title": TITLE})
    accepted, result, trace, record = _acquire(location, CONFLICT_PDF)
    assert not accepted
    candidate = next(c for c in trace["candidates"]
                     if c.get("identity_evidence_kind") == "pdf_front_matter_observation")
    assert candidate["observed"]["authors"] == ["Avery Quint", "Lena Moor"]
    assert {c["field_name"]: c["outcome"] for c in candidate["comparisons"]}["author"] == "material_conflict"
    assert record["outcome"] == "bibliographic_conflict"


def test_preflight_blocks_landing_corroboration_when_authors_conflict(monkeypatch):
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    conflict = SourceResolver._landing_author_conflict(
        {"observed": {"title": TITLE, "authors": ["Quint, Avery", "Moor, Lena"]},
         "content_sha256": "0" * 64},
        expected_doi=None, expected_title=TITLE, expected_author=CITED, expected_year=YEAR)
    assert conflict["basis"] == "landing_page"
    assert SourceResolver._landing_author_conflict(
        {"observed": {"title": TITLE, "authors": ["Halvard, Sven"]}},
        expected_doi=None, expected_title=TITLE, expected_author=CITED, expected_year=YEAR) is None
    assert SourceResolver._landing_author_conflict(
        {"observed": {"title": "A Different Coastal Study Entirely", "authors": ["Quint, Avery"]}},
        expected_doi=None, expected_title=TITLE, expected_author=CITED, expected_year=YEAR) is None

    def preflight(**extra):
        result = RetrievalResult(source_name="web_search", success=True, representation=SourceRepresentation(
            kind=RepresentationKind.PDF, media_type="application/pdf", content=CONTROL_PDF,
            source_url=PDF_URL))
        outcome = resolver._preflight_acquired_representation(
            result, expected_doi=None, expected_title=TITLE, expected_author=CITED,
            expected_year=YEAR, **extra)
        return outcome, result.metadata
    # The file's own byline names the cited authors: the landing page cannot override it.
    (accepted, _, _), metadata = preflight(landing_author_conflict=conflict)
    assert metadata["identity_confidence"] == "high" and "author_conflict" not in metadata
    # A file that does not name them inherits the conflict.
    monkeypatch.setattr("app.services.source_resolver.validate_retrieved_pdf",
                        lambda *a, **k: validator.ValidationResult(
                            accept=False, identity_confidence="high", completeness="complete",
                            text_quality="digital", reason="stub",
                            observed_identity={"policy": validator.OPENING_BYLINE_POLICY,
                                               "title": TITLE, "authors": [], "doi": None}))
    (accepted, outcome, _), metadata = preflight(landing_author_conflict=conflict)
    assert not accepted and outcome == "identity_unconfirmed"
    assert metadata["author_conflict"]["basis"] == "landing_page"


# ── (c) reuse never confirms a reference against its own fields ─────────

class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def upload(self, data, key):
        self.objects[key] = data
        return key

    def download(self, key):
        try:
            return self.objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc

    def delete(self, key):
        self.objects.pop(key, None)
        return True

    def exists(self, key):
        return key in self.objects

    def list_keys(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def repository(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    storage = MemoryStorage()
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_ENABLED", True)
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "invented-scope")
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = storage
    resolver._repository_session_factory = factory
    yield resolver, factory, storage
    engine.dispose()


def _admit(factory, storage, content, *, kind=RepresentationKind.PDF, evidence=None,
           provenance="web_search", title=TITLE):
    with factory() as session:
        record = admit_representation(session, storage, AdmissionRequest(
            work=WorkIdentity(title=title, work_type="report", author=CITED, year=YEAR),
            representation=SourceRepresentation(
                kind=kind, media_type="application/pdf" if kind is RepresentationKind.PDF else "text/plain",
                content=content, source_url=PDF_URL),
            provenance=provenance, license_class="open_access", scope_type="personal_owner",
            scope_id="invented-scope", identity_verdict="verified", identity_confidence=1.0,
            completeness_verdict="complete", cleanliness_verdict="clean",
            admitted_by="retrieval_pipeline", validation_evidence=evidence or {}))
        commit_source_admissions(session)
        return str(record.id)


def _reuse_outcome(resolver, title=TITLE):
    result = resolver._check_local_cache(None, None, title, author=CITED, year=YEAR,
                                         expected_source_kind=SourceKindAssessment())
    token = _ACTIVE_DISCOVERY_TRACE.set({
        "reference_id": "invented-reference",
        "expected": ExpectedBibliographicFields(title=title, authors=[CITED], year=YEAR),
        "required": set(), "queries": [], "attempts": [], "candidates": [], "limitations": []})
    try:
        SourceResolver._record_discovery_attempt(category="durable_repository",
                                                 provider="local_cache", result=result, required=False)
        _trace, record = SourceResolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    return result, (record or {}).get("outcome")


def test_reuse_of_a_legacy_pdf_rechecks_its_bytes(repository):
    resolver, factory, storage = repository
    _admit(factory, storage, CONFLICT_PDF)
    result, outcome = _reuse_outcome(resolver)
    assert result.success is False and outcome != "confirmed"


def test_reuse_of_a_legacy_pdf_that_still_holds_is_confirmed_from_its_bytes(repository):
    resolver, factory, storage = repository
    _admit(factory, storage, CONTROL_PDF)
    result, outcome = _reuse_outcome(resolver)
    assert result.success is True and result.authors == [CITED]
    assert result.metadata["reuse_observed_identity"]["policy"] == validator.OPENING_BYLINE_POLICY
    assert outcome == "confirmed"


def test_citation_echo_fields_never_agree_on_reuse(repository):
    resolver, factory, storage = repository
    # The work record repeats the citation exactly, but the source itself
    # showed none of those fields.
    _admit(factory, storage, CONFLICT_PDF, evidence={"observed_identity": {
        "policy": validator.OPENING_BYLINE_POLICY, "title": None, "authors": [],
        "year": None, "doi": None}})
    result, outcome = _reuse_outcome(resolver)
    assert result.success is True and result.title is None and result.authors == []
    assert outcome != "confirmed"


def test_reused_text_reports_only_what_its_opening_shows(repository):
    resolver, factory, storage = repository
    text = ("Coastal trade policy overview.\n" + (BODY + "\n") * 60 + FOOTNOTE + "\n" + TITLE).encode()
    _admit(factory, storage, text, kind=RepresentationKind.PLAIN_TEXT)
    result, outcome = _reuse_outcome(resolver)
    assert result.success is True and result.title is None and result.authors == []
    assert outcome != "confirmed"


# ── (d) read-only recheck of admitted search-found PDFs ─────────────────

def test_recheck_lists_admitted_web_search_pdfs_no_longer_high(repository):
    _resolver, factory, storage = repository
    flagged_id = _admit(factory, storage, CONFLICT_PDF)
    _admit(factory, storage, CONTROL_PDF)
    _admit(factory, storage, CONFLICT_PDF, provenance="core")
    with factory() as session:
        summary = recheck_admitted_representations(session, storage)
    assert summary["checked"] == 2
    assert [item["representation_id"] for item in summary["flagged"]] == [flagged_id]
    item = summary["flagged"][0]
    assert item["content_sha256"] == hashlib.sha256(CONFLICT_PDF).hexdigest()
    assert item["author_conflict"] is True and item["identity_confidence"] == "medium"
    assert set(item) == {"representation_id", "canonical_work_id", "content_sha256",
                         "identity_confidence", "author_conflict", "reason_code"}


def test_affiliation_marks_against_byline_names_still_credit_the_author():
    pdf = note("Sven Halvard1, Jonas Brisk2*",
               footnote="2 Other, A. (2016). Trade measures since the downturn.")
    assert check(pdf).identity_confidence == "high"


def manuscript_with_author_note(note_heading="Mini Bio"):
    """An invented accepted manuscript: no byline, the writer named in an author note."""
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Title:", fontsize=10)
        page.insert_text((50, 80), TITLE, fontsize=14)
        y = 110
        page.insert_text((50, y), "Abstract", fontsize=10)
        for index in range(18):
            y += 14
            page.insert_text((50, y), BODY[index * 7:index * 7 + 90], fontsize=10)
        y += 24
        page.insert_text((50, y), note_heading, fontsize=10)
        page.insert_text((50, y + 14), "Dr. Sven Halvard teaches trade policy at a coastal institute.",
                         fontsize=10)
        page.insert_text((50, 780), "2017", fontsize=7)
        for _ in range(3):
            body = doc.new_page()
            body.insert_text((50, 60), BODY[:90], fontsize=10)
        return doc.tobytes()


@pytest.mark.parametrize("heading", ["Mini Bio", "About the author", "Notes on contributor"])
def test_an_author_note_credits_a_manuscript_without_a_byline(heading):
    assert check(manuscript_with_author_note(heading)).identity_confidence == "high"


def test_a_surname_far_below_the_title_without_an_author_note_gets_no_credit():
    assert check(manuscript_with_author_note("Further reading")).identity_confidence != "high"


def test_a_cited_surname_inside_a_hyphenated_byline_name_is_not_credited():
    front = ("An Introduction to the Larkmere-Vance Conjecture\nO. Quillby\n"
             "Notes for a lecture course on curves.")
    credited, *_ = validator._author_observation(front, None, "Introduction", "Vance, R.",
                                                 validator._normalize_identity_text(front))
    assert not credited
    whole, *_ = validator._author_observation(front.replace("O. Quillby", "R. Larkmere-Vance"),
                                              None, "Introduction", "Larkmere-Vance, R.", "")
    assert whole


def test_a_one_word_title_inside_a_longer_title_is_not_prominent():
    spans = [{'text': 'An Introduction to the Larkmere-Vance Conjecture',
              'bbox': (50, 40, 500, 60), 'size': 18.0},
             {'text': 'Notes for a lecture course on curves. ' * 5,
              'bbox': (50, 400, 500, 420), 'size': 10.0}]
    assert not validator._title_in_layout_spans(spans, 800.0, "Introduction")
    alone = [{**spans[0], 'text': 'Introduction'}, spans[1]]
    assert validator._title_in_layout_spans(alone, 800.0, "Introduction")
