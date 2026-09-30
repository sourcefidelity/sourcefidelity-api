"""Page-level text check, OCR repair receipts and rejection of unusable text (owner decision 2026-09-28)."""
import hashlib

import fitz
import pytest

from app.config import settings
from app.services import page_ocr_repair as repair
from app.services import text_quality as tq

WORDS = frozenset("""the of and to in a is that for it as with was on be by are this which from or at an not
have has were their but its they these also can been more than such other into film films studio studios
system industry changed changes after war audiences television hollywood would could only would director
directors production companies market markets gradually since decade decline power new old period
first findings efficiency different important example with them were there each year many large small
cinema theatres audience studied between during because while under over early late product products eel eels
""".split())


@pytest.fixture(autouse=True)
def word_list(monkeypatch):
    monkeypatch.setattr(tq, "_word_list", lambda: (WORDS, "w" * 64))
    monkeypatch.setattr(tq, "word_list_sha256", lambda: "w" * 64)


CLEAN = ("After the war the studio system changed and the industry lost its power over the market. "
         "Television took many audiences from the theatres and the studios would sell old films to it. ") * 4
DAMAGED = ("After tl1e war tlie studio system changed and tlie industry lost its power over tl1e market. "
           "Television took many audiences from tlie theatres and tlie studios wouJd sell old fiJms to it. ") * 4
FRENCH = ("Après la guerre le système des studios a changé et l'industrie a perdu son pouvoir sur le marché. "
          "La télévision a pris beaucoup de spectateurs aux salles de cinéma et les studios ont vendu. ") * 4


def test_clean_english_page_is_clean():
    assert tq.assess_page(CLEAN).status == "clean"


def test_ocr_confusions_make_a_page_damaged():
    result = tq.assess_page(DAMAGED)
    assert result.status == "damaged" and result.damaged_share >= tq.DAMAGED_WORD_SHARE


@pytest.mark.parametrize("text", [
    # Unknown words that are only unknown (jargon, URLs, identifiers) are not damage.
    CLEAN + " transmedia intercompositional doi https://doi.org/10.1234/abc task1 491b55f9 pre-war " * 3,
])
def test_jargon_links_and_identifiers_are_not_damage(text):
    assert tq.assess_page(text).status == "clean"


def test_a_dropped_ligature_is_damage():
    text = CLEAN + " The ndings show e ciency was di erent. " * 12
    assert tq.assess_page(text).status == "damaged"


def test_other_languages_and_sparse_pages_are_not_assessed():
    assert tq.assess_page(FRENCH).status == "not_assessed"
    assert tq.assess_page("Figure 3. Box office, 1946-1960.").status == "not_assessed"


def _pdf(*pages):
    document = fitz.open()
    for text in pages:
        document.new_page().insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


def _fake_ocr(texts, confidence=95.0, fallback=None):
    """texts: page -> OCR text in mode 6; fallback: page -> (text, confidence) in mode 3."""
    def ocr(data, indexes, mode=6):
        def page(i):
            text, conf = (texts[i], confidence) if mode == 6 else (fallback or {}).get(i, (texts[i], confidence))
            return {"page_index": i, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "character_count": len(text), "mean_word_confidence": conf, "render_sha256": "r" * 64}
        return {"engine": "tesseract", "engine_version": "tesseract 5.3.0", "language": "eng", "render_dpi": 300,
                "page_segmentation_mode": mode, "parent_content_sha256": hashlib.sha256(data).hexdigest(),
                "pages": [page(i) for i in indexes]}
    return ocr


def test_a_clean_pdf_needs_no_repair():
    data = _pdf(CLEAN, CLEAN)
    receipt = repair.build_receipt(data, ocr_pages=lambda *_: pytest.fail("no OCR expected"))
    assert receipt["status"] == "clean" and receipt["damaged"] == []
    assert repair.validated_repairs(receipt, hashlib.sha256(data).hexdigest()) is None


def test_damaged_pages_are_repaired_and_bound_to_the_bytes():
    data = _pdf(CLEAN, DAMAGED)
    receipt = repair.build_receipt(data, ocr_pages=_fake_ocr({1: CLEAN}))
    assert receipt["status"] == "repaired" and [d["page_index"] for d in receipt["damaged"]] == [1]
    repairs = repair.validated_repairs(receipt, hashlib.sha256(data).hexdigest())
    assert repairs.texts == {1: CLEAN}
    # Any edit to the receipt, or other bytes, voids it.
    assert repair.validated_repairs(receipt, "0" * 64) is None
    receipt["repaired"][0]["text"] = "tampered"
    assert repair.validated_repairs(receipt, hashlib.sha256(data).hexdigest()) is None


@pytest.mark.parametrize("ocr_text,confidence,reason", [
    (DAMAGED, 95.0, "still_damaged_after_ocr"),
    (CLEAN, 55.0, "ocr_confidence_below_floor"),
])
def test_a_page_that_stays_damaged_makes_the_source_unusable(ocr_text, confidence, reason):
    receipt = repair.build_receipt(_pdf(CLEAN, DAMAGED), ocr_pages=_fake_ocr({1: ocr_text}, confidence))
    assert receipt["status"] == "unusable" and receipt["unusable"][0]["reason"] == reason


def test_a_page_with_pictures_is_repaired_by_the_recorded_layout_fallback():
    """Film stills made mode 6 read noise on three Thompson pages; mode 3 read them cleanly."""
    ocr = _fake_ocr({1: DAMAGED}, confidence=55.0, fallback={1: (CLEAN, 96.0)})
    receipt = repair.build_receipt(_pdf(CLEAN, DAMAGED), ocr_pages=ocr)
    assert receipt["status"] == "repaired"
    assert receipt["repaired"][0]["page_segmentation_mode"] == 3


def test_an_ocr_failure_makes_the_source_unusable():
    from app.services.ocr_derivative import OcrDerivativeError

    def fail(*_):
        raise OcrDerivativeError("down")
    receipt = repair.build_receipt(_pdf(DAMAGED), ocr_pages=fail)
    assert receipt["status"] == "unusable" and receipt["unusable"][0]["reason"] == "ocr_failed"


# --- authorization and extraction -------------------------------------------------------------

def _admitted(tmp_session, content):
    from tests.unit.test_verification_evidence import MemoryStorage, _admit
    storage = MemoryStorage()
    return storage, _admit(tmp_session, storage, content)


@pytest.fixture
def check_on(monkeypatch):
    monkeypatch.setattr(settings, "SOURCE_TEXT_QUALITY_CHECK_ENABLED", True)


def _session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.models import Base
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _receipt(content, ocr_text=CLEAN):
    return repair.build_receipt(content, ocr_pages=_fake_ocr({1: ocr_text}))


def _authorize(session, storage, record):
    from app.services.verification_evidence import authorize_representation
    return authorize_representation(session, storage, representation_id=record.id,
                                    scope_type="personal_owner", scope_id="owner-1")


def test_repaired_text_is_what_extraction_reads(check_on, monkeypatch):
    from app.services.verification_evidence import _extract_pages
    original = repair.ensure_receipt
    monkeypatch.setattr(repair, "ensure_receipt",
                        lambda s, rid, content, sha: original(s, rid, content, sha, build=_receipt))
    with _session() as session:
        storage, record = _admitted(session, _pdf(CLEAN, DAMAGED))
        pages, limitations = _extract_pages(_authorize(session, storage, record))
    assert "tl1e" not in pages[1].text and "tlie" not in pages[1].text
    assert any("re-read by local OCR" in item for item in limitations)


def test_an_unusable_source_is_rejected_and_never_authorized(check_on, monkeypatch):
    from app.models.source_repository import SourceRepresentationRecord
    from app.services.verification_evidence import EvidenceAuthorizationError
    original = repair.ensure_receipt
    monkeypatch.setattr(repair, "ensure_receipt", lambda s, rid, content, sha: original(
        s, rid, content, sha, build=lambda data: _receipt(data, ocr_text=DAMAGED)))
    with _session() as session:
        storage, record = _admitted(session, _pdf(CLEAN, DAMAGED))
        with pytest.raises(EvidenceAuthorizationError):
            _authorize(session, storage, record)
        session.expire_all()
        stored = session.get(SourceRepresentationRecord, record.id)
        assert stored.admission_state == "rejected" and stored.cleanliness_verdict == "unusable_text"
        # Accepting it again by hand does not bring the damaged text back into use.
        stored.admission_state = "accepted"
        session.commit()
        with pytest.raises(EvidenceAuthorizationError):
            _authorize(session, storage, record)



@pytest.mark.parametrize("text,expected", [
    ("the stu-\ndio system", "the studio system"),
    ("a well-\nknown film", "a well-known film"),       # both halves are words; the join is not
    ("one line\nnext line", "one line next line"),
    ("Light-\ning and tonality", "Lighting and tonality"),
])
def test_readable_text_joins_line_breaks(monkeypatch, text, expected):
    words = WORDS | {"studio", "well", "known", "lighting", "light", "line", "next", "one", "tonality"}
    monkeypatch.setattr(tq, "_word_list", lambda: (frozenset(words), "w" * 64))
    assert tq.readable_text(text) == expected


def test_words_a_fresh_ocr_also_reads_are_confirmed_not_damage():
    """Pallant (2010): "cel"/"cels" (animation cels) look like OCR errors but a fresh
    OCR reads them too, so the page is kept as it is and the source is not rejected."""
    page = CLEAN.replace("studio", "cel", 3).replace("films", "cels", 3)
    receipt = repair.build_receipt(_pdf(CLEAN, page), ocr_pages=_fake_ocr({1: page}))
    assert receipt["damaged"] and receipt["status"] == "clean"
    assert receipt["confirmed"][0]["confirmed_words"] == ["cel", "cels"] and not receipt["repaired"]


def test_agreement_does_not_excuse_words_the_ocr_read_differently():
    receipt = repair.build_receipt(_pdf(CLEAN, DAMAGED), ocr_pages=_fake_ocr({1: CLEAN}))
    assert receipt["status"] == "repaired" and not receipt["confirmed"]


def test_transient_sources_repair_once_per_content():
    calls = []
    data = _pdf(CLEAN, DAMAGED)
    sha = hashlib.sha256(data).hexdigest()
    repair._TRANSIENT_RECEIPTS.clear()

    def build(content):
        calls.append(1)
        return _receipt(content)
    first = repair.transient_receipt(data, sha, build=build)
    second = repair.transient_receipt(data, sha, build=build)
    assert first["status"] == "repaired" and second is first and len(calls) == 1


def test_unusable_text_is_its_own_refusal_so_the_paper_continues():
    from app.services.verification_evidence import EvidenceAuthorizationError, SourceTextUnusable
    from app.services.verification_run import TransientSourceTextUnusable, VerificationRunError
    assert issubclass(SourceTextUnusable, EvidenceAuthorizationError)
    assert issubclass(TransientSourceTextUnusable, VerificationRunError)
    import inspect
    from app.services import paper_workflow
    source = inspect.getsource(paper_workflow)
    assert source.count('reason_code": "source_text_unusable"') == 2
    assert 'reason_code="source_text_unusable"' in source
