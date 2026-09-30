"""Page-level text check and OCR repair for PDF sources (owner decision 2026-09-28).

Every page of a PDF is scored by `text_quality.assess_page`. A damaged page is
re-OCRed locally (Tesseract, the validated English mode, in an isolated child
process). The repaired page is used only if its OCR text passes the same check
and the existing OCR confidence floor. If any damaged page cannot be repaired,
the source's text cannot be used and the representation is rejected; nobody is
asked to review it (owner decision: no human approval inside the app).

The result is a receipt stored on the representation's validation evidence,
bound to the exact PDF bytes, the word lists, the engine and each page's render
and text hashes (the extraction contract's per-page fallback rule). Extraction
substitutes a repaired page only from a receipt that validates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import fitz

from app.config import settings
from app.services import text_quality

logger = logging.getLogger(__name__)

PAGE_REPAIR_VERSION = "page-ocr-repair-v2"
RECEIPT_KEY = "page_text_quality"
REPAIR_DPI = 300
MAX_REPAIR_PAGES = 600
TOTAL_TIMEOUT_SECONDS = 1800
PAGE_TIMEOUT_SECONDS = 45
_SUBSTANTIVE_CHARACTERS = 100   # the existing OCR intake floor applies at or above this


class PageTextUnusable(RuntimeError):
    """The source has pages whose text is damaged and could not be repaired."""


@dataclass(frozen=True)
class PageRepairs:
    """Validated page substitutions for extraction: page index -> repaired text."""
    texts: dict[int, str]
    manifest_sha256: str


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_sha256(receipt: dict) -> str:
    return _sha256(_canonical({k: v for k, v in receipt.items() if k != "manifest_sha256"}))


def _confident(page: dict) -> bool:
    confidence = page.get("mean_word_confidence")
    return (page.get("character_count") or 0) < _SUBSTANTIVE_CHARACTERS or (
        type(confidence) in (int, float) and 0 <= confidence <= 100
        and confidence >= settings.PURE_SCAN_OCR_MIN_MEAN_WORD_CONFIDENCE)


def _agreed(native: str, page: dict) -> frozenset[str]:
    """Suspect words both the embedded text and the fresh OCR read: they are on the page."""
    both = frozenset(text_quality.damaged_words(native)) & frozenset(
        text_quality.damaged_words(page.get("text") or ""))
    return frozenset(word for word in both if text_quality.confirmable(word))


def _confirms(native: str, page: dict) -> bool:
    """An independent confident OCR read the same suspect words: the embedded page is right."""
    agreed = _agreed(native, page)
    return bool(agreed) and _confident(page) and \
        text_quality.assess_page(native, agreed).status != "damaged"


def _usable_repair(page: dict, agreed: frozenset[str] = frozenset()) -> tuple[bool, str, float | None]:
    quality = text_quality.assess_page(page.get("text") or "", agreed)
    if quality.status == "damaged":
        return False, "still_damaged_after_ocr", quality.damaged_share
    confidence = page.get("mean_word_confidence")
    floor = settings.PURE_SCAN_OCR_MIN_MEAN_WORD_CONFIDENCE
    if (page.get("character_count") or 0) >= _SUBSTANTIVE_CHARACTERS and (
            type(confidence) not in (int, float) or not 0 <= confidence <= 100 or confidence < floor):
        return False, "ocr_confidence_below_floor", quality.damaged_share
    return True, "repaired", quality.damaged_share


def build_receipt(pdf_bytes: bytes, *, ocr_pages=None) -> dict:
    """Assess every page and repair the damaged ones. `ocr_pages` is injectable for tests."""
    from app.services.ocr_derivative import OcrDerivativeError, ocr_pdf_pages_isolated
    ocr_pages = ocr_pages or (lambda data, indexes, mode=6: ocr_pdf_pages_isolated(
        data, indexes, dpi=REPAIR_DPI, page_segmentation_mode=mode,
        timeout_seconds_per_page=PAGE_TIMEOUT_SECONDS, total_timeout_seconds=TOTAL_TIMEOUT_SECONDS))
    receipt = {
        "version": PAGE_REPAIR_VERSION, "text_quality_version": text_quality.TEXT_QUALITY_VERSION,
        "word_list_sha256": text_quality.word_list_sha256(), "threshold": text_quality.DAMAGED_WORD_SHARE,
        "parent_content_sha256": _sha256(pdf_bytes), "pages_total": 0, "pages_assessed": 0,
        "damaged": [], "confirmed": [], "repaired": [], "unusable": [], "ocr": None, "status": "clean",
    }
    native: dict[int, str] = {}
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        receipt["pages_total"] = document.page_count
        for index, page in enumerate(document):
            quality = text_quality.assess_page(page.get_text())
            if quality.status == "not_assessed":
                continue
            receipt["pages_assessed"] += 1
            if quality.status == "damaged":
                receipt["damaged"].append({"page_index": index, "damaged_share": quality.damaged_share})
                native[index] = page.get_text()
    finally:
        document.close()
    damaged = [item["page_index"] for item in receipt["damaged"]]
    if damaged:
        before = {item["page_index"]: item["damaged_share"] for item in receipt["damaged"]}
        if len(damaged) > MAX_REPAIR_PAGES:
            receipt["unusable"] = [{"page_index": i, "reason": "too_many_damaged_pages",
                                    "damaged_share_before": before[i]} for i in damaged]
        else:
            outputs = {}
            try:
                outputs[6] = ocr_pages(pdf_bytes, damaged)
            except OcrDerivativeError as exc:
                logger.warning("Page OCR repair failed (type=%s)", type(exc).__name__)
            first = outputs.get(6)
            if first is None or first.get("parent_content_sha256") != receipt["parent_content_sha256"]:
                receipt["unusable"] = [{"page_index": i, "reason": "ocr_failed",
                                        "damaged_share_before": before[i]} for i in damaged]
            else:
                receipt["ocr"] = {k: first.get(k) for k in ("engine", "engine_version", "language", "render_dpi")}
                results = {page["page_index"]: (6, page) for page in first.get("pages") or []}
                # Recorded per-page fallback: automatic layout (mode 3) for pages where
                # pictures made mode 6 read noise (measured on Thompson, 2026-09-28).
                retry = [i for i, (_, page) in results.items()
                         if not _confirms(native[i], page) and not _usable_repair(page, _agreed(native[i], page))[0]]
                if retry:
                    try:
                        second = ocr_pages(pdf_bytes, retry, 3)
                    except OcrDerivativeError as exc:
                        logger.warning("Page OCR fallback failed (type=%s)", type(exc).__name__)
                        second = None
                    if second is not None and second.get("parent_content_sha256") == receipt["parent_content_sha256"]:
                        for page in second.get("pages") or []:
                            index = page["page_index"]
                            if _confirms(native[index], page) or _usable_repair(page, _agreed(native[index], page))[0]:
                                results[index] = (3, page)
                for index in damaged:
                    if index not in results:
                        receipt["unusable"].append({"page_index": index, "reason": "ocr_missing_page",
                                                    "damaged_share_before": before[index]})
                        continue
                    mode, page = results[index]
                    agreed = _agreed(native[index], page)
                    if _confirms(native[index], page):
                        # Not damage: the flagged words are really on the page ("cel").
                        receipt["confirmed"].append({
                            "page_index": index, "page_segmentation_mode": mode,
                            "damaged_share_before": before[index], "confirmed_words": sorted(agreed)[:20],
                            "mean_word_confidence": page.get("mean_word_confidence"),
                            "render_sha256": page.get("render_sha256"), "text_sha256": page.get("text_sha256")})
                        continue
                    ok, reason, after = _usable_repair(page, agreed)
                    entry = {"page_index": index, "page_segmentation_mode": mode,
                             "damaged_share_before": before[index], "damaged_share_after": after,
                             "mean_word_confidence": page.get("mean_word_confidence"),
                             "render_sha256": page.get("render_sha256"), "text_sha256": page.get("text_sha256")}
                    if ok:
                        receipt["repaired"].append({**entry, "text": page.get("text") or ""})
                    else:
                        receipt["unusable"].append({**entry, "reason": reason})
        receipt["status"] = ("unusable" if receipt["unusable"]
                             else "repaired" if receipt["repaired"] else "clean")
    receipt["manifest_sha256"] = _manifest_sha256(receipt)
    return receipt


def validated_repairs(receipt: dict | None, content_sha256: str) -> PageRepairs | None:
    """The page substitutions a receipt authorizes, or None if it does not validate."""
    if not isinstance(receipt, dict) or receipt.get("version") != PAGE_REPAIR_VERSION:
        return None
    if receipt.get("parent_content_sha256") != content_sha256 or receipt.get("status") != "repaired":
        return None
    if receipt.get("manifest_sha256") != _manifest_sha256(receipt):
        return None
    texts = {}
    for item in receipt.get("repaired") or []:
        text = item.get("text")
        if not isinstance(text, str) or _sha256(text.encode("utf-8")) != item.get("text_sha256"):
            return None
        texts[int(item["page_index"])] = text
    return PageRepairs(texts=texts, manifest_sha256=receipt["manifest_sha256"])


def current_receipt(validation_evidence: dict | None, content_sha256: str) -> dict | None:
    """A stored receipt for these bytes under the current check, if there is one."""
    receipt = (validation_evidence or {}).get(RECEIPT_KEY)
    if (not isinstance(receipt, dict) or receipt.get("version") != PAGE_REPAIR_VERSION
            or receipt.get("text_quality_version") != text_quality.TEXT_QUALITY_VERSION
            or receipt.get("parent_content_sha256") != content_sha256
            or receipt.get("manifest_sha256") != _manifest_sha256(receipt)):
        return None
    return receipt


def ensure_receipt(session, record_id, content: bytes, content_sha256: str, *, build=build_receipt) -> dict:
    """Return the stored receipt, building and storing it once if missing.

    Runs in its own session on the caller's engine so the row lock and the
    commit do not touch the caller's transaction. An unusable result rejects
    the representation: it stays rejected whatever its admission state says.
    """
    from sqlalchemy.orm import Session
    from app.models.source_repository import SourceRepresentationRecord

    def store(target, record) -> dict:
        receipt = current_receipt(record.validation_evidence, content_sha256)
        if receipt is None:
            receipt = build(content)
            evidence = dict(record.validation_evidence or {})
            evidence[RECEIPT_KEY] = receipt
            record.validation_evidence = evidence
            if receipt["status"] == "unusable":
                record.admission_state = "rejected"
                record.cleanliness_verdict = "unusable_text"
        return receipt

    with Session(bind=session.get_bind()) as own:
        record = own.query(SourceRepresentationRecord).filter_by(id=record_id).with_for_update().one_or_none()
        if record is not None:
            receipt = store(own, record)
            own.commit()
            return receipt
    # Admitted in the caller's still-open transaction: record it there.
    record = session.get(SourceRepresentationRecord, record_id)
    receipt = store(session, record)
    session.flush()
    return receipt


_TRANSIENT_RECEIPTS: dict[str, dict] = {}
_TRANSIENT_CACHE_SIZE = 8


def transient_receipt(content: bytes, content_sha256: str, *, build=build_receipt) -> dict:
    """The receipt for a transient source, which has no stored record to keep it on.

    Cached in this process by content hash, so one run's repeated loads repair once.
    """
    receipt = _TRANSIENT_RECEIPTS.get(content_sha256)
    if receipt is None or current_receipt({RECEIPT_KEY: receipt}, content_sha256) is None:
        receipt = build(content)
        if len(_TRANSIENT_RECEIPTS) >= _TRANSIENT_CACHE_SIZE:
            _TRANSIENT_RECEIPTS.pop(next(iter(_TRANSIENT_RECEIPTS)))
        _TRANSIENT_RECEIPTS[content_sha256] = receipt
    return receipt
