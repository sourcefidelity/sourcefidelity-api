"""Opt-in, fail-closed preparation of pure-scan OCR source pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math

from app.config import settings
from app.services.ocr_derivative import (
    OcrDerivative,
    OcrDerivativeError,
    build_isolated_pdf_ocr_derivative,
)
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.source_validator import (
    ValidationResult,
    _check_completeness,
    _check_text_quality,
    validate_ocr_derivative_text,
)


@dataclass(frozen=True)
class PreparedPureScanOcr:
    status: str
    reason: str
    parent: SourceRepresentation | None = None
    derivative: SourceRepresentation | None = None
    derivative_record: OcrDerivative | None = None
    validation: ValidationResult | None = None
    page_labels: tuple[str | None, ...] = ()
    page_mapping_method: str = "not_assessed"
    page_mapping_sha256: str | None = None


def _page_mapping(
    labels: tuple[str | None, ...],
    *,
    expected_page_range: tuple[int, int] | None,
) -> tuple[tuple[str | None, ...], str, str]:
    numeric = [
        (index, int(label))
        for index, label in enumerate(labels)
        if label is not None and label.isdigit()
    ]
    effective = list(labels)
    method = "observed_only"
    if len(numeric) >= 3:
        offsets = Counter(label - index for index, label in numeric)
        offset, support = offsets.most_common(1)[0]
        if support == len(numeric):
            if expected_page_range is not None:
                start, end = expected_page_range
                for index in range(len(effective)):
                    inferred = offset + index
                    if effective[index] is None and start <= inferred <= end:
                        effective[index] = str(inferred)
                observed_range = {
                    int(label)
                    for label in effective
                    if label is not None and label.isdigit() and start <= int(label) <= end
                }
                if observed_range != set(range(start, end + 1)):
                    raise OcrDerivativeError(
                        "OCR page mapping did not cover the expected source range."
                    )
                method = "observed_plus_consistent_expected_range_inference"
            else:
                first_index = min(index for index, _label in numeric)
                last_index = max(index for index, _label in numeric)
                for index in range(first_index, last_index + 1):
                    if effective[index] is None:
                        effective[index] = str(offset + index)
                method = "observed_plus_bounded_interpolation"
    payload = {
        "method": method,
        "labels": effective,
        "observed_labels": list(labels),
    }
    mapping_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(effective), method, mapping_sha256


def prepare_pure_scan_ocr(
    pdf_bytes: bytes,
    *,
    safety_verified: bool,
    source_url: str | None = None,
    expected_doi: str | None = None,
    expected_title: str | None = None,
    expected_author: str | None = None,
    expected_year: str | None = None,
    expected_isbn: str | None = None,
    document_kind: str = "unknown",
    expected_page_range: tuple[int, int] | None = None,
    expected_source_kind: str | None = None,
    expected_source_kind_confidence: str = "unknown",
    expected_source_kind_evidence: tuple[str, ...] = (),
    enabled: bool | None = None,
) -> PreparedPureScanOcr:
    """Create a validated parent/derivative pair or return a neutral failure."""
    active = settings.PURE_SCAN_OCR_ENABLED if enabled is None else enabled
    if not active:
        return PreparedPureScanOcr("not_enabled", "Pure-scan OCR is disabled.")
    if not safety_verified:
        return PreparedPureScanOcr(
            "safety_required",
            "Pure-scan OCR requires completed hostile-file inspection.",
        )
    if _check_text_quality(pdf_bytes) != "pure_scan":
        return PreparedPureScanOcr(
            "not_pure_scan",
            "The source is not an eligible image-only PDF.",
        )
    is_article = document_kind == "article"
    completeness, page_count = _check_completeness(
        pdf_bytes,
        is_article,
        document_kind=document_kind,
        expected_page_range=expected_page_range,
        isbn=expected_isbn,
        title=expected_title,
        author=expected_author,
    )
    if completeness != "complete":
        return PreparedPureScanOcr(
            "completeness_unconfirmed",
            "The immutable parent PDF did not establish complete-source coverage.",
        )
    try:
        derivative = build_isolated_pdf_ocr_derivative(
            pdf_bytes,
            language=settings.PURE_SCAN_OCR_LANGUAGE,
            dpi=settings.PURE_SCAN_OCR_DPI,
            page_segmentation_mode=settings.PURE_SCAN_OCR_PAGE_SEGMENTATION_MODE,
            max_pages=settings.PURE_SCAN_OCR_MAX_PAGES,
            max_pixels_per_page=settings.PURE_SCAN_OCR_MAX_PIXELS_PER_PAGE,
            max_total_pixels=settings.PURE_SCAN_OCR_MAX_TOTAL_PIXELS,
            timeout_seconds_per_page=settings.PURE_SCAN_OCR_PAGE_TIMEOUT_SECONDS,
            total_timeout_seconds=settings.PURE_SCAN_OCR_TOTAL_TIMEOUT_SECONDS,
            max_derivative_bytes=settings.PURE_SCAN_OCR_MAX_DERIVATIVE_MB * 1_000_000,
            executable=settings.PURE_SCAN_OCR_EXECUTABLE,
        )
        substantive_confidences = [
            page.mean_word_confidence
            for page in derivative.page_results
            if page.character_count >= 100
        ]
        if not substantive_confidences or not all(
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0 <= confidence <= 100
            and confidence >= settings.PURE_SCAN_OCR_MIN_MEAN_WORD_CONFIDENCE
            for confidence in substantive_confidences
        ):
            raise OcrDerivativeError(
                "OCR confidence did not satisfy the configured evidence floor."
            )
        labels, mapping_method, mapping_sha256 = _page_mapping(
            derivative.page_labels,
            expected_page_range=expected_page_range,
        )
    except OcrDerivativeError:
        return PreparedPureScanOcr(
            "ocr_failed",
            "The OCR derivative did not satisfy the configured safety and quality checks.",
        )
    validation = validate_ocr_derivative_text(
        derivative.content.decode("utf-8", "strict"),
        expected_doi=expected_doi,
        expected_title=expected_title,
        expected_author=expected_author,
        expected_year=expected_year,
        expected_source_kind=expected_source_kind,
        expected_source_kind_confidence=expected_source_kind_confidence,
        expected_source_kind_evidence=expected_source_kind_evidence,
        completeness=completeness,
        page_count=page_count,
    )
    if not validation.accept:
        return PreparedPureScanOcr(
            "identity_unconfirmed",
            validation.reason,
            derivative_record=derivative,
            validation=validation,
            page_labels=labels,
            page_mapping_method=mapping_method,
            page_mapping_sha256=mapping_sha256,
        )
    parent = SourceRepresentation(
        kind=RepresentationKind.PDF,
        media_type="application/pdf",
        content=pdf_bytes,
        source_url=source_url,
        completeness=completeness,
    )
    derivative_representation = SourceRepresentation(
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
        content=derivative.content,
        source_url=source_url,
        original_kind=RepresentationKind.PDF,
        completeness=completeness,
        metadata={
            "ocr_derivative_version": derivative.manifest["version"],
            "parent_content_sha256": derivative.parent_content_sha256,
            "derivation_manifest_sha256": derivative.manifest_sha256,
            "page_labels": list(labels),
            "page_mapping_method": mapping_method,
            "page_mapping_sha256": mapping_sha256,
            "ocr_engine": derivative.manifest["engine"],
            "ocr_engine_version": derivative.manifest["engine_version"],
            "ocr_language": derivative.manifest["language"],
        },
    )
    return PreparedPureScanOcr(
        "ready",
        validation.reason,
        parent=parent,
        derivative=derivative_representation,
        derivative_record=derivative,
        validation=validation,
        page_labels=labels,
        page_mapping_method=mapping_method,
        page_mapping_sha256=mapping_sha256,
    )
