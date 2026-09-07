"""Opt-in pure-scan OCR intake boundary regressions."""

import hashlib

from app.services import pure_scan_ocr
from app.services.ocr_derivative import OcrDerivative, OcrPageResult
from app.services.pure_scan_ocr import _page_mapping, prepare_pure_scan_ocr
from app.services.source_validator import validate_ocr_derivative_text


def _derivative(*, confidence: float = 91.0) -> OcrDerivative:
    texts = (
        "Rejecting the Center: Radical Grassroots Politics in the 1970s\n"
        "Joshua Zeitz\nJournal article source material published in 2008.\n",
        "673\nSubstantive article body text with enough words for verification.\n",
        "674\nAdditional substantive source text continues on this page.\n",
    )
    content = "\f".join(texts).encode("utf-8")
    page_results = tuple(
        OcrPageResult(
            page_index=index,
            page_label=None if index == 0 else str(672 + index),
            render_sha256=f"{index + 1:x}" * 64,
            page_label_render_sha256=f"{index + 4:x}" * 64,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            character_count=len(text),
            mean_word_confidence=confidence,
        )
        for index, text in enumerate(texts)
    )
    manifest = {
        "version": "local-pdf-ocr-derivative-v1",
        "parent_content_sha256": "a" * 64,
        "derivative_content_sha256": hashlib.sha256(content).hexdigest(),
        "engine": "tesseract",
        "engine_version": "test",
        "language": "eng",
        "pages": [item.__dict__ for item in page_results],
    }
    return OcrDerivative(
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
        parent_content_sha256="a" * 64,
        page_labels=tuple(item.page_label for item in page_results),
        page_results=page_results,
        manifest=manifest,
        manifest_sha256="b" * 64,
    )


def test_pure_scan_ocr_is_disabled_without_work(monkeypatch) -> None:
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(pure_scan_ocr, "build_isolated_pdf_ocr_derivative", unexpected)

    result = prepare_pure_scan_ocr(b"%PDF-private", safety_verified=True, enabled=False)

    assert result.status == "not_enabled"
    assert not called


def test_pure_scan_ocr_requires_prior_safety_inspection() -> None:
    result = prepare_pure_scan_ocr(b"%PDF-private", safety_verified=False, enabled=True)

    assert result.status == "safety_required"


def test_expected_range_page_mapping_uses_consistent_observed_anchors() -> None:
    labels = (
        None,
        None,
        None,
        "675",
        None,
        "677",
        None,
        "679",
        None,
        "681",
        None,
        "683",
        None,
        "685",
        None,
        "687",
        None,
    )

    effective, method, _digest = _page_mapping(
        labels, expected_page_range=(673, 688)
    )

    assert effective[0] is None
    assert effective[1:] == tuple(str(value) for value in range(673, 689))
    assert method == "observed_plus_consistent_expected_range_inference"


def test_pure_scan_ocr_prepares_validated_parent_derivative_pair(monkeypatch) -> None:
    derivative = _derivative()
    monkeypatch.setattr(pure_scan_ocr, "_check_text_quality", lambda _content: "pure_scan")
    monkeypatch.setattr(
        pure_scan_ocr,
        "_check_completeness",
        lambda *_args, **_kwargs: ("complete", 3),
    )
    monkeypatch.setattr(
        pure_scan_ocr,
        "build_isolated_pdf_ocr_derivative",
        lambda *_args, **_kwargs: derivative,
    )

    result = prepare_pure_scan_ocr(
        b"%PDF-parent-bytes",
        safety_verified=True,
        expected_title="Rejecting the Center: Radical Grassroots Politics in the 1970s",
        expected_author="Zeitz, Joshua",
        expected_year="2008",
        document_kind="article",
        enabled=True,
    )

    assert result.status == "ready"
    assert result.parent is not None and result.parent.content.startswith(b"%PDF")
    assert result.derivative is not None
    assert result.derivative.original_kind.value == "pdf"
    assert result.validation is not None and result.validation.accept


def test_pure_scan_ocr_rejects_low_confidence_derivative(monkeypatch) -> None:
    monkeypatch.setattr(pure_scan_ocr, "_check_text_quality", lambda _content: "pure_scan")
    monkeypatch.setattr(
        pure_scan_ocr,
        "_check_completeness",
        lambda *_args, **_kwargs: ("complete", 3),
    )
    monkeypatch.setattr(
        pure_scan_ocr,
        "build_isolated_pdf_ocr_derivative",
        lambda *_args, **_kwargs: _derivative(confidence=20.0),
    )

    result = prepare_pure_scan_ocr(
        b"%PDF-parent-bytes", safety_verified=True, enabled=True
    )

    assert result.status == "ocr_failed"


def test_ocr_identity_matches_given_name_surname_to_initialled_byline() -> None:
    result = validate_ocr_derivative_text(
        "Rejecting the Center: Radical Grassroots Politics in the 1970s — "
        "Second-Wave Feminism as a Case Study\n"
        "Author(s): J. Zeitz\n"
        + "Complete article body text. " * 10,
        expected_title=(
            "Rejecting the Center: Radical Grassroots Politics in the 1970s—"
            "Second-Wave Feminism as a Case Study"
        ),
        expected_author="Joshua Zeitz",
        completeness="complete",
        page_count=17,
    )

    assert result.accept
    assert result.identity_confidence == "high"
