"""Development-only identity observations, never replacement source evidence.

Callers retain source authorization/lifecycle responsibility. This builder scans
the exact bytes before rendering, grants no identity/admission status, and has
no repository, report, or evidence-package integration.
"""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Literal
import csv
import io
import math

import fitz
from pydantic import BaseModel, ConfigDict, Field

from app.services.file_safety import SafetyVerdict, inspect_uploaded_pdf
from app.services.ocr_derivative import (
    OcrDerivativeError, _normalize_ocr_text, _run_tesseract_page, _tesseract_version,
)


# Decimal MB; shared by every identity-observation caller. Raster, page and
# processing-time bounds below remain independent of the source-file ceiling.
MAX_IDENTITY_OCR_SOURCE_BYTES = 25_000_000


class IdentityOcrObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["identity-ocr-observation-v1"] = "identity-ocr-observation-v1"
    purpose: Literal["identity_observation_only"] = "identity_observation_only"
    identity_status: Literal["unverified"] = "unverified"
    grants_admission: Literal[False] = False
    grants_evidence_use: Literal[False] = False
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_index: Literal[0] = 0
    region: tuple[float, float, float, float]
    dpi: Literal[250] = 250
    mode: Literal[3] = 3
    engine_version: str
    renderer_version: str
    text: str = Field(min_length=1, max_length=20000, repr=False)
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IdentityOcrWord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    bbox: tuple[float, float, float, float]
    line: tuple[int, int, int]


class LayoutIdentityOcrObservation(IdentityOcrObservation):
    version: Literal["identity-ocr-observation-v2"] = "identity-ocr-observation-v2"
    words: tuple[IdentityOcrWord, ...] = Field(max_length=10000)


class FrontMatterIdentityOcrObservation(LayoutIdentityOcrObservation):
    version: Literal["identity-ocr-observation-v3"] = "identity-ocr-observation-v3"
    page_index: Literal[0, 1, 2] = 0


def _digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()).hexdigest()


def validate_identity_observation(observation: IdentityOcrObservation, source: bytes) -> None:
    # Revalidation also rejects model_copy/update attempts to change fixed grants.
    model = (FrontMatterIdentityOcrObservation if isinstance(observation, FrontMatterIdentityOcrObservation)
             else LayoutIdentityOcrObservation if isinstance(observation, LayoutIdentityOcrObservation)
             else IdentityOcrObservation)
    value = model.model_validate(observation.model_dump()).model_dump()
    expected = value.pop("observation_sha256")
    if (_digest(value) != expected or sha256(source).hexdigest() != value["source_sha256"]
            or sha256(value["text"].encode()).hexdigest() != value["text_sha256"]):
        raise OcrDerivativeError("Identity observation binding changed.")
    if isinstance(observation, LayoutIdentityOcrObservation):
        previous = 0
        rx0, ry0, rx1, ry1 = observation.region
        if not all(math.isfinite(v) for v in observation.region) or rx1 <= rx0 or ry1 <= ry0:
            raise OcrDerivativeError("Invalid OCR viewport.")
        for word in observation.words:
            x0, y0, x1, y1 = word.bbox
            if (not previous <= word.start < word.end <= len(observation.text)
                    or observation.text[previous:word.start].strip()
                    or not all(math.isfinite(v) for v in word.bbox)
                    or not rx0 <= x0 < x1 <= rx1 + 0.3
                    or not ry0 <= y0 < y1 <= ry1 + 0.3):
                raise OcrDerivativeError("Invalid OCR word binding.")
            previous = word.end
        if observation.text[previous:].strip():
            raise OcrDerivativeError("OCR layout does not cover the observed text.")


def build_identity_observation(source: bytes, *, include_layout: bool = False,
                               page_index: int = 0) -> IdentityOcrObservation:
    """Observe one bounded front-matter page; default v1/v2 behavior unchanged."""
    if type(page_index) is not int or page_index not in (0, 1, 2) or (page_index and not include_layout):
        raise OcrDerivativeError("Identity observation requires a bounded layout page.")
    if not source or len(source) > MAX_IDENTITY_OCR_SOURCE_BYTES:
        raise OcrDerivativeError("Identity observation source exceeds its bound.")
    safety = inspect_uploaded_pdf(source)
    if any(verdict != SafetyVerdict.CLEAN for verdict in
           (safety.verdict, safety.structural_verdict, safety.malware_verdict)):
        raise OcrDerivativeError("Identity observation requires clean source bytes.")
    with fitz.open(stream=source, filetype="pdf") as document:
        if document.needs_pass or page_index >= len(document):
            raise OcrDerivativeError("Identity observation needs an accessible requested page.")
        page = document[page_index]
        matrix = fitz.Matrix(250 / 72, 250 / 72)
        pixels = (page.rect * matrix).irect
        if not 0 < pixels.width * pixels.height <= 25_000_000:
            raise OcrDerivativeError("Identity observation raster exceeds its bound.")
        region = tuple(page.rect)
        png = page.get_pixmap(matrix=matrix, alpha=False).tobytes("png")
    engine = _tesseract_version("tesseract")
    with tempfile.TemporaryDirectory(prefix="identity-ocr-observation-") as directory:
        path = Path(directory) / "page.png"
        path.write_bytes(png)
        text, _ = _run_tesseract_page(path, Path(directory) / "observation",
                                    executable="tesseract", language="eng", dpi=250,
                                    page_segmentation_mode=3, timeout_seconds=10)
        text = _normalize_ocr_text(text)
        words = ()
        if include_layout:
            tsv_path = Path(directory) / "observation.tsv"
            if not tsv_path.exists() or tsv_path.stat().st_size > 2_000_000:
                raise OcrDerivativeError("OCR layout missing or exceeds its bound.")
            words = _bind_layout_words(tsv_path.read_text(encoding="utf-8"), text, region)
    text = _normalize_ocr_text(text)
    model = (FrontMatterIdentityOcrObservation if page_index else
             LayoutIdentityOcrObservation if include_layout else IdentityOcrObservation)
    observation = model(
        source_sha256=sha256(source).hexdigest(), render_sha256=sha256(png).hexdigest(),
        text_sha256=sha256(text.encode()).hexdigest(), region=region,
        engine_version=engine, renderer_version=fitz.VersionBind, text=text,
        observation_sha256="0" * 64,
        page_index=page_index,
        **({"words": words} if include_layout else {}),
    )
    value = observation.model_dump(); value.pop("observation_sha256")
    observation = observation.model_copy(update={"observation_sha256": _digest(value)})
    validate_identity_observation(observation, source)
    return observation


def _bind_layout_words(tsv: str, text: str, region: tuple) -> tuple[IdentityOcrWord, ...]:
    """Bind engine word positions to exact normalized text, or fail closed."""
    words = []
    cursor = 0
    try:
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
            if row["level"] != "5" or not row["text"].strip():
                continue
            token = _normalize_ocr_text(row["text"]).strip()
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if not token or not text.startswith(token, cursor):
                raise OcrDerivativeError("OCR text and layout differ.")
            left, top, width, height = (int(row[k]) for k in ("left", "top", "width", "height"))
            scale = 72 / 250
            words.append(IdentityOcrWord(
                start=cursor, end=cursor + len(token),
                bbox=(region[0] + left * scale, region[1] + top * scale,
                      region[0] + (left + width) * scale, region[1] + (top + height) * scale),
                line=tuple(int(row[k]) for k in ("block_num", "par_num", "line_num")),
            ))
            cursor += len(token)
            if len(words) > 10000:
                raise OcrDerivativeError("OCR word count exceeds its bound.")
    except (KeyError, TypeError, ValueError) as exc:
        raise OcrDerivativeError("Invalid OCR layout output.") from exc
    if text[cursor:].strip():
        raise OcrDerivativeError("OCR layout is incomplete.")
    return tuple(words)


def compare_authorized_identity_observation(session, backend, *, principal,
                                            representation_id: str,
                                            observation: IdentityOcrObservation,
                                            expected_fields: dict[str, str],
                                            expected_source_kind: str | None = None,
                                            expected_source_kind_confidence: str = "unknown",
                                            expected_source_kind_evidence: tuple[str, ...] = ()):
    return compare_authorized_identity_observations(
        session, backend, principal=principal, representation_id=representation_id,
        observations=(observation,), expected_fields=expected_fields,
        expected_source_kind=expected_source_kind,
        expected_source_kind_confidence=expected_source_kind_confidence,
        expected_source_kind_evidence=expected_source_kind_evidence,
    )


def compare_authorized_identity_observations(session, backend, *, principal,
                                             representation_id: str,
                                             observations: tuple[IdentityOcrObservation, ...],
                                             expected_fields: dict[str, str],
                                             expected_source_kind: str | None = None,
                                             expected_source_kind_confidence: str = "unknown",
                                             expected_source_kind_evidence: tuple[str, ...] = ()):
    """Adapt authorized OCR observations to the existing source validator.

    Development-only: front-matter observations cannot establish completeness or
    evidence readability. No repository writes, admission grants or new policy.
    """
    from app.security import REPORT_SOURCE_CAPABILITY
    from app.services.verification_evidence import authorize_representation
    from app.services.source_validator import validate_ocr_derivative_text

    principal.require(REPORT_SOURCE_CAPABILITY)
    source = authorize_representation(session, backend, representation_id=representation_id,
                                     scope_type=principal.scope_type, scope_id=principal.scope_id)
    if not 1 <= len(observations) <= 3 or tuple(o.page_index for o in observations) != tuple(range(len(observations))):
        raise OcrDerivativeError("Identity observations must be contiguous opening pages, at most three.")
    for observation in observations:
        validate_identity_observation(observation, source.content)
    if (set(expected_fields) - {"title", "author", "year", "doi"}
            or any(not isinstance(v, str) or len(v) > 2000 for v in expected_fields.values())):
        raise ValueError("Invalid expected identity fields.")
    layout = {}
    pages = []
    for observation in observations:
        if not isinstance(observation, LayoutIdentityOcrObservation):
            pages.append(([], 0))
            continue
        spans = [{"text": observation.text[w.start:w.end],
                  "bbox": (w.bbox[0] - observation.region[0], w.bbox[1] - observation.region[1],
                           w.bbox[2] - observation.region[0], w.bbox[3] - observation.region[1])}
                 for w in observation.words]
        lines = {}
        for word, span in zip(observation.words, spans):
            group = lines.setdefault(word.line, [])
            group.append(span)
        pages.append((spans, observation.region[3] - observation.region[1]))
        if observation.page_index == 0:
            layout = {"_layout_spans": spans,
                  "_layout_height": observation.region[3] - observation.region[1],
                  "_layout_lines": [(" ".join(s["text"] for s in group),
                                     min(s["bbox"][1] for s in group))
                                    for group in lines.values()]}
    return validate_ocr_derivative_text(
        "\f".join(o.text for o in observations), completeness="uncertain", page_count=len(observations),
        _layout_pages=pages,
        expected_source_kind=expected_source_kind,
        expected_source_kind_confidence=expected_source_kind_confidence,
        expected_source_kind_evidence=expected_source_kind_evidence,
        **{f"expected_{k}": v for k, v in expected_fields.items()}, **layout,
    )
