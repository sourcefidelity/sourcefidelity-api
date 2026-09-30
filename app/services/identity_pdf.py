"""Temporary, safety-gated PDF front-matter observations, not source admission."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import re

import fitz
import httpx

from app.config import settings
from app.services.candidate_budget import require_source_candidate, CandidateBudgetExceeded
from app.services.file_safety import inspect_uploaded_pdf, SafetyVerdict, FileSafetyUnavailable
from app.services.pdf_verifier import _extract_title_from_first_page
from app.services.reference_discovery import build_reference_discovery_candidate
from app.services.reference_review_scope import scope
from app.services.retrieval.base import RetrievalResult
from app.services.safe_fetch import (
    safe_request, UnsafeUrlError, ResponseTooLargeError, ResponseMediaTypeError,
)
from app.services.source_validator import (
    _has_prominent_front_title_support, _detect_nonwork_listing,
    _explicit_first_page_document_years, _normalize_identity_text,
)

POLICY = "identity-only-pdf-metadata-v1"
MAX_FILES = 2
MAX_BYTES = 25 * 1024 * 1024
_ALLOWANCE = ContextVar("identity_pdf_allowance", default=None)


def remaining_inspections():
    allowance = _ALLOWANCE.get()
    return MAX_FILES if allowance is None else max(0, MAX_FILES - allowance[0])


def bounded_pdf_inspection(function):
    @wraps(function)
    def run(*args, **kwargs):
        if _ALLOWANCE.get() is not None:
            return function(*args, **kwargs)
        token = _ALLOWANCE.set([0])
        try:
            return function(*args, **kwargs)
        finally:
            _ALLOWANCE.reset(token)
    return run


def _observed_fields(content):
    """Native three-page front matter only; no reference or search-field fill."""
    with fitz.open(stream=content, filetype="pdf") as document:
        pages = [document[i].get_text()[:12000] for i in range(min(3, len(document)))]
        metadata = document.metadata or {}
    front = "\n".join(pages)[:20000]
    if _detect_nonwork_listing(front, str(metadata.get("title") or "")):
        return None
    for index, text in enumerate(pages):
        # Embedded metadata alone is not evidence that this PDF contains the work.
        title = str(metadata.get("title") or "")[:300].strip()
        if (not title or _normalize_identity_text(title) not in _normalize_identity_text(text)
                or not _has_prominent_front_title_support(content, title)):
            title = _extract_title_from_first_page(content, page_index=index) or ""
        if not title or not _has_prominent_front_title_support(content, title):
            continue
        author = str(metadata.get("author") or "")[:300].strip()
        opening = _normalize_identity_text(text[:2500])
        if not author or _normalize_identity_text(author) not in opening:
            match = re.search(r"(?im)^by[ \t]+([^\n]{3,150})$", text[:2500])
            author = match.group(1).strip() if match else ""
        # A title or generic heading alone cannot adjudicate a work's identity.
        if not author or len(re.findall(r"[^\W\d_]+", author)) < 2:
            continue
        years = _explicit_first_page_document_years(content) if index == 0 else set()
        return dict(title=title, authors=[author], year=next(iter(years)) if len(years) == 1 else None)
    return None


def inspect(location, expected):
    attempt = dict(url=location.url, outcome="not_attempted", reason_code="identity_pdf_limit")
    allowance = _ALLOWANCE.get()
    if allowance is not None:
        if allowance[0] >= MAX_FILES:
            return attempt
        allowance[0] += 1
    response = None
    try:
        require_source_candidate(location.url)
        response = safe_request(location.url, usage_label="file download", timeout=20,
            max_bytes=min(MAX_BYTES, settings.MAX_FILE_SIZE_MB * 1024 * 1024),
            headers={"Accept": "application/pdf"},
            allowed_media_types=frozenset({"application/pdf"}))
        if not 200 <= response.status_code < 300:
            raise ValueError("identity_pdf_http_status")
        safety = inspect_uploaded_pdf(response.content)
        if safety.verdict != SafetyVerdict.CLEAN:
            attempt.update(outcome="identity_unconfirmed", reason_code="identity_pdf_safety_not_clean")
            return attempt
        fields = _observed_fields(response.content)
        if not fields:
            attempt.update(outcome="identity_unconfirmed", reason_code="identity_pdf_bibliography_unavailable")
            return attempt
        candidate = build_reference_discovery_candidate(attempt_id="pdf-inspection",
            provider="independent_pdf_metadata", expected=expected,
            result=RetrievalResult(source_name="independent_pdf_metadata", success=True, **fields))
        different = (any(c.field_name == "title" and c.outcome == "material_conflict"
                         for c in candidate.comparisons)
                     and scope(expected.title, fields["title"]) == "outside_bound")
        confirmed = (not expected.reference_parse_review and candidate.plausible_identity_match
            and not candidate.has_material_conflict and not candidate.has_unresolved_supplied_identity_fields
            and candidate.agreement_count >= 2)
        observation = dict(observed=fields, content_sha256=hashlib.sha256(response.content).hexdigest(),
            source_url=str(response.url), identity_evidence_kind=(
                "pdf_front_matter_metadata" if confirmed else "pdf_front_matter_observation"))
        # Reuse the independent-observation envelope and transient cleanup, not
        # a source representation. The typed kind distinguishes PDF from HTML.
        attempt.update(outcome="unavailable" if confirmed else "identity_rejected" if different else "identity_unconfirmed",
            reason_code="identity_pdf_metadata_observed",
            **{"landing_metadata_identity" if confirmed else "landing_metadata_observation": observation})
    except CandidateBudgetExceeded:
        attempt["reason_code"] = "candidate_budget_exhausted"
    except (ResponseMediaTypeError, ResponseTooLargeError, FileSafetyUnavailable) as exc:
        reason = ("non_pdf_response" if isinstance(exc, ResponseMediaTypeError) else
                  "response_too_large" if isinstance(exc, ResponseTooLargeError) else "safety_unavailable")
        attempt.update(outcome="unavailable", reason_code="identity_pdf_" + reason)
    except UnsafeUrlError:
        attempt.update(outcome="transport_failure", reason_code="identity_pdf_unsafe_url")
    except (httpx.TimeoutException, TimeoutError):
        attempt.update(outcome="transport_failure", reason_code="identity_pdf_timeout")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        attempt.update(outcome="access_restricted" if status in {401, 403} else "transport_failure",
                       reason_code=f"identity_pdf_http_{status}")
    except httpx.RequestError:
        attempt.update(outcome="transport_failure", reason_code="identity_pdf_transport_failure")
    except Exception:
        attempt.update(outcome="identity_unconfirmed", reason_code="identity_pdf_inspection_failed")
    finally:
        # No filesystem/cache/repository writes; release the in-memory body on
        # success, failure and interruption. This is not cryptographic erasure.
        if response is not None:
            response.close()
            response._content = b""
    return attempt
