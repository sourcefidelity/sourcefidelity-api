"""Fail-closed authenticated-principal boundary for protected report delivery."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import quote
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import settings


REPORT_VIEW_CAPABILITY = "report:view"
REPORT_PAPER_CAPABILITY = "report:paper:view"
REPORT_SOURCE_CAPABILITY = "authorized_source_content"
# Starting or retrying a Judgment run (paid model calls; no notice since 2026-09-28).
REPORT_JUDGMENT_CAPABILITY = "report:judgment:run"
PAPER_CHECK_CAPABILITY = "paper:check"
PAPER_STATUS_CAPABILITY = "paper:status:view"
SOURCE_REPOSITORY_READ_CAPABILITY = "source_repository:read"
SOURCE_REPOSITORY_WRITE_CAPABILITY = "source_repository:write"
EDITION_REVIEW_CAPABILITY = "source:edition:review"
# Instructor/administrator: record that marks are released for an assessment
# (assessment_marks.py). Never granted to the report browser session.
ASSESSMENT_MARKS_RELEASE_CAPABILITY = "assessment:marks:release"

_bearer = HTTPBearer(auto_error=False)
REPORT_SESSION_COOKIE = "sourcefidelity_report_session"
_SESSION_VERSION = 1


class AuthenticatedPrincipal(BaseModel):
    """One deployment-resolved identity with an exact authorization scope."""

    provider: str
    subject: str = Field(min_length=1, max_length=255)
    scope_type: str = Field(min_length=1, max_length=100)
    scope_id: str = Field(min_length=1, max_length=255)
    capabilities: frozenset[str]

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def get_report_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedPrincipal:
    """Resolve a report principal; never infer identity from request parameters."""
    return _resolve_report_principal(request, credentials)


def get_authenticated_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedPrincipal:
    """Resolve the authenticated deployment principal for non-report APIs."""
    return _resolve_report_principal(request, credentials)


def require_same_origin_request(request: Request) -> None:
    """Reject browser cross-origin state changes while preserving CLI clients."""
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site == "cross-site":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get(
            "host", ""
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    elif fetch_site not in {None, "none", "same-origin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def get_report_browser_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedPrincipal:
    """Redirect an unauthenticated browser to same-origin Personal login."""
    try:
        return _resolve_report_principal(request, credentials)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED and credentials is None:
            destination = quote(request.url.path, safe="/")
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": f"/report/login?next={destination}"},
            ) from None
        raise


def _resolve_report_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> AuthenticatedPrincipal:
    _require_personal_mode()
    if settings.REPORT_AUTH_MODE == "personal_local":
        return _personal_principal(provider="personal_local")
    if credentials is not None:
        supplied = (
            credentials.credentials
            if credentials.scheme.casefold() == "bearer"
            else ""
        )
        return authenticate_personal_bearer(supplied)
    session_token = request.cookies.get(REPORT_SESSION_COOKIE, "")
    if session_token:
        return verify_report_session(session_token)
    raise _unauthorized()


def authenticate_personal_bearer(supplied: str) -> AuthenticatedPrincipal:
    """Validate the configured Personal credential without logging it."""
    _require_personal_mode()
    expected = _configured_personal_secret()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise _unauthorized()
    return _personal_principal()


def ensure_report_auth_configured() -> None:
    """Fail closed before presenting a login surface that cannot succeed."""
    _require_personal_mode()
    if settings.REPORT_AUTH_MODE == "personal_bearer":
        _configured_personal_secret()


def create_report_session(
    principal: AuthenticatedPrincipal,
    *,
    now: int | None = None,
) -> str:
    """Create a short-lived signed session containing no bearer credential."""
    expected = _personal_principal()
    if principal != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    session_principal = _report_session_principal(provider=principal.provider)
    issued_at = int(time.time() if now is None else now)
    payload = {
        "v": _SESSION_VERSION,
        "provider": session_principal.provider,
        "subject": session_principal.subject,
        "scope_type": session_principal.scope_type,
        "scope_id": session_principal.scope_id,
        "capabilities": sorted(session_principal.capabilities),
        "iat": issued_at,
        "exp": issued_at + settings.REPORT_SESSION_TTL_SECONDS,
        "nonce": secrets.token_hex(16),
    }
    encoded = _b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = _b64encode(hmac.digest(_session_key(), encoded.encode("ascii"), "sha256"))
    return f"{encoded}.{signature}"


def verify_report_session(
    token: str,
    *,
    now: int | None = None,
) -> AuthenticatedPrincipal:
    """Verify signature, lifetime, current scope and the bounded capability set."""
    _require_personal_mode()
    try:
        encoded, signature = token.split(".", 1)
        expected_signature = _b64encode(
            hmac.digest(_session_key(), encoded.encode("ascii"), "sha256")
        )
        if not secrets.compare_digest(signature, expected_signature):
            raise ValueError("signature")
        payload = json.loads(_b64decode(encoded))
        principal = AuthenticatedPrincipal(
            provider=payload["provider"],
            subject=payload["subject"],
            scope_type=payload["scope_type"],
            scope_id=payload["scope_id"],
            capabilities=frozenset(payload["capabilities"]),
        )
        current = int(time.time() if now is None else now)
        if (
            payload.get("v") != _SESSION_VERSION
            or not isinstance(payload.get("iat"), int)
            or not isinstance(payload.get("exp"), int)
            or payload["iat"] > current + 30
            or payload["exp"] <= current
            or payload["exp"] - payload["iat"] != settings.REPORT_SESSION_TTL_SECONDS
            or principal != _report_session_principal()
        ):
            raise ValueError("claims")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _unauthorized() from None
    return principal


def _require_personal_mode() -> None:
    mode = settings.REPORT_AUTH_MODE
    if mode == "disabled":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authenticated report delivery is not configured",
        )
    if mode == "institutional_adapter":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Institutional authentication adapter is not configured",
        )
    if mode == "personal_local" and settings.API_BIND_HOST not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Passwordless Personal reports require a loopback API bind",
        )


def _configured_personal_secret() -> str:
    configured = settings.REPORT_PERSONAL_ACCESS_TOKEN
    expected = configured.get_secret_value() if configured is not None else ""
    if len(expected) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authenticated report delivery is not configured",
        )
    if not 60 <= settings.REPORT_SESSION_TTL_SECONDS <= 3_600:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authenticated report delivery is not configured",
        )
    return expected


def _personal_principal(*, provider: str = "personal_bearer") -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        provider=provider,
        subject="personal-owner",
        scope_type="personal_owner",
        scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
        capabilities=frozenset(
            {
                REPORT_VIEW_CAPABILITY,
                REPORT_PAPER_CAPABILITY,
                REPORT_SOURCE_CAPABILITY,
                REPORT_JUDGMENT_CAPABILITY,
                PAPER_CHECK_CAPABILITY,
                PAPER_STATUS_CAPABILITY,
                SOURCE_REPOSITORY_READ_CAPABILITY,
                SOURCE_REPOSITORY_WRITE_CAPABILITY,
                EDITION_REVIEW_CAPABILITY,
                # No marks release in Personal (owner decision 2026-09-29).
            }
        ),
    )


def _report_session_principal(
    *, provider: str = "personal_bearer"
) -> AuthenticatedPrincipal:
    """Limit a Personal browser cookie to report surfaces and owner review."""
    return AuthenticatedPrincipal(
        provider=provider,
        subject="personal-owner",
        scope_type="personal_owner",
        scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
        capabilities=frozenset(
            {
                REPORT_VIEW_CAPABILITY,
                REPORT_PAPER_CAPABILITY,
                REPORT_SOURCE_CAPABILITY,
                REPORT_JUDGMENT_CAPABILITY,
                EDITION_REVIEW_CAPABILITY,
            }
        ),
    )


def _session_key() -> bytes:
    return hashlib.sha256(
        b"sourcefidelity-report-session-v1\x00"
        + _configured_personal_secret().encode("utf-8")
    ).digest()


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)
