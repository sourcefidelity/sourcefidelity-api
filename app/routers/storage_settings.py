"""Storage settings page (owner decision 2026-10-13; wording owner-approved).

The Personal owner sets their own scope; an Institutional administrator sets
the deployment-wide row every course and assessment inherits
(`storage_policy.settings_scope_for`). Instructors never hold the capability.
Only the approved text is shown: no helper sentences, no confirmation line.
"""
from html import escape
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import (
    STORAGE_SETTINGS_CAPABILITY,
    AuthenticatedPrincipal,
    get_authenticated_principal,
    require_same_origin_request,
)
from app.services.storage_policy import StoragePolicy, policy_for, set_policy, settings_scope_for

router = APIRouter(prefix="/settings", tags=["settings"])

PAPER_LONGER_THAN_REPORT = "The paper cannot be kept longer than its report."
DAYS_OR_UNTIL_DELETED = "Enter a number of days or choose Until deleted."
_PUBLIC_CHOICES = (("all", "All"), ("open_access_only", "Open access only"), ("none", "None"))


def _days(name: str, label: str, value: int, *, until_deleted: bool) -> str:
    keep = until_deleted and value == 0
    shown = "" if keep else str(value)
    field = (f'<label>{escape(label)} <input type="number" name="{name}" min="1" step="1" '
             f'value="{shown}"{"" if until_deleted else " required"}> days</label>')
    if until_deleted:
        field += (f'<label><input type="checkbox" name="{name}_until_deleted" value="true"'
                  f'{" checked" if keep else ""}> Until deleted</label>')
    return field


def _page(policy: StoragePolicy, error: str | None = None) -> HTMLResponse:
    nonce = secrets.token_urlsafe(24)
    sources, papers, reports = policy.sources, policy.papers, policy.reports
    public = "".join(
        f'<label><input type="radio" name="keep_public_copies" value="{value}"'
        f'{" checked" if sources.keep_public_copies == value else ""}> {label}</label>'
        for value, label in _PUBLIC_CHOICES)
    licensed = "".join(
        f'<label><input type="radio" name="keep_licensed_copies" value="{value}"'
        f'{" checked" if sources.keep_licensed_copies == (value == "on") else ""}> {label}</label>'
        for value, label in (("on", "On"), ("off", "Off")))
    alert = f'<p role="alert">{escape(error)}</p>' if error else ""
    body = (
        '<h1>Storage</h1>' + alert + '<form method="post" action="/settings/storage">'
        '<fieldset><legend>Sources</legend>'
        f'<fieldset><legend>Keep downloaded sources</legend>{public}</fieldset>'
        + _days("public_copy_days", "Keep downloaded sources for", sources.public_copy_days, until_deleted=True)
        + f'<fieldset><legend>Keep library-licensed sources</legend>{licensed}</fieldset>'
        + _days("licensed_copy_days", "Keep library-licensed sources for", sources.licensed_copy_days,
                until_deleted=True)
        + '</fieldset><fieldset><legend>Papers</legend>'
        + _days("paper_copy_days", "Keep the paper beside its report for", papers.paper_copy_days,
                until_deleted=False)
        + '</fieldset><fieldset><legend>Reports</legend>'
        + _days("report_days", "Keep reports for", reports.report_days, until_deleted=True)
        + '</fieldset><button type="submit">Save</button></form>')
    return HTMLResponse(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Storage</title>'
        f'<style nonce="{nonce}">body{{max-width:42rem;margin:3rem auto;padding:1rem;font:17px/1.6 system-ui;'
        'color:#18212b}fieldset{margin:1rem 0;padding:.75rem 1rem}label{display:block;margin:.4rem 0}'
        'input[type=number]{width:6rem}button{padding:.6rem 1rem}[role=alert]{color:#9b1c1c}</style>'
        f'</head><body>{body}</body></html>',
        status_code=status.HTTP_400_BAD_REQUEST if error else status.HTTP_200_OK,
        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                 'Content-Security-Policy': f"default-src 'none'; style-src 'nonce-{nonce}'; "
                                            "form-action 'self'; base-uri 'none'; frame-ancestors 'self'"})


class _MissingDays(Exception):
    """A days box left empty with Until deleted unticked."""


def _changes(form) -> dict:
    """The submitted form as a full storage policy body; HTTP 400 when malformed."""
    def days(name: str, *, until_deleted: bool) -> int:
        if until_deleted and form.get(f"{name}_until_deleted") == "true":
            return 0
        raw = str(form.get(name) or "").strip()
        if not raw:
            raise _MissingDays(name)
        try:
            value = int(raw)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from None
        if value < 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
        return value

    public = form.get("keep_public_copies")
    licensed = form.get("keep_licensed_copies")
    if public not in {value for value, _ in _PUBLIC_CHOICES} or licensed not in {"on", "off"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return {
        "sources": {"keep_public_copies": public,
                    "public_copy_days": days("public_copy_days", until_deleted=True),
                    "keep_licensed_copies": licensed == "on",
                    "licensed_copy_days": days("licensed_copy_days", until_deleted=True)},
        "papers": {"paper_copy_days": days("paper_copy_days", until_deleted=False)},
        "reports": {"report_days": days("report_days", until_deleted=True)},
    }


@router.get("/storage", response_class=HTMLResponse)
def storage_settings_page(principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
                          session: Session = Depends(get_db)):
    principal.require(STORAGE_SETTINGS_CAPABILITY)
    return _page(policy_for(session, *settings_scope_for(principal.scope_type, principal.scope_id)))


@router.post("/storage", response_class=HTMLResponse)
async def save_storage_settings(request: Request,
                                principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
                                session: Session = Depends(get_db)):
    require_same_origin_request(request)
    principal.require(STORAGE_SETTINGS_CAPABILITY)
    scope = settings_scope_for(principal.scope_type, principal.scope_id)
    try:
        changes = _changes(await request.form())
    except _MissingDays:
        return _page(policy_for(session, *scope), DAYS_OR_UNTIL_DELETED)
    try:
        StoragePolicy.model_validate({**changes, "version": "storage-policy-v1"})
    except ValidationError:
        submitted = policy_for(None, *scope).model_copy(update={})
        shown = StoragePolicy.model_construct(
            sources=submitted.sources.model_copy(update=changes["sources"]),
            papers=submitted.papers.model_copy(update=changes["papers"]),
            reports=submitted.reports.model_copy(update=changes["reports"]))
        return _page(shown, PAPER_LONGER_THAN_REPORT)
    set_policy(session, *scope, changes, updated_by=f"{principal.provider}:{principal.subject}")
    session.commit()
    return RedirectResponse("/settings/storage", status_code=status.HTTP_303_SEE_OTHER)
