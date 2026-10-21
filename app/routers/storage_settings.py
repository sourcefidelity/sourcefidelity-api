"""Storage settings page (owner decisions 2026-10-13, 2026-10-19 and 2026-10-20; wording owner-approved).

The deployment's owner sets them. Only the approved text is shown: no helper
sentences, no confirmation line after a save. Tightening which sources are
kept shows how many kept copies would be deleted and asks to confirm first.
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
from app.services.source_retention import apply_source_setting, copies_to_delete
from app.services.storage_policy import (
    POLICY_VERSION,
    SUBMITTED_FILE_OUTLIVES_REPORT,
    StoragePolicy,
    policy_for,
    set_policy,
)

router = APIRouter(prefix="/settings", tags=["settings"])

PAPER_LONGER_THAN_REPORT = "The paper cannot be kept longer than its report."
DAYS_OR_UNTIL_DELETED = "Enter a number of days or choose Until deleted."
SUBMITTED_FILE_LONGER_THAN_REPORT = "The submitted file cannot be kept longer than its report."
_SOURCE_CHOICES = (("all", "All"), ("open_access_only", "Open access only"), ("none", "None"))
_MONTH_CHOICES = (("3", "3 months"), ("6", "6 months"), ("9", "9 months"), ("12", "12 months"),
                  ("0", "Permanently"))


def _deleted_line(count: int) -> str:
    return f"{count} kept source will be deleted." if count == 1 else f"{count} kept sources will be deleted."


def _radios(name: str, choices, chosen: str) -> str:
    return "".join(
        f'<label><input type="radio" name="{name}" value="{value}"'
        f'{" checked" if chosen == value else ""}> {label}</label>' for value, label in choices)


def _days(name: str, label: str, value: int, *, until_deleted: bool) -> str:
    keep = until_deleted and value == 0
    shown = "" if keep else str(value)
    field = (f'<label>{escape(label)} <input type="number" name="{name}" min="1" step="1" '
             f'value="{shown}"{"" if until_deleted else " required"}> days</label>')
    if until_deleted:
        field += (f'<label><input type="checkbox" name="{name}_until_deleted" value="true"'
                  f'{" checked" if keep else ""}> Until deleted</label>')
    return field


def _respond(body: str, *, error: bool = False) -> HTMLResponse:
    nonce = secrets.token_urlsafe(24)
    return HTMLResponse(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Storage</title>'
        f'<style nonce="{nonce}">body{{max-width:42rem;margin:3rem auto;padding:1rem;font:17px/1.6 system-ui;'
        'color:#18212b}fieldset{margin:1rem 0;padding:.75rem 1rem}label{display:block;margin:.4rem 0}'
        'input[type=number]{width:6rem}button,a.button{padding:.6rem 1rem;margin-right:.5rem}'
        '[role=alert]{color:#9b1c1c}</style>'
        f'</head><body>{body}</body></html>',
        status_code=status.HTTP_400_BAD_REQUEST if error else status.HTTP_200_OK,
        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                 'Content-Security-Policy': f"default-src 'none'; style-src 'nonce-{nonce}'; "
                                            "form-action 'self'; base-uri 'none'; frame-ancestors 'self'"})


def _page(policy: StoragePolicy, error: str | None = None) -> HTMLResponse:
    sources, papers, reports = policy.sources, policy.papers, policy.reports
    alert = f'<p role="alert">{escape(error)}</p>' if error else ""
    on_off = (("on", "On"), ("off", "Off"))
    body = (
        '<h1>Storage</h1>' + alert + '<form method="post" action="/settings/storage">'
        '<fieldset><legend>Sources</legend>'
        f'<fieldset><legend>Keep sources</legend>{_radios("keep_copies", _SOURCE_CHOICES, sources.keep_copies)}'
        '</fieldset>'
        '<fieldset><legend>Keep each source after its last use for</legend>'
        f'{_radios("keep_months", _MONTH_CHOICES, str(sources.keep_months))}</fieldset>'
        '</fieldset><fieldset><legend>Papers</legend>'
        + _days("paper_copy_days", "Keep the paper beside its report for", papers.paper_copy_days,
                until_deleted=False)
        + '<fieldset><legend>Keep the submitted file</legend>'
        + _radios("keep_submitted_file", on_off, "on" if papers.keep_submitted_file else "off") + '</fieldset>'
        + _days("submitted_file_days", "Keep the submitted file for", papers.submitted_file_days,
                until_deleted=True)
        + '</fieldset><fieldset><legend>Reports</legend>'
        + _days("report_days", "Keep reports for", reports.report_days, until_deleted=True)
        + '<fieldset><legend>Keep only the latest report</legend>'
        + _radios("keep_latest_only", on_off, "on" if reports.keep_latest_only else "off") + '</fieldset>'
        + '</fieldset><fieldset><legend>Searches</legend><fieldset><legend>Finish incomplete searches automatically'
        '</legend>' + _radios("finish_incomplete_automatically", on_off,
                              "on" if policy.searches.finish_incomplete_automatically else "off")
        + '</fieldset></fieldset><button type="submit">Save</button></form>')
    return _respond(body, error=bool(error))


def _confirm_page(form, count: int) -> HTMLResponse:
    """The submitted form again, held until the user confirms the deletion."""
    hidden = "".join(f'<input type="hidden" name="{escape(str(key))}" value="{escape(str(value))}">'
                     for key, value in form.multi_items() if key != "confirm")
    body = ('<h1>Storage</h1>' f'<p role="alert">{escape(_deleted_line(count))}</p>'
            '<form method="post" action="/settings/storage">' + hidden
            + '<input type="hidden" name="confirm" value="delete">'
            '<button type="submit">Delete and save</button>'
            '<a class="button" href="/settings/storage">Cancel</a></form>')
    return _respond(body)


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

    keep = form.get("keep_copies")
    months = form.get("keep_months")
    kept = form.get("keep_submitted_file")
    if (keep not in {value for value, _ in _SOURCE_CHOICES} or months not in {value for value, _ in _MONTH_CHOICES}
            or kept not in {"on", "off"}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    papers = {"paper_copy_days": days("paper_copy_days", until_deleted=False), "keep_submitted_file": kept == "on"}
    # With the file not kept, an empty days box leaves the stored period as it is.
    if (kept == "on" or str(form.get("submitted_file_days") or "").strip()
            or form.get("submitted_file_days_until_deleted") == "true"):
        papers["submitted_file_days"] = days("submitted_file_days", until_deleted=True)
    searches = {}
    finish = form.get("finish_incomplete_automatically")
    if finish is not None:
        # Absent (an older page): the stored choice stays.
        if finish not in {"on", "off"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
        searches = {"finish_incomplete_automatically": finish == "on"}
    reports = {"report_days": days("report_days", until_deleted=True)}
    latest_only = form.get("keep_latest_only")
    if latest_only is not None:
        # Absent (an older page): the stored choice stays.
        if latest_only not in {"on", "off"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
        reports["keep_latest_only"] = latest_only == "on"
    return {
        **({"searches": searches} if searches else {}),
        "sources": {"keep_copies": keep, "keep_months": int(months)},
        "papers": papers,
        "reports": reports,
    }


@router.get("/storage", response_class=HTMLResponse)
def storage_settings_page(principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
                          session: Session = Depends(get_db)):
    principal.require(STORAGE_SETTINGS_CAPABILITY)
    return _page(policy_for(session, principal.scope_type, principal.scope_id))


@router.post("/storage", response_class=HTMLResponse)
async def save_storage_settings(request: Request,
                                principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
                                session: Session = Depends(get_db)):
    require_same_origin_request(request)
    principal.require(STORAGE_SETTINGS_CAPABILITY)
    scope = (principal.scope_type, principal.scope_id)
    form = await request.form()
    try:
        changes = _changes(form)
    except _MissingDays:
        return _page(policy_for(session, *scope), DAYS_OR_UNTIL_DELETED)
    current = policy_for(session, *scope)
    try:
        proposed = StoragePolicy.model_validate({
            "version": POLICY_VERSION, "sources": changes["sources"],
            "papers": {**current.papers.model_dump(), **changes["papers"]},
            "reports": changes["reports"],
            "searches": {**current.searches.model_dump(), **changes.get("searches", {})}})
    except ValidationError as exc:
        shown = StoragePolicy.model_construct(
            sources=current.sources.model_copy(update=changes["sources"]),
            papers=current.papers.model_copy(update=changes["papers"]),
            reports=current.reports.model_copy(update=changes["reports"]),
            searches=current.searches)
        return _page(shown, SUBMITTED_FILE_LONGER_THAN_REPORT if SUBMITTED_FILE_OUTLIVES_REPORT in str(exc)
                     else PAPER_LONGER_THAN_REPORT)
    count = copies_to_delete(session, *scope, proposed)
    if count and form.get("confirm") != "delete":
        return _confirm_page(form, count)
    policy = set_policy(session, *scope, changes, updated_by=f"{principal.provider}:{principal.subject}")
    apply_source_setting(session, *scope, policy)
    session.commit()
    return RedirectResponse("/settings/storage", status_code=status.HTTP_303_SEE_OTHER)
