"""Passive, snapshot-bound submitted-link observations; never initiate requests."""
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import socket
import ssl
import time
from urllib.parse import urlsplit
import ipaddress
import re
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def binding(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def is_site_homepage(html: str, url: str) -> bool:
    """Structural home destination; caller separately establishes title mismatch.

    Root addresses alone are insufficient: article metadata or a visible
    article container blocks this observation. No network request occurs here.
    """
    from bs4 import BeautifulSoup
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.query or not re.fullmatch(r'/(?:[a-z]{2}(?:-[a-z]{2})?/){0,2}(?:index\.(?:html?|aspx?))?', parsed.path or '/', re.I):
        return False
    soup = BeautifulSoup(html, 'html.parser')
    if soup.select('article, [itemprop="articleBody"], meta[name="citation_title"], meta[property="article:published_time"]'):
        return False
    kind = soup.find('meta', attrs={'property': 'og:type'})
    if not kind or str(kind.get('content', '')).casefold() != 'website':
        return False
    return bool(soup.title and soup.title.get_text(strip=True))


class LinkHop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    destination_sha256: Hash
    http_status: int = Field(ge=100, le=599)
    location_sha256: Hash | None = None
    destination_origin: str | None = None


class LinkIdentityDifference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal['title', 'author', 'year', 'doi']
    submitted: str = Field(min_length=1, max_length=2000)
    destination: str = Field(min_length=1, max_length=2000)


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    started_at: datetime
    completed_at: datetime | None = None
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    request_sha256: Hash
    destination_sha256: Hash | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    outcome: Literal["response", "not_found", "removed", "authentication_required",
        "proxy_authentication_required", "access_refused", "rate_limited", "server_failure",
        "legal_restriction_reported", "timeout", "dns_failure", "tls_failure",
        "connection_failure", "safety_refused", "size_limit", "redirect_limit",
        "operational_failure"] = "operational_failure"
    hops: list[LinkHop] = Field(default_factory=list)
    response_sha256: Hash | None = None
    safety_reason: Literal['not_recorded', 'dns_failure', 'non_public_destination',
        'url_credentials', 'invalid_connected_peer', 'non_public_connected_peer'] = 'not_recorded'
    destination_origin: str | None = None
    identity_evidence_sha256: Hash | None = None
    identity_fields: list[Literal['title', 'author', 'year', 'doi']] = Field(default_factory=list)
    identity_differences: list[LinkIdentityDifference] = Field(default_factory=list, max_length=4)
    page_observation: Literal['not_assessed', 'pdf_route_required', 'cross_script_title_unresolved',
        'page_title_mismatch_unconfirmed', 'readable_text_unavailable', 'source_kind_unconfirmed', 'site_homepage'] = 'not_assessed'
    page_evidence_sha256: Hash | None = None
    representation_sha256: Hash | None = None
    representation_id: str | None = None
    source_validation: Literal['not_assessed', 'safety_rejected', 'safety_unavailable',
        'completeness_uncertain', 'completeness_rejected'] = 'not_assessed'
    source_validation_sha256: Hash | None = None
    # HTTP observations alone cannot establish these independent axes.
    destination_identity: Literal["not_assessed", "confirmed", "bibliographic_conflict", "unconfirmed"] = "not_assessed"
    admitted_content: Literal["not_assessed", "admitted"] = "not_assessed"

    @model_validator(mode="after")
    def valid_times(self):
        if self.identity_differences and (self.destination_identity != 'bibliographic_conflict'
                or any(d.field not in self.identity_fields for d in self.identity_differences)):
            raise ValueError('Field differences require response-bound conflicting identity fields')
        if self.started_at.tzinfo is None or (self.completed_at is not None and
                (self.completed_at.tzinfo is None or self.completed_at < self.started_at)):
            raise ValueError("Request times must be ordered and timezone-aware")
        if self.destination_identity != 'not_assessed' and (
            not self.response_sha256 or self.identity_evidence_sha256 != self.response_sha256
        ):
            raise ValueError('Destination identity requires the exact observed response')
        if self.admitted_content == 'admitted' and (
            self.destination_identity != 'confirmed' or not self.representation_sha256 or not self.representation_id
        ):
            raise ValueError('Admitted content requires independent identity and representation binding')
        if self.page_observation != 'not_assessed' and (
            not self.response_sha256 or self.page_evidence_sha256 != self.response_sha256
        ):
            raise ValueError('Page observation requires exact response binding')
        if self.source_validation != 'not_assessed' and (
            not self.response_sha256 or self.source_validation_sha256 != self.response_sha256
        ):
            raise ValueError('Source validation limitation requires exact response binding')
        for origin in [self.destination_origin, *(hop.destination_origin for hop in self.hops)]:
            if origin is not None and safe_origin(origin) != origin:
                raise ValueError('Only sanitized public origins may be retained')
        return self


class SubmittedLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["submitted-link-v1", "submitted-link-v2"] = "submitted-link-v2"
    reference_id: str
    reference_snapshot_sha256: Hash
    kind: Literal["url", "doi"]
    submitted_sha256: Hash
    request_sha256: Hash
    state: Literal["not_checked", "observed", "historical_unknown"] = "not_checked"
    requests: list[LinkRequest] = Field(default_factory=list)
    observations_truncated: bool = False
    address_extraction: Literal['not_assessed', 'possible_truncation'] = 'not_assessed'
    not_checked_reason: Literal['not_recorded', 'not_visited_by_resolution', 'uncited_reference', 'resolution_not_run',
        'authorized_reuse', 'capability_disabled', 'library_locator_only', 'candidate_budget_exhausted',
        'unsupported_source_type'] = 'not_recorded'

    @model_validator(mode="after")
    def valid_state(self):
        if bool(self.requests) != (self.state == "observed"):
            raise ValueError("Observed state requires actual request records")
        if any(r.completed_at is None for r in self.requests):
            raise ValueError("Persist only finished request observations")
        if any(r.request_sha256 != self.request_sha256 for r in self.requests):
            raise ValueError("Request observations must belong to this submitted link")
        return self


ACTIVE_LINKS = ContextVar("submitted_link_observations", default=None)
ACTIVE_REQUEST = ContextVar("submitted_link_request", default=None)


def link_not_visited(url, reason):
    """Record an explicit branch decision; never override actual requests."""
    allowed = {'authorized_reuse', 'capability_disabled', 'library_locator_only',
               'candidate_budget_exhausted', 'unsupported_source_type'}
    if reason not in allowed:
        return
    for row in ACTIVE_LINKS.get() or []:
        if row.state == 'not_checked' and (url is None or row.request_sha256 == binding(str(url))):
            row.not_checked_reason = reason


def safe_origin(value):
    """Retain origin only: no userinfo, path, query, fragment or private address."""
    try:
        parsed = urlsplit(str(value))
        host = parsed.hostname
        if parsed.scheme not in {'http', 'https'} or not host or parsed.username or parsed.password:
            return None
        if host == 'localhost' or host.endswith(('.local', '.internal', '.localhost')) or '.' not in host:
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        if not all(c.isascii() and (c.isalnum() or c in '.-') for c in host):
            return None
        port = parsed.port
        return f'{parsed.scheme}://{host}' + (f':{port}' if port else '')
    except ValueError:
        return None


def request_url(value, kind, *, legacy=False):
    if legacy:
        return value if kind == 'url' else 'https://doi.org/' + value
    value = value.strip()
    if kind == 'doi':
        for prefix in ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:'):
            if value.lower().startswith(prefix):
                value = value[len(prefix):].strip()
                break
        return 'https://doi.org/' + value
    return 'https://' + value if value.lower().startswith('www.') else value


def address_may_be_truncated(value, raw_reference):
    """Recognize a retained URL prefix followed by a URL-shaped continuation."""
    return bool(value and re.search(re.escape(value) +
        r'\s+(?=[A-Za-z0-9._~/%?=&+-]*[-/_?=&])[A-Za-z0-9._~/%?=&+-]+', raw_reference or ''))


def initial_observations(reference, *, historical=False, legacy=False):
    snapshot = binding(reference.model_dump_json())
    rows = []
    for kind in ("url", "doi"):
        value = getattr(reference, kind, "") or ""
        if not value:
            continue
        request = request_url(value, kind, legacy=legacy)
        # A URL-shaped continuation after the extracted prefix warrants a
        # caveat, never an invented repair or another request.
        possible_truncation = bool(kind == 'url' and not getattr(reference, 'url_repair', None)
            and address_may_be_truncated(value, getattr(reference, 'raw_ref', '')))
        rows.append(SubmittedLink(reference_id=reference.reference_id,
            reference_snapshot_sha256=snapshot, kind=kind,
            submitted_sha256=binding(value), request_sha256=binding(request),
            state="historical_unknown" if historical else "not_checked",
            address_extraction='possible_truncation' if possible_truncation and not historical else 'not_assessed',
            version='submitted-link-v1' if legacy else 'submitted-link-v2'))
    return rows


def observe_reference(method):
    @wraps(method)
    def wrapped(self, reference, *args, **kwargs):
        rows = initial_observations(reference)
        for row in rows:
            row.not_checked_reason = 'not_visited_by_resolution'
        token = ACTIVE_LINKS.set(rows)
        try:
            result = method(self, reference, *args, **kwargs)
        except Exception as exc:
            exc.submitted_link_observations = [r.model_dump(mode="json") for r in rows]
            raise
        else:
            result.metadata = result.metadata or {}
            result.metadata["submitted_link_observations"] = [r.model_dump(mode="json") for r in rows]
            return result
        finally:
            ACTIVE_LINKS.reset(token)
    return wrapped


def response_observed(url, status, location=None):
    request = ACTIVE_REQUEST.get()
    if request is None:
        return
    request.http_status = status
    request.destination_sha256 = binding(str(url))
    request.destination_origin = safe_origin(url)
    # Only hashes, never response headers, destinations, credentials or signed tokens.
    if len(request.hops) < 32:
        request.hops.append(LinkHop(destination_sha256=binding(str(url)), http_status=status,
                                   location_sha256=binding(location) if location else None,
                                   destination_origin=safe_origin(url)))


def identity_observed(url, content_sha256, reason, fields=(), *, differences=()):
    """Consume the existing verifier, never infer identity from HTTP or a title."""
    outcome = {'bibliographic_identity_confirmed': 'confirmed',
               'bibliographic_fields_conflict': 'bibliographic_conflict',
               'bibliographic_identity_unconfirmed': 'unconfirmed'}.get(reason)
    if not outcome:
        return
    for row in ACTIVE_LINKS.get() or []:
        if row.request_sha256 != binding(str(url)):
            continue
        for request in row.requests:
            if request.response_sha256 == content_sha256:
                request.destination_identity = outcome
                request.identity_evidence_sha256 = content_sha256
                request.identity_fields = sorted({f for f in fields if f in {'title', 'author', 'year', 'doi'}})
                request.identity_differences = ([LinkIdentityDifference.model_validate(d) for d in differences]
                    if outcome == 'bibliographic_conflict' else [])


def page_observed(url, content_sha256, reason):
    if reason not in {'pdf_route_required', 'cross_script_title_unresolved',
        'page_title_mismatch_unconfirmed', 'readable_text_unavailable', 'source_kind_unconfirmed', 'site_homepage'}:
        return
    for row in ACTIVE_LINKS.get() or []:
        if row.request_sha256 != binding(str(url)):
            continue
        for request in row.requests:
            if request.response_sha256 == content_sha256:
                request.page_observation = reason
                request.page_evidence_sha256 = content_sha256


def bind_authorized_admission(values, metadata, representation_id):
    """Called only after ordinary current-scope representation authorization."""
    admission = metadata.get('durable_admission') or {}
    diagnostic = metadata.get('web_fetch_diagnostic') or {}
    if (admission.get('state') != 'accepted'
        or str(admission.get('representation_id')) != str(representation_id)
        or not metadata.get('accepted_representation_sha256')):
        return values
    try:
        rows = [SubmittedLink.model_validate(value) for value in values or []]
        for row in rows:
            for request in row.requests:
                web_bound = (row.request_sha256 == metadata.get('requested_url_sha256')
                    and diagnostic.get('reason') == 'bibliographic_identity_confirmed'
                    and request.response_sha256 == diagnostic.get('observed_content_sha256'))
                direct_bound = any(
                    item.get('request_sha256') == row.request_sha256
                    and item.get('response_sha256') == request.response_sha256
                    and item.get('response_sha256') == metadata['accepted_representation_sha256']
                    for item in metadata.get('submitted_response_identity', []) if isinstance(item, dict))
                if request.destination_identity == 'confirmed' and (web_bound or direct_bound):
                    request.admitted_content = 'admitted'
                    request.representation_sha256 = metadata['accepted_representation_sha256']
                    request.representation_id = str(representation_id)
        return [SubmittedLink.model_validate(row.model_dump()).model_dump(mode='json') for row in rows]
    except (ValueError, TypeError):
        return values


def validated_response_identity(url, content_sha256, reason, fields=()):
    """Bind the existing verifier to exact fetched bytes, including redirects.

    Converted/OCR text and a PDF linked from a landing page cannot masquerade
    as the landing response. Those require their own derivation provenance.
    """
    outcome = {'high': 'confirmed', 'rejected': 'bibliographic_conflict',
               'medium': 'unconfirmed', 'low': 'unconfirmed'}.get(reason)
    if not url or not outcome:
        return []
    records = []
    for row in ACTIVE_LINKS.get() or []:
        for request in row.requests:
            if (request.outcome != 'response' or request.http_status is None
                or not 200 <= request.http_status < 300
                or request.response_sha256 != content_sha256
                or binding(str(url)) not in {request.request_sha256, request.destination_sha256}):
                continue
            request.destination_identity = outcome
            request.identity_evidence_sha256 = content_sha256
            request.identity_fields = sorted({f for f in fields if f in {'title', 'author', 'year', 'doi'}})
            if outcome == 'confirmed':
                records.append({'request_sha256': request.request_sha256, 'response_sha256': content_sha256})
    return records


def source_validation_observed(url, content_sha256, reason):
    """Record an existing source gate's limitation, never a new decision."""
    if not url or reason not in {'safety_rejected', 'safety_unavailable',
                                'completeness_uncertain', 'completeness_rejected'}:
        return
    for row in ACTIVE_LINKS.get() or []:
        for request in row.requests:
            if (request.outcome == 'response' and request.http_status is not None
                and 200 <= request.http_status < 300 and request.response_sha256 == content_sha256
                and binding(str(url)) in {request.request_sha256, request.destination_sha256}):
                request.source_validation = reason
                request.source_validation_sha256 = content_sha256


def response_body_observed(content):
    request = ACTIVE_REQUEST.get()
    if request is not None:
        request.response_sha256 = hashlib.sha256(content).hexdigest()


def _outcome(exc, status):
    if exc is not None and not isinstance(exc, httpx.HTTPStatusError):
        if type(exc).__name__ == 'UnsafeUrlError' and getattr(exc, 'reason', None) == 'dns_failure':
            return 'dns_failure'
        if type(exc).__name__ == "UnsafeUrlError": return "safety_refused"
        if type(exc).__name__ == "ResponseTooLargeError": return "size_limit"
        if isinstance(exc, httpx.TooManyRedirects): return "redirect_limit"
        if isinstance(exc, httpx.TimeoutException): return "timeout"
        seen=set(); cause=exc
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause,socket.gaierror): return "dns_failure"
            if isinstance(cause,ssl.SSLError): return "tls_failure"
            cause=cause.__cause__ or cause.__context__
        if isinstance(exc,httpx.ConnectError): return "connection_failure"
        return "operational_failure"
    return {404:"not_found",410:"removed",401:"authentication_required",
        407:"proxy_authentication_required",403:"access_refused",429:"rate_limited",
        451:"legal_restriction_reported"}.get(status,
        "server_failure" if status is not None and status>=500 else "response")


def observe_request(function):
    @wraps(function)
    def wrapped(url, *args, **kwargs):
        matches=[r for r in (ACTIVE_LINKS.get() or []) if r.request_sha256==binding(str(url))]
        if not matches:
            return function(url,*args,**kwargs)
        request=LinkRequest(started_at=datetime.now(timezone.utc),request_sha256=binding(str(url)))
        token=ACTIVE_REQUEST.set(request); started=time.monotonic(); error=None
        try:
            response=function(url,*args,**kwargs)
            request.response_sha256=hashlib.sha256(response.content).hexdigest()
            return response
        except Exception as exc:
            error=exc
            raise
        finally:
            request.completed_at=datetime.now(timezone.utc)
            request.elapsed_seconds=max(0,time.monotonic()-started)
            request.outcome=_outcome(error,request.http_status)
            if type(error).__name__ == 'UnsafeUrlError':
                reason = getattr(error, 'reason', 'not_recorded')
                if reason in {'dns_failure', 'non_public_destination', 'url_credentials',
                              'invalid_connected_peer', 'non_public_connected_peer'}:
                    request.safety_reason = reason
            for row in matches:
                if len(row.requests)<32:
                    row.requests.append(request.model_copy(deep=True)); row.state="observed"
                else:
                    row.observations_truncated=True
            ACTIVE_REQUEST.reset(token)
    return wrapped


def project_observations(references, source_results, *, cited_reference_ids=None):
    """Neutral report data for all entries; no new checks or issue counts."""
    by_id={r.get("reference_id"):r for r in source_results}
    projected=[]
    for reference in references:
        saved=by_id.get(reference.reference_id)
        fallback=initial_observations(reference,historical=saved is not None)
        raw=(saved or {}).get("submitted_link_observations")
        if raw is not None:
            try:
                rows=[SubmittedLink.model_validate(r) for r in raw]
                expected_rows = initial_observations(reference, legacy=bool(rows) and all(r.version == 'submitted-link-v1' for r in rows))
                expected={(r.kind,r.submitted_sha256,r.reference_snapshot_sha256,r.request_sha256) for r in expected_rows}
                actual={(r.kind,r.submitted_sha256,r.reference_snapshot_sha256,r.request_sha256) for r in rows}
                if actual!=expected or len(rows)!=len(fallback) or any(r.reference_id!=reference.reference_id for r in rows):
                    raise ValueError("Observation snapshot mismatch")
            except (ValueError,TypeError):
                rows=fallback
        else:
            rows=fallback
        if saved is None and cited_reference_ids is not None:
            for row in rows:
                row.not_checked_reason = ('uncited_reference' if reference.reference_id not in cited_reference_ids else 'resolution_not_run')
        projected.extend(r.model_dump(mode="json") for r in rows)
    return projected
