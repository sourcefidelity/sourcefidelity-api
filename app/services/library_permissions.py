"""Library permissions: six switches for every service the application calls.

ARCHITECTURE §4 *Library permissions*: search, availability/link, abstract
use, acquisition, retention and remote-model processing are separate
permissions; turning one on never turns on another. Every service has all six
(owner decision 2026-10-18: the application makes no copyright decision for
its users, and its own reading of a provider's terms may be wrong or out of
date). The person who deploys the application sets them: the Personal user
for their own deployment, the administrator for an Institutional one.

The first setting is where the application is used (`NETWORK_LOCATION`:
on a campus network, off campus, or both/not sure, which is treated as on
campus). On a campus network a publisher may serve subscribed text to any
request, so a full-text PDF or adapter text without open-access or
public-domain evidence counts as licensed text.

Defaults: a public service keeps working as before (every switch on). The
campus DOI resolver and Elsevier's institutional text start off wherever the
application is used, because both reach subscribed text off campus too. The
publisher-PDF constructor reaches only free PDFs off campus, so there it
works as before; on a campus network it starts off (owner decision
2026-10-18). Licensed text is used only with switches the deployment turned
on itself, never with a default.

`LIBRARY_PERMISSIONS` is a JSON object of the switches the deployment set,
for example ``{"openalex": {"remote_model": false},
"publisher_pdf": {"acquisition": true, "remote_model": true}}``. A service
left out keeps its defaults; an unknown service or switch is ignored.

Licensed text needs remote-model processing as well as acquisition: the
local-only path for licensed text (ARCHITECTURE §4 *Licensed text without
remote-model permission*) is not built, and without it the run would send the
text to the configured model arms. For the same reason a service with
remote-model processing off contributes no abstract or full text at all.
"""
from __future__ import annotations

import json
import logging
from functools import wraps

from app.config import settings

logger = logging.getLogger(__name__)

PERMISSIONS = ("search", "availability", "abstract", "acquisition", "retention", "remote_model")

# Licensed services: their purpose is subscribed full text.
PUBLISHER_PDF = "publisher_pdf"
DOI_RESOLVER = "doi_resolver"
ELSEVIER_INSTITUTIONAL = "elsevier_institutional"
LICENSED_SERVICES = (PUBLISHER_PDF, DOI_RESOLVER, ELSEVIER_INSTITUTIONAL)

# The resolver's own routes, beside the registered adapters. Their search and
# availability switches are not enforced yet (STATE *Library permissions*);
# their text switches act on the result they return.
PUBLIC_DOI = "public_doi"
STUDENT_URL = "student_url"
WEB_PAGES = "web_pages"
OTHER_ROUTES = "other_routes"
ROUTE_SERVICES = (STUDENT_URL, PUBLIC_DOI, WEB_PAGES, "google_books", "internet_archive",
                  "media_history_archive", "mdpi_file_server", OTHER_ROUTES)

# Result source names that belong to a route service.
_ROUTE_SOURCE_NAMES = {
    "student_url": STUDENT_URL, "submitted_link": STUDENT_URL,
    "web_fetch": WEB_PAGES, "landing_page": WEB_PAGES,
    "independent_landing_metadata": WEB_PAGES, "independent_content": WEB_PAGES,
    "google_books": "google_books", "internet_archive": "internet_archive",
    "lantern": "media_history_archive", "media_history_archive": "media_history_archive",
    "mdpi_file_server": "mdpi_file_server",
}
# Copies the application already holds; their switches were applied when
# they were obtained.
_HELD_SOURCE_NAMES = frozenset({"local_cache", "retrieval_lookup_cache"})

_RESTRICTED = "access restricted by library permission"


def adapter_services() -> tuple[str, ...]:
    from app.services.retrieval import _RETRIEVER_CLASSES
    return tuple(_RETRIEVER_CLASSES)


def services() -> tuple[str, ...]:
    """Every service with switches: registered adapters, the resolver's own
    routes and the licensed services."""
    return adapter_services() + ROUTE_SERVICES + LICENSED_SERVICES


def explicit(value: str | None = None) -> dict[str, dict[str, bool]]:
    """The switches the deployment set; invalid entries are ignored."""
    raw = settings.LIBRARY_PERMISSIONS if value is None else value
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        logger.warning("Ignored LIBRARY_PERMISSIONS: not a JSON object")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignored LIBRARY_PERMISSIONS: not a JSON object")
        return {}
    known = set(services())
    switches: dict[str, dict[str, bool]] = {}
    for service, values in parsed.items():
        if service not in known or not isinstance(values, dict):
            logger.warning("Ignored an unknown LIBRARY_PERMISSIONS service")
            continue
        for name, on in values.items():
            if name not in PERMISSIONS or not isinstance(on, bool):
                logger.warning("Ignored an invalid LIBRARY_PERMISSIONS switch")
                continue
            switches.setdefault(service, {})[name] = on
    return switches


def on_campus() -> bool:
    return settings.NETWORK_LOCATION != "off_campus"


def default(service: str) -> bool:
    if service == PUBLISHER_PDF:
        return not on_campus()
    return service not in LICENSED_SERVICES


def permitted(service: str, permission: str) -> bool:
    return explicit().get(service, {}).get(permission, default(service))


def explicitly_on(service: str, permission: str) -> bool:
    return explicit().get(service, {}).get(permission) is True


def table() -> dict[str, dict[str, bool]]:
    """Every service's six switches as they apply now."""
    set_switches = explicit()
    return {service: {name: set_switches.get(service, {}).get(name, default(service))
                      for name in PERMISSIONS} for service in services()}


def signature() -> str | None:
    """What a lookup cache must be keyed on; None while nothing was set."""
    set_switches = explicit()
    if not set_switches and not on_campus():
        return None
    return json.dumps({"location": settings.NETWORK_LOCATION,
                       "switches": set_switches}, sort_keys=True)


def licensed_text_refusal(service: str) -> str | None:
    """Why `service` may not supply licensed text, or None when it may.
    Licensed text needs switches the deployment turned on itself."""
    if not explicitly_on(service, "acquisition"):
        return "licensed_acquisition_not_permitted"
    if not explicitly_on(service, "remote_model"):
        return "licensed_text_local_processing_not_built"
    return None


def licensed_switch(service: str, permission: str) -> bool:
    """A licensed service's switch: one it starts with on follows its
    switches; otherwise only a switch the deployment turned on counts. Off
    campus the publisher-PDF constructor starts on; it reaches only free PDFs
    there."""
    return permitted(service, permission) if default(service) else explicitly_on(service, permission)


def acquisition_refusal(service: str) -> str | None:
    """Why a licensed service may not acquire, or None when it may."""
    if not licensed_switch(service, "acquisition"):
        return "licensed_acquisition_not_permitted"
    if not licensed_switch(service, "remote_model"):
        return "licensed_text_local_processing_not_built"
    return None


def campus_resolver_active() -> bool:
    """The configured campus DOI resolver acquires licensed text, so it runs
    only with its switches on; otherwise the public DOI route runs."""
    return bool(settings.DOI_RESOLVER_URL) and acquisition_refusal(DOI_RESOLVER) is None


def service_for(result) -> str | None:
    """The service a result came from; None for a copy already held."""
    name = (getattr(result, "source_name", "") or "").strip()
    if name in _HELD_SOURCE_NAMES:
        return None
    if name == "doi_resolver":
        # The campus and public DOI routes never both run for one deployment.
        return DOI_RESOLVER if campus_resolver_active() else PUBLIC_DOI
    if name in adapter_services():
        return name
    return _ROUTE_SOURCE_NAMES.get(name, OTHER_ROUTES)


def _withhold(result, reason: str, *, text: bool, abstract: bool, links: bool) -> None:
    result.metadata = result.metadata or {}
    withheld = result.metadata.setdefault("library_permission_withheld", [])
    if text and (result.full_text is not None or result.representation is not None):
        result.full_text = None
        result.representation = None
        result.parent_representation = None
        withheld.append({"part": "text", "reason": reason})
    if abstract and result.abstract:
        result.abstract = None
        withheld.append({"part": "abstract", "reason": reason})
    if links and (result.locations or result.full_text_url):
        result.locations = []
        result.full_text_url = None
        withheld.append({"part": "links", "reason": reason})
    if not withheld:
        del result.metadata["library_permission_withheld"]


def _is_open(result) -> bool:
    from app.services.source_resolver import _check_is_oa
    return _check_is_oa(result) or (result.metadata or {}).get("license_class") == "public_domain" or (
        result.source_name in {"gutenberg", "wikisource"})


def _counts_as_licensed(result) -> bool:
    if not on_campus() or _is_open(result):
        return False
    from app.services.retrieval.base import RepresentationKind
    representation = result.representation
    if representation is not None and representation.kind is RepresentationKind.PDF:
        return True
    # Text an adapter returned itself (Elsevier XML, for example).
    return representation is not None and result.source_name in adapter_services()


def apply(result, service: str | None = None):
    """Withhold what `service`'s switches do not permit from `result`."""
    if result is None:
        return result
    service = service or service_for(result)
    if service is None:
        return result
    remote = permitted(service, "remote_model")
    if not permitted(service, "availability"):
        _withhold(result, "availability_off", text=False, abstract=False, links=True)
    if not permitted(service, "acquisition"):
        _withhold(result, "acquisition_off", text=True, abstract=False, links=True)
    if not permitted(service, "abstract"):
        _withhold(result, "abstract_off", text=False, abstract=True, links=False)
    if not remote:
        _withhold(result, "remote_model_off", text=True, abstract=True, links=True)
    if result.full_text is not None or result.representation is not None:
        # Text a licensed service supplied follows that service's switches.
        licensed = (result.metadata or {}).get("library_service")
        refusal = (acquisition_refusal(licensed) if licensed in LICENSED_SERVICES
                   else licensed_text_refusal(service) if _counts_as_licensed(result) else None)
        if refusal:
            _withhold(result, refusal, text=True, abstract=False, links=False)
    return result


def restrict(source):
    """Apply a registered adapter's switches to every result it returns.

    With search off, each search answers that access is restricted, which
    keeps a required route unattempted (`search_incomplete`), never a miss.
    """
    service = source.name
    if service not in adapter_services() or all(table()[service].values()):
        return source

    def restricted(*args, **kwargs):
        from app.services.retrieval.base import RetrievalResult
        return RetrievalResult(source_name=service, success=False, error=_RESTRICTED,
                               metadata={"library_permission_off": "search"})

    searches = ("search_by_doi", "search_by_title_author", "search_by_isbn", "search_reference",
                "search_after_failed_candidates", "paper_by_id")
    for name in searches:
        method = getattr(source, name, None)
        if not callable(method):
            continue
        if not permitted(service, "search"):
            setattr(source, name, restricted)
        else:
            setattr(source, name, _applying(method, service))
    if not permitted(service, "search"):
        for name in ("prefetch_dois", "prefetch_titles"):
            if callable(getattr(source, name, None)):
                setattr(source, name, lambda *args, **kwargs: 0)
    download = getattr(source, "download_full_text", None)
    if callable(download):
        setattr(source, "download_full_text", _applying(download, service))
    return source


def _applying(method, service):
    @wraps(method)
    def call(*args, **kwargs):
        return apply(method(*args, **kwargs), service)
    return call
