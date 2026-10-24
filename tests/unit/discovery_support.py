"""Test support: a discovery recorder built from the fields a test names (discovery-recorder-v1)."""
from app.services.discovery_recorder import DiscoveryRecorder
from app.services.reference_discovery import ExpectedBibliographicFields

_CONSTRUCTOR = ("reference_id", "expected", "required", "submitted_reference", "search_policy_version",
                "search_retention_policy", "required_web_providers")
_SEEDED = ("queries", "attempts", "candidates", "limitations")
_SLOTS = ("journal_checks", "native_script_route", "periodical_archive_match", "container_identity",
          "doi_title_mismatch", "declared_document_url", "search_memo_reused")


def recorder_from(fields: dict | None = None, **more) -> DiscoveryRecorder:
    """A recorder for tests. Seeded queries, attempts, candidates and
    limitations are taken as given, the way a fixture states them."""
    fields = {**(fields or {}), **more}
    unknown = set(fields) - set(_CONSTRUCTOR) - set(_SEEDED) - set(_SLOTS)
    if unknown:
        raise KeyError(f"Not a recorder field: {sorted(unknown)}")
    recorder = DiscoveryRecorder(
        reference_id=fields.get("reference_id", "fixture"),
        expected=fields.get("expected") or ExpectedBibliographicFields(title=""),
        required=fields.get("required") or (),
        submitted_reference=fields.get("submitted_reference", ""),
        search_policy_version=fields.get("search_policy_version", "configured-search-v1"),
        search_retention_policy=fields.get("search_retention_policy"),
        required_web_providers=fields.get("required_web_providers") or (),
    )
    for name in _SEEDED:
        setattr(recorder, name, list(fields.get(name) or []))
    for name in _SLOTS:
        if name in fields:
            setattr(recorder, name, fields[name])
    return recorder
