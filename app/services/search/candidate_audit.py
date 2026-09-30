"""Development-only audit of web-search candidate links.

Enabled only by ``SEARCH_CANDIDATE_AUDIT_URLS``. A developer can then open each
Exa, Tavily or SearXNG candidate the resolver considered and check whether its
disposition was right. Brave candidates are never recorded here: the Brave
Search API terms (§3.2(i)) forbid storing search results beyond transient
operational use, so Brave keeps its count-only transient audit.

When the setting is off nothing is added to any trace or phase record.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AUDIT_POLICY = "dev-candidate-audit-v1"
# An allowlist, not a denylist: a provider is recorded only after its terms
# were checked for result retention. Brave is deliberately absent.
# SearXNG relays other engines (Google CSE here) whose terms are unchecked, so it is not recorded.
AUDITABLE_PROVIDERS = frozenset({"exa", "tavily"})
MAX_AUDIT_CANDIDATES = 256
MAX_AUDIT_URL_LENGTH = 2000

AuditDisposition = Literal[
    "acquired",
    "acquired_fallback",
    "identity_rejected",
    "identity_unconfirmed",
    "completeness_rejected",
    "type_rejected",
    "type_unconfirmed",
    "transport_failure",
    "access_restricted",
    "unavailable",
    "not_attempted",
    "unknown",
]
_DISPOSITIONS = frozenset(AuditDisposition.__args__)


class SearchCandidateAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["exa", "tavily", "searxng"]
    query_id: str | None = Field(default=None, max_length=128)
    rank: int | None = Field(default=None, ge=1, le=10_000)
    url: str = Field(min_length=1, max_length=MAX_AUDIT_URL_LENGTH)
    url_truncated: bool = False
    disposition: AuditDisposition
    reason_code: str | None = Field(default=None, max_length=100)


class SearchCandidateAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audit_policy: Literal["dev-candidate-audit-v1"] = AUDIT_POLICY
    candidates: list[SearchCandidateAuditEntry] = Field(
        default_factory=list, max_length=MAX_AUDIT_CANDIDATES
    )
    omitted_count: int = Field(default=0, ge=0)


def audit_enabled() -> bool:
    from app.config import settings

    return bool(getattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", False))


def auditable_provider(provider: object) -> bool:
    return audit_enabled() and str(provider or "").strip().lower() in AUDITABLE_PROVIDERS


def location_audit_fields(location) -> dict:
    """Search query and rank to carry with a candidate, only while auditing."""
    metadata = getattr(location, "metadata", None) or {}
    if not auditable_provider(metadata.get("search_provider")):
        return {}
    return {
        key: metadata[key]
        for key in ("search_query", "search_rank")
        if metadata.get(key) is not None
    }


def _disposition(record: dict) -> tuple[str, str | None]:
    outcome = str(record.get("outcome") or "unknown")
    reason = str(record.get("reason_code") or "") or None
    if outcome == "metadata_only":
        # Returned to the resolver but acquisition never ran on it.
        return "not_attempted", reason or "acquisition_not_run"
    return (outcome if outcome in _DISPOSITIONS else "unknown"), reason


def build_candidate_audit(
    retrieval_trace: list[dict], query_ids: dict[tuple[str, str], str]
) -> SearchCandidateAudit | None:
    """Collect auditable candidates from a web route's retrieval phases.

    ``query_ids`` maps (raw query, provider) to the trace's query id. Returns
    None when auditing is off or no auditable candidate was considered.
    """
    if not audit_enabled():
        return None
    records: dict[str, dict] = {}
    for phase in retrieval_trace or []:
        for item in phase.get("candidate_locations", []) or []:
            url = str(item.get("url") or "")
            if url:
                records.setdefault(url, dict(item))
        for item in phase.get("location_attempts", []) or []:
            url = str(item.get("url") or "")
            if url:
                records.setdefault(url, {}).update(item)
    entries: list[SearchCandidateAuditEntry] = []
    omitted = 0
    for url, record in records.items():
        provider = str(record.get("discovery_provider") or "").strip().lower()
        if provider not in AUDITABLE_PROVIDERS:
            continue  # Brave, unknown and non-search locations are never recorded.
        if len(entries) >= MAX_AUDIT_CANDIDATES:
            omitted += 1
            continue
        query = str(record.get("search_query") or "").strip()
        query_id = query_ids.get((query, provider)) if query else None
        # The provider-tier ranking position, not the acquisition order that
        # location attempts record as "rank". Derived locations have none.
        rank = record.get("search_rank")
        rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        disposition, reason = _disposition(record)
        entries.append(SearchCandidateAuditEntry(
            provider=provider,
            query_id=query_id[:128] if query_id else None,
            rank=rank if rank is not None and 1 <= rank <= 10_000 else None,
            url=url[:MAX_AUDIT_URL_LENGTH],
            url_truncated=len(url) > MAX_AUDIT_URL_LENGTH,
            disposition=disposition,
            reason_code=reason[:100] if reason else None,
        ))
    if not entries:
        return None
    return SearchCandidateAudit(candidates=entries, omitted_count=omitted)
