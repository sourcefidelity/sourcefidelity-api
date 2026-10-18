"""Source reuse: whether an earlier check's result, or a text already in the
source library, stands in for a new search (2026-10-15).

Two questions with one rule set behind them:

* ``earlier_result``: may a reference reuse a previous check's whole search
  and source result (``paper-search-reuse-v5``, owner decisions 2026-10-01 to
  2026-10-14)?
* ``stored_copy``: may a text already in the source library stand in for the
  cited work?

Both answer with ``Reuse``, whose ``reason`` names a refusal so a reference
searched again says why (provenance only). They share ``stored_text_today``,
which re-tests a stored text against today's admission rules, and
``ROUTE_VERSIONS``, the one table of route versions and what bumping each
reads again. Whole-paper reuse (``paper_reuse``) is a different question and
stays separate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reuse:
    reusable: bool
    reason: str = ""


REUSE = Reuse(True)


def refuse(reason: str) -> Reuse:
    return Reuse(False, reason)


@dataclass(frozen=True)
class RouteVersion:
    version: str
    rereads: str


# Every route version reuse reads, and what bumping it reads again. Bumping
# changes which references are searched again, and so cost: a decision, not
# a refactor.
ROUTE_VERSIONS = {
    # v2, 2026-10-14: Media History Digital Library pages, dead-link answers,
    # linked Chinese abstract pages; v3, the same day: such abstracts by the
    # cross-language naming standard, the one page-identity rule for DOI
    # pages; v4, Chinese author-field labels are not names; v5, the model's
    # same-work answer for a page naming the work also keeps its abstract.
    "student_link_routes": RouteVersion(
        "student-link-routes-v5",
        "an earlier result with a student link and no text, or a search-found page's text, "
        "is searched again once; a stored copy or text the student's own link gave is kept"),
    # Owner decision 2026-10-14: pages separated by form feeds with printed page numbers.
    "lantern_pages": RouteVersion(
        "lantern-page-labels-v1",
        "a stored Lantern text without printed page numbers is read again once, from an "
        "earlier result or the source library"),
    # 2026-10-16 (Rita Hayworth review): the magazine archive search, the
    # chapter-title namesake rule and look-alike letter folding.
    "search_rules": RouteVersion(
        "search-rules-v6",
        "an earlier result without text for a dated periodical article, a chapter whose search "
        "ended in a record conflict or whose book's authorship was unresolved, or a title with "
        "look-alike letters is searched again once"),
}
STUDENT_LINK_ROUTES_VERSION = ROUTE_VERSIONS["student_link_routes"].version
SEARCH_RULES_VERSION = ROUTE_VERSIONS["search_rules"].version
LANTERN_PAGES_VERSION = ROUTE_VERSIONS["lantern_pages"].version

_REUSABLE_STATUSES = frozenset({"durable_authorized", "unavailable", "abstract_only", "metadata_only"})
_INCOMPLETE_REASONS = frozenset({"full_text_search_incomplete", "transient_verification_run_unavailable",
                                 "transient_source_admission_failed", "durable_representation_not_authorized"})
# Routes that read the student's own link (its page, its file, its DOI, its
# Media History Digital Library pages).
_STUDENT_LINK_SOURCES = frozenset({"student_url", "web_fetch", "lantern", "doi_resolver"})


# ---------------------------------------------------------------- shared


def stored_text_today(text: str, *, title: str | None, completeness: str) -> tuple[Reuse, str]:
    """A stored plain text against today's admission rules (2026-10-13): an
    index record is not the work, and a page of notes or listings is never its
    whole text. Returns the decision and the completeness to use."""
    from app.services.abstract_only_pages import mostly_notes_or_listings
    from app.services.source_validator import _detect_nonwork_listing
    if _detect_nonwork_listing(" ".join(text.split()), title):
        return refuse("stored_text_is_a_listing"), completeness
    if str(completeness or "").casefold() == "complete" and mostly_notes_or_listings(text):
        return REUSE, "not_assessed"
    return REUSE, completeness


def lantern_copy_stale(record) -> bool:
    """A stored Lantern text from before its pages carried printed page numbers."""
    return (getattr(record, "provenance", None) == "lantern"
            and not (getattr(record, "validation_evidence", None) or {}).get("printed_page_labels"))


# ---------------------------------------------------------------- stored copy


def stored_copy(record, data: bytes, *, kind, title, author, year, normalize_author,
                edition_check=None) -> tuple[Reuse, str]:
    """May this library text stand in for the cited work? Returns the decision
    and the completeness to use. `edition_check`, given when the lookup matched
    only after an edition qualifier was dropped from the title, re-checks the
    copy itself (the resolver's PDF validation). The lookup and the identity
    re-check stay with the resolver."""
    import hashlib
    from app.services.retrieval.base import RepresentationKind
    from app.services.source_repository import uploaded_limited_text
    if uploaded_limited_text(record):
        pass  # the user's own upload as limited text (owner decision 2026-10-18)
    elif record.completeness_verdict == 'incomplete':
        from app.services.publisher_preview import valid_preview_receipt
        if (not author or not year or record.canonical_work.year != year
                or normalize_author(record.canonical_work.author) != normalize_author(author)
                or not valid_preview_receipt((record.validation_evidence or {}).get('publisher_preview'),
                                             hashlib.sha256(data).hexdigest(), record.source_url)):
            return refuse("limited_preview_not_revalidated"), record.completeness_verdict
    if edition_check is not None:
        if kind != RepresentationKind.PDF:
            return refuse("edition_candidate_not_pdf"), record.completeness_verdict
        if not edition_check():
            return refuse("edition_candidate_not_revalidated"), record.completeness_verdict
    if kind == RepresentationKind.PLAIN_TEXT:
        return stored_text_today(data.decode("utf-8", "replace"), title=title,
                                 completeness=record.completeness_verdict)
    return REUSE, record.completeness_verdict


# ---------------------------------------------------------------- earlier result


def public_copy(item: dict) -> dict | None:
    """The earlier run's verified open-access address and file hash, if kept."""
    access = item.get("public_source_access") or {}
    href, digest = str(access.get("href") or ""), str(access.get("content_sha256") or "")
    if (access.get("version") == "verified-public-source-access-v1" and href.startswith("https://")
            and len(digest) == 64 and item.get("source_name")):
        return {"href": href, "content_sha256": digest, "provider": str(item["source_name"])}
    return None


def downloadable(item: dict) -> bool:
    """A verified open-access copy can be downloaded again: a transient text,
    or one whose admission lapsed during a long run after it was verified
    (`transient_verification_run_unavailable`, the long article, 2026-10-08)."""
    return public_copy(item) is not None and (
        item.get("status") == "transient_authorized"
        or item.get("reason_code") == "transient_verification_run_unavailable")


def earlier_result(item: dict, reference: dict | None = None) -> Reuse:
    """May this reference reuse a previous check's result instead of searching?"""
    if _student_link_routes_due(item, reference):
        return refuse("student_link_routes_changed")
    if item.get("source_name") == "lantern" and item.get("lantern_pages") != LANTERN_PAGES_VERSION:
        return refuse("lantern_pages_changed")
    if search_rules_due(item, reference):
        # Before the stored-text exits: a stored copy is found again at once,
        # and its book's authorship is what changed (2026-10-16).
        return refuse("search_rules_changed")
    if item.get("status") == "transient_authorized" or downloadable(item):
        return REUSE if downloadable(item) else refuse("transient_text_not_kept")
    if item.get("status") == "durable_authorized":
        # The text itself is stored; how the search ended does not matter.
        return REUSE if item.get("representation_id") else refuse("stored_copy_missing")
    if native_script_route_due(item, reference):
        return refuse("native_script_search_due")
    discovery = item.get("reference_discovery") or {}
    if (isinstance(reference, dict) and reference.get("source_kind") == "book_section"
            and (reference.get("container_title") or "").strip()
            and discovery.get("outcome") not in {"confirmed", "confirmed_with_minor_differences"}
            and not discovery.get("container_identity")):
        # An unconfirmed chapter whose book was never identified is searched
        # again, so a corrected book lookup reaches it (Gershon, 2026-10-04).
        return refuse("chapter_book_unidentified")
    if item.get("status") == "link_check_only":
        # An uncited web page's own search (webpage-verification-v1): reused
        # once its web providers answered (paper 10's Statista, 2026-10-04).
        return (REUSE if item.get("web_page_check") and _web_search_answered(discovery)
                else refuse("web_page_check_unanswered"))
    if item.get("status") not in _REUSABLE_STATUSES or item.get("reason_code") in _INCOMPLETE_REASONS:
        return refuse("result_incomplete")
    if item.get("full_text_search_incomplete_providers"):
        return refuse("full_text_search_incomplete")
    if item.get("retryable_provider_dependencies"):
        # An optional provider left a retry, but every search the reference's
        # kind requires finished (paper 10: OpenAlex re-searched each run, 2026-10-04).
        return (REUSE if _required_searches_completed(item, reference)
                else refuse("required_search_unfinished"))
    if _network_failed(discovery):
        return refuse("network_failed")
    if discovery.get("outcome") == "search_incomplete":
        return (REUSE if _required_searches_completed(item, reference) or _no_provider_at_fault(discovery)
                else refuse("search_incomplete"))
    return REUSE if discovery else refuse("no_search_record")


def search_rules_due(item: dict, reference) -> bool:
    """An earlier result without text that today's search rules would change
    (`search_rules` above): searched again once, then reused as before."""
    if not isinstance(reference, dict) or item.get("search_rules") == SEARCH_RULES_VERSION:
        return False
    import re
    from app.services.lookalike_letters import fold_lookalikes
    discovery = item.get("reference_discovery") or {}
    periodical = (reference.get("source_kind") in {"journal_article", "news_article", "unknown"}
                  and (reference.get("container_title") or "").strip()
                  and re.match(r"^\d{4}", str(reference.get("year") or "")))
    container = discovery.get("container_identity") or {}
    chapter = (reference.get("source_kind") == "book_section"
               and (reference.get("container_title") or "").strip())
    namesake = (chapter
                and (not container  # a confirmed chapter whose book was never checked
                     or discovery.get("outcome") == "bibliographic_conflict"
                     # The book's authorship was unread: Open Library's main title
                     # and added-material statements are now read.
                     or (container and (container.get("responsibility") or {}).get("classification") == "unresolved")
                     # A single-authored book without the catalogue's own record:
                     # its text is now sought as the book (v5, 2026-10-18).
                     or (container.get("is_monograph") and not container.get("located_record"))
                     # v6: its text not recorded as the book's (and any copy
                     # taken from a repository metadata export), 2026-10-18.
                     or (container.get("is_monograph") and item.get("status") == "transient_authorized"
                         and not item.get("held_as_book_of_cited_part"))))
    lookalike = fold_lookalikes(reference.get("title") or "") != (reference.get("title") or "")
    # A reference that already has its text needs only its book's authorship
    # checked; the search rules for finding a text do not apply to it.
    has_text = item.get("status") in {"durable_authorized", "transient_authorized"} or downloadable(item)
    return bool(namesake or (not has_text and (periodical or lookalike)))


def native_script_route_due(item: dict, reference) -> bool:
    """A likely other-language work whose earlier result has no completed
    search in its own script, and was not confirmed, is searched again
    (native-script-route-v1): reusing it would keep the reference neutral
    forever, or reuse a search that never looked where the work is."""
    if not isinstance(reference, dict):
        return False
    from app.services.other_language import other_language_signals
    if not other_language_signals(reference):
        return False
    discovery = item.get("reference_discovery") or {}
    if discovery.get("outcome") in {"confirmed", "confirmed_with_minor_differences"}:
        return False
    from app.services.reference_verification import NATIVE_SCRIPT_ROUTE_POLICIES
    route = (discovery.get("native_script_route")
             or (item.get("reference_discovery_trace") or {}).get("native_script_route") or {})
    return not (route.get("policy_version") in NATIVE_SCRIPT_ROUTE_POLICIES and route.get("status") == "completed")


def _student_link_routes_due(item: dict, reference: dict | None) -> bool:
    """A reference with a student link whose earlier result was searched under
    older student-link handling, and holds no text or holds a search-found
    page's text: searched again once, so newer link routes run (the Jean
    Harlow paper's Lantern links, the Regulation paper's Chinese abstract
    pages and Kinzius et al.'s DOI page, 2026-10-14). A stored copy, or text
    the student's own link gave, is reused as before."""
    if not isinstance(reference, dict) or not (reference.get("url") or reference.get("doi")):
        return False
    if item.get("student_link_routes") == STUDENT_LINK_ROUTES_VERSION:
        return False
    if item.get("status") == "durable_authorized":
        return False
    if item.get("status") == "transient_authorized":
        return item.get("source_name") not in _STUDENT_LINK_SOURCES
    return True


def _required_searches_completed(item: dict, reference: dict | None) -> bool:
    """reference-verification found every required search complete."""
    if not reference:
        return False
    from app.services.reference_verification import COMPLETED_WITHOUT_MATCH_REASONS, assess_reference_verification
    from app.services.schemas import ParsedReference
    try:
        parsed = ParsedReference.model_validate(reference)
    except Exception:
        return False
    verdict = assess_reference_verification(parsed, item.get("reference_discovery"))
    return verdict.get("reason_code") in COMPLETED_WITHOUT_MATCH_REASONS


def _no_provider_at_fault(discovery: dict) -> bool:
    """Incomplete only for reasons a repeat would meet again: every required
    provider answered, and the gaps are filtered index records, unresolved
    web-search leads or pages that did not confirm the work (owner decision
    2026-10-07; Franchise 2's 8 such references were searched again each run).
    A provider whose failure holds the search incomplete still forces a new
    search; so does a reference whose parse was incomplete."""
    from app.services.paper_workflow import _blocking_providers_from_record
    if (discovery.get("expected") or {}).get("reference_parse_review"):
        return False
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    if not any(a.get("route_category") == "bounded_web" for a in attempts):
        return False
    return not _blocking_providers_from_record(discovery)


def _web_search_answered(discovery: dict) -> bool:
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    web = [a for a in attempts if a.get("route_category") == "bounded_web"]
    return bool(web) and all(a.get("outcome") in {"candidate_found", "no_match"} for a in web)


def _network_failed(discovery: dict) -> bool:
    """The earlier run could not reach the student's link, or most indexes:
    a network failure, not a finished search (2026-10-02)."""
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    # A failed student link matters only when nothing else confirmed the work:
    # a confirmed reference is not searched again for it (paper 4 re-run,
    # 2026-10-02: 15 confirmed references re-searched, USD 0.35).
    if (discovery.get("outcome") not in {"confirmed", "confirmed_with_minor_differences"}
            and any(a.get("route_category") == "student_url" and a.get("outcome") == "operational_failure"
                    for a in attempts)):
        return True
    adapters = [a for a in attempts if a.get("route_category") == "academic_adapter"
                and a.get("outcome") != "unavailable"]
    failed = sum(1 for a in adapters if a.get("outcome") == "operational_failure")
    return bool(adapters) and failed * 2 >= len(adapters)
