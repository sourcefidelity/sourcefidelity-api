"""Web page identity: whether a page, or an own-script index record, is the cited work (web-page-identity-v1).

Owner decisions 2026-10-21 (architecture review candidate 2, ARCHITECTURE §4).
Every route that reaches a page (a student's link, its archived copy, the
uncited-link check, a web-search page, the own-script route, an uploaded HTML
page) decides through this module:

- `ask_model` is the one model comparison: the same question on every route
  (the reference's title, author, year and container title; the page's titles,
  date and opening text), at most `CROSS_SCRIPT_PAGE_JUDGMENTS_PER_REFERENCE`
  per reference across all routes, none when
  `CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED` is off. A page asked about twice
  for one reference gets its first answer without a second call.
- `confirm_page` is the shared field comparison followed by the page-level
  rules that need the page itself, in one fixed order.

The model sits behind a seam: `DeepSeekJudge` in the application,
`FixedAnswerJudge` in tests (`use_judge`). A model yes never overrides a
conflict; the deterministic cross-language thresholds are unchanged.
"""
from __future__ import annotations

import contextvars
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field

VERSION = "web-page-identity-v1"


class DeepSeekJudge:
    """The configured model's comparison (cross-script-identity-v1)."""

    def __call__(self, reference: dict, page: dict) -> dict:
        from app.services.cross_script_identity import judge_cross_script_identity
        return judge_cross_script_identity(reference, page)


@dataclass
class FixedAnswerJudge:
    """Tests: the same answer for every question, or one answer per call in order."""
    answer: dict | list[dict] = field(default_factory=lambda: {"same_work": "unsure"})
    calls: list[tuple[dict, dict]] = field(default_factory=list)

    def __call__(self, reference: dict, page: dict) -> dict:
        self.calls.append((reference, page))
        answer = self.answer[min(len(self.calls), len(self.answer)) - 1] if isinstance(self.answer, list) else self.answer
        return {"version": "fixed-answer", "author": "not_shown", "page_authors": [], "call": "answered", **answer}


_JUDGE: contextvars.ContextVar = contextvars.ContextVar("web_page_identity_judge", default=None)


@contextmanager
def use_judge(judge):
    """Put a judge at the seam for the duration (tests)."""
    token = _JUDGE.set(judge)
    try:
        yield judge
    finally:
        _JUDGE.reset(token)


def _allowance() -> dict | None:
    """The current reference's model allowance, kept on its discovery trace."""
    from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE
    return _ACTIVE_DISCOVERY_TRACE.get()


def ask_model(*, title: str, author: str | None, year: str | None, page_titles: list,
              page_date: str | None, page_text: str | None, container_title: str | None = None) -> dict | None:
    """The model's comparison of the reference with a page, or None when it may
    not be asked (setting off, allowance spent, nothing to compare)."""
    from app.config import settings
    titles = [str(t) for t in (page_titles or []) if str(t).strip()]
    if not settings.CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED or not title or not titles:
        return None
    reference = {"title": title, "author": author or "", "year": year or "", "container_title": container_title or ""}
    page = {"titles": titles, "date": page_date or "", "text": page_text or ""}
    from app.services.cross_script_identity import _payload
    key = hashlib.sha256(_payload(reference, page).encode()).hexdigest()
    trace = _allowance()
    asked = trace.setdefault("web_page_identity_answers", {}) if trace is not None else {}
    if key in asked:
        return asked[key]
    if trace is not None:
        used = trace.get("cross_script_page_judgments", 0)
        if used >= settings.CROSS_SCRIPT_PAGE_JUDGMENTS_PER_REFERENCE:
            return None
        trace["cross_script_page_judgments"] = used + 1
    judged = (_JUDGE.get() or DeepSeekJudge())(reference, page)
    asked[key] = judged
    return judged


def model_admits(judged: dict | None, cited_author: str | None) -> bool:
    """The model judged the page the cited work and found no cited person
    missing (an organisation's name is a translation and never counts against it)."""
    if (judged or {}).get("same_work") != "yes":
        return False
    if judged.get("author") != "differs":
        return True
    from app.services.source_validator import _cited_personal_surnames
    return bool(cited_author) and not _cited_personal_surnames(cited_author)


@dataclass
class PageVerdict:
    identity: object
    comparisons: list
    confirmed: bool
    basis: str | None = None
    authors: list[str] | None = None

    @property
    def conflict(self) -> bool:
        return bool(self.identity.has_material_conflict)


def confirm_page(result, *, title, author, year, doi, page_text: str | None = None, html: str | None = None,
                 page_titles: list | None = None, romanised_page: str | None = None,
                 judged: dict | None = None) -> PageVerdict:
    """The shared field comparison, then the page-level rules in a fixed order:
    an organisation author named in the page text, the model's same-work answer,
    a Han byline beside the page's title. Each applies only while unconfirmed
    and never over a conflict."""
    from app.services.source_resolver import (
        _organisation_author_in_page, _romanised_author_near_title, _web_identity_comparison,
    )
    identity, comparisons, confirmed = _web_identity_comparison(result, title=title, author=author, year=year, doi=doi)
    verdict = PageVerdict(identity, comparisons, confirmed)
    if confirmed or identity.has_material_conflict:
        return verdict
    if _organisation_author_in_page(author, comparisons, page_text):
        # "Presented by the UCLA Film & Television Archive": an organisation
        # author is named in the page text, not as a byline (2026-10-01).
        verdict.confirmed, verdict.basis = True, "web-organisation-author-in-page-v1"
        return verdict
    if model_admits(judged, author):
        verdict.confirmed, verdict.basis = True, "cross-script-identity-v1"
        return verdict
    fields = {c.field_name: c.outcome for c in comparisons}
    if (romanised_page is not None and html and fields.get("title") in {"agreement", "minor_difference"}
            and fields.get("author") == "unknown"):
        # A page in the work's own language names its author beside the
        # title, outside its metadata ("潘越" for "Pan, Y.", 2026-10-12).
        people = _romanised_author_near_title(html, list(page_titles or []), author)
        if people:
            verdict.confirmed, verdict.basis, verdict.authors = True, "cross-language-identity-v1", people
    return verdict



def nearby_publication_year(comparisons, *, expected_kind, observed_kind, expected_year, observed_year) -> bool:
    """A one-year bibliographic error must not make a clearly identified
    journal article unreadable: title and author agree, the page is confidently
    a journal article, and the years differ by one. The year conflict stays in
    the discovery and link evidence; this is work-level acquisition only."""
    import re
    fields = {c.field_name: c.outcome for c in comparisons}
    return bool(
        expected_kind == "journal_article"
        and observed_kind.kind == "journal_article" and observed_kind.confidence == "high"
        and fields.get("title") == fields.get("author") == "agreement"
        and {c.field_name for c in comparisons if c.outcome == "material_conflict"} == {"year"}
        and re.fullmatch(r"\d{4}", str(expected_year or ""))
        and re.fullmatch(r"\d{4}", str(observed_year or ""))
        and abs(int(expected_year) - int(observed_year)) == 1
        and all(c.outcome in {"agreement", "minor_difference"} for c in comparisons if c.field_name != "year")
    )
