"""Bounded journal registration checks, not an exhaustive publisher archive."""
from __future__ import annotations
import hashlib
import json
import re
import time
import unicodedata
from pydantic import BaseModel, Field
from app.log_safety import safe_exception_code


def normalized(value):
    return ' '.join(re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', value).casefold()).split())


class JournalClaim(BaseModel):
    journal: str = Field(max_length=1000)
    year: str
    volume: str
    issue: str
    pages: str
    reference_sha256: str


class JournalCheck(BaseModel):
    policy_version: str = 'journal-registration-v1'
    claim: JournalClaim
    stage: str
    query: str
    outcome: str
    elapsed_seconds: float = 0
    provider_calls: int = 0
    response_sha256: str | None = None
    issn: str | None = None
    returned_records: int = 0
    total_registered_records: int | None = None
    coverage: str = 'registered_records_only_not_complete_journal_archive'
    page_overlap_dois: list[str] = Field(default_factory=list, max_length=50)
    excluded_records: dict[str, int] = Field(default_factory=dict)


def journal_claim(reference):
    """Original APA volume(issue), pages only; no guessed journal from a topic."""
    if reference.needs_review or reference.source_kind != 'journal_article' or not reference.title:
        return None
    raw = reference.raw_ref
    if raw.count(reference.title) != 1 or not re.fullmatch(r'(?:18|19|20)\d{2}[a-z]?', reference.year or ''):
        return None
    tail = raw.split(reference.title, 1)[1]
    match = re.fullmatch(r'\s*[.]?\s*(?P<journal>[^,\n]{3,250}),\s*(?P<volume>\d{1,4})'
        r'\s*\((?P<issue>\d{1,3})\)\s*,\s*(?P<pages>\d{1,5}\s*[-–]\s*\d{1,5})'
        r'\s*[.]?\s*(?:(?:https?://|doi:)\S+)?\s*', tail)
    if not match or re.search(r'https?://|\bdoi\b', match['journal'], re.I):
        return None
    start, end = [int(n) for n in re.split(r'[-–]', match['pages'])]
    if end < start:
        return None
    return JournalClaim(**match.groupdict(), year=reference.year[:4],
        reference_sha256=hashlib.sha256(raw.encode()).hexdigest())


def registration_checks(adapter, claim, title):
    """Yield at most three recorded requests. Caller uses normal identity/budgets."""
    from app.services.retrieval.crossref import CROSSREF_BASE
    issn = None
    stages = [('journal', '/journals', {'query': claim.journal, 'rows': 5}),
              ('title', None, {'query.title': title, 'rows': 5}),
              ('year_issue', None, {'filter': f'from-pub-date:{claim.year}-01-01,until-pub-date:{claim.year}-12-31', 'rows': 50})]
    for stage, path, params in stages:
        if stage != 'journal':
            if not issn: return
            path = f'/journals/{issn}/works'
        query = json.dumps({'path':path,'params':params}, sort_keys=True)
        check = JournalCheck(claim=claim, stage=stage, query=query, outcome='operational_failure', issn=issn)
        started = time.monotonic()
        rows = []
        try:
            check.provider_calls = 1
            response = adapter._get(CROSSREF_BASE+path, params)
            response.raise_for_status()
            payload = response.json()
            message = payload['message']; items = message['items']
            if payload.get('status') != 'ok' or not isinstance(items, list) or len(items)>params['rows']:
                raise ValueError('Invalid bounded registry response')
            check.response_sha256 = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            check.returned_records = len(items)
            total = message.get('total-results')
            check.total_registered_records = total if type(total) is int and total >= 0 else None
            if stage == 'journal':
                matches = [r for r in items if normalized(str(r.get('title',''))) == normalized(claim.journal)]
                if len(matches) == 1:
                    identifiers = [v for v in matches[0].get('ISSN',[]) if re.fullmatch(r'\d{4}-\d{3}[\dXx]', v)]
                    if identifiers: issn = sorted(identifiers)[0]
                check.issn = issn
                check.outcome = 'journal_identified' if issn else 'journal_unresolved'
            else:
                for item in items:
                    reason = None
                    if item.get('type') != 'journal-article': reason = 'not_journal_article'
                    elif not any(normalized(v) == normalized(claim.journal) for v in item.get('container-title', [])): reason = 'journal_not_matched'
                    elif issn not in item.get('ISSN',[]): reason = 'issn_not_matched'
                    elif stage == 'year_issue' and (str(item.get('volume','')) != claim.volume or str(item.get('issue','')) != claim.issue): reason = 'issue_not_matched'
                    if reason:
                        check.excluded_records[reason] = check.excluded_records.get(reason, 0) + 1
                        continue
                    rows.append(adapter._parse_message(item))
                    if stage == 'year_issue' and rows[-1].year == claim.year:
                        page = re.fullmatch(r'(\d+)[-–](\d+)', str(item.get('page','')))
                        low,high = [int(v) for v in re.split('[-–]', claim.pages)]
                        if page and int(page[1]) <= high and int(page[2]) >= low and item.get('DOI'):
                            check.page_overlap_dois.append(str(item['DOI']))
                check.outcome = 'candidates_found' if rows else 'no_registered_candidates'
        except Exception as exc:
            check.outcome = safe_exception_code(exc)
            rows = []
        finally:
            check.elapsed_seconds = time.monotonic()-started
        yield check, rows
        if check.outcome not in {'journal_identified','candidates_found','no_registered_candidates'}:
            return
