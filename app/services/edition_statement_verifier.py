"""Bounded deterministic publication-statement verifier, shadow v1.

Only identifies explicit reprint/reissue statements on already authorized
publication pages. It does not establish work identity, unchanged content,
page correspondence, source admission or any claim's usability.
"""
from __future__ import annotations
import hashlib
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

VERSION = 'explicit-reprint-statement-v1'
MAX_PAGE_CHARACTERS = 6000
MAX_TOTAL_CHARACTERS = 20000


class PublicationPage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    pdf_page_index: int = Field(ge=0, le=5)
    text: str = Field(min_length=1, max_length=MAX_PAGE_CHARACTERS)


class ReprintStatementFinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    version: Literal['explicit-reprint-statement-v1', 'publication-history-v2'] = VERSION
    status: Literal['documented_reprint', 'unresolved']
    reason: str
    representation_sha256: str
    submitted_reference_sha256: str
    cited_year: str
    retrieved_year: str
    page_index: int | None = None
    character_start: int | None = None
    character_end: int | None = None
    exact_statement: str | None = None
    statement_sha256: str | None = None
    pages_sha256: tuple[str, ...] = ()
    whole_work_equivalence: Literal[False] = False
    task_usability_granted: Literal[False] = False
    indexed_input_sha256: str | None = None


def verify_reprint_statement(pages: list[PublicationPage], *,
                             representation_sha256: str,
                             submitted_reference_sha256: str,
                             cited_year: str, retrieved_year: str) -> ReprintStatementFinding:
    """Require a single explicit dated relationship; ambiguity abstains.

    Caller must bind pages to the source bytes, authenticate scope and establish
    work identity independently. Supplied hashes are not proof of those facts.
    """
    if any(not re.fullmatch('[0-9a-f]{64}', value) for value in
           (representation_sha256, submitted_reference_sha256)):
        raise ValueError('invalid_source_or_reference_hash')
    if any(not re.fullmatch(r'(?:19|20)\d{2}', value) for value in (cited_year,retrieved_year)):
        raise ValueError('invalid_year')
    if not pages or len(pages)>6 or sum(len(p.text) for p in pages)>MAX_TOTAL_CHARACTERS:
        raise ValueError('publication_page_budget')
    if len({p.pdf_page_index for p in pages}) != len(pages):
        raise ValueError('duplicate_publication_page')
    base=dict(representation_sha256=representation_sha256,
              submitted_reference_sha256=submitted_reference_sha256,
              cited_year=cited_year,retrieved_year=retrieved_year,
              pages_sha256=tuple(hashlib.sha256(p.text.encode()).hexdigest() for p in pages))
    def unresolved(reason):
        return ReprintStatementFinding(status='unresolved',reason=reason,**base)
    if int(retrieved_year)<=int(cited_year):
        return unresolved('not_a_later_manifestation')
    text='\n'.join(p.text for p in pages)
    if re.search(r'\b(?:revised|abridged|translated|translation|adapted|expanded|corrected)\b',text,re.I):
        return unresolved('revision_or_translation_requires_review')
    if re.search(r'(?im)^\s*(?:catalogue|catalog|book review|review of|bibliography)\s*[:.]?\s*$',text):
        return unresolved('nonwork_or_listing_requires_review')
    # Deliberately narrow positive grammar; no inferred connection between
    # unrelated copyright, printing and historical dates.
    pattern=re.compile(
        r'(?im)^[ \t]*(?:This (?:edition|volume) (?:is |was )?)?'
        r'(?:[Aa]n? )?(?:unabridged |unchanged )?(?:reprint|reissue)'
        r'(?: published)?(?: in)? '+re.escape(retrieved_year)+
        r' of (?:the )?'+re.escape(cited_year)+r' edition[. \t]*$')
    candidates=[]
    for page in pages:
        for match in pattern.finditer(page.text):
            candidates.append((page,match))
    if len(candidates)!=1:
        return unresolved('missing_or_ambiguous_explicit_statement')
    page,match=candidates[0]
    # Other reprint years can indicate an ambiguous manifestation chain.
    for line in text.splitlines():
        if re.search(r'\breprint|\breissue',line,re.I):
            if set(re.findall(r'\b(?:19|20)\d{2}\b',line)) - {cited_year,retrieved_year}:
                return unresolved('conflicting_reprint_years')
    statement=match.group()
    return ReprintStatementFinding(status='documented_reprint',
        reason='explicit_publication_statement_only',page_index=page.pdf_page_index,
        character_start=match.start(),character_end=match.end(),exact_statement=statement,
        statement_sha256=hashlib.sha256(statement.encode()).hexdigest(),**base)


def verify_reprint_history(pages: list[PublicationPage], *,
                          representation_sha256: str,
                          submitted_reference_sha256: str,
                          cited_year: str, retrieved_year: str) -> ReprintStatementFinding:
    """V2 shadow screen for adjacent publication/reprint history lines.

    This verifies a documented printing history, not which printing supplied
    bytes came from. It grants no alternate-edition attestation. Existing V1
    callers and receipts retain their policy and behavior. Authorization,
    source safety, page extraction and work binding remain caller obligations.
    """
    import json

    validated = verify_reprint_statement(
        pages, representation_sha256=representation_sha256,
        submitted_reference_sha256=submitted_reference_sha256,
        cited_year=cited_year, retrieved_year=retrieved_year)
    base = validated.model_dump()
    base.update(version='publication-history-v2', status='unresolved',
                reason='missing_or_ambiguous_publication_history', page_index=None,
                character_start=None, character_end=None, exact_statement=None,
                statement_sha256=None)
    # Unlike a list of text hashes, this also binds each page's actual index.
    payload = dict(pages=[p.model_dump() for p in pages],
                   representation_sha256=representation_sha256,
                   submitted_reference_sha256=submitted_reference_sha256,
                   cited_year=cited_year, retrieved_year=retrieved_year,
                   version='publication-history-v2')
    base['indexed_input_sha256'] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def result(reason, **updates):
        return ReprintStatementFinding(**dict(base, reason=reason, **updates))

    if int(retrieved_year) <= int(cited_year):
        return result('not_a_later_manifestation')
    text = '\n'.join(p.text for p in pages)
    if re.search(r'\b(?:revised|abridged|expanded|corrected|adapted|translated)\s+'
                 r'(?:edition|version|reprint|text)\b|\bthis translation\b|'
                 r'\btranslated (?:by|from)\b', text, re.I):
        return result('revision_or_translation_requires_review')
    if re.search(r'(?im)^\s*(?:catalogue|catalog|book review|review of|bibliography)\s*[:.]?\s*$', text):
        return result('nonwork_or_listing_requires_review')
    pattern = re.compile(
        r'(?im)^[ \t]*First published[ \t]+' + re.escape(cited_year) +
        r'[ \t]*\r?\n[ \t]*Reprinted[ \t]+' + re.escape(retrieved_year) +
        r'[ \t]*$')
    matches = [(page, match) for page in pages for match in pattern.finditer(page.text)]
    if len(matches) != 1:
        return result('missing_or_ambiguous_publication_history')
    page, match = matches[0]
    if not re.search(r'©|\bcopyright\b', page.text, re.I) or not re.search(r'\bISBN\b', page.text, re.I):
        return result('publication_page_context_not_established')
    # Any additional dated edition/publication statement may refer to another
    # manifestation. Do not attach a historical reprint to the current copy.
    remaining = '\n'.join(
        p.text[:match.start()] + p.text[match.end():] if p is page else p.text
        for p in pages)
    if re.search(r'(?im)^.*\b(?:edition|reprint\w*|reissu\w*|first published|'
                 r'digital printing)\b[^\r\n]*\b(?:19|20)\d{2}\b', remaining):
        return result('additional_manifestation_requires_review')
    statement = match.group()
    return result('adjacent_publication_history_only', status='documented_reprint',
                  page_index=page.pdf_page_index, character_start=match.start(),
                  character_end=match.end(), exact_statement=statement,
                  statement_sha256=hashlib.sha256(statement.encode()).hexdigest())
