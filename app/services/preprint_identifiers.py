"""An arXiv identifier carried in a reference is a retrieval route, not a title.

Reference lists commonly end a preprint entry with `arXiv:1609.08144` or
`arXiv preprint arXiv:2302.09210`. When the parser leaves that token inside the
title, three things fail at once and none of them look like a parsing problem:

  * every title-led provider scores the real work as a keyword coincidence,
    because the query carries an identifier the indexed title does not;
  * no identifier route runs at all, since the `doi` field stays empty; and
  * the reference has no venue, so it classifies as `unknown` kind.

Under `bounded-reference-review-v9` an unclassified reference is reviewed, so
the combined effect was a *fabrication finding against a real, heavily cited
preprint*. arXiv has registered DataCite DOIs for its whole corpus, so the
identifier yields a direct, deterministic route.
"""

import re

ARXIV_DOI_PREFIX = "10.48550/arXiv."

_MODERN = r"\d{4}\.\d{4,5}"
_LEGACY = r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}"
# The version suffix identifies a revision, not a different work; arXiv's DOI
# is registered against the unversioned identifier.
# "arXiv preprint arXiv:1706.03762" repeats the label; "arXiv 2302.09210" and
# "arXiv:1609.08144" do not. One pattern serves both the free search and the
# title-adjacent match so the two cannot drift apart.
_ARXIV_TOKEN_SRC = (
    rf"(?:arxiv\s*(?:preprint)?\s*[:\s]\s*)(?:arxiv\s*[:\s]\s*)?"
    rf"({_MODERN}|{_LEGACY})(v\d+)?\b"
)
_ARXIV_TOKEN = re.compile(rf"\b{_ARXIV_TOKEN_SRC}", re.IGNORECASE)
_ARXIV_TOKEN_AFTER_TITLE = re.compile(rf"[\s.,;:]*{_ARXIV_TOKEN_SRC}", re.IGNORECASE)
_TRAILING_SEPARATORS = " .,;:-–—"


def arxiv_identifier(text: str | None) -> str | None:
    """Return the unversioned arXiv identifier carried in this text, if any."""
    if not text:
        return None
    match = _ARXIV_TOKEN.search(text)
    return match.group(1) if match else None


def arxiv_doi(identifier: str) -> str:
    return f"{ARXIV_DOI_PREFIX}{identifier}"


def strip_arxiv_identifier(text: str | None) -> str:
    """Remove the identifier token, leaving the title it was appended to."""
    if not text:
        return ""
    cleaned = _ARXIV_TOKEN.sub("", text)
    # "arXiv preprint arXiv:1706.03762" leaves a bare label behind once the
    # identifier is removed; it is not part of the title either.
    cleaned = re.sub(
        r"[\s.,;:]*\barxiv(?:\s+preprint)?\b[\s.,;:]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned.strip(_TRAILING_SEPARATORS).strip()


def _identifier_adjacent_to_title(raw_ref: str | None, title: str | None) -> str | None:
    """An identifier that immediately follows the title is the entry's own.

    The whole reference string is not searched: an entry can *mention* another
    work's identifier ("Reprinted from arXiv:1111.2222", "see also arXiv:...")
    and assigning that as this reference's DOI would confirm it against the
    wrong work through the highest-trust route. Only the position the parser
    was observed to leave the token in - directly after the title - counts.
    """
    if not raw_ref or not title:
        return None
    normalized_raw = " ".join(raw_ref.split())
    normalized_title = " ".join(title.split())
    index = normalized_raw.casefold().find(normalized_title.casefold())
    if index < 0:
        return None
    tail = normalized_raw[index + len(normalized_title):]
    match = _ARXIV_TOKEN_AFTER_TITLE.match(tail)
    return match.group(1) if match else None


def apply_preprint_identity(reference):
    """Move an arXiv identifier out of the title and into a resolvable DOI.

    Mutates in place and returns the reference. A DOI the parser already found
    is authoritative and is never replaced; only an absent one is supplied.
    """
    if reference is None:
        return reference
    title = getattr(reference, "title", "") or ""
    identifier = arxiv_identifier(title) or _identifier_adjacent_to_title(
        getattr(reference, "raw_ref", ""), title
    )
    if not identifier:
        return reference
    if arxiv_identifier(title):
        stripped = strip_arxiv_identifier(title)
        # Never trade a usable title for an empty one.
        if stripped:
            reference.title = stripped
    if not (getattr(reference, "doi", "") or "").strip():
        reference.doi = arxiv_doi(identifier)
    return reference
