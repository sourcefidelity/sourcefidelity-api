"""Title-relevance scoring for retrieval result filtering.

When a reference lacks a DOI/URL and falls back to title search in OpenAlex or
CORE, the results can be keyword-coincidence matches rather than the actual
cited work — e.g. the film "Rain Man" matches a diabetology paper whose title
contains "Man". This module scores how likely a matched title is to be the
genuine cited work, so junk matches can be rejected.

Approach: tokenise both the query (cited title) and the matched title, then
measure what fraction of the query's *significant* tokens appear in the match.
Short queries (1-2 significant tokens) require complete coverage; longer
queries allow one or two missing tokens. Common stopwords don't count.
"""

import re
from dataclasses import dataclass

# Stopwords excluded from significance scoring.
_STOPWORDS = frozenset(
    """
    a an the and or of in on at to for with from by as is are was were be been
    being this that these those it its their his her our your my we us you they
    them he she i which who whom whose what when where why how all any both each
    few more most other some such no nor not only own same so than too very can
    will just should now
    """.split()
)

# Tokens must contain at least one letter; pure numbers excluded from
# significance (years, volumes) but can be checked separately if needed.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")


@dataclass
class RelevanceScore:
    """Result of comparing a cited title to a matched title.

    Attributes:
        is_relevant: whether the match should be accepted as likely-genuine.
        score: 0.0-1.0, fraction of query's significant tokens found in match.
        detail: human-readable explanation.
    """

    is_relevant: bool
    score: float
    detail: str


def score_title_relevance(query_title: str, matched_title: str) -> RelevanceScore:
    """Score how likely ``matched_title`` is the work cited by ``query_title``.

    Based on coverage of the query's significant (non-stopword) tokens.
    Threshold scales with query length: short queries require ~100% coverage;
    longer queries allow one or two missing tokens.

    Examples:
        "Rain Man" vs "Homeostasis model assessment: insulin resistance..."
            -> NOT relevant (only "man" matched, "rain" missing)
        "Harry Potter" vs "Harry Potter and the Prisoner of Azkaban"
            -> relevant (all query tokens present)
    """
    q_tokens = _significant_tokens(query_title)
    m_tokens = set(_significant_tokens(matched_title))

    if not q_tokens:
        # No significant tokens to match on (e.g. query was all stopwords).
        # Can't judge relevance -- default to accepting (let caller decide).
        return RelevanceScore(True, 1.0, "no significant tokens in query; accepting by default")

    q_set = set(q_tokens)
    covered = q_set & m_tokens
    missing = q_set - m_tokens
    coverage = len(covered) / len(q_set)

    # Threshold scales with query length.
    n = len(q_set)
    if n <= 2:
        # Very short query: require all significant tokens present.
        threshold = 1.0
    elif n <= 4:
        # Allow one missing token.
        threshold = (n - 1) / n
    else:
        # Allow up to two missing tokens for longer queries.
        threshold = max(0.6, (n - 2) / n)

    is_relevant = coverage >= threshold

    if is_relevant:
        detail = (
            f"relevant: {len(covered)}/{len(q_set)} significant tokens "
            f"({', '.join(sorted(covered))})"
        )
        if missing:
            detail += f"; missing: {', '.join(sorted(missing))}"
    else:
        detail = (
            f"NOT relevant: {len(covered)}/{len(q_set)} significant tokens "
            f"({', '.join(sorted(covered)) if covered else 'none'}); "
            f"missing: {', '.join(sorted(missing))}; needed >= {threshold:.0%}"
        )

    return RelevanceScore(is_relevant, coverage, detail)


def _significant_tokens(text: str) -> list[str]:
    """Extract significant (non-stopword) lowercase word tokens from text."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Author verification
# ---------------------------------------------------------------------------

# Minimum surname similarity for an author match (configurable, but 0.7 is
# a reasonable default — handles "Smith" vs "Smyth" but rejects "Smith" vs "Jones").
_AUTHOR_SIMILARITY_THRESHOLD = 0.7


def extract_surnames(author_string: str) -> list[str]:
    """Extract surname(s) from an author string in various citation formats.

    Handles:
      "Dickens, C."           -> ["dickens"]
      "Charles Dickens"       -> ["dickens"]
      "Smith, J. & Jones, A." -> ["smith", "jones"]
      "Parc, J., Messerlin, P., & Kim, K." -> ["parc", "messerlin", "kim"]
    """
    if not author_string:
        return []

    surnames: list[str] = []
    # Split on common multi-author separators
    authors = re.split(r"(?:,?\s*(?:&|and)\s*|;\s*)", author_string)
    for a in authors:
        a = a.strip().strip(".")
        if not a:
            continue
        # Format "Surname, Initials" -> surname is before the comma
        if "," in a:
            surname = a.split(",")[0].strip()
        else:
            # Format "First Last" -> surname is the last word
            parts = a.split()
            surname = parts[-1] if parts else a
        # Clean: lowercase, strip non-alpha
        surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
        if len(surname) > 1:
            surnames.append(surname)
    return surnames


def _surname_similarity(s1: str, s2: str) -> float:
    """Similarity between two surnames (0.0–1.0) using SequenceMatcher."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio()


def verify_authors(
    cited_authors: str | list[str] | None,
    matched_authors: str | list[str] | None,
) -> tuple[bool, float, str]:
    """Check whether the cited author(s) plausibly match the matched work's author(s).

    Compares surnames (format-independent) with a similarity threshold.
    A match is accepted if ANY cited surname is similar (≥0.7) to ANY matched
    surname — handles multi-author works where one author suffices for identity.

    Args:
        cited_authors: The student's cited author string (e.g. "Dickens, C.").
            Can also be a list of author name strings.
        matched_authors: The retrieved work's author string(s).

    Returns:
        (passes: bool, similarity: float, detail: str)
    """
    # Normalize to surname lists
    if isinstance(cited_authors, list):
        cited_surnames = []
        for a in cited_authors:
            cited_surnames.extend(extract_surnames(a))
    else:
        cited_surnames = extract_surnames(cited_authors or "")

    if isinstance(matched_authors, list):
        matched_surnames = []
        for a in matched_authors:
            matched_surnames.extend(extract_surnames(a))
    else:
        matched_surnames = extract_surnames(matched_authors or "")

    if not cited_surnames:
        # No cited author to compare — can't verify, accept by default
        return True, 1.0, "no cited author; skipping author check"
    if not matched_surnames:
        # Matched work has no author data — accept by default
        return True, 1.0, "matched work has no author data; skipping"

    # Find the best surname match across all cited/matched pairs
    best_sim = 0.0
    best_pair = ("", "")
    for cs in cited_surnames:
        for ms in matched_surnames:
            sim = _surname_similarity(cs, ms)
            if sim > best_sim:
                best_sim = sim
                best_pair = (cs, ms)

    passes = best_sim >= _AUTHOR_SIMILARITY_THRESHOLD
    if passes:
        detail = f"author match: {best_pair[0]!r} ~ {best_pair[1]!r} ({best_sim:.0%})"
    else:
        detail = (
            f"author MISMATCH: cited {cited_surnames} vs matched {matched_surnames} "
            f"(best: {best_pair[0]!r} ~ {best_pair[1]!r} = {best_sim:.0%}, "
            f"need ≥{_AUTHOR_SIMILARITY_THRESHOLD:.0%})"
        )
    return passes, best_sim, detail


def score_relevance(
    query_title: str,
    matched_title: str,
    cited_authors: str | list[str] | None = None,
    matched_authors: str | list[str] | None = None,
) -> RelevanceScore:
    """Combined title + author relevance check.

    A match is relevant only if BOTH the title tokens are sufficiently covered
    AND the cited author plausibly matches the matched work's author.
    """
    # Title check (reuse existing logic)
    title_score = score_title_relevance(query_title, matched_title)
    if not title_score.is_relevant:
        return title_score  # title already failed; no point checking author

    # Author check
    author_passes, author_sim, author_detail = verify_authors(
        cited_authors, matched_authors
    )

    if not author_passes:
        return RelevanceScore(
            False,
            title_score.score,
            f"{title_score.detail}; AUTHOR REJECTED: {author_detail}",
        )

    return RelevanceScore(
        True,
        title_score.score,
        f"{title_score.detail}; {author_detail}",
    )

