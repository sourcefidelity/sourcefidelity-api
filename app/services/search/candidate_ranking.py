"""Discovery priority is not bibliographic verification or source admission."""
from contextvars import ContextVar
from functools import wraps
import re
import unicodedata
from urllib.parse import unquote
from app.services.reference_review_scope import catalog_title_match

POLICY = "candidate-purpose-ranking-v1"
PURPOSE = ContextVar("candidate_ranking_purpose", default="text")


def identity_candidate_ranking(function):
    @wraps(function)
    def run(*args, **kwargs):
        token = PURPOSE.set("identity")
        try:
            return function(*args, **kwargs)
        finally:
            PURPOSE.reset(token)
    return run


def _words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text or "").casefold())


def candidate_score(candidate, doi, title, author, year, *, purpose=None):
    """Order hints lexically; never turn them into identity agreements.

    Strong work-title hints dominate topical overlap and format. Snippet words
    cannot supply a title match; they only supply weak author/date tie breakers.
    Catalogs are identity opportunities, not intrinsically bad locations.
    """
    purpose = purpose or PURPOSE.get()
    visible = " ".join(_words(candidate.title))
    expected = " ".join(_words(title))
    haystack = unquote(" ".join((candidate.url, candidate.title, candidate.snippet))).casefold()
    exact_title = bool(expected and (visible == expected or any(
        " ".join(_words(part)) == expected for part in re.split(r"\s+[|–—]\s+", candidate.title))))
    exact_title = exact_title or catalog_title_match(title, candidate.title)
    significant = lambda value: set(_words(value)) - {"a", "an", "the", "and", "of", "in", "on", "for", "to", "with"}
    wanted, shown = significant(title), significant(candidate.title)
    overlap = len(wanted & shown) / len(wanted) if wanted else 0
    names = _words((author or "").split(",", 1)[0])
    author_hint = bool(names and names[-1] in set(_words(haystack)))
    identifier = bool(doi and re.search(r"(?<!\w)" + re.escape(doi.casefold()) + r"(?![\w./-])", haystack))
    # Same vocabulary in a different title remains below an exact title.
    tier = 5 if identifier else 4 if exact_title and author_hint else 3 if exact_title else 1 if overlap >= .8 and author_hint else 0
    catalog = any(marker in haystack for marker in ("catalog", "worldcat", "books.google.", "openlibrary.org"))
    access_gate = any(marker in haystack for marker in ("/login", "sign in", "search results"))
    text_hint = not catalog and not access_gate  # HTML and PDF are equal here.
    purpose_hint = int(catalog) if purpose == "identity" else int(text_hint)
    return tier * 1000 + round(100 * overlap) + 30 * author_hint + 10 * bool(year and year in haystack) + 5 * purpose_hint - 5 * access_gate
