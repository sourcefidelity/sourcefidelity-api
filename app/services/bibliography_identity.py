"""Routing eligibility only; credibility remains the shared evidence-bound rule."""

from app.services.source_type import is_bibliographically_searchable

POLICY = "bibliography-identity-only-v2"


def eligible_identity_only(reference):
    # An unclassified reference is a parser gap, not an unsearchable work.
    # Author, title and year settle identity for every kind the indexes carry,
    # so eligibility turns on what the reference supplies, not on whether the
    # classifier managed to name its kind.
    return bool(
        is_bibliographically_searchable(reference.source_kind)
        and not reference.needs_review
        and reference.title.strip()
        and reference.author.strip()
    )
