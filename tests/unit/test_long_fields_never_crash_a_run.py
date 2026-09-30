"""Owner decision, 2026-09-23: an unexpectedly long field must not crash a run.

"Missing a reference or a citation is not the end of the world" - losing detail
on one item is acceptable where losing a whole submission is not. This pins
that rule across every model a paper run constructs from paper or provider
data, because the failure it replaces was real: a 223-character author list
ended a 65-reference submission at persist time with no report.

Each model degrades in the way its own contract allows. A plain text field
keeps a prefix, since these values are matched by token overlap rather than
equality. A field paired with character offsets cannot simply be clipped - the
offsets would then describe a span the text does not support, and a report
highlight would land on the wrong words - so it either drops to an unresolved
binding or clamps text and span together.
"""
import pytest

BIG = "X" * 50_000


def test_a_parsed_reference_survives_a_malformed_parse():
    from app.services.schemas import ParsedReference, declared_max_length

    reference = ParsedReference(
        raw_ref=BIG, title=BIG, author=BIG, container_title=BIG,
        publisher=BIG, volume=BIG, issue=BIG, pages=BIG,
    )
    for name in ("container_title", "publisher", "volume", "issue", "pages"):
        limit = declared_max_length(ParsedReference, name)
        assert len(getattr(reference, name)) == limit, name


def test_observed_provider_metadata_survives_an_absurd_record():
    from app.services.reference_discovery import ExpectedBibliographicFields

    observed = ExpectedBibliographicFields(
        title=BIG, container_title=BIG, publisher=BIG, doi=BIG, isbn=BIG,
        volume=BIG, issue=BIG, pages=BIG, year=BIG, authors=["A"] * 5_000,
    )
    assert observed.title and len(observed.authors) == 64


def test_claim_evidence_survives_a_misparsed_locator():
    from app.services.verification_evidence import ClaimEvidence

    claim = ClaimEvidence(
        claim_id="c", paper_version_id="pv", text="t",
        page_locator=BIG, citation_marker=BIG, reference_ids=["r"],
    )
    assert len(claim.page_locator) == 100


class TestSpanBearingFieldsDegradeWithoutLying:
    """Text paired with offsets must never be clipped while the offsets stand."""

    def test_an_unrepresentable_marker_becomes_unresolved(self):
        from app.services.verification_evidence import CitationSourceBinding

        binding = CitationSourceBinding(
            status="exact", reference_id="r", cited_author_label="A",
            marker_text=BIG, marker_local_start=0, marker_local_end=len(BIG),
        )
        assert binding.status == "unresolved"
        assert binding.marker_text == ""
        assert (binding.marker_local_start, binding.marker_local_end) == (-1, -1)

    def test_an_ordinary_marker_keeps_its_exact_span(self):
        from app.services.verification_evidence import CitationSourceBinding

        marker = "(Paniagua et al., 2022)"
        binding = CitationSourceBinding(
            status="exact", reference_id="r", cited_author_label="Paniagua",
            marker_text=marker, marker_local_start=10,
            marker_local_end=10 + len(marker),
        )
        assert binding.status == "exact"
        assert binding.marker_text == marker

    def test_a_marker_member_clamps_text_and_span_together(self):
        from app.services.schemas import CitationMarkerMember, declared_max_length

        limit = declared_max_length(CitationMarkerMember, "text")
        member = CitationMarkerMember(
            text=BIG, local_start=40, local_end=40 + len(BIG)
        )
        assert len(member.text) == limit
        # The invariant the model enforces must still hold afterwards.
        assert member.local_end - member.local_start == len(member.text)
        # And the span must still start where the citation actually starts.
        assert member.local_start == 40

    def test_an_ordinary_marker_member_is_untouched(self):
        from app.services.schemas import CitationMarkerMember

        text = "(Alharbi, 2023; Dorst et al., 2022)"
        member = CitationMarkerMember(text=text, local_start=5, local_end=5 + len(text))
        assert member.text == text
        assert member.local_end - member.local_start == len(text)
