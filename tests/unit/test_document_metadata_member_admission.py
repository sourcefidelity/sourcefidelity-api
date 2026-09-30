"""One admission rule for opening-page title blocks, shared by both sides.

A `document_metadata` role is geometry-derived: it comes from page layout, not
from the words. Re-reading the block as plain text therefore cannot reproduce
it, and the validator has to decide which plain-text readings are consistent
with the role rather than evidence that it was supplied from outside the
application.

Reading as `publication_metadata` is the *expected* result for a title/byline
block. Excluding it refused a block the producer had already offered, and the
refusal is fatal: it fails the whole paper at persist time. Observed on a real
65-reference submission, where exactly one such block out of 21 acquired
sources ended the run with `candidate_passage_role_is_not_application_derived`
and no report.
"""
import pytest

from app.services.verification_evidence import (
    DOCUMENT_METADATA_MEMBER_PAGE_LIMIT,
    document_metadata_member_admissible,
    passage_role_from_text,
)

TITLE_BLOCK = "Journal of Regulatory Economics\nVolume 31, Number 2\nDOI 10.1007/s11149-006-9016-6"


def test_the_observed_failure_is_admissible():
    # Precondition: this block really does read as publication metadata, which
    # is what the former rule refused.
    assert passage_role_from_text(TITLE_BLOCK) == "publication_metadata"
    assert document_metadata_member_admissible(page_index=0, text=TITLE_BLOCK)


@pytest.mark.parametrize("page_index", range(DOCUMENT_METADATA_MEMBER_PAGE_LIMIT))
def test_weaker_readings_stay_admissible_on_opening_pages(page_index):
    text = "Some words that carry no strong structural signal at all"
    assert passage_role_from_text(text) in {"unknown", "body_prose"}
    assert document_metadata_member_admissible(page_index=page_index, text=text)


def test_a_block_past_the_opening_pages_is_not_document_level_evidence():
    assert not document_metadata_member_admissible(
        page_index=DOCUMENT_METADATA_MEMBER_PAGE_LIMIT, text=TITLE_BLOCK
    )
    assert not document_metadata_member_admissible(page_index=None, text=TITLE_BLOCK)


def test_a_reference_list_block_is_never_document_level_evidence():
    references = (
        "References\n"
        "Berg, S. (2007). The economic analysis of regulation. Cambridge.\n"
        "Khan, L. (2017). Amazon's antitrust paradox. Yale Law Journal.\n"
        "Noam, E. (2009). Media ownership and concentration. Oxford.\n"
    )
    assert passage_role_from_text(references) == "reference_list"
    assert not document_metadata_member_admissible(page_index=0, text=references)


def test_producer_and_validator_share_one_rule():
    """The asymmetry itself was the defect: a candidate the producer offered
    could be refused downstream, so the failure mode was a dead paper rather
    than an unoffered passage."""
    import inspect

    from app.services import verification_evidence, verification_report

    producer = inspect.getsource(verification_evidence._candidate_union_candidates)
    validator = inspect.getsource(verification_report._validate_candidate_passage_retrieval)
    assert "document_metadata_member_admissible" in producer
    assert "document_metadata_member_admissible" in validator
    # Both sides must also count a multi-source aggregate the same way.
    caller = inspect.getsource(verification_evidence)
    assert "include_document_metadata=len(set(artifact.claim.reference_ids)) > 1" in caller
    assert "len(set(artifact.claim.reference_ids)) > 1" in validator


class TestOverlapUnionKeepsItsRole:
    """A consolidated union must not inherit a role its text no longer has.

    When two overlapping candidates are merged, the union span covers page text
    neither candidate was scored on, yet the merged passage inherited the longer
    candidate's `passage_role`. Each candidate can sit below the reference-entry
    threshold on its own while their union crosses it, so the merged passage
    reads as a reference list while still labelled body prose. The validator's
    role check is fatal: the whole paper fails at persist time with no report.

    Observed on the owner's 65-reference submission as
    `stored=body_prose; derived=reference_list; method=bm25_concept`, which the
    bounded failure detail identified directly. No block returned by
    `_source_blocks` carries that disagreement, so the union was the only
    possible producer.
    """

    BODY = "The regulator reviewed the tariff schedule. It then published a finding."
    # The middle entry is deliberately the longest: the two candidates overlap
    # on it, and the overlap must exceed half the shorter span to be merged.
    ENTRIES = [
        "Berg, S. (2007). The economic analysis of regulation. Cambridge University Press.",
        "Khan, L. (2017). Amazon's antitrust paradox and the future of competition "
        "policy in digital markets. Yale Law Journal, 126(3).",
        "Noam, E. (2009). Media ownership. Oxford.",
    ]

    def _page(self):
        return self.BODY + "\n" + "\n".join(self.ENTRIES) + "\n"

    def _candidate(self, page_text, start, end, score):
        from app.services.verification_evidence import _PassageCandidate

        return (
            _PassageCandidate(
                page_index=0, page_label="1", start=start, end=end,
                text=page_text[start:end], method="bm25_concept",
                score=score, passage_role="body_prose",
            ),
            {"candidate_bm25_concept"},
        )

    def test_a_union_crossing_into_a_reference_list_keeps_the_attested_passage(self):
        from app.services.verification_evidence import (
            SUBSTANTIAL_PASSAGE_OVERLAP_RATIO,
            _consolidate_nested_passage_entries,
            _passage_overlap_ratio,
            passage_role_from_text,
        )

        page_text = self._page()
        first_end = page_text.index(self.ENTRIES[1]) + len(self.ENTRIES[1])
        second_start = page_text.index(self.ENTRIES[1])
        second_end = page_text.index(self.ENTRIES[2]) + len(self.ENTRIES[2])
        first = self._candidate(page_text, 0, first_end, 0.95)
        second = self._candidate(page_text, second_start, second_end, 0.80)

        # Preconditions: each candidate is honestly body prose on its own text,
        # they overlap substantially enough to be merged, and only their union
        # crosses the reference-entry threshold.
        assert passage_role_from_text(first[0].text) == "body_prose"
        assert passage_role_from_text(second[0].text) == "body_prose"
        assert (
            _passage_overlap_ratio(first[0], second[0])
            >= SUBSTANTIAL_PASSAGE_OVERLAP_RATIO
        )
        assert passage_role_from_text(page_text) == "reference_list"

        out = _consolidate_nested_passage_entries(
            [first, second], page_text_by_index={0: page_text}
        )

        assert out, "consolidation must not discard every candidate"
        for candidate, _channels in out:
            if candidate.passage_role == "document_metadata":
                continue
            assert passage_role_from_text(candidate.text) == candidate.passage_role, (
                candidate.passage_role,
                passage_role_from_text(candidate.text),
            )
