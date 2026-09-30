"""Naming the reference details that disagree with the located record.

The report previously said only "Bibliographic fields conflict with the
located record" and never which field. Khan's reference gets journal, volume,
issue, year and pages wrong; the reader was told none of them.

The hard constraint comes from measurement: when title or author materially
conflicts, the located record is a DIFFERENT WORK. Reporting its year as the
reader's correct value would assert that the cited work exists with those
details. Every large year gap among articles in the corpus was exactly that.
"""
import hashlib

import pytest

from app.services.reference_formatting import bibliographic_field_conflicts
from app.services.schemas import ParsedReference


def _hash(value: str) -> str:
    from app.services.reference_discovery import _value_hash
    return _value_hash(value)


REFERENCE = ParsedReference(
    reference_id="r1", author="Khan, L.", year="2018",
    title="The separation of platforms and commerce",
    container_title="Harvard Law Review", volume="131", issue="5", pages="100-180",
    raw_ref="Khan, L. (2018). The separation of platforms and commerce. "
            "Harvard Law Review, 131(5), 100-180.")

OBSERVED = {"title": "The separation of platforms and commerce", "authors": ["Khan, L."],
            "year": "2019", "container_title": "Columbia Law Review",
            "volume": "119", "issue": "4", "pages": "973-1098"}


def _comparison(field, submitted, observed, outcome):
    return {"field_name": field, "expected_sha256": _hash(submitted),
            "observed_sha256": _hash(observed), "outcome": outcome,
            "reason_code": "normalized_component_field_match"}


def _discovery(*, title_outcome="agreement", author_outcome="agreement",
               conflicts=("year", "container_title", "volume", "issue", "pages"),
               observed=None, credible=True, permitted=True, reference_id="r1"):
    observed = {**OBSERVED, **(observed or {})}
    comparisons = [
        _comparison("title", REFERENCE.title, observed["title"], title_outcome),
        _comparison("author", REFERENCE.author, "Khan, L.", author_outcome),
    ]
    for field in ("year", "container_title", "volume", "issue", "pages"):
        comparisons.append(_comparison(
            field, getattr(REFERENCE, field), observed[field],
            "material_conflict" if field in conflicts else "agreement"))
    return {
        "reference_id": reference_id,
        "created_at": "2026-09-22T00:00:00Z",
        "outcome": "confirmed",
        "contributes_to_neutral_pattern": False,
        "expected": {"title": REFERENCE.title, "authors": [REFERENCE.author],
                     "year": REFERENCE.year, "reference_parse_review": False},
        "required_route_categories": [],
        "queries": [], "limitations": [], "candidates_complete": True,
        "attempts": [{"attempt_id": "a1", "provider": "crossref", "permitted": permitted,
                      "required": True, "started_at": "2026-09-22T00:00:00Z",
                      "completed_at": "2026-09-22T00:00:00Z", "outcome": "candidate_found",
                      "route_category": "academic_adapter"}],
        "candidates": [{
            "candidate_id": "c1", "attempt_id": "a1", "provider": "crossref",
            "discovery_provider": "crossref",
            "observed": observed,
            "comparisons": comparisons,
            "plausible_identity_match": credible,
            "authoritative_identifier_match": False,
            # A candidate agreeing on title and author is credible by the
            # record's own rule even without a plausible-match flag, so a
            # rejected identity is what genuine non-credibility looks like.
            "acquisition_outcome": "metadata_only" if credible else "identity_rejected",
            "location_available": False,
        }],
    }


class TestNamingTheConflicts:
    def test_every_reportable_differing_field_is_named(self):
        finding = bibliographic_field_conflicts(REFERENCE, _discovery())
        assert finding is not None
        named = {d["field_name"] for d in finding["field_differences"]}
        assert named == {"volume", "issue"}

    def test_pages_are_not_named_although_they_differ(self):
        """Measured on 2026-09-23 over 3,645 stored candidates.

        A pages rule fired 20 times and every sampled firing was an Elsevier
        article number ("102305") compared against a page range. The defect is
        in the input, and a finding built on it would tell a reader a correct
        reference is wrong. Restore pages only with the article-number case
        handled and re-measured.
        """
        finding = bibliographic_field_conflicts(REFERENCE, _discovery())
        assert "pages" not in {d["field_name"] for d in finding["field_differences"]}

    def test_the_journal_is_not_named_although_it_differs(self):
        """Measured 2026-09-23: one firing across 3,687 candidates, and wrong.

        A reference gave "Economics and Law" where Crossref records "Ekonomia
        i Prawo" -- one journal under two languages. Nothing in the strings
        separates that from a different journal.
        """
        finding = bibliographic_field_conflicts(REFERENCE, _discovery())
        named = {d["field_name"] for d in finding["field_differences"]}
        assert "container_title" not in named

    def test_both_values_are_reported_without_a_verdict(self):
        finding = bibliographic_field_conflicts(REFERENCE, _discovery())
        volume = next(d for d in finding["field_differences"]
                      if d["field_name"] == "volume")
        assert volume["submitted_value"] and volume["located_value"]
        assert volume["submitted_value"] != volume["located_value"]
        text = finding["finding"].lower()
        for accusation in ("fabricat", "false", "incorrect", "wrong", "misciting", "error"):
            assert accusation not in text

    def test_only_the_conflicting_fields_are_named(self):
        finding = bibliographic_field_conflicts(REFERENCE, _discovery(conflicts=("volume",)))
        assert [d["field_name"] for d in finding["field_differences"]] == ["volume"]


class TestADifferentWorkIsNotACitationError:
    """The measured false positive: a match on another paper entirely."""

    def test_a_title_conflict_abstains(self):
        assert bibliographic_field_conflicts(
            REFERENCE, _discovery(title_outcome="material_conflict")) is None

    def test_an_author_conflict_abstains(self):
        assert bibliographic_field_conflicts(
            REFERENCE, _discovery(author_outcome="material_conflict")) is None

    def test_a_minor_author_difference_still_reports(self):
        assert bibliographic_field_conflicts(
            REFERENCE, _discovery(author_outcome="minor_difference")) is not None


class TestAbstentions:
    def test_no_conflicting_field_reports_nothing(self):
        assert bibliographic_field_conflicts(REFERENCE, _discovery(conflicts=())) is None

    def test_a_year_difference_alone_is_never_reported(self):
        """Measured: 15 of 18 year differences were the online-first pattern.

        On a record already anchored by title and author, a differing year was
        never a citation error in the corpus. Monograph years remain covered by
        `book_publication_year_discrepancy`, which corroborates across two
        independent edition records before it says anything.
        """
        assert bibliographic_field_conflicts(
            REFERENCE, _discovery(conflicts=("year",))) is None

    def test_a_rejected_identity_is_ignored(self):
        """A record the resolver rejected cannot supply a discrepancy."""
        assert bibliographic_field_conflicts(REFERENCE, _discovery(credible=False)) is None

    def test_an_unpermitted_attempt_is_ignored(self):
        assert bibliographic_field_conflicts(REFERENCE, _discovery(permitted=False)) is None

    def test_a_reference_marked_for_review_abstains(self):
        assert bibliographic_field_conflicts(
            REFERENCE.model_copy(update={"needs_review": True}), _discovery()) is None

    def test_a_record_for_another_reference_abstains(self):
        assert bibliographic_field_conflicts(
            REFERENCE, _discovery(reference_id="other")) is None

    def test_a_stale_comparison_abstains(self):
        """The reference has been re-read since the comparison was stored."""
        edited = REFERENCE.model_copy(update={"container_title": "Yale Law Journal"})
        finding = bibliographic_field_conflicts(edited, _discovery())
        named = {d["field_name"] for d in (finding or {}).get("field_differences", [])}
        assert "container_title" not in named

    def test_an_empty_located_value_is_not_a_discrepancy(self):
        finding = bibliographic_field_conflicts(
            REFERENCE, _discovery(observed={"volume": ""}))
        named = {d["field_name"] for d in (finding or {}).get("field_differences", [])}
        assert "volume" not in named


class TestContradictingRecords:
    def test_records_disagreeing_on_the_located_value_abstain_for_that_field(self):
        discovery = _discovery()
        second = {**discovery["candidates"][0], "candidate_id": "c2"}
        second["observed"] = {**OBSERVED, "container_title": "Stanford Law Review"}
        second["comparisons"] = [
            c if c["field_name"] != "container_title"
            else _comparison("container_title", REFERENCE.container_title,
                             "Stanford Law Review", "material_conflict")
            for c in discovery["candidates"][0]["comparisons"]]
        discovery["candidates"].append(second)
        finding = bibliographic_field_conflicts(REFERENCE, discovery)
        named = {d["field_name"] for d in (finding or {}).get("field_differences", [])}
        assert "container_title" not in named
        # The other fields are unaffected by one field's contradiction.
        assert "volume" in named


class TestObservedSideIsSurfaced:
    """The located record must supply the fields, or nothing can be compared.

    The submitted side was fixed first; every observed container, volume and
    issue was still empty because only the ERIC adapter ever set one, so Khan's
    journal comparison resolved `unknown` on a run where his reference finally
    carried "Harvard Law Review", "131" and "5".
    """

    def test_crossref_surfaces_the_journal_volume_and_issue(self):
        from app.services.retrieval.crossref import _first_container, _scalar
        message = {"container-title": ["Columbia Law Review"], "volume": "119", "issue": "4"}
        assert _first_container(message) == "Columbia Law Review"
        assert _scalar(message["volume"]) == "119"
        assert _scalar(message["issue"]) == "4"

    def test_an_ambiguous_container_list_is_not_used(self):
        from app.services.retrieval.crossref import _first_container
        assert _first_container({"container-title": ["A", "B"]}) == ""
        assert _first_container({"container-title": []}) == ""
        assert _first_container({}) == ""

    def test_non_scalar_values_are_refused(self):
        from app.services.retrieval.crossref import _scalar
        assert _scalar(["119"]) == ""
        assert _scalar({"v": 1}) == ""
        assert _scalar(None) == ""
        assert _scalar("x" * 200) == ""

    def test_volume_and_issue_reach_the_comparison(self):
        import inspect
        from app.services import reference_discovery
        source = inspect.getsource(reference_discovery.build_reference_discovery_candidate)
        assert "('volume', volume)" in source
        assert "('issue', issue)" in source
