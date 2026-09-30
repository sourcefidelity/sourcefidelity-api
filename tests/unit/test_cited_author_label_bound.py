"""An over-long author list must not fail a whole submission.

Observed on the owner's `Academic Article - Long.pdf`: the run failed at
verification with `ValidationError` and produced no report for 65 references.
The cause was `CitationSourceBinding.cited_author_label` exceeding
`max_length=200` for Wu et al. (2016), which lists dozens of authors.

It surfaced only after the arXiv identifier fix made that reference resolvable:
previously it was never acquired, so no binding was ever built for it. A bare
`ValidationError` in `error_message` gave no field, which is what motivated
`safe_exception_detail`; the detail then named the field on the first run.
"""
import pytest
from pydantic import ValidationError

from app.services.verification_evidence import (
    CITED_AUTHOR_LABEL_LIMIT,
    CitationSourceBinding,
)

WU = (
    "Wu, Y., Schuster, M., Chen, Z., Le, Q.V., Norouzi, M., Macherey, W., "
    "Krikun, M., Cao, Y., Gao, Q., Macherey, K., Klingner, J., Shah, A., "
    "Johnson, M., Liu, X., Kaiser, L., Gouws, S., Kato, Y., Kudo, T., Kazawa, H."
)


def _binding(label):
    return CitationSourceBinding(
        status="unresolved", reference_id="ref-0061", cited_author_label=label
    )


def test_the_observed_reference_no_longer_fails_the_paper():
    assert len(WU) > CITED_AUTHOR_LABEL_LIMIT
    binding = _binding(WU)
    assert len(binding.cited_author_label) <= CITED_AUTHOR_LABEL_LIMIT
    assert binding.cited_author_label


def test_an_ordinary_author_list_is_untouched():
    for label in ("Berg, S. V., & Forsyth, P.", "Khan, L.", "Smith, J."):
        assert _binding(label).cited_author_label == label


def test_the_bound_label_is_a_prefix_of_what_was_cited():
    """Never invent a name that the reference did not carry."""
    bounded = _binding(WU).cited_author_label
    assert WU.startswith(bounded)


def test_the_leading_author_survives_truncation():
    """Surname matching uses the leading authors, so they must be kept."""
    bounded = _binding(WU).cited_author_label
    assert bounded.startswith("Wu, Y.")


def test_a_single_overlong_token_still_yields_a_label():
    """No comma to cut back to; a hard clip beats failing the submission."""
    label = "A" * (CITED_AUTHOR_LABEL_LIMIT + 50)
    bounded = _binding(label).cited_author_label
    assert 0 < len(bounded) <= CITED_AUTHOR_LABEL_LIMIT


def test_an_empty_label_is_still_rejected():
    """Bounding must not weaken the field's own contract."""
    with pytest.raises(ValidationError):
        _binding("")


class TestBoundedBibliographicFieldsAudit:
    """Audit result: overlong bibliographic values were bounded at call sites.

    Three fields carried the same arrangement - the limit enforced where the
    value was produced, and the model left to reject anything that arrived by
    another route. `CitationSourceBinding.cited_author_label` is what failed a
    65-reference paper. `ExpectedBibliographicFields.authors` had the same
    defect and was patched at three call sites earlier in the same session
    (`_bounded_observed_authors` twice, plus a slice in the Semantic Scholar
    adapter). `StageCandidateCase.cited_author_label` was safe only because
    `_citation_author_label` truncates, with the limit written out twice.

    Measured against the corpus: longest author string 223 characters, longest
    citation marker 328, and collaboration author lists run to the hundreds.
    """

    def test_a_large_collaboration_does_not_fail_the_run(self):
        from app.services.reference_discovery import (
            OBSERVED_AUTHOR_LIMIT,
            ExpectedBibliographicFields,
        )

        authors = [f"Author{n}, A." for n in range(500)]
        observed = ExpectedBibliographicFields(title="A big collaboration", authors=authors)
        assert len(observed.authors) == OBSERVED_AUTHOR_LIMIT
        # The leading authors are the identifying ones and must be the ones kept.
        assert observed.authors[0] == "Author0, A."
        assert observed.authors == authors[:OBSERVED_AUTHOR_LIMIT]

    def test_an_ordinary_author_list_is_untouched(self):
        from app.services.reference_discovery import ExpectedBibliographicFields

        authors = ["Berg, S. V.", "Forsyth, P."]
        assert ExpectedBibliographicFields(title="T", authors=authors).authors == authors

    def test_the_stage_case_bounds_its_own_label(self):
        from app.services.relationship_stage_evaluation import (
            CITED_AUTHOR_LABEL_LIMIT,
            _citation_author_label,
        )

        # A marker of this length exists in the corpus (328 characters).
        marker = "(" + "; ".join(f"Surname{n}, {n + 1990}" for n in range(40)) + ")"
        assert len(marker) > CITED_AUTHOR_LABEL_LIMIT
        assert len(_citation_author_label(marker)) <= CITED_AUTHOR_LABEL_LIMIT

    def test_the_limit_is_defined_once_per_field(self):
        """The producer and the model must not carry separate magic numbers."""
        import inspect

        from app.services import relationship_stage_evaluation as rse

        producer = inspect.getsource(rse._citation_author_label)
        assert "CITED_AUTHOR_LABEL_LIMIT" in producer
        assert "[:200]" not in producer


class TestSiblingTextFieldsAreBoundedToo:
    """Authors were bounded first because a real reference overran them; the
    sibling text fields carried the same arrangement and would have failed the
    same way on a malformed provider record. `title` was the live risk at 629
    characters observed against a 1,000 limit - and before the arXiv identifier
    fix, titles were carrying appended identifiers, which is exactly how a
    title grows unexpectedly.
    """

    def test_overlong_text_fields_no_longer_fail_the_run(self):
        from app.services.reference_discovery import ExpectedBibliographicFields

        observed = ExpectedBibliographicFields(
            title="T" * 4000, container_title="C" * 4000, publisher="P" * 900,
            doi="D" * 900, isbn="I" * 400, volume="V" * 400, issue="S" * 400,
            pages="P" * 400, year="Y" * 400,
        )
        for name in ("title", "container_title", "publisher", "doi", "isbn",
                     "volume", "issue", "pages", "year"):
            value = getattr(observed, name)
            limit = ExpectedBibliographicFields.model_fields[name].metadata
            declared = next(
                (getattr(m, "max_length") for m in limit
                 if getattr(m, "max_length", None) is not None), None
            )
            assert declared is not None, name
            assert len(value) == declared, name

    def test_ordinary_values_are_untouched(self):
        from app.services.reference_discovery import ExpectedBibliographicFields

        observed = ExpectedBibliographicFields(
            title="The economic analysis of regulation",
            container_title="Journal of Regulatory Economics",
            publisher="Cambridge University Press", volume="31", issue="2",
            pages="113-138", year="2007",
        )
        assert observed.title == "The economic analysis of regulation"
        assert observed.pages == "113-138"

    def test_marker_text_is_deliberately_not_truncated(self):
        """`marker_text` carries a coordinate contract that truncation breaks.

        `CitationSourceBinding` requires
        `marker_local_end - marker_local_start == len(marker_text)`, so clipping
        the text silently invalidates the span it points at. A marker too long
        to represent exactly is an evidence question - the binding should be
        `unresolved` rather than claim a span it cannot support - not a length
        question, so it is left for an explicit decision. Observed markers
        reach 328 characters against a 1,000 limit.
        """
        import pytest
        from pydantic import ValidationError

        from app.services.verification_evidence import CitationSourceBinding

        text = "M" * 2000
        with pytest.raises(ValidationError):
            CitationSourceBinding(
                status="exact", reference_id="r", cited_author_label="A",
                marker_text=text[:1000], marker_local_start=0, marker_local_end=2000,
            )
