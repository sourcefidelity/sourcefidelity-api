"""Owner decision 2026-10-04 (option A): a follow-on sentence joins the citation before it."""
from app.services.paper_extraction import join_adjacent_continuations
from app.services.schemas import InTextCitation

BODY = ("It relates to identity as Smith (2019) outlines. Mara's capacity points to a developing identity. "
        "Wilson (2022) studies exploration.\n\nWilson stresses the power of living abroad.")


def _cite(text, refs, marker):
    start = BODY.index(text)
    return InTextCitation(reference_ids=refs, text=text, citation_marker=marker,
                          passage_start=start, passage_end=start + len(text))


def test_an_adjacent_continuation_of_the_same_source_joins_its_citation():
    smith = _cite("It relates to identity as Smith (2019) outlines.", ["ref-smith"], "Smith (2019)")
    follow = _cite("Mara's capacity points to a developing identity.", ["ref-smith"], "implicit_continuation")
    wilson = _cite("Wilson (2022) studies exploration.", ["ref-wilson"], "Wilson (2022)")
    after_break = _cite("Wilson stresses the power of living abroad.", ["ref-wilson"], "implicit_continuation")
    result = join_adjacent_continuations([smith, follow, wilson, after_break], BODY)
    active = [c for c in result if c.drop_reason is None]
    assert [c.text for c in active] == [
        "Wilson (2022) studies exploration.", "Wilson stresses the power of living abroad.",
        "It relates to identity as Smith (2019) outlines. Mara's capacity points to a developing identity."]
    assert {c.drop_reason for c in result if c.drop_reason} == {
        "base_of_joined_continuation_v1", "joined_to_preceding_citation_v1"}


def test_a_continuation_of_another_source_is_not_joined():
    smith = _cite("It relates to identity as Smith (2019) outlines.", ["ref-smith"], "Smith (2019)")
    other = _cite("Mara's capacity points to a developing identity.", ["ref-other"], "implicit_continuation")
    assert join_adjacent_continuations([smith, other], BODY) == [smith, other]


def test_a_gap_that_names_the_author_is_joined_too():
    from types import SimpleNamespace
    body = ("It relates to identity as Smith (2019) outlines. Smith contends that people create hybrid identities. "
            "Mara's capacity points to a developing identity. Students often agree.")
    refs = [SimpleNamespace(reference_id="ref-smith", author="Smith, J")]
    def cite(text, marker):
        start = body.index(text)
        return InTextCitation(reference_ids=["ref-smith"], text=text, citation_marker=marker,
                              passage_start=start, passage_end=start + len(text))
    smith = cite("It relates to identity as Smith (2019) outlines.", "Smith (2019)")
    follow = cite("Mara's capacity points to a developing identity.", "implicit_continuation")
    joined = [c for c in join_adjacent_continuations([smith, follow], body, refs) if c.drop_reason is None]
    assert [c.text for c in joined] == [body[:body.index(" Students")]]
    unnamed = body.replace("Smith contends", "Many argue")
    smith2 = InTextCitation(reference_ids=["ref-smith"], text=smith.text, citation_marker="Smith (2019)",
                            passage_start=0, passage_end=len(smith.text))
    start = unnamed.index(follow.text)
    follow2 = follow.model_copy(update={"passage_start": start, "passage_end": start + len(follow.text)})
    assert join_adjacent_continuations([smith2, follow2], unnamed, refs) == [smith2, follow2]


def test_a_sentence_naming_the_author_joins_without_a_model_continuation():
    from types import SimpleNamespace
    from app.services.paper_extraction import join_author_naming_sentences
    body = ("It relates to identity as Smith (2019) outlines. Smith contends that people create hybrid identities. "
            "Students often agree.\n\nSmith is cited again here.")
    refs = [SimpleNamespace(reference_id="ref-smith", author="Smith, J")]
    text = "It relates to identity as Smith (2019) outlines."
    smith = InTextCitation(reference_ids=["ref-smith"], text=text, citation_marker="Smith (2019)",
                           passage_start=0, passage_end=len(text))
    active = [c for c in join_author_naming_sentences([smith], [], body, refs) if c.drop_reason is None]
    assert [c.text for c in active] == [
        "It relates to identity as Smith (2019) outlines. Smith contends that people create hybrid identities."]
