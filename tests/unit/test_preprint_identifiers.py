"""An arXiv identifier in the title caused a fabrication finding against a real work.

Observed on the owner's own submission (job `17e3b5a8`): Wu et al.'s
*Google's neural machine translation system* and Hendy et al.'s *How good are
GPT models at machine translation?* were both flagged
`potentially_fabricated_reference`. Both are real, heavily cited preprints that
OpenAlex indexes; querying OpenAlex with the identifier removed returns each on
the first result. Three failures compounded: the polluted title made every
title-led provider score the real work as a keyword coincidence, the empty
`doi` field meant no identifier route ran at all, and the absent venue left the
reference `unknown` kind, which `bounded-reference-review-v9` reviews.
"""
import pytest

from app.services.preprint_identifiers import (
    ARXIV_DOI_PREFIX,
    apply_preprint_identity,
    arxiv_identifier,
    strip_arxiv_identifier,
)
from app.services.schemas import ParsedReference

OBSERVED = [
    (
        "How good are gpt models at machine translation? A comprehensive evaluation. arXiv 2302.09210",
        "2302.09210",
        "How good are gpt models at machine translation? A comprehensive evaluation",
    ),
    (
        "Google’s neural machine translation system: Bridging the gap between "
        "human and machine translation. arXiv:1609.08144",
        "1609.08144",
        "Google’s neural machine translation system: Bridging the gap between "
        "human and machine translation",
    ),
    (
        "Attention is all you need. arXiv preprint arXiv:1706.03762v5",
        "1706.03762",
        "Attention is all you need",
    ),
    ("Some older work. arXiv:math/0309136", "math/0309136", "Some older work"),
]


@pytest.mark.parametrize("title,identifier,cleaned", OBSERVED)
def test_the_identifier_moves_out_of_the_title_and_into_a_doi(title, identifier, cleaned):
    reference = ParsedReference(
        raw_ref=f"Author, A. (2023). {title}", title=title, author="Author, A."
    )
    apply_preprint_identity(reference)
    assert reference.title == cleaned
    assert reference.doi == f"{ARXIV_DOI_PREFIX}{identifier}"
    # A version suffix names a revision, not a different work; arXiv registers
    # its DOI against the unversioned identifier.
    import re

    assert not re.search(r"v\d+$", reference.doi)
    assert not re.search(r"\barxiv\b", reference.title, re.IGNORECASE)


def test_an_ordinary_reference_is_untouched():
    title = "The economic analysis of regulation"
    reference = ParsedReference(
        raw_ref=f"Berg, S. (2007). {title}. Cambridge.", title=title, author="Berg, S."
    )
    apply_preprint_identity(reference)
    assert reference.title == title
    assert not reference.doi


def test_merely_naming_arxiv_is_not_an_identifier():
    for text in ("A paper about the arXiv repository itself", "arXiv at thirty"):
        assert arxiv_identifier(text) is None
        assert strip_arxiv_identifier(text) == text


def test_a_doi_the_parser_already_found_is_authoritative():
    reference = ParsedReference(
        raw_ref="A. (2023). Work. arXiv:2302.09210",
        title="Work. arXiv:2302.09210",
        author="A.",
        doi="10.1234/publisher.version",
    )
    apply_preprint_identity(reference)
    assert reference.doi == "10.1234/publisher.version"
    assert reference.title == "Work"


def test_a_title_that_is_only_an_identifier_is_not_emptied():
    """Never trade a usable title for an empty one."""
    reference = ParsedReference(
        raw_ref="A. (2023). arXiv:2302.09210", title="arXiv:2302.09210", author="A."
    )
    apply_preprint_identity(reference)
    assert reference.title == "arXiv:2302.09210"
    assert reference.doi == f"{ARXIV_DOI_PREFIX}2302.09210"


def test_every_extraction_path_normalizes_identifiers():
    """The first version of this test checked `parse_reference_batch`, which is
    the *legacy* path. The live regex-first path returns separately, so the
    normalization never ran on a real paper and the rerun still produced an
    empty `doi` and a polluted title. Assert on every exit instead."""
    import ast
    import inspect

    from app.services import reference_parser

    source = inspect.getsource(reference_parser.extract_and_parse_references)
    tree = ast.parse(source.lstrip())
    returns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert returns, "no return statements found"
    normalized = 0
    for node in returns:
        text = ast.unparse(node)
        # An empty-list early exit carries no references to normalize.
        if text.strip() in {"return []", "return list()"}:
            normalized += 1
            continue
        assert "_apply_identifier_normalization" in text, text
        normalized += 1
    assert normalized == len(returns)


class TestOnlyTheEntrysOwnIdentifierIsAdopted:
    """Review finding, 2026-09-24: the whole reference string was searched.

    An entry can *mention* another work's identifier - "Reprinted from
    arXiv:1111.2222", "see also arXiv:..." - and adopting it as this entry's DOI
    would confirm the reference against the wrong work through the highest-trust
    route. Only the title itself, or the position directly after the title where
    the parser was observed to leave the token, may supply it.
    """

    @pytest.mark.parametrize("raw,title,expected", [
        ("H. (2023). How good are GPT models. arXiv 2302.09210", "How good are GPT models", "2302.09210"),
        ("W. (2016). GNMT system. arXiv:1609.08144", "GNMT system", "1609.08144"),
        ("V. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762v5",
         "Attention is all you need", "1706.03762"),
    ])
    def test_a_token_directly_after_the_title_is_adopted(self, raw, title, expected):
        reference = ParsedReference(raw_ref=raw, title=title, author="X")
        apply_preprint_identity(reference)
        assert reference.doi == f"{ARXIV_DOI_PREFIX}{expected}"
        assert reference.title == title

    @pytest.mark.parametrize("raw,title", [
        ("S. (2020). A real title. Journal, 3(2), 1-9. Reprinted from arXiv:1111.2222", "A real title"),
        ("D. (2021). Some title. Press. See also arXiv:9999.00001", "Some title"),
        ("E. (2019). Ordinary article. Journal, 1(1). Cf. arXiv:math/0309136 for the proof", "Ordinary article"),
    ])
    def test_a_mentioned_identifier_is_not_adopted(self, raw, title):
        reference = ParsedReference(raw_ref=raw, title=title, author="X")
        apply_preprint_identity(reference)
        assert not reference.doi
        assert reference.title == title
