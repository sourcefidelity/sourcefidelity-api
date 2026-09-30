"""Every adapter states whether its failure blocks a "not found" claim.

Until 2026-09-23 this was derived from `deferred`, which means "batches DOI
prefetch" — an unrelated property. `semantic_scholar` is the only adapter that
declares it, so every other adapter was required by default and no one had
decided it. That default is load-bearing: `_search_is_incomplete` returns true
if **any** required attempt carries a failure outcome, so one adapter failing
makes the whole reference `search_incomplete`.

The direction matters. `True` is SAFE — a required route that fails blocks a
potentially-fabricated-reference finding. Setting an adapter to `False` removes
that block, so it is a judgment about evidence, never a way to go faster.
"""
import pytest

from app.services.retrieval import _RETRIEVER_CLASSES
from app.services.retrieval.base import RetrievalSource


def test_every_registered_adapter_declares_a_value() -> None:
    """The point of the change: no adapter may inherit this silently."""
    undeclared = sorted(
        name for name, cls in _RETRIEVER_CLASSES.items()
        if cls.required_for_search_completion is None
    )

    assert undeclared == [], (
        f"these adapters have not decided whether their failure blocks a "
        f"'not found' claim: {undeclared}"
    )


def test_the_declaration_is_a_bool_not_an_accident() -> None:
    for name, cls in _RETRIEVER_CLASSES.items():
        assert isinstance(cls.required_for_search_completion, bool), name


def test_semantic_scholar_and_core_are_the_non_blocking_routes() -> None:
    """Semantic Scholar's previous behaviour, and CORE by owner decision 2026-09-29.

    Crossref and OpenAlex are the main article sources; CORE augments them, so
    a failed CORE search is not a reason to call a search incomplete.
    """
    assert _RETRIEVER_CLASSES["semantic_scholar"].blocks_search_completion() is False
    assert _RETRIEVER_CLASSES["core"].blocks_search_completion() is False
    others = {name for name, cls in _RETRIEVER_CLASSES.items()
              if not cls.blocks_search_completion()}
    assert others == {"semantic_scholar", "core"}


def test_the_declaration_no_longer_rides_on_the_batching_flag() -> None:
    """`deferred` is about DOI prefetch batching and must not decide evidence."""
    class Batched(RetrievalSource):
        name = "batched"
        deferred = True
        required_for_search_completion = True
        def search_by_doi(self, doi): ...
        def search_by_title_author(self, title, author=None): ...

    assert Batched.blocks_search_completion() is True


def test_an_undeclared_source_keeps_the_previous_behaviour() -> None:
    """A duck-typed source outside this registry is unaffected by the change."""
    from app.services.source_resolver import _route_blocks_completion

    class Duck:
        name = "duck"

    class DeferredDuck:
        name = "deferred_duck"
        deferred = True

    assert _route_blocks_completion(Duck()) is True
    assert _route_blocks_completion(DeferredDuck()) is False


def test_the_current_declarations_preserve_the_previous_routing() -> None:
    """This refactor changed who decides, not what happens.

    Each value equals what `not deferred` produced before, so no reference's
    outcome moves. Changing any of them is a separate, measured decision.
    """
    for name, cls in _RETRIEVER_CLASSES.items():
        if name == "core":
            continue  # Changed by owner decision 2026-09-29 (see above).
        assert cls.blocks_search_completion() is not getattr(cls, "deferred", False), name


def test_a_non_boolean_declaration_is_not_treated_as_a_decision() -> None:
    """A stub or Mock must not become a truthy "required" by accident."""
    from unittest.mock import Mock

    from app.services.source_resolver import _route_blocks_completion

    stub = Mock()
    stub.blocks_search_completion.return_value = Mock()

    # Falls back to the old derivation rather than trusting the stub's answer.
    assert _route_blocks_completion(stub) is False


def test_no_adapter_call_site_hardcodes_the_requirement() -> None:
    """The declaration governs routing only if every call site reads it.

    Added 2026-09-24 after the first version of this refactor was found
    incomplete: the tests above checked the class attribute, while two call
    sites -- the enriched-DOI lookup and `_try_source_sequence` -- still passed
    `required=True` for any adapter. Six Semantic Scholar attempts in the
    corpus run were recorded as required despite its `False` declaration.

    This is a structural check on the call-site pattern: any academic-adapter
    attempt recorded for a generic `source.name` must take its requirement from
    the adapter. Fixed-provider sites (crossref journal check, google_books,
    internet_archive, caches, student URLs) and the bounded-web policy route
    name their provider literally and keep their own explicit values.
    """
    import ast
    import inspect

    from app.services import source_resolver

    tree = ast.parse(inspect.getsource(source_resolver))
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_record_discovery_attempt"):
            continue
        kwargs = {k.arg: k.value for k in node.keywords}
        provider, category = kwargs.get("provider"), kwargs.get("category")
        generic = (isinstance(provider, ast.Attribute) and provider.attr == "name"
                   and isinstance(provider.value, ast.Name) and provider.value.id == "source")
        academic = isinstance(category, ast.Constant) and category.value == "academic_adapter"
        if not (generic and academic):
            continue
        required = kwargs.get("required")
        reads_declaration = (isinstance(required, ast.Call)
                             and getattr(required.func, "id", None) == "_route_blocks_completion")
        if not reads_declaration:
            offenders.append(node.lineno)

    assert offenders == [], f"call sites bypass the adapter declaration at lines {offenders}"

