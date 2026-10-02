import hashlib

import pytest

from app.services.reference_parser import extract_and_parse_references, extract_reference_section
from app.services.reference_url_repair import repair_reference_urls
from app.services.schemas import ParsedReference
from app.services.text_extractor import _clean_text, extract_qualified_text_from_bytes


def parse(layout):
    return extract_and_parse_references(
        extract_reference_section(_clean_text(layout), "apa"),
        format_hint="apa", use_llm_fallback=False,
    )


@pytest.mark.parametrize("wrapped,expected", [
    ("https://catalog.example-\nroute.test:443/login?AN=x\nrecord.123&site=library&scope=all",
     "https://catalog.example-route.test:443/login?AN=xrecord.123&site=library&scope=all"),
    ("https://example.test/books/\nvolumes/123?format=pdf", "https://example.test/books/volumes/123?format=pdf"),
    ("https://example.test/find?a=1\n&b=2", "https://example.test/find?a=1&b=2"),
])
def test_structured_wrap_is_bound_and_raw_is_unchanged(wrapped, expected):
    layout = f"References\nAdams, A. (2020). A study. Press. {wrapped}\nBaker, B. (2021). Another study. Press."
    refs = parse(layout)
    original = [r.model_dump() for r in refs]
    repaired = repair_reference_urls(refs, layout)
    assert len(repaired) == 2
    assert repaired[0].url == expected
    assert [r.model_dump() for r in refs] == original
    assert repaired[0].raw_ref == refs[0].raw_ref
    assert repaired[1] == refs[1]
    record = repaired[0].url_repair
    assert record is not None
    assert layout[record.span_start:record.span_end] == record.observed_span
    assert record.layout_sha256 == hashlib.sha256(layout.encode()).hexdigest()
    assert record.raw_reference_sha256 == hashlib.sha256(refs[0].raw_ref.encode()).hexdigest()
    assert record.original_url == refs[0].url
    assert repair_reference_urls(repaired, layout) == repaired
    assert ParsedReference.model_validate(repaired[0].model_dump()) == repaired[0]


@pytest.mark.parametrize("suffix", [
    "\nOrdinary prose with spaces.", "\nConclusion", "\nopaqueidentifier",
    "\n\n&other=1", " &other=1", "\nhttps://other.test/path",
    "\nBaker, B. (2021). Another study. https://other.test/path",
])
def test_ambiguous_or_nonlocal_continuations_are_not_joined(suffix):
    layout = "References\nAdams, A. (2020). A study. Press. https://example.test/find?a=x" + suffix
    refs = parse(layout)
    assert repair_reference_urls(refs, layout) == refs


def test_duplicate_bindings_and_missing_raw_span_abstain():
    layout = "References\nAdams, A. (2020). A study. Press. https://example.test/find?a=x\n&b=2"
    refs = parse(layout)
    assert repair_reference_urls(refs * 2, layout) == refs * 2
    unbound = [refs[0].model_copy(update={"raw_ref": "unrelated"})]
    assert repair_reference_urls(unbound, layout) == unbound
    assert repair_reference_urls(refs, layout + "\n" + layout) == refs


def test_credentials_are_not_repaired():
    layout = "References\nAdams, A. (2020). A study. Press. https://user:secret@example.test/find?a=x\n&b=2"
    refs = parse(layout)
    assert repair_reference_urls(refs, layout) == refs


def test_native_boundaries_do_not_change_semantic_text(monkeypatch):
    from app.services import text_extractor as module
    layout = "References\nAdams, A. (2020). A study. Press. https://example.test/find?a=x\nrecord&b=2"
    monkeypatch.setattr(module, "_extract_pdfplumber_bytes", lambda _: layout)
    monkeypatch.setattr(module, "_extract_pymupdf_bytes", lambda _: layout)
    qualified = extract_qualified_text_from_bytes(b"bounded fixture", "paper.pdf")
    assert all(c.layout_text == layout and c.text == _clean_text(layout) for c in qualified.candidates)
    assert all(c["layout_sha256"] == hashlib.sha256(layout.encode()).hexdigest()
               for c in qualified.evidence()["candidates"])


def test_paper_boundary_applies_repair_without_models():
    from app.services.paper_extraction import extract_paper_evidence
    layout = "An ordinary body paragraph.\n\nReferences\nAdams, A. (2020). A study. Press. https://example.test/find?a=x\nrecord&b=2"
    args = dict(paper_version_id="url-wrap-test", format_hint="apa", use_llm_boundaries=False,
                use_llm_atomizer=False, use_llm_reference_fallback=False)
    old = extract_paper_evidence(_clean_text(layout), **args)
    new = extract_paper_evidence(_clean_text(layout), reference_layout_text=layout, **args)
    assert new.references[0].url == "https://example.test/find?a=xrecord&b=2"
    assert new.references[0].reference_id == old.references[0].reference_id
    assert new.references[0].raw_ref == old.references[0].raw_ref


def test_an_opaque_continuation_joins_only_when_the_pdf_link_target_proves_it():
    layout = ("References\nAdams, A. (2020). A study. Press. https://example.test/publication/1_Theorising_the_P\n"
              "ractice_of_Media\nBaker, B. (2021). Another study. Press.")
    full = "https://example.test/publication/1_Theorising_the_Practice_of_Media"
    refs = parse(layout)
    assert repair_reference_urls(refs, _clean_text(layout))[0].url != full
    assert repair_reference_urls(refs, _clean_text(layout), link_targets=frozenset({full}))[0].url == full
    other = frozenset({"https://example.test/publication/1_Theorising_the_Practice_of_Media_extra"})
    assert repair_reference_urls(refs, _clean_text(layout), link_targets=other)[0].url != full
