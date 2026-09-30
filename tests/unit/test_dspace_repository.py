"""DSpace 7 repository pages: the item and its PDF come from the same host's API."""
from app.services.ref_field_extractor import extract_fields_apa
from app.services.retrieval.dspace import is_dspace_shell, item_api_url, repository_item
from app.services.source_type import classify_reference_source_kind

PAGE = "https://studenttheses.uu.nl/handle/20.500.12932/30205"
ITEM = "https://studenttheses.uu.nl/server/api/pid/find?id=20.500.12932/30205"
BUNDLES = "https://studenttheses.uu.nl/server/api/core/items/abc/bundles"
STREAMS = "https://studenttheses.uu.nl/server/api/core/bundles/orig/bitstreams"


def _responses(content="https://studenttheses.uu.nl/server/api/core/bitstreams/x/content"):
    return {
        ITEM: {"type": "item", "metadata": {
            "dc.title": [{"value": "Disneyfication of Alice’s Adventures in Wonderland"}],
            "dc.contributor.author": [{"value": "Meelker, C.M."}],
            "dc.date.issued": [{"value": "2017"}]},
            "_links": {"bundles": {"href": BUNDLES}}},
        BUNDLES: {"_embedded": {"bundles": [
            {"name": "LICENSE", "_links": {"bitstreams": {"href": "https://studenttheses.uu.nl/l"}}},
            {"name": "ORIGINAL", "_links": {"bitstreams": {"href": STREAMS}}}]}},
        STREAMS: {"_embedded": {"bitstreams": [
            {"name": "thesis.pdf", "sizeBytes": 1053148, "_links": {"content": {"href": content}}},
            {"name": "notes.txt", "_links": {"content": {"href": content + "2"}}}]}},
    }


def test_the_shell_and_the_item_address_are_recognised():
    assert is_dspace_shell("<html><body><ds-app></ds-app></body></html>")
    assert not is_dspace_shell("<html><body><main>Article</main></body></html>")
    assert item_api_url(PAGE) == ITEM
    assert item_api_url("https://repo.example/items/0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b").endswith(
        "/server/api/core/items/0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b")
    assert item_api_url("http://studenttheses.uu.nl/handle/1/2") is None


def test_the_item_metadata_and_its_pdf_are_read_from_the_api():
    responses = _responses()
    record = repository_item(PAGE, responses.__getitem__)
    assert record["title"] == "Disneyfication of Alice’s Adventures in Wonderland"
    assert record["authors"] == ["Meelker, C.M."] and record["year"] == "2017"
    assert [f["name"] for f in record["files"]] == ["thesis.pdf"]


def test_a_file_on_another_host_is_never_followed():
    record = repository_item(PAGE, _responses("https://elsewhere.example/file.pdf").__getitem__)
    assert record["files"] == []


def test_a_thesis_descriptor_is_not_part_of_the_title():
    ref = ("Meelker, C. M. (2017). Disneyfication of Alice’s Adventures in Wonderland (Bachelor's thesis, Utrecht "
           "University, Netherlands). Retrieved from: https://studenttheses.uu.nl/handle/20.500.12932/30205")
    assert extract_fields_apa(ref).title == "Disneyfication of Alice’s Adventures in Wonderland"
    assert extract_fields_apa("Smith, J. (2019). Film and memory [Doctoral dissertation, University of Leeds]. "
                              "White Rose.").title == "Film and memory"
    assert extract_fields_apa("Doe, A. (2020). A study of film (2nd ed.). Press.").title == "A study of film (2nd ed.)"
    assert classify_reference_source_kind(ref, title="Disneyfication", url="").kind == "thesis"
