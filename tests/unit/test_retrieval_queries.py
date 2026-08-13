from app.services.retrieval.core import _build_title_query


def test_core_query_uses_significant_title_tokens_and_author_surname() -> None:
    query = _build_title_query(
        "New Media Giants and the Changing Media Landscape",
        "Croteau, D.",
    )

    assert query == "Media Giants Changing Media Landscape Croteau"
    assert 'title:' not in query
    assert '"' not in query
