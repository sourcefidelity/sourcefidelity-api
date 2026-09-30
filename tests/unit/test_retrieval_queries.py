"""CORE title queries (measured against the live API 2026-09-25).

An unfielded query searches every field and matched 11-31 million records for
well-known titles; a quoted title phrase with an author returned only the right
record, and per-word title fields still found a slightly misquoted title.
"""
import json
from unittest.mock import Mock

import httpx
from pydantic import SecretStr

from app.services.retrieval.core import CoreRetriever, _author_clause, _title_queries


def test_title_phrase_first_then_per_word_title_fields_with_author() -> None:
    queries = _title_queries("New Media Giants and the Changing Media Landscape", "Croteau, D.")
    assert queries == [
        'title:"New Media Giants and the Changing Media Landscape" AND authors:Croteau',
        'title:Media AND title:Giants AND title:Changing AND title:Media AND title:Landscape AND authors:Croteau',
    ]


def test_every_query_is_fielded() -> None:
    for query in _title_queries("Attention is all you need", "Vaswani, A."):
        assert all(part.startswith(('title:', 'authors:')) for part in query.split(' AND '))


def test_quotes_and_backslashes_cannot_break_the_phrase() -> None:
    [phrase, _] = _title_queries('The "so-called" crisis\\ revisited', None)
    assert phrase == 'title:"The so-called crisis revisited"'


def test_subtitle_colon_and_other_punctuation_leave_the_phrase() -> None:
    # A colon inside the quoted phrase made CORE answer HTTP 500 (2026-09-25).
    [phrase, _] = _title_queries("Children's media: A review (2nd ed.) & more?", None)
    assert phrase == "title:\"Children's media A review 2nd ed more\""


def test_multi_word_and_unicode_surnames_are_quoted_or_kept() -> None:
    assert _author_clause("van Dijk, T.") == 'authors:"van Dijk"'
    assert _author_clause("Ménard, J.") == "authors:Ménard"
    assert _author_clause("") == ""


def test_no_author_and_short_titles_still_query_safely() -> None:
    assert _title_queries("Deep Residual Learning for Image Recognition", None)[1] == (
        "title:Deep AND title:Residual AND title:Learning AND title:Image AND title:Recognition")
    assert _title_queries("On it", None) == ['title:"On it"']


def _json_response(results):
    response = httpx.Response(200, request=httpx.Request("GET", "https://api.core.ac.uk/v3/search/outputs/"))
    response._content = json.dumps({"totalHits": len(results), "results": results}).encode()
    return response


def test_phrase_hit_needs_one_request(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.core.settings.CORE_API_KEY", SecretStr("test-key"))
    request = Mock(return_value=_json_response([{"id": 1, "title": "Attention Is All You Need",
                                                  "authors": [{"name": "Vaswani, Ashish"}], "yearPublished": 2017}]))
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    result = CoreRetriever().search_by_title_author("Attention is all you need", "Vaswani, A.")
    assert result.success and request.call_count == 1
    assert request.call_args.args[1]["q"].startswith('title:"Attention is all you need"')


def test_misquoted_title_falls_back_to_per_word_title_fields(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.core.settings.CORE_API_KEY", SecretStr("test-key"))
    found = {"id": 2, "title": "Attention Is All You Need", "authors": [{"name": "Vaswani, Ashish"}],
             "yearPublished": 2017}
    request = Mock(side_effect=[_json_response([]), _json_response([found])])
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    result = CoreRetriever().search_by_title_author("Attention is all we need", "Vaswani, A.")
    assert request.call_count == 2
    assert request.call_args_list[1].args[1]["q"] == "title:Attention AND title:need AND authors:Vaswani"
    assert result.title == "Attention Is All You Need"


def test_nothing_from_either_query_is_an_unqualified_empty(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.core.settings.CORE_API_KEY", SecretStr("test-key"))
    request = Mock(side_effect=[_json_response([]), _json_response([])])
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    result = CoreRetriever().search_by_title_author("A study that does not exist anywhere", "Nobody, N.")
    assert not result.success and result.error == "No results"
    assert result.metadata["identity_search_reason_code"] == "metadata_empty_response_unqualified"
