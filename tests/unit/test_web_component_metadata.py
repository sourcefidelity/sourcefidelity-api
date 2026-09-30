"""Explicit HTML component fields remain observations, not identity acceptance."""
import pytest
from app.services.web_source_metadata import extract_web_source_metadata


def extract(extra):
    return extract_web_source_metadata(
        '<html><head><meta name="citation_title" content="A chapter">'
        + extra + '</head><body></body></html>', 'https://publisher.example/chapter')


def test_explicit_component_metadata_is_retained_without_invented_fields():
    observed = extract('<meta name="citation_inbook_title" content="A collection">'
                       '<meta name="citation_firstpage" content="179">'
                       '<meta name="citation_lastpage" content="203">')
    assert observed['container_title'] == 'A collection'
    assert observed['pages'] == '179-203'
    assert observed['authors'] == [] and observed['year'] is None


@pytest.mark.parametrize('first,last', [('203', '179'), ('x', '203'), ('179', ''), ('1-5', '20')])
def test_ambiguous_or_incomplete_page_ranges_abstain(first, last):
    assert extract(f'<meta name="citation_firstpage" content="{first}">'
                   f'<meta name="citation_lastpage" content="{last}">')['pages'] is None


def test_conflicting_container_values_abstain():
    observed = extract('<meta name="citation_inbook_title" content="One">'
                       '<meta name="citation_inbook_title" content="Two">')
    assert observed['container_title'] is None


def test_bibliographic_citation_date_precedes_page_upload_date():
    observed = extract('<meta name="citation_date" content="1981/03/01">'
                       '<meta property="article:published_time" content="2009-02-22">')
    assert observed['year'] == '1981'
    assert observed['publication_date'] == '1981/03/01'
