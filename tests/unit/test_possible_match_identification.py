"""A possible match must survive more than a title phrase.

Stardom's report offered a University of Hertfordshire thesis, "Working Below
the Line in the Studio System", as a possible match for Belton's chapter "The
Studio System": a different author, a different year and a different
publication history, joined only by a phrase one title happened to contain.
"""
from types import SimpleNamespace

import fitz
import pytest

from app.services import source_resolver as resolver_module
from app.services.source_resolver import SourceResolver, _PROVISIONAL_CANDIDATES
from app.services.file_safety import SafetyVerdict
from app.services.pdf_verifier import _title_matches
from app.services.retrieval.base import RetrievalResult

THESIS = ("Working Below the Line in the Studio System: Exploring Labour "
          "Processes in the UK Film Industry 1927-1950")


@pytest.mark.parametrize('cited,observed,expected', [
    ('The Studio System', THESIS, False),
    ('The Studio System', 'The Studio System: A History of Hollywood', True),
    ('The Studio System', 'The studio system', True),
    ('Studio system', 'The Studio System: A History', True),
    ('The Studio System: A History', 'The Studio System', True),
    ('The antitrust paradox', 'The Antitrust Paradox: A Policy at War with Itself', True),
    ('Competition policy', 'Merger review and competition policy in Canada', False),
])
def test_a_cited_title_may_lose_a_subtitle_but_never_float_inside_another(cited, observed, expected):
    assert _title_matches(cited, {'title': observed}) is expected


def _cover(title, author, year):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 140), title, fontsize=22)
    page.insert_text((72, 200), 'by ' + author, fontsize=12)
    page.insert_text((72, 240), str(year), fontsize=12)
    page.insert_text((72, 300), 'The opening paragraph of the work follows.', fontsize=11)
    content = document.tobytes()
    document.close()
    return content


@pytest.mark.parametrize('author,year,retained', [
    ('Rebecca Harrison', 2015, 0),   # neither the cited author nor the cited year
    ('John Belton', 2015, 1),        # the copy names the cited author
    ('Rebecca Harrison', 2013, 1),   # the copy carries the cited year
])
def test_an_extended_title_needs_the_cited_author_or_year(monkeypatch, author, year, retained):
    monkeypatch.setattr(resolver_module, 'inspect_uploaded_pdf',
                        lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN))
    result = RetrievalResult(source_name='test', success=True,
                             full_text=_cover('The Studio System: A History', author, year))
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        object.__new__(SourceResolver)._retain_provisional_candidate(
            result, confidence='medium', reason='Year uncertain', completeness='complete',
            text_quality='digital', kind_verdict='unknown',
            expected_title='The Studio System', expected_author='Belton, J',
            expected_year='2013')
        assert len(_PROVISIONAL_CANDIDATES.get()) == retained
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)


def test_an_identical_title_still_needs_no_corroboration(monkeypatch):
    """The existing route for a matching cover is unchanged."""
    monkeypatch.setattr(resolver_module, 'inspect_uploaded_pdf',
                        lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN))
    result = RetrievalResult(source_name='test', success=True,
                             full_text=_cover('The Studio System', 'Rebecca Harrison', 2015))
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        object.__new__(SourceResolver)._retain_provisional_candidate(
            result, confidence='medium', reason='Year uncertain', completeness='complete',
            text_quality='digital', kind_verdict='unknown',
            expected_title='The Studio System', expected_author='Belton, J',
            expected_year='2013')
        assert len(_PROVISIONAL_CANDIDATES.get()) == 1
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)


@pytest.mark.parametrize('identity,best,tone', [
    ('possible_match', None, 'not_assessed'),
    ('verified', None, 'retrieved_no_connection'),
    ('possible_match', {'text': 'A retained passage.'}, 'evidence_available'),
])
def test_no_displayed_passage_never_reads_as_evidence_available(identity, best, tone):
    from app.services.evidence_report import member_tone
    assert member_tone({'coverage_level': 'full_text', 'relevance_status': 'not_assessed',
                        'identity_status': identity, 'best_evidence': best}) == tone


@pytest.mark.parametrize('cited,page_titles,confirms', [
    ('The Studio System', [THESIS], False),
    ('The Studio System', ['The Studio System | University Press'], True),
    ('The Studio System', ['The Studio System: A History - Publisher'], True),
    ('The Disney Dilemma: Modernized Fairy Tales or Modern Disaster?',
     ['The Disney Dilemma: Modernized Fairy Tales or Modern Disaster? by Ivy V. Doster'], True),
    ('The Disney Dilemma: Modernized Fairy Tales or Modern Disaster?', ['Login required'], False),
])
def test_a_landing_page_confirms_identity_only_on_an_aligned_title(cited, page_titles, confirms):
    from app.services.source_resolver import _landing_title_confirms
    assert _landing_title_confirms(cited, page_titles) is confirms


def test_the_rejection_filter_stays_permissive():
    """Confirmation was tightened; rejecting a page was not."""
    from app.services.source_resolver import _html_title_matches
    assert _html_title_matches('The Studio System', [THESIS]) is True
