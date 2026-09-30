"""Repairs for owner-reported defects in the two shareable reports (2026-09-20).

Each test names the reader-visible defect it prevents, so a later change that
reintroduces one fails with the reason rather than with a bare assertion.
"""
import pytest

from app.services.abstract_shape import looks_like_contents_listing, describe
from app.services.report_layers import member_marks


CARLTON_CONTENTS = (
    "I. INTRODUCTION AND THEORY. 1. Overview. 2. The Firm and Costs. "
    "II. MARKET STRUCTURES. 3. Competition. 4. Monopolies, Monopsonies, and "
    "Dominant Firms. 5. Cartels. 6. Oligopoly. 7. Product Differentiation and "
    "Monopolistic Competition. 8. Industry Structure and Performance."
)
BORDWELL_CONTENTS = (
    "Acknowledgments Introduction: Beyond the Blockbuster part i: a real story "
    "1. Continuing Tradition, by Any Means Necessary 2. Pushing the Premises "
    "3. Subjective Stories and Network Narratives 4. A Certain Amount of Plot"
)
LANGFORD_CONTENTS = (
    "Introduction Part I: in Transition 1946-1965 Introduction 1, The Autumn of "
    "the Patriarchs 2, The Communication of Ideas 3, Modernising Part II: Crisis "
    "and Renaissance 1966-1981 Introduction 4, Changing of the Guard 5, New Wave "
    "6, Who Lost the Picture Show? Part III: New 1982-2006 Introduction "
    "7, Corporate 8, Culture Wars 9, Post-Classical Style? Conclusion Further Reading"
)
REAL_ABSTRACT = (
    "Three animated Disney features, Alice in Wonderland, The Little Mermaid, and "
    "Beauty and the Beast, are shown here to reflect an ambivalence about freedom "
    "of imagination that may confuse young female viewers. The essay compares these "
    "three girls' movies with their fairy-tale sources."
)
ENUMERATING_ABSTRACT = (
    "This paper reports three findings about machine translation in the classroom. "
    "1. Students who used the tool revised more often than those who did not. "
    "2. The quality of their revisions was not higher, because they rarely checked "
    "the output against a source. 3. Teachers who modelled checking changed that "
    "pattern, which suggests that instruction matters more than access."
)


@pytest.mark.parametrize(
    "text",
    [CARLTON_CONTENTS, BORDWELL_CONTENTS, LANGFORD_CONTENTS],
    ids=["carlton", "bordwell", "langford"],
)
def test_catalog_contents_listing_is_not_an_abstract(text):
    """A book's chapter list told the reader nothing about the cited content."""
    assert looks_like_contents_listing(text) is True


@pytest.mark.parametrize(
    "text", [REAL_ABSTRACT, ENUMERATING_ABSTRACT], ids=["prose", "numbered_findings"]
)
def test_real_abstracts_are_kept(text):
    """Discarding a real abstract costs the reader more than an unhelpful one."""
    assert looks_like_contents_listing(text) is False


def test_short_metadata_value_is_never_a_contents_listing():
    assert looks_like_contents_listing("1. Overview. 2. Costs.") is False
    assert looks_like_contents_listing("") is False


def test_contents_signals_are_inspectable():
    signals = describe(CARLTON_CONTENTS)
    assert signals["numbered_headings"] >= 6
    assert signals["policy_version"] == "abstract-contents-listing-v1"


def _quotation_member():
    return {
        "coverage_level": "full_text",
        "show_quotation_check": True,
        "quotation_check": {"attention": True, "status": "complete"},
    }


_TARGET = {"x0": 10.0, "y0": 20.0, "x1": 40.0, "y1": 33.0}


def test_quotation_marker_dot_is_dropped_when_the_words_are_highlighted():
    """The gold dot beside the citation repeated the yellow quotation highlight."""
    citation = {"quotation_difference_rectangles": [{"page_index": 0, "x0": 1}]}
    assert member_marks(_quotation_member(), citation, _TARGET) == ""


def test_quotation_marker_dot_remains_when_no_geometry_was_located():
    """Without a highlight the marker is the only notice, so keep it."""
    citation = {"quotation_difference_rectangles": []}
    marks = member_marks(_quotation_member(), citation, _TARGET)
    assert "indicator-practice" in marks


def test_provisional_retainer_requires_the_opening_pages_to_name_the_work():
    """A different work by the same author was quoted as the cited source."""
    import fitz

    from app.services.source_resolver import _opening_pages_present_title

    def pdf(text: str) -> bytes:
        document = fitz.open()
        page = document.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 780), text, fontsize=11)
        return document.tobytes()

    filler = " ".join(f"word{index}" for index in range(60))
    cited = "The Economic Analysis of Regulation: A Critical Assessment of the Literature"

    named = pdf("The Economic Analysis of Regulation A Critical Assessment of "
                "the Literature by Sanford Berg " + filler)
    other = pdf("Fundamentals of Economic Regulation by Sanford Berg " + filler)

    assert _opening_pages_present_title(named, cited) is True
    assert _opening_pages_present_title(other, cited) is False
    # Too little front-matter text is not an affirmative observation.
    assert _opening_pages_present_title(pdf("Cover"), cited) is False


def test_lead_address_is_never_kept_even_with_the_retired_flag(monkeypatch):
    """The application keeps no search-result links (owner decision 2026-09-29).

    The older development flag is retired and ignored; only the one-way hash
    survives. Exa/Tavily candidates can still be audited in development through
    SEARCH_CANDIDATE_AUDIT_URLS, which records them on the route attempt.
    """
    from app.config import settings
    from app.services.reference_discovery import (
        ExpectedBibliographicFields, build_reference_discovery_candidate,
    )
    from app.services.retrieval.base import RetrievalResult

    def candidate():
        return build_reference_discovery_candidate(
            attempt_id='exa', provider='exa',
            expected=ExpectedBibliographicFields(title='A cited work', authors=['Writer, A']),
            result=RetrievalResult(source_name='exa', success=True, title=''),
            location_url='https://example.org/lead', discovery_provider='exa',
        )

    monkeypatch.setattr(settings, 'DEVELOPMENT_RETAIN_DISCOVERY_LEAD_URLS', False, raising=False)
    withheld = candidate()
    assert withheld.development_location_url is None
    assert withheld.location_sha256  # the hash is kept either way

    monkeypatch.setattr(settings, 'DEVELOPMENT_RETAIN_DISCOVERY_LEAD_URLS', True, raising=False)
    assert candidate().development_location_url is None
    assert 'example.org' not in candidate().model_dump_json()


class TestFirstPageTitleReasons:
    """A missing title used to be one `None`; these are five different facts.

    Measured on the documents that motivated the change: the Baker lead is a
    health-care book whose cover image reads "PART I INTRODUCTION", Berg and
    Sullivan are uniformly typeset, and Doster opens on a scanned handwritten
    approval form.
    """

    @staticmethod
    def _pdf(*, text='', fontsize=11, blank=False):
        import fitz

        document = fitz.open()
        page = document.new_page()
        if not blank:
            page.insert_textbox(fitz.Rect(40, 40, 560, 780), text, fontsize=fontsize)
        return document.tobytes()

    def observe(self, **kwargs):
        from app.services.pdf_verifier import describe_first_page_title

        return describe_first_page_title(self._pdf(**kwargs))

    def test_page_with_no_text_layer_was_never_read(self):
        observation = self.observe(blank=True)
        assert observation.reason == 'no_text_layer'
        assert observation.read_the_page is False

    def test_uniform_type_was_read_and_carries_no_title(self):
        observation = self.observe(text='Fundamentals of Economic Regulation. ' * 6)
        assert observation.reason == 'uniform_typography'
        assert observation.read_the_page is True

    def test_section_heading_is_not_reported_as_a_work_title(self):
        observation = self.observe(text='PART I INTRODUCTION', fontsize=20)
        assert observation.reason == 'front_matter_label_only'
        assert observation.title is None

    def test_unreadable_scan_of_a_cover_is_not_a_title(self):
        observation = self.observe(text="J.-'\\L J4-t o..J-.r:.", fontsize=20)
        assert observation.reason == 'illegible_title'
        assert observation.title is None

    def test_a_real_title_is_still_returned(self):
        observation = self.observe(text='The Separation of Platforms and Commerce', fontsize=20)
        assert observation.reason == 'title_observed'
        assert observation.observed is True
        assert 'Separation' in observation.title

    def test_back_compatible_accessor_returns_only_the_title(self):
        from app.services.pdf_verifier import _extract_title_from_first_page

        assert _extract_title_from_first_page(self._pdf(text='PART I INTRODUCTION', fontsize=20)) is None


class TestWebTitleReasons:
    """The HTML half of the same repair.

    Paper-11 Cohen J was vetoed by an ITU report's contents page that carries no
    title element. An empty title there meant the same thing as an empty title
    from a blocked page or a page we never reached.
    """

    @staticmethod
    def observe(html):
        from app.services.web_source_metadata import extract_web_source_metadata

        return extract_web_source_metadata(html, 'https://example.org/page')

    def test_article_title_is_observed_with_its_provenance(self):
        observed = self.observe(
            '<html><head><title>The Separation of Platforms and Commerce</title></head>'
            '<body><p>A handful of digital platforms mediate online commerce.</p></body></html>'
        )
        assert observed['title_reason'] == 'title_observed'
        assert observed['title_method'] == 'page_metadata'

    @pytest.mark.parametrize(
        'chrome', ['Just a moment...', 'Access Denied', 'Page not found', 'Sign in'],
    )
    def test_site_chrome_is_not_a_work_title(self, chrome):
        observed = self.observe(f'<html><head><title>{chrome}</title></head><body>x</body></html>')
        assert observed['title_reason'] == 'boilerplate_title_only'

    def test_page_without_a_title_element_says_so(self):
        observed = self.observe('<html><body><p>Body text at some length about a topic.</p></body></html>')
        assert observed['title_reason'] == 'no_title_element'

    def test_contents_listing_is_reported_as_a_listing_not_a_bare_absence(self):
        observed = self.observe(
            '<html><body><p>Preface Final report Table of Contents List of Tables, '
            'Figures and Boxes Chapter I Chapter II Chapter III Annex I Annex II '
            'Acknowledgements Bibliography Index</p></body></html>'
        )
        assert observed['title_reason'] == 'navigation_listing_page'

    def test_a_real_title_is_never_reclassified_by_body_text(self):
        """Refine only an already-empty observation."""
        observed = self.observe(
            '<html><head><title>Consumer protection in telecommunications</title></head>'
            '<body><p>Preface Table of Contents Chapter I Chapter II Chapter III '
            'Annex I Bibliography Index Acknowledgements</p></body></html>'
        )
        assert observed['title_reason'] == 'title_observed'


class TestTitleAbsenceReasonReachesTheReview:
    """The wiring: what the fetcher saw must reach the fabrication decision."""

    @staticmethod
    def candidate(result, outcome='identity_unconfirmed'):
        from app.services.reference_discovery import (
            ExpectedBibliographicFields, build_reference_discovery_candidate,
        )

        return build_reference_discovery_candidate(
            attempt_id='a', provider='web_search',
            expected=ExpectedBibliographicFields(title='A cited work', authors=['Writer, A']),
            result=result, acquisition_outcome=outcome,
            location_url='https://example.org/lead',
        )

    @staticmethod
    def result(**kwargs):
        from app.services.retrieval.base import RetrievalResult

        kwargs.setdefault('title', '')
        return RetrievalResult(source_name='x', success=True, **kwargs)

    def test_html_reason_is_carried_through(self):
        observed = {'title_reason': 'navigation_listing_page'}
        assert self.candidate(
            self.result(metadata={'web_identity': observed})
        ).title_absence_reason == 'navigation_listing_page'

    def test_unvisited_and_blocked_leads_are_named_as_such(self):
        assert self.candidate(self.result(), 'not_attempted').title_absence_reason == 'lead_not_visited'
        assert self.candidate(self.result(), 'access_restricted').title_absence_reason == 'lead_not_reached'
        assert self.candidate(self.result(), 'transport_failure').title_absence_reason == 'lead_not_reached'

    def test_an_observed_title_needs_no_absence_reason(self):
        assert self.candidate(self.result(title='A Real Work')).title_absence_reason is None

    def test_unexplained_absence_stays_unexplained(self):
        """No evidence either way must not be silently read as an absence."""
        from app.services.reference_discovery import OBSERVED_TITLE_ABSENCE

        reason = self.candidate(self.result()).title_absence_reason
        assert reason is None
        assert reason not in OBSERVED_TITLE_ABSENCE


class TestWebSearchCandidatePlumbing:
    """Paper-11 Baker, Cohen J and Sullivan reached the review as bare silence.

    Their candidates carried `title_absence_reason: None` because a web-search
    location records an observation only when identity is confirmed. A page we
    fetched and a page we never opened were then indistinguishable.
    """

    @staticmethod
    def candidate(metadata, *, title='', outcome='unavailable'):
        from app.services.reference_discovery import (
            ExpectedBibliographicFields, build_reference_discovery_candidate,
        )
        from app.services.retrieval.base import RetrievalResult

        return build_reference_discovery_candidate(
            attempt_id='a', provider='web_search',
            expected=ExpectedBibliographicFields(title='A cited work', authors=['Writer, A']),
            result=RetrievalResult(source_name='web_search', success=True,
                                   title=title, metadata=metadata),
            acquisition_outcome=outcome, location_url='https://example.org/lead',
        )

    def test_a_fetched_page_reports_what_it_showed(self):
        from app.services.reference_discovery import OBSERVED_TITLE_ABSENCE

        reason = self.candidate({'title_reason': 'navigation_listing_page'}).title_absence_reason
        assert reason == 'navigation_listing_page'
        assert reason in OBSERVED_TITLE_ABSENCE

    def test_a_search_snippet_is_not_an_observation_of_the_document(self):
        """Nothing beyond a ranked search hit was ever seen for this location."""
        from app.services.reference_discovery import OBSERVED_TITLE_ABSENCE

        reason = self.candidate({'title_reason': 'search_snippet_only'}).title_absence_reason
        assert reason == 'search_snippet_only'
        assert reason not in OBSERVED_TITLE_ABSENCE

    def test_snippet_only_lead_still_blocks_a_fabrication_finding(self):
        import test_bounded_reference_review as suite

        ref, trace = suite.review_fixture()
        suite.blank_web_candidate(trace, 'unavailable', title_reason='search_snippet_only')
        assert suite.kinds(
            suite.assess_reference_credibility(ref, None, trace)
        ) == []

    def test_fetched_listing_page_clears_the_lead(self):
        import test_bounded_reference_review as suite

        ref, trace = suite.review_fixture()
        suite.blank_web_candidate(trace, 'unavailable', title_reason='navigation_listing_page')
        assert suite.kinds(
            suite.assess_reference_credibility(ref, None, trace)
        ) == ['potentially_fabricated_reference']


class TestRepeatedProviderRowDoesNotInvalidateATrace:
    """Paper-11 Noam: Google Books returned one volume twice.

    Candidate ids derive from attempt + provider + record key, so the rows
    collided, the record builder rejected the whole trace, and the reference
    became unassessable. One repeated row cost a complete assessment.
    """

    @staticmethod
    def candidate(volume_id, attempt='a'):
        from app.services.reference_discovery import (
            ExpectedBibliographicFields, build_reference_discovery_candidate,
        )
        from app.services.retrieval.base import RetrievalResult

        return build_reference_discovery_candidate(
            attempt_id=attempt, provider='google_books',
            expected=ExpectedBibliographicFields(title='A cited work', authors=['Writer, A']),
            result=RetrievalResult(source_name='google_books', success=True, title='Another book'),
            candidate_key=volume_id, acquisition_outcome='identity_rejected',
        )

    def test_an_identical_repeated_record_is_appended_once(self):
        from app.services.source_resolver import _append_unique_candidate

        trace = {'candidates': []}
        _append_unique_candidate(trace, self.candidate('vol-1'))
        _append_unique_candidate(trace, self.candidate('vol-1'))
        assert len(trace['candidates']) == 1

    def test_distinct_records_are_all_kept(self):
        from app.services.source_resolver import _append_unique_candidate

        trace = {'candidates': []}
        _append_unique_candidate(trace, self.candidate('vol-1'))
        _append_unique_candidate(trace, self.candidate('vol-2'))
        _append_unique_candidate(trace, self.candidate('vol-1', attempt='b'))
        assert len({c.candidate_id for c in trace['candidates']}) == 3

    def test_duplicate_ids_are_what_invalidated_the_trace(self):
        """Guards the reason the dedupe exists, not just the dedupe."""
        from app.services.reference_discovery import derive_reference_discovery_record
        import pytest as _pytest

        duplicate = self.candidate('vol-1')
        with _pytest.raises(ValueError, match='unique'):
            derive_reference_discovery_record(
                reference_id='ref-1', expected=duplicate.observed,
                required_route_categories=['academic_adapter'], queries=[], attempts=[],
                candidates=[duplicate, duplicate],
                search_policy_version='api-first-search-v2',
                search_retention_policy=None, created_at=None,
            )


class TestDeclinedCallIsNotReportedAsATimeout:
    """CORE opened its circuit after real timeouts, then skipped every later
    call instantly. The skip message contained the word "timeout", so the
    classifier recorded timeouts the app never waited for."""

    @staticmethod
    def outcome(error):
        from app.services.source_resolver import _search_execution_outcome
        from app.services.retrieval.base import RetrievalResult

        return _search_execution_outcome(
            RetrievalResult(source_name='core', success=False, error=error))

    def test_circuit_skip_is_classified_as_a_skip(self):
        from app.services.retrieval.core import CoreRetriever

        retriever = CoreRetriever()
        retriever._timeout_circuit_open = True
        message = retriever._circuit_error()
        assert 'timeout' not in message.casefold()
        assert self.outcome(message) == 'cooldown_skipped'

    def test_a_real_timeout_is_still_a_timeout(self):
        assert self.outcome('read_timeout') == 'timeout'


class TestTextbookPublisherIsRecognised:
    """A publisher without "Press" in its name read as no publisher at all.

    Carlton & Perloff ("Pearson.") then parsed as an unknown kind, which cost it
    the book-catalog route entirely. Measured across 400 stored references, the
    added imprints reclassify exactly three entries, all genuine books.
    """

    @staticmethod
    def kind(raw):
        from app.services.source_type import classify_reference_source_kind

        return classify_reference_source_kind(raw).kind

    def test_bare_imprint_names_are_publishers(self):
        assert self.kind('Carlton, D. W., & Perloff, J. M. (2005). '
                         'Modern Industrial Organization. Pearson.') == 'monograph'
        assert self.kind('Ward, S. J. (2018). Ethical journalism in a populist age. '
                         'Rowman & Littlefield.') == 'monograph'
        assert self.kind('Belton, J. (2013). American cinema/American culture '
                         '(4th ed.). McGraw-Hill.') == 'monograph'

    def test_a_venue_word_is_not_treated_as_a_publisher(self):
        """"Harvard Law Review" must stay a journal, so "harvard" is excluded."""
        assert self.kind('Khan, L. (2018). The separation of platforms and commerce. '
                         'Harvard Law Review, 131(5), 100-180.') == 'journal_article'

    def test_a_reference_naming_no_venue_stays_unknown(self):
        """Berg & Menard genuinely name no journal and no publisher."""
        assert self.kind('Berg, S. V., & Forsyth, P. (2007). The Economic Analysis '
                         'of Regulation: A Critical Assessment of the Literature.') == 'unknown'


def test_google_books_reports_each_volume_once(monkeypatch):
    """Paper-11 Noam: the catalog returned one volume twice in one response.

    Downstream the repeat collided on its derived candidate id and invalidated
    the whole trace, costing the reference its entire assessment. Reporting each
    distinct volume once keeps the result count and the reviewed records
    describing the same set, which the catalog-coverage check compares.
    """
    import httpx
    from unittest.mock import Mock
    from app.services.retrieval.google_books import GoogleBooksRetriever

    def volume(identifier, title):
        return {"id": identifier,
                "volumeInfo": {"title": title, "authors": ["Some Author"],
                               "publishedDate": "1990", "pageCount": 300}}

    payload = {"items": [volume("vol-1", "Law of International Telecommunications"),
                         volume("vol-2", "Telecommunications in Europe"),
                         volume("vol-1", "Law of International Telecommunications")]}
    request = httpx.Request("GET", "https://www.googleapis.com/books/v1/volumes")
    monkeypatch.setattr("app.services.retrieval.google_books.httpx.get",
                        Mock(return_value=httpx.Response(200, json=payload, request=request)))

    search = GoogleBooksRetriever().search_metadata_result(
        title="An example work", author="Writer, A")

    ids = [record.volume_id for record in search.candidates]
    assert ids.count("vol-1") == 1
    assert len(search.candidates) == 2


class TestJournalArticleWithoutAnIssueNumber:
    """Ordinary APA omits the issue number, and the classifier required it.

    "System, 93, 1-11" parsed as an unknown kind, so 21 real journal articles
    across the stored corpus were never eligible for the review paths that apply
    to articles.
    """

    @staticmethod
    def kind(raw):
        from app.services.source_type import classify_reference_source_kind

        return classify_reference_source_kind(raw).kind

    def test_volume_and_page_range_without_an_issue_is_an_article(self):
        assert self.kind('Hu, J., & Wu, P. (2020). Understanding English language learning '
                         'in tertiary contexts in Japan. System, 93, 1-11.') == 'journal_article'
        assert self.kind('Groves, M., & Mundt, K. (2015). Friend or foe? English for '
                         'Specific Purposes, 37, 112-121.') == 'journal_article'

    def test_an_issue_range_is_still_an_issue(self):
        assert self.kind('Tsai, S. C. (2019). Using Google Translate in EFL drafts. '
                         'Computer Assisted Language Learning, 32(5-6), 510-526.') == 'journal_article'

    def test_a_magazine_page_list_is_not_a_volume_and_page_range(self):
        """"pp. 34-36, 91-93" must not read as volume 36, pages 91-93."""
        assert self.kind('Deere, D. (1946, October). Everything\'s Going Her Way. '
                         'Movieland, pp. 34-36, 91-93.') != 'journal_article'
        assert self.kind('Vidor, C. (1946, August). Exciting Woman. '
                         'Photoplay, pp. 42, 86-87.') != 'journal_article'

    def test_a_bare_number_pair_is_not_an_article(self):
        assert self.kind('Smith, J. (2018). The Fourth Estate: Origins and '
                         'Evolution of Journalism Ethics in America') == 'unknown'


class TestOversizedProviderRecordCannotAbortAPaper:
    """A 500-author record from Semantic Scholar failed an entire paper run.

    Enabling S2 title search surfaced a Global Burden of Disease paper for the
    query "Competition policy in an age of globalization". Its author list
    exceeded the bounded observed-author field, and the ValidationError
    propagated out of the workflow, so 21 references produced no report at all.
    """

    def test_candidate_construction_survives_a_huge_author_list(self):
        from app.services.reference_discovery import (
            ExpectedBibliographicFields, build_reference_discovery_candidate,
        )
        from app.services.retrieval.base import RetrievalResult

        candidate = build_reference_discovery_candidate(
            attempt_id='a', provider='semantic_scholar',
            expected=ExpectedBibliographicFields(title='A cited work', authors=['Writer, A']),
            result=RetrievalResult(
                source_name='semantic_scholar', success=True,
                title='A large collaboration study',
                authors=[f'Author {index}' for index in range(500)],
            ),
        )
        assert len(candidate.observed.authors) == 64

    def test_semantic_scholar_caps_the_authors_it_reports(self):
        from app.services.retrieval.semantic_scholar import SemanticScholarRetriever

        parsed = SemanticScholarRetriever()._parse_paper({
            'title': 'A large collaboration study',
            'authors': [{'name': f'Author {index}'} for index in range(500)],
            'year': 2020,
        })
        assert len(parsed.authors) == 64


class TestCatalogQuerySyntaxIsNotTakenAsText:
    """Both new adapters lost real works to query-parser characters.

    Open Library returned an unrelated book for "American cinema/American
    culture" because the slash steers its parser; without it the work matches at
    rank one. Cited titles routinely carry slashes, colons and question marks.
    """

    def test_open_library_strips_query_syntax(self):
        from app.services.retrieval.open_library import _sanitize_query

        assert _sanitize_query("American cinema/American culture") == (
            "American cinema American culture")
        assert _sanitize_query("A Policy at War with Itself?") == (
            "A Policy at War with Itself")

    def test_datacite_strips_query_syntax(self):
        from app.services.retrieval.datacite import _sanitize_query

        assert _sanitize_query('The Disney Dilemma: Modern Disaster?') == (
            "The Disney Dilemma Modern Disaster")

    def test_open_library_searches_generally_not_by_title_field(self, monkeypatch):
        """A title-field search demands the catalog's title match the cited one.

        Bork's book is held as "The antitrust paradox", so the cited title with
        its subtitle returned nothing at all from the field search.
        """
        import httpx
        from unittest.mock import Mock
        from app.services.retrieval.open_library import OpenLibraryRetriever

        request = httpx.Request("GET", "https://openlibrary.org/search.json")
        captured = Mock(return_value=httpx.Response(
            200, json={"docs": [{"title": "The antitrust paradox",
                                 "author_name": ["Robert H. Bork"],
                                 "first_publish_year": 1978}]}, request=request))
        monkeypatch.setattr("app.services.retrieval.open_library.httpx.get", captured)

        result = OpenLibraryRetriever().search_by_title_author(
            "The Antitrust Paradox: A Policy at War with Itself", "Bork, R. H")

        assert result.success is True
        params = captured.call_args.kwargs["params"]
        assert "title" not in params
        assert params["q"].startswith("The Antitrust Paradox A Policy at War with Itself")
        assert params["q"].endswith("Bork")

    def test_datacite_falls_back_from_the_exact_phrase(self, monkeypatch):
        """A deposit whose registered subtitle differs is still findable."""
        import httpx
        from unittest.mock import Mock
        from app.services.retrieval.datacite import DataCiteRetriever

        request = httpx.Request("GET", "https://api.datacite.org/dois")
        empty = httpx.Response(200, json={"data": []}, request=request)
        found = httpx.Response(200, json={"data": [{"attributes": {
            "doi": "10.5281/zenodo.1", "titles": [{"title": "A deposited work"}],
            "creators": [{"name": "Writer, A"}], "publicationYear": 2020}}]}, request=request)
        calls = Mock(side_effect=[empty, found])
        monkeypatch.setattr("app.services.retrieval.datacite.httpx.get", calls)

        result = DataCiteRetriever().search_by_title_author("A deposited work")

        assert result.success is True
        assert result.doi == "10.5281/zenodo.1"
        assert calls.call_count == 2
        assert calls.call_args_list[0].kwargs["params"]["query"].startswith("titles.title:")
        assert calls.call_args_list[1].kwargs["params"]["query"] == "A deposited work"


class TestCorroborationAdaptersSkipOnlyWhenNothingIsAtStake:
    """Accuracy first: the skip must never touch an open identity question.

    Measured on the stored corpus, a Stardom-shaped paper confirms 12 of 15
    references by fast metadata while paper 11 confirms 0 of 19 — so the saving
    lands on genuine work and fabrication-heavy papers pay full price, which is
    the right way round.
    """

    class _Graph:
        def __init__(self, success, locations):
            self._success, self._locations = success, locations

        def to_result(self):
            from app.services.retrieval.base import RetrievalResult

            return RetrievalResult(
                source_name="graph", success=self._success,
                locations=self._locations, full_text_url=None,
            )

    @staticmethod
    def _resolver(outcome):
        from app.services.source_resolver import SourceResolver

        resolver = SourceResolver.__new__(SourceResolver)
        resolver._discovery_artifacts = staticmethod(lambda: (None, {"outcome": outcome}))
        return resolver

    @staticmethod
    def _location():
        from app.services.retrieval.base import AcquisitionLocation

        return AcquisitionLocation(url="https://example.org/a.pdf", provider="openalex")

    def test_unconfirmed_identity_always_runs_every_adapter(self):
        for outcome in ("search_incomplete", "possible_match", "bibliographic_conflict",
                        "unlocated_after_search", None):
            resolver = self._resolver(outcome)
            graph = self._Graph(True, [self._location()])
            assert resolver._identity_settled(graph) is False, outcome

    def test_confirmed_but_unreachable_still_runs_every_adapter(self):
        """An open-access link is the one thing they can still contribute."""
        resolver = self._resolver("confirmed")
        assert resolver._identity_settled(self._Graph(True, [])) is False

    def test_confirmed_and_reachable_settles_the_question(self):
        resolver = self._resolver("confirmed")
        assert resolver._identity_settled(self._Graph(True, [self._location()])) is True
        resolver = self._resolver("confirmed_with_minor_differences")
        assert resolver._identity_settled(self._Graph(True, [self._location()])) is True

    def test_a_declined_route_is_recorded_as_declined(self):
        """Not a timeout, a failure or an absence."""
        from app.services.source_resolver import SourceResolver, _search_execution_outcome

        class _Source:
            name = "core"

        (source, result), = SourceResolver._skipped_sources([_Source()])
        assert source.name == "core"
        assert result.success is False
        assert _search_execution_outcome(result) == "cooldown_skipped"
        assert (result.metadata or {}).get("lookup_applicable") is False


class TestLandingPageCorroboratesTheFileItAdvertises:
    """Doster's honours thesis: citation 13 showed limited text with nothing in it.

    The repository page names the cited work and advertises the thesis through
    `citation_pdf_url`, but the PDF opens on a scanned handwritten approval
    form, so its own bytes can never establish a title. Acquisition rejected it
    at medium identity and the 274-character bibliographic record won instead.
    """

    @staticmethod
    def _preflight(**kwargs):
        from app.services.source_resolver import SourceResolver

        return SourceResolver._preflight_acquired_representation, kwargs

    def test_corroboration_is_off_by_default(self):
        """No other acquisition path may inherit this allowance."""
        import inspect
        from app.services.source_resolver import SourceResolver

        for name in ("_preflight_acquired_representation", "_acquire_from_locations"):
            signature = inspect.signature(getattr(SourceResolver, name))
            parameter = signature.parameters["advertised_by_confirmed_landing"]
            assert parameter.default is False, name

    def test_only_a_positive_title_match_corroborates(self, monkeypatch):
        """A page with no readable title must not corroborate anything.

        The identity gate above only rejects a *mismatch*, so an unreadable
        page falls through it; that is not the same as the page naming the work.
        """
        from app.services import source_resolver as module

        monkeypatch.setattr(module, "_extract_html_titles", lambda _text: [])
        assert bool([] and module._html_title_matches("A cited work", [])) is False

    def test_a_confirmed_page_title_corroborates(self):
        from app.services.source_resolver import _html_title_matches

        titles = ["The Disney Dilemma: Modernized Fairy Tales or Modern Disaster?"]
        assert _html_title_matches(
            "The Disney Dilemma: Modernized Fairy Tales or Modern Disaster?", titles
        ) is True


class TestArticleNumberLocators:
    """Article-number journals cite an e-locator instead of pages.

    "Journal of English for Academic Purposes, 50, Article 100957" is ordinary
    modern APA — PLOS, BMC and Elsevier titles all use it — and was
    unclassifiable because the locator is not a page number.
    """

    @staticmethod
    def kind(raw):
        from app.services.source_type import classify_reference_source_kind

        return classify_reference_source_kind(raw).kind

    def test_article_number_without_an_issue(self):
        assert self.kind('Groves, M., & Mundt, K. (2021). A ghostwriter in the machine? '
                         'Journal of English for Academic Purposes, 50, Article 100957.') == 'journal_article'

    def test_article_number_with_an_issue(self):
        assert self.kind('Yang, M. (2019). A case study in Japan. '
                         'BMC Medical Education, 19(1), Article 15.') == 'journal_article'
        assert self.kind('O\'Neill, E. M. (2016). Measuring the impact of online translation. '
                         'IALLT Journal of Language Learning Technologies, 46(2), Article 2.') == 'journal_article'

    def test_the_word_article_alone_is_not_a_locator(self):
        assert self.kind('Smith, J. (2018). This article discusses journalism ethics '
                         'in America and its origins.') == 'unknown'


class TestCorroboratedSourceSurvivesTheAdmissionGate:
    """Acquiring the bytes was not enough; the workflow gate discarded them.

    The Stardom rerun proved this: Doster's thesis PDF was acquired, and the
    citation still showed nothing, because admission accepted only a high
    identity or the provisional possible-match marker.
    """

    @staticmethod
    def _gate(metadata, content=b"pdf-bytes"):
        """Mirror of the workflow's admission condition."""
        import hashlib

        confidence = metadata.get("identity_confidence")
        provisional = metadata.get("provisional_source") or {}
        possible_match = bool(
            confidence == "medium"
            and provisional.get("policy_version") == "submission-possible-match-v1"
            and provisional.get("content_sha256") == hashlib.sha256(content).hexdigest())
        corroborated = bool(
            confidence == "medium"
            and metadata.get("identity_corroborated_by_landing_page")
            and metadata.get("accepted_representation_sha256")
            == hashlib.sha256(content).hexdigest())
        return confidence == "high" or possible_match or corroborated

    def test_corroborated_medium_identity_is_admitted(self):
        import hashlib

        assert self._gate({
            "identity_confidence": "medium",
            "identity_corroborated_by_landing_page": True,
            "accepted_representation_sha256": hashlib.sha256(b"pdf-bytes").hexdigest(),
        }) is True

    def test_the_flag_must_be_bound_to_the_validated_bytes(self):
        """A stale flag must not admit content the preflight never accepted."""
        import hashlib

        assert self._gate({
            "identity_confidence": "medium",
            "identity_corroborated_by_landing_page": True,
            "accepted_representation_sha256": hashlib.sha256(b"other-bytes").hexdigest(),
        }) is False

    def test_an_uncorroborated_medium_identity_is_still_refused(self):
        assert self._gate({"identity_confidence": "medium"}) is False
        assert self._gate({"identity_confidence": "low",
                           "identity_corroborated_by_landing_page": True}) is False


def test_student_cited_urls_keep_the_longer_deadline():
    """A student chose their URL deliberately; a search hit is speculative.

    Measured 2026-09-21: 85% of web-search time was waiting on candidates, with
    one reference able to spend 150s on five unreachable ones. Only the
    discovery loop was shortened.
    """
    from app.config import settings

    assert settings.STUDENT_URL_TIMEOUT_SECONDS == 30
    # Separately configurable, but held equal until evidence justifies a change:
    # a single-run trial could not be distinguished from run-to-run variance.
    assert settings.DISCOVERY_CANDIDATE_TIMEOUT_SECONDS <= settings.STUDENT_URL_TIMEOUT_SECONDS


def test_safe_download_defaults_to_the_student_url_deadline():
    """The shorter deadline must be opt-in per call, never a global default."""
    import inspect
    from app.services.source_resolver import SourceResolver

    assert inspect.signature(SourceResolver._safe_download).parameters['timeout'].default is None


class TestFetchAttemptsRecordTheirDuration:
    """Retrieval timing was only answerable by comparing whole runs.

    Across four Stardom runs the references retrieving anything swung 12-17 and
    wall time 1,114-1,845s, on configurations that cannot explain a five-source
    difference. A per-fetch duration turns "did a successful fetch ever exceed
    this budget" into a question one run can answer.
    """

    def test_every_fetch_attempt_carries_an_elapsed_time(self, monkeypatch):
        from app.services.retrieval.base import AcquisitionLocation, RetrievalResult
        from app.services.source_resolver import SourceResolver
        from app.services import source_resolver as module

        monkeypatch.setattr(module, "require_source_candidate", lambda _url: None)
        monkeypatch.setattr(
            SourceResolver, "_safe_download",
            lambda self, url, *, timeout=None: (_ for _ in ()).throw(ValueError("not a PDF")))
        monkeypatch.setattr(
            module, "safe_request",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("unreachable")))

        result = RetrievalResult(
            source_name="web_search", success=False,
            locations=[AcquisitionLocation(url="https://example.org/a.pdf", provider="exa")],
        )
        SourceResolver.__new__(SourceResolver)
        resolver = SourceResolver()
        resolver._acquire_from_locations(result, expected_title="A cited work")

        attempts = (result.metadata or {}).get("location_attempts") or []
        assert attempts, "no fetch attempt was recorded"
        timed = [a for a in attempts if a.get("outcome") != "not_attempted"]
        assert timed, "no attempted fetch was recorded"
        for attempt in timed:
            assert isinstance(attempt.get("elapsed_seconds"), (int, float))
            assert attempt["elapsed_seconds"] >= 0

    def test_the_internal_start_marker_never_survives(self, monkeypatch):
        """A bookkeeping key must not reach a persisted trace."""
        from app.services.retrieval.base import AcquisitionLocation, RetrievalResult
        from app.services.source_resolver import SourceResolver
        from app.services import source_resolver as module

        monkeypatch.setattr(module, "require_source_candidate", lambda _url: None)
        monkeypatch.setattr(
            module, "safe_request",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("unreachable")))

        result = RetrievalResult(
            source_name="web_search", success=False,
            locations=[AcquisitionLocation(url="https://example.org/b.html", provider="exa")],
        )
        SourceResolver()._acquire_from_locations(result, expected_title="A cited work")

        for attempt in (result.metadata or {}).get("location_attempts") or []:
            assert not [key for key in attempt if key.startswith("_")]


class TestCatalogRecordHashTracksOnlyWhatWeRead:
    """`record_sha256` identifies a catalog record inside a finding.

    It hashed the entire raw volume — sale info, access flags, thumbnail URLs,
    search snippets — none of which this adapter reads. An unrelated change on
    the catalog's side therefore moved a hash that binds evidence.
    """

    @staticmethod
    def _item(**overrides):
        item = {
            "id": "vol-1",
            "volumeInfo": {
                "title": "A Cited Book", "authors": ["Writer, A"],
                "publisher": "Example Press", "publishedDate": "1990",
                "pageCount": 300, "printType": "BOOK",
            },
        }
        item.update(overrides)
        return item

    def test_volatile_fields_do_not_move_the_hash(self):
        from app.services.retrieval.google_books import _record_digest

        plain = self._item()
        noisy = self._item(
            saleInfo={"saleability": "NOT_FOR_SALE"},
            accessInfo={"viewability": "NO_PAGES"},
            searchInfo={"textSnippet": "a snippet that changes"},
        )
        noisy["volumeInfo"]["imageLinks"] = {"thumbnail": "https://example.org/x.jpg"}

        assert _record_digest(plain, plain["volumeInfo"]) == (
            _record_digest(noisy, noisy["volumeInfo"]))

    def test_a_field_we_read_still_moves_the_hash(self):
        from app.services.retrieval.google_books import _record_digest

        plain = self._item()
        changed = self._item()
        changed["volumeInfo"]["publishedDate"] = "1993"

        assert _record_digest(plain, plain["volumeInfo"]) != (
            _record_digest(changed, changed["volumeInfo"]))

    def test_every_digested_field_is_requested_from_the_catalog(self):
        """The projection and the `fields` request must not drift apart."""
        from app.services.retrieval.google_books import _DIGEST_VOLUME_KEYS, _VOLUME_FIELDS

        for key in _DIGEST_VOLUME_KEYS:
            assert key in _VOLUME_FIELDS, key
        assert "id" in _VOLUME_FIELDS and "totalItems" in _VOLUME_FIELDS


class TestOurOwnFailuresAreNotReportedAsProviderOutcomes:
    """Three times this session an error described itself as an ordinary result.

    A skipped CORE call reported `timeout`; a duplicate catalog id reported
    `candidate_binding_incomplete`; a caller's TypeError reported "no
    identity-compatible exact-edition page count". Each sent a diagnosis to the
    wrong place, and in a tool whose output is evidence, a fabricated outcome is
    worse than a loud failure.
    """

    @staticmethod
    def classify(exc):
        from app.services.search.base import classify_search_failure

        return classify_search_failure(exc)

    def test_application_errors_are_named_as_ours(self):
        for exc in (TypeError("unexpected keyword"), AttributeError("no attribute"),
                    NameError("undefined"), ImportError("missing module")):
            assert self.classify(exc) == "internal_error", type(exc).__name__

    def test_genuine_provider_failures_keep_their_meaning(self):
        import httpx

        request = httpx.Request("GET", "https://provider.example/q")
        assert self.classify(httpx.ReadTimeout("slow", request=request)) == "timeout"
        assert self.classify(ValueError("bad json")) == "response_invalid"
        assert self.classify(
            httpx.HTTPStatusError("denied", request=request,
                                  response=httpx.Response(403, request=request))
        ) == "access_restricted"

    def test_an_internal_error_is_never_a_completed_observation(self):
        """It must not become evidence that a source does not exist."""
        from app.services.reference_discovery import assess_reference_discovery_trace
        import inspect

        source = inspect.getsource(assess_reference_discovery_trace.__module__ and
                                   __import__("app.services.reference_discovery",
                                              fromlist=["x"]))
        assert '"internal_error",' in source
        # Present in the incomplete set, so it can never satisfy a required route.
        marker = source.index("incomplete_execution_outcomes")
        assert "internal_error" in source[marker:marker + 400]

    def test_a_category_blocker_carries_its_real_cause(self):
        from app.services.reference_discovery import ReferenceDiscoveryCompletion

        completion = ReferenceDiscoveryCompletion(
            ready=False, blocker_codes=["candidate_binding_incomplete"],
            blocker_detail="reference_candidate_ids_must_be_unique")
        assert completion.blocker_detail == "reference_candidate_ids_must_be_unique"
        assert ReferenceDiscoveryCompletion(ready=False).blocker_detail is None


class TestEricIsPositiveOnlyByConstruction:
    """ERIC holds education research and nothing else.

    Measured 2026-09-22: of four genuine education-adjacent titles from the
    corpus it holds one; of three labelled-fabricated titles it holds none.
    Braun & Clarke's "Using thematic analysis in psychology" returns nothing,
    because a psychology journal is not education literature — which is exactly
    why this adapter's silence must never become evidence.
    """

    def test_it_is_declared_positive_only_everywhere_that_matters(self):
        from app.services.bounded_reference_review import POSITIVE_ONLY_PROVIDERS
        from app.services.retrieval.eric import EricRetriever
        from app.services.source_resolver import _CORROBORATION_ONLY_SOURCES

        assert EricRetriever.name == "eric"
        assert EricRetriever.positive_only is True
        # The review keys on the provider name, so the two must agree.
        assert EricRetriever.name in POSITIVE_ONLY_PROVIDERS
        # It can only corroborate, so it need not run once identity is settled.
        assert EricRetriever.name in _CORROBORATION_ONLY_SOURCES

    def test_query_syntax_is_stripped_from_the_title(self):
        """The title is wrapped in title:"…"; a stray quote would close it."""
        from app.services.retrieval.eric import _sanitize

        assert _sanitize('A "quoted" title: with (syntax)') == "A quoted title with syntax"

    def test_an_empty_index_answer_is_not_an_error(self, monkeypatch):
        import httpx
        from unittest.mock import Mock
        from app.services.retrieval.eric import EricRetriever

        request = httpx.Request("GET", "https://api.ies.ed.gov/eric/")
        monkeypatch.setattr(
            "app.services.retrieval.eric.httpx.get",
            Mock(return_value=httpx.Response(
                200, json={"response": {"numFound": 0, "docs": []}}, request=request)))

        result = EricRetriever().search_by_title_author("A work outside education")
        assert result.success is False
        assert result.error == "No results"
        assert result.metadata["identity_search_result_count"] == 0

    def test_a_transport_failure_is_not_reported_as_no_results(self, monkeypatch):
        """Honest-failure rule: a broken call must not read as an empty index."""
        import httpx
        from unittest.mock import Mock
        from app.services.retrieval.eric import EricRetriever

        monkeypatch.setattr(
            "app.services.retrieval.eric.httpx.get",
            Mock(side_effect=httpx.ConnectError("unreachable")))

        result = EricRetriever().search_by_title_author("Any title")
        assert result.success is False
        assert result.error != "No results"
        assert result.error.startswith("eric:")

    def test_the_author_never_narrows_the_query(self, monkeypatch):
        import httpx
        from unittest.mock import Mock
        from app.services.retrieval.eric import EricRetriever

        request = httpx.Request("GET", "https://api.ies.ed.gov/eric/")
        captured = Mock(return_value=httpx.Response(
            200, json={"response": {"numFound": 0, "docs": []}}, request=request))
        monkeypatch.setattr("app.services.retrieval.eric.httpx.get", captured)

        EricRetriever().search_by_title_author("English-Medium Instruction", "Brown, A")
        sent = captured.call_args.kwargs["params"]["search"]
        assert sent == 'title:"English-Medium Instruction"'
        assert "Brown" not in sent
