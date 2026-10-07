"""Owner decisions 2026-10-06: a student's link that names the cited work keeps
the reference from "Cannot be verified" (P5 review of the APA papers)."""
from types import SimpleNamespace

from app.services.cross_script_identity import judge_cross_script_identity, validate_answer
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate
from app.services.reference_verification import registered_unread_doi
from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.internet_archive import item_text_identity
from app.services.source_resolver import chapter_editors


def _doi_observation(first_status=302, origin='https://doi.org'):
    return [{'kind': 'doi', 'requests': [{'hops': [
        {'destination_origin': origin, 'http_status': first_status, 'location_sha256': 'a' * 64},
        {'destination_origin': 'https://records.example.test', 'http_status': 200, 'location_sha256': None}]}]}]


def test_a_registered_doi_no_index_reads_is_never_unverifiable():
    reference = SimpleNamespace(doi='10.9999/j.issn.1234-5678.2005.03.007', year='2005', issue='3')
    assert registered_unread_doi(reference, {'candidates': []}, _doi_observation()) == 'agrees'
    # The suffix's issue disagrees, or carries no year: registered, not agreeing.
    assert registered_unread_doi(SimpleNamespace(**{**vars(reference), 'issue': '4'}), {}, _doi_observation()) == 'registered'
    assert registered_unread_doi(SimpleNamespace(**{**vars(reference), 'year': '2011'}), {}, _doi_observation()) == 'registered'
    # doi.org answering 404 is no registration; an index that read the DOI decides instead.
    assert registered_unread_doi(reference, {}, _doi_observation(first_status=404)) is None
    read = {'candidates': [{'observed': {'doi': reference.doi.upper()}}]}
    assert registered_unread_doi(reference, read, _doi_observation()) is None
    assert registered_unread_doi(SimpleNamespace(doi='', year='2005', issue=''), {}, _doi_observation()) is None


ISSUE_META = {'identifier': 'Gazette-1931-02', 'title': 'Picture Gazette 1931-02 Vol. 9, No. 4',
              'date': '1931-02', 'volume': '9', 'creator': 'Picture Gazette'}
ISSUE_TEXT = ("CONTENTS ... THE QUIET STAGE DOOR. By Ada Morrow ... 22\n"
              "page after page of other features\nThe Quiet Stage Door — continued from page 23")


def test_an_archived_issue_names_the_article_in_its_text():
    observed = item_text_identity(ISSUE_META, ISSUE_TEXT, title='The quiet stage door', author='Morrow, A.')
    assert observed['title'] == 'The quiet stage door' and observed['authors'] == ['Morrow, A.']
    assert (observed['year'], observed['container_title'], observed['volume'], observed['issue']) == ('1931', 'Picture Gazette', '9', '4')
    # OCR that garbles the name leaves the author unknown, never a conflict.
    assert item_text_identity(ISSUE_META, ISSUE_TEXT, title='The quiet stage door', author='Marlow, A.')['authors'] == []
    assert item_text_identity(ISSUE_META, ISSUE_TEXT, title='A different feature entirely', author='Morrow, A.') is None
    assert item_text_identity(ISSUE_META, ISSUE_TEXT, title='Stage', author='Morrow, A.') is None


def test_a_model_comparison_across_scripts_becomes_the_title_and_author_comparison():
    expected = ExpectedBibliographicFields(title='Rivers of the northern plain', authors=['Lin, Q.'], year='2021')

    def candidate(same_work, author='agrees'):
        result = RetrievalResult(source_name='web_fetch', success=False, title='北方平原的河流', authors=['林青'],
                                 year='2021', metadata={'cross_script_judgment': {'same_work': same_work, 'author': author}})
        built = build_reference_discovery_candidate(attempt_id='a', provider='student_url_html', expected=expected, result=result)
        return {c.field_name: (c.outcome, c.reason_code) for c in built.comparisons}

    same = candidate('yes')
    assert same['title'] == ('agreement', 'cross_script_model_same_work') and same['author'][0] == 'agreement'
    assert candidate('no', 'differs')['title'][0] == 'material_conflict'
    assert candidate('unsure', 'not_shown')['title'] == ('unknown', 'cross_script_model_unsure')


def test_the_model_answer_is_validated_and_a_failed_call_decides_nothing():
    assert validate_answer({'same_work': 'maybe'}) is None
    assert validate_answer({'same_work': 'yes', 'author': 'agrees', 'page_authors': []})['author'] == 'not_shown'
    reference, page = {'title': 'Rivers', 'author': 'Lin, Q.', 'year': '2021'}, {'titles': ['北方平原的河流'], 'text': 'x' * 5000}
    seen = {}

    def answered(system, payload, **kwargs):
        seen['payload'] = payload
        kwargs['receipt'].update(returned_model='model-x', total_tokens=10)
        return {'same_work': 'yes', 'author': 'agrees', 'page_authors': ['林青']}

    judged = judge_cross_script_identity(reference, page, completion=answered)
    assert judged['same_work'] == 'yes' and judged['returned_model'] == 'model-x' and len(judged['payload_sha256']) == 64
    assert seen['payload'].count('x') <= 1500 + 10      # bounded page excerpt

    def failing(*_args, **_kwargs):
        raise RuntimeError('down')
    assert judge_cross_script_identity(reference, page, completion=failing)['same_work'] == 'unsure'


def test_an_unsure_cross_script_page_leaves_the_reference_not_assessed():
    from app.services.reference_verification import assess_reference_verification
    discovery = {'outcome': 'search_incomplete', 'expected': {'title': 'Rivers of the northern plain', 'source_kind': 'journal_article'},
                 'attempts': [{'route_category': 'student_url'}],
                 'candidates': [{'provider': 'student_url_html', 'comparisons': [
                     {'field_name': 'title', 'outcome': 'unknown', 'reason_code': 'cross_script_model_unsure'}]}]}
    reference = SimpleNamespace(reference_id='r1', source_kind='journal_article', title='Rivers of the northern plain', raw_ref='')
    verdict = assess_reference_verification(reference, discovery)
    assert verdict['status'] == 'not_assessed' and verdict['reason_code'] == 'cross_script_page_unresolved'


def test_reversed_editor_initials_are_read_as_a_name():
    assert chapter_editors('Ames, T. (2017). A part. In B, Poore (Ed.), The book (pp. 1–9). Press.') == 'Poore, B.'


def test_a_series_volume_note_is_not_title_wording():
    # Wu's repository page, 2026-10-06: "(Vol. 15)" after the cited title.
    expected = ExpectedBibliographicFields(title='The open harbour: Trade in a new age (Vol. 15)', authors=['Ames, T.'], year='2018')
    result = RetrievalResult(source_name='web_fetch', success=False, title='The Open Harbour: Trade in a New Age',
                             authors=['Ames, Tom'], year='2018')
    built = build_reference_discovery_candidate(attempt_id='a', provider='student_url_html', expected=expected, result=result)
    assert {c.field_name: c.reason_code for c in built.comparisons}['title'] == 'title_match_without_volume_note'
    assert built.is_credible


def test_a_title_cut_at_vs_is_parsed_whole_and_its_doi_record_is_a_possible_match():
    from app.services.reference_parser import extract_and_parse_references
    from app.services.reference_verification import _possible_match
    [ref] = extract_and_parse_references(
        'References\nLin, Q. (2000). Harbour Rights vs. Inland Claims: Two Coastal Strategies. Journal of Ports, 5(2), '
        '155–182. https://doi.org/10.9999/jp0502_1\n', format_hint='apa', use_regex_first=True,
        use_llm_fallback=False, paper_version_id='x')
    assert ref.title == 'Harbour Rights vs. Inland Claims: Two Coastal Strategies'
    record = {'observed': {'title': 'Harbour Rights vs. Inland Claims: Two Coastal Strategies'},
              'comparisons': [{'field_name': 'doi', 'outcome': 'agreement'}, {'field_name': 'year', 'outcome': 'agreement'},
                              {'field_name': 'title', 'outcome': 'material_conflict'},
                              {'field_name': 'author', 'outcome': 'material_conflict'}]}
    assert _possible_match(record, {'title': 'Harbour Rights vs'})
    assert not _possible_match(record, {'title': 'Different Words Entirely'})
