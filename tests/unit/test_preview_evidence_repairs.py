import fitz
import pytest

from app.services.completeness_checker import check_completeness, COMPLETE, INCOMPLETE
from app.services.verification_evidence import passage_role_from_text
from app.services.evidence_report import _eligible_display_passages
from app.services.source_type import classify_reference_source_kind


def numbered_book(numbers):
    with fitz.open() as doc:
        for i, number in enumerate(numbers):
            page = doc.new_page()
            page.insert_text((72, 180), 'Index' if i == len(numbers)-1 else 'Substantive chapter prose.')
            if number is not None:
                page.insert_text((300, 810), str(number))
        return doc.tobytes()


def test_copied_index_does_not_establish_complete_book_with_interior_gap():
    result = check_completeness(numbered_book([1, 2, 3, None, 201, 202, 203]),
                                document_kind='book', external_lookup=False)
    assert result.verdict == INCOMPLETE


@pytest.mark.parametrize('numbers', [[1,2,3,4,5,6,7], [21,22,23,1,2,3,4], [1,2,3,None,5,6,7]])
def test_continuity_and_pagination_reset_do_not_trigger_gap(numbers):
    from app.services.completeness_checker import _signal_book_pagination_gap
    assert _signal_book_pagination_gap(numbered_book(numbers))['vote'] is None


def test_collected_component_numbers_do_not_apply_book_gap_rule():
    result = check_completeness(numbered_book([1,2,3,None,201,202,203]),
                                document_kind='chapter', external_lookup=False)
    assert result.verdict == COMPLETE


def test_compressed_publisher_is_book_without_changing_reference():
    raw = 'Lee, A. (2012). An extended biography. ExampleUniversityPress. https://example.org/sample.pdf'
    assert classify_reference_source_kind(raw, title='An extended biography').kind == 'monograph'


def test_copyright_catalog_leaf_excluded_even_with_permissions_prose():
    text = ('© 2005 Example Press. Library of Congress Cataloging-in-Publication Data. '
            'ISBN 0-520-24422-2. ' + 'An earlier contribution was reproduced by permission. ' * 25)
    assert passage_role_from_text(text) == 'publication_metadata'


def test_discussion_of_cataloging_and_isbn_remains_prose():
    assert passage_role_from_text('We examined Cataloging-in-Publication Data and ISBN identifiers. '
                                  'These practices affect the circulation of books.') == 'body_prose'


def test_three_partial_candidates_reach_existing_context_cap():
    passages = [dict(passage_id=str(i), text=f'The source addresses distinct facet {i}.') for i in range(3)]
    gate = dict(status='complete', assessments=[dict(passage_id=str(i), relevance='partially_relevant',
               evidence_role='source_own_claim_or_finding') for i in range(3)])
    assert _eligible_display_passages(passages, gate) == passages


def test_summary_shows_every_category_and_no_empty_wording():
    from app.services.evidence_report import _render_report_summary
    html = _render_report_summary(dict(evidence=['First issue','Second issue'], reference_formatting=[]))
    # Owner request 2026-09-28: an empty column shows only its heading.
    assert '<h2>Citation and Reference Formatting</h2>' in html and 'No issue was established' not in html
    assert 'First issue' in html and 'Second issue' in html
