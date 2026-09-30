from app.services.reference_formatting import reference_order_projection
from app.services.report_references import (
    reference_catalog, reference_list_finding, reference_numbers, reference_ordinal,
)


def rid(n):
    return f"ref-{n:04d}-{n:012x}"


def order_findings(observed, expected, flagged=None):
    """Findings as reference_style_findings emits them: one per shifted entry."""
    return [dict(finding_type='reference_order', reference_id=r,
                 expected_reference_order=expected, observed_reference_order=observed, rectangles=[])
            for position, r in enumerate(observed)
            if (flagged is None and expected.index(r) != position) or (flagged and r in flagged)]


def test_reference_n_follows_bibliography_ordinal():
    view = {'citations': [{'members': [{'reference_id': rid(12)}, {'reference_id': rid(3)}]}],
            'reference_practice': [{'reference_id': rid(7)}]}
    assert reference_ordinal(rid(12)) == 12 and reference_ordinal('r') is None
    assert reference_numbers(view) == {rid(12): 12, rid(3): 3, rid(7): 7}


def test_synthetic_ids_fall_back_to_bibliography_then_paper_position():
    view = {'bibliography': [{'reference_id': 'b'}, {'reference_id': 'a'}],
            'citations': [{'members': [{'reference_id': 'a'}]}]}
    assert reference_numbers(view) == {'b': 1, 'a': 2}
    located = {'paper_surface': {'reference_locations': {
        'x': {'rectangles': [{'page_index': 1, 'y0': 10, 'x0': 0}]},
        'y': {'rectangles': [{'page_index': 0, 'y0': 90, 'x0': 0}]}}},
        'reference_practice': [{'reference_id': 'x'}, {'reference_id': 'y'}, {'reference_id': 'z'}]}
    assert reference_numbers(located) == {'y': 1, 'x': 2, 'z': 3}
    # One non-conforming id makes ordinals unusable for the whole list.
    mixed = {'reference_practice': [{'reference_id': rid(4)}, {'reference_id': 'odd'}]}
    assert reference_numbers(mixed) == {rid(4): 1, 'odd': 2}


def test_projection_numbers_are_reused_by_the_catalog():
    view = {'reference_numbers': {'q': 5}, 'reference_practice': [{'reference_id': 'q', 'source': {'raw_reference': 'Q.'}}]}
    assert reference_numbers(view) == {'q': 5}
    assert reference_catalog(view)[0]['template_id'] == 'reference-entry-panel-5'


def test_catalog_prefers_member_source_with_links_and_lists_citations():
    linked = {'raw_reference': 'A. Linked.', 'submitted_hyperlinks': ['https://example.org/a']}
    view = {
        'citations': [
            {'members': [{'reference_id': rid(1), 'source': {'raw_reference': 'A. Plain.'}, 'coverage_level': 'abstract_only'}]},
            {'members': [{'reference_id': rid(1), 'source': linked, 'coverage_level': 'full_text'}]},
        ],
        'reference_practice': [
            {'reference_id': rid(1), 'finding_type': 'reference_title_style', 'rectangles': [{'page_index': 0}]},
            {'reference_id': rid(1), 'finding_type': 'reference_order', 'rectangles': []},
            {'reference_id': rid(1), 'finding_type': 'body_title_style', 'rectangles': [{'page_index': 0}]},
        ],
        'bibliography': [{'reference_id': rid(2), 'source': {'raw_reference': 'B. Uncited.'}}],
        'paper_surface': {'reference_locations': {rid(1): {'rectangles': [{'page_index': 0, 'x0': 1, 'y0': 1, 'x1': 2, 'y1': 2}]}}},
    }
    first, second = reference_catalog(view)
    assert first['source'] is linked and first['citation_numbers'] == [1, 2]
    assert first['member']['coverage_level'] == 'full_text'
    # Only located reference-list findings join the window; body text keeps its own.
    assert first['finding_indexes'] == [1] and first['location']
    assert second['number'] == 2 and second['citation_numbers'] == [] and second['location'] is None
    assert not reference_list_finding({'reference_id': 'x', 'finding_type': 'required_quotation_locator_missing'})


def test_one_moved_entry_flags_only_that_entry():
    expected = [rid(n) for n in range(1, 11)]
    observed = [expected[-1]] + expected[:-1]          # last entry moved to the front
    findings = order_findings(observed, expected)
    assert len(findings) == 10                          # every position differs
    kept, pervasive = reference_order_projection(findings, reference_ordinal)
    assert not pervasive and [f['reference_id'] for f in kept] == [expected[-1]]


def test_shuffled_list_is_reported_once_without_flags():
    expected = [rid(n) for n in range(1, 9)]
    observed = list(reversed(expected))
    kept, pervasive = reference_order_projection(order_findings(observed, expected), reference_ordinal)
    assert pervasive and kept == []


def test_swap_of_two_and_two_entry_list():
    expected = [rid(n) for n in range(1, 7)]
    observed = expected[:2] + [expected[3], expected[2]] + expected[4:]
    kept, pervasive = reference_order_projection(order_findings(observed, expected), reference_ordinal)
    assert not pervasive and len(kept) == 1
    pair = [rid(1), rid(2)]
    kept, pervasive = reference_order_projection(order_findings(pair[::-1], pair), reference_ordinal)
    assert not pervasive and len(kept) == 1


def test_historical_findings_recover_observed_order_from_ordinals():
    expected = [rid(n) for n in (2, 3, 4, 1)]           # entry 1 belongs last
    observed = [rid(n) for n in (1, 2, 3, 4)]
    findings = [dict(f, observed_reference_order=None) for f in order_findings(observed, expected)]
    kept, pervasive = reference_order_projection(findings, reference_ordinal)
    assert not pervasive and [f['reference_id'] for f in kept] == [rid(1)]


def test_unknown_observed_order_leaves_findings_unchanged():
    findings = [dict(finding_type='reference_order', reference_id=r, expected_reference_order=['b', 'a'],
                     rectangles=[]) for r in ('a', 'b')]
    kept, pervasive = reference_order_projection(findings, reference_ordinal)
    assert kept == findings and not pervasive
