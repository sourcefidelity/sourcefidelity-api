from datetime import datetime, timezone
import pytest
from app.services.ref_field_extractor import extract_incomplete_apa_chapter
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate, derive_reference_discovery_record, ReferenceRouteAttempt
from app.services.retrieval.base import RetrievalResult

RAW = 'National sovereignty, 2018–Present. In Studies of public media (pp.179-203). Example Press.'


def test_visible_chapter_fields_do_not_invent_author_or_publication_year():
    r = extract_incomplete_apa_chapter(RAW)
    assert r.title == 'National sovereignty, 2018–Present'
    assert r.container_title == 'Studies of public media' and r.pages == '179-203'
    assert r.author == '' and r.year == 'n.d.' and r.needs_review
    assert r.raw_ref == RAW and r.extraction_method == 'partial_regex'


@pytest.mark.parametrize('raw', [
    'Smith, J. (2024). National sovereignty. In Studies of public media (pp.179-203). Example Press.',
    'Smith, J. National sovereignty. In Studies of public media (pp.179-203). Example Press.',
    'A fragment with no container or pages',
])
def test_ambiguous_or_complete_entries_are_not_recovered_as_authorless(raw):
    assert extract_incomplete_apa_chapter(raw) is None


def candidate(expected, **metadata):
    return build_reference_discovery_candidate(attempt_id='test',provider='fixture',expected=expected,
        result=RetrievalResult(source_name='fixture',success=True,title='A longer title about national sovereignty, 2018–Present',
            authors=['Jane Smith'],year='2024',doi='10.1234/example',metadata=metadata))


def test_review_title_is_not_conflict_and_component_agreements_do_not_confirm():
    e = ExpectedBibliographicFields(title='National sovereignty, 2018–Present', reference_parse_review=True,
        container_title='Studies of public media',pages='179-203',year='n.d.')
    c = candidate(e,container_title=e.container_title,pages=e.pages)
    comparison={x.field_name:x for x in c.comparisons}
    assert comparison['title'].outcome == 'unknown'
    assert comparison['author'].outcome == comparison['year'].outcome == 'unknown'
    assert comparison['container_title'].outcome == comparison['pages'].outcome == 'agreement'
    assert c.agreement_count == 0 and not c.has_material_conflict
    assert c.observed.container_title == e.container_title and c.observed.pages == e.pages


def test_crossref_container_preserved_and_multiple_containers_abstain():
    e = ExpectedBibliographicFields(title='Title')
    c = candidate(e,message={'container-title':['Book']},page='1-20')
    assert c.observed.container_title == 'Book' and c.observed.pages == '1-20'
    assert not candidate(e,message={'container-title':['Book','Other']}).observed.container_title


def test_review_reference_doi_cannot_override_incomplete_parse():
    e = ExpectedBibliographicFields(title='A longer title about national sovereignty, 2018–Present',
        authors=['Jane Smith'],year='2024',doi='10.1234/example',reference_parse_review=True)
    c = candidate(e)
    now=datetime.now(timezone.utc)
    a=ReferenceRouteAttempt(attempt_id='test',provider='fixture',route_category='academic_adapter',required=True,
        permitted=True,outcome='candidate_found',started_at=now,completed_at=now)
    r=derive_reference_discovery_record(reference_id='ref',expected=e,required_route_categories=['academic_adapter'],queries=[],attempts=[a],candidates=[c])
    assert r.outcome == 'possible_match' and not r.contributes_to_neutral_pattern


def test_partial_recovery_is_not_cached_and_legacy_fields_default_empty(monkeypatch):
    from app.config import settings
    from app.services import reference_parser
    from app.services.schemas import ParsedReference
    monkeypatch.setattr(settings, 'CACHE_ENABLED', True)
    writes=[]
    monkeypatch.setattr(reference_parser.doi_cache, 'cache_reference', lambda **kw:writes.append(kw))
    results=reference_parser._extract_fields_regex_first([RAW], 'apa', use_llm_fallback=False)
    assert results[0].extraction_method == 'partial_regex' and not writes
    old=ParsedReference.model_validate({'raw_ref':RAW,'needs_review':True,'extraction_method':'fallback'})
    assert old.title == old.container_title == old.pages == ''
    assert not ExpectedBibliographicFields.model_validate({'title':'Existing'}).reference_parse_review


@pytest.mark.parametrize('observed_changes', [
    {},
    {'container_title': 'Studies of publicmedia'},
    {'pages': '204-225'},
    {'container_title': 'A different collection'},
    {'title': 'A different chapter on national sovereignty'},
    {'year': '2025'},
])
def test_unresolved_partial_candidates_never_become_completed_negatives(observed_changes):
    """Offline controls are not acquired-source identity acceptance."""
    expected = ExpectedBibliographicFields(
        title='National sovereignty, 2018–Present', reference_parse_review=True,
        container_title='Studies of public media', pages='179-203',
        source_kind='book_section', year='n.d.',
    )
    before = expected.model_dump()
    fields = dict(title='A longer title about national sovereignty, 2018–Present',
                  year='2024', container_title=expected.container_title, pages=expected.pages)
    fields.update(observed_changes)
    result = RetrievalResult(
        source_name='fixture', success=True, title=fields.pop('title'),
        year=fields.pop('year'), authors=['Jane Smith'],
        metadata={**fields, 'observed_source_kind': 'book_section'},
    )
    c = build_reference_discovery_candidate(
        attempt_id='test', provider='fixture', expected=expected, result=result)
    now = datetime.now(timezone.utc)
    attempt = ReferenceRouteAttempt(
        attempt_id='test', provider='fixture', route_category='academic_adapter',
        required=True, permitted=True, outcome='candidate_found',
        started_at=now, completed_at=now)
    record = derive_reference_discovery_record(
        reference_id='ref', expected=expected, required_route_categories=['academic_adapter'],
        queries=[], attempts=[attempt], candidates=[c])
    assert record.outcome == 'search_incomplete'
    assert not record.contributes_to_neutral_pattern
    assert not c.validated_identity_content_sha256
    assert expected.model_dump() == before
    assert type(record).model_validate_json(record.model_dump_json()) == record
