"""Review stopping bounds must not become nonexistence or source-use evidence."""
from copy import deepcopy
import pytest

from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate, assess_reference_discovery_trace
from app.services.reference_review_scope import scope, screen_metadata, ReviewScreen, input_hash, POLICY
from app.services.retrieval.base import RetrievalResult, AcquisitionLocation
from app.services.search.transient import finalize_transient_brave
from test_reference_credibility_cumulative import fixture


def review_fixture():
    ref, trace = fixture()
    trace['credibility_policy_version'] = 'reference-credibility-v4'
    trace['bounded_review_policy_version'] = POLICY
    return ref, trace


def kinds(result):
    return [f['finding_type'] for f in result['findings']]


def test_v3_distinct_short_title_and_fuzzy_surname_do_not_hold_every_lead_open():
    from app.services.reference_review_scope import scope_for
    title = 'The history of cinema and racial representation'
    assert scope_for('bounded-reference-review-v2')(title,'Cinema histories','Green, S.',['Joel Greenberg']) == 'material'
    assert scope(title,'Cinema histories','Green, S.',['Joel Greenberg']) == 'outside_bound'
    assert scope_for('bounded-reference-review-v4')(title,'Cinema histories','Green, S.',['Susan Green']) == 'material'
    assert scope(title,'Cinema histories','Green, S.',['Susan Green']) == 'outside_bound'
    assert scope(title,'Cinema histories','Green, S.',['James Green']) == 'outside_bound'
    assert scope(title,title,'Green, S.',['James Green']) == 'material'
    assert scope(title,'Access denied') == 'unknown'
    assert scope(title,'Cinema') == 'unknown'


def test_v2_scope_dispatch_preserves_short_title_uncertainty():
    from app.services.reference_review_scope import scope_for
    assert scope_for('bounded-reference-review-v2')('A detailed history of polar exploration','Ocean journeys') == 'unknown'
    assert scope('A detailed history of polar exploration','Ocean journeys') == 'outside_bound'


def test_author_bearing_clipped_candidate_does_not_gain_authorless_prefix_protection():
    from app.services.reference_review_scope import scope_for
    args=('Consumer protection in telecommunications: A foundation for competition',
          'Consumer Protection Section ... annual report','River, J.',
          ["State Attorney General's Consumer Protection Section"])
    assert scope_for('bounded-reference-review-v3')(*args) == 'material'
    assert scope(*args) == 'outside_bound'
    assert scope(args[0],'Consumer protection in telecommunications ...',args[2],[]) == 'material'


def test_catalog_responsibility_separator_preserves_main_title_correction():
    assert scope('Alex Example: A narrative and stylistic analysis (2nd ed.)',
                 'Alex Example /','Writer, B.',['Example, C.']) == 'material'
    assert scope('Alex Example: A narrative and stylistic analysis',
                 'Ocean journeys /','Writer, B.',['Other, C.']) == 'outside_bound'


def test_completed_bound_can_flag_without_a_nonexistence_assertion():
    ref, trace = review_fixture()
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == ['potentially_fabricated_reference']
    assert result['findings'][0]['evidence_basis'] == 'bounded_identity_nonverification'


def web_candidate(trace, title, acquisition='not_attempted'):
    candidate = build_reference_discovery_candidate(attempt_id='exa', provider='exa',
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=RetrievalResult(source_name='exa', success=True, title=title),
        acquisition_outcome=acquisition, location_url='https://example.org/work', discovery_provider='exa')
    trace['candidates'].append(candidate.model_dump(mode='json'))
    trace['attempts'][3]['outcome'] = 'candidate_found'
    trace['queries'][3].update(execution_outcome='results', result_count=1)


@pytest.mark.parametrize('acquisition', ['not_attempted','access_restricted','transport_failure','identity_unconfirmed'])
def test_unrelated_unopened_candidate_is_not_an_indefinite_veto(acquisition):
    ref, trace = review_fixture()
    web_candidate(trace, 'Volcanic sediment dynamics in alpine reservoirs', acquisition)
    assert kinds(assess_reference_credibility(ref,None,trace)) == ['potentially_fabricated_reference']
    assert assess_reference_discovery_trace(trace).record.outcome == 'search_incomplete'
    historical=deepcopy(trace); historical['credibility_policy_version']='reference-credibility-v3'
    assert not assess_reference_credibility(ref,None,historical)['findings']


# A blank observation left this list under bounded-reference-review-v7: an
# interstitial page still names something, but a record with no title, author or
# identifier at all reports nothing that could be the cited work. See
# test_lead_with_no_observed_fields_is_absence_of_information_not_a_match.
@pytest.mark.parametrize('title', ['Youth culture and cinematic violence',
    'Youth culture and cinema violence', 'Youth culture and…', 'Just a moment', 'Access denied by server'])
def test_material_and_unknown_candidates_continue_to_block(title):
    ref, trace = review_fixture(); web_candidate(trace,title)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def add_failed_metadata(trace):
    q=deepcopy(trace['queries'][1]);q.update(query_id='core',provider='core',execution_provider='core',execution_outcome='timeout',result_count=None)
    a=deepcopy(trace['attempts'][1]);a.update(attempt_id='core',provider='core',query_ids=['core'],outcome='operational_failure')
    trace['queries'].append(q);trace['attempts'].append(a)


def test_failed_redundant_metadata_route_does_not_erase_completed_quorum():
    ref,trace=review_fixture();add_failed_metadata(trace)
    result=assess_reference_credibility(ref,None,trace)
    assert kinds(result)==['potentially_fabricated_reference']
    assert result['findings'][0]['search_evidence']['uncounted_failure_query_ids']==['core']
    assert assess_reference_discovery_trace(trace).record.outcome=='search_incomplete'


@pytest.mark.parametrize('provider_index',[0,1])
def test_missing_material_route_still_blocks(provider_index):
    """A metadata adapter that did not complete still blocks the finding."""
    ref,trace=review_fixture();trace['queries'][provider_index]['execution_outcome']='timeout'
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('provider_index,surviving',[(2,'exa'),(3,'brave')])
def test_one_web_provider_failing_does_not_remove_the_finding(provider_index,surviving):
    """Owner decision 2026-09-21: a vendor outage must not silently delete findings.

    Measured that day: Exa failed 10 of 20 queries and six paper-11 findings
    disappeared, five of which Brave had already answered with a certified
    empty result. The surviving provider still has to complete, and the
    narrowed coverage is recorded on the finding.
    """
    ref,trace=review_fixture();trace['queries'][provider_index]['execution_outcome']='timeout'
    result=assess_reference_credibility(ref,None,trace)
    assert kinds(result)==['potentially_fabricated_reference']
    finding=result['findings'][0]
    assert finding['search_evidence']['web_providers']==[surviving]
    assert any('Only one title-led web provider' in limitation
               for limitation in finding['limitations'])
    assert surviving in finding['evidence_explanation']


def test_both_web_providers_failing_still_blocks_the_finding():
    """The category quorum is one provider, not zero."""
    ref,trace=review_fixture()
    for index in (2,3):
        trace['queries'][index]['execution_outcome']='timeout'
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_filtered_metadata_candidates_are_inspectable_scope_decisions():
    ref,trace=review_fixture()
    trace['queries'][1].update(result_count=1,reason_code='metadata_candidates_filtered',
        bounded_review_screen=screen_metadata(ref.title,ref.author,[RetrievalResult(
            source_name='openalex',success=True,title='Volcanic sediment dynamics',authors=['Stone'])]))
    assert kinds(assess_reference_credibility(ref,None,trace))==['potentially_fabricated_reference']
    trace['queries'][1]['bounded_review_screen']['observations'][0]['title']=ref.title
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_material_filtered_metadata_not_lost_when_adapter_returns_no_match():
    ref,trace=review_fixture()
    trace['queries'][1].update(result_count=1,reason_code='metadata_candidates_filtered',
        bounded_review_screen=screen_metadata(ref.title,ref.author,[RetrievalResult(
            source_name='openalex',success=True,title=ref.title,authors=['Stone'])]))
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_same_author_can_write_a_different_work():
    from app.services.retrieval.crossref import CrossrefRetriever
    ref,trace=review_fixture()
    result=CrossrefRetriever()._parse_message(dict(DOI='10.1234/other',title=['Volcanic sediment dynamics'],
        author=[dict(family='River',given='A.')],issued={'date-parts':[[2002]]}))
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),result=result)
    trace['candidates']=[c.model_dump(mode='json')];trace['attempts'][0]['outcome']='candidate_found'
    trace['queries'][0].update(execution_outcome='results',result_count=1,
        bounded_review_screen=screen_metadata(ref.title,ref.author,[result]))
    assert kinds(assess_reference_credibility(ref,None,trace))==['potentially_fabricated_reference']


@pytest.mark.parametrize('kind',['monograph','film','television_series','book_section'])
def test_media_exclusion_and_book_catalog_requirement_remain(kind):
    ref,trace=review_fixture();ref.source_kind=kind;trace['expected']['source_kind']=kind
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_brave_screen_retains_only_balanced_operational_counts():
    ref,trace=review_fixture()
    result=RetrievalResult(source_name='web_search',success=True,
        locations=[AcquisitionLocation(url='https://private-result.example/unvisited',provider='web_search',
            metadata={'bounded_review_scope':'outside_bound'})],
        metadata={'search_retention_policy':'brave-operational-transient-v1',
            'bounded_review_input_sha256':input_hash(ref.title,ref.author),
            'transient_unselected_count':1,'transient_unselected_review_counts':{'outside_bound':1}})
    finalize_transient_brave(result)
    assert not result.locations
    audit=result.metadata['transient_search_audits'][0]
    assert audit['not_attempted']==2 and audit['bounded_review_screen']['outside_bound']==2
    assert 'private-result' not in str(result.metadata)
    assert audit['bounded_review_screen']['observations'] is None
    trace['attempts'][2]['transient_search_audits']=[audit]
    trace['attempts'][2]['outcome']='candidates_processed'
    trace['queries'][2].update(execution_outcome='results',result_count=2)
    trace['search_retention_policy']='brave-operational-transient-v1'
    assert kinds(assess_reference_credibility(ref,None,trace))==['potentially_fabricated_reference']
    audit['bounded_review_screen'].update(outside_bound=1,material=1)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_cross_script_and_short_titles_not_discarded_as_unrelated():
    assert scope('世界政治の歴史','History of global political thought')=='unknown'
    assert scope('Star Wars','Volcanic sediment dynamics')=='unknown'


def test_topical_title_is_not_a_plausible_title_identity():
    from app.services.reference_review_scope import scope_v1
    title = 'Youth violence and A Clockwork Orange'
    other = 'Youth Violence, Free Will, and the Creative Cycle in A Clockwork Orange'
    assert scope_v1(title, other) == 'material'
    assert scope(title, other) == 'outside_bound'
    for candidate in [title, title + ': A study', title.replace('violence','violenc')]:
        assert scope(title,candidate) == 'material'
    from app.services.reference_review_scope import scope_for
    assert scope_for('bounded-reference-review-v4')(title,other,'Smith',['Smith']) == 'material'
    assert scope(title,other,'Smith',['Smith']) == 'outside_bound'


@pytest.mark.parametrize('author',['River, A.','Different, B.'])
def test_real_author_and_journal_do_not_rescue_a_different_work(author):
    ref,trace=review_fixture()
    # Returned title is clearly a different work, even in the same journal and
    # by the submitted author. No registration/admission is inferred from it.
    result=RetrievalResult(source_name='openalex',success=True,
        title='Volcanic sediment dynamics',authors=[author],year=ref.year,
        metadata={'container_title':'Journal of Cinema'})
    trace['queries'][1].update(execution_outcome='results',result_count=1,
        bounded_review_screen=screen_metadata(ref.title,ref.author,[result]))
    assert kinds(assess_reference_credibility(ref,None,trace))==['potentially_fabricated_reference']


@pytest.mark.parametrize('title',['Youth culture and cinematic violence',
    'Youth culture and cinematic violence: A study','Youth culture and cinematic violenc'])
def test_near_work_identity_still_blocks_with_wrong_author_or_year(title):
    ref,trace=review_fixture()
    trace['queries'][1].update(execution_outcome='results',result_count=1,
        bounded_review_screen=screen_metadata(ref.title,ref.author,[RetrievalResult(
            source_name='openalex',success=True,title=title,authors=['Different, B.'],year='1999')]))
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_clipped_title_scope_preserved():
    title = 'Youth violence and A Clockwork Orange'
    assert scope(title,'[PDF] Juvenile Justice Caught between the Exorcist and a Clockwork ...') == 'outside_bound'
    assert scope(title,'[PDF] Youth violence and A Clockwork ...') == 'material'
    assert scope(title,'A Clockwork Orange from Burgess to Kubrick','Jeffries, S.',['GREGORI F.']) == 'outside_bound'


def test_matching_isbn_protects_possible_title_error():
    ref,trace=review_fixture()
    trace['expected']['isbn']='9780195070521'
    result=RetrievalResult(source_name='google_books',success=True,title='Volcanic sediment dynamics',
        authors=[ref.author],metadata={'isbn':'9780195070521'})
    candidate=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),result=result)
    assert any(c.field_name=='isbn' and c.outcome=='agreement' for c in candidate.comparisons)
    trace['attempts'][0]['outcome']='candidate_found'
    trace['candidates']=[candidate.model_dump(mode='json')]
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('rank,identifier_match,expected', [(11,False,True),(5,False,True),(11,True,False),(11,None,True)])
def test_uninformative_lead_is_excluded_by_what_it_observed_not_by_its_rank(rank,identifier_match,expected):
    """v7 replaces the rank/reason-code bounds with the observation itself.

    Measured on paper 11: leads at ranks 4 and 5, with acquisition outcomes
    other than `bounded_location_limit`, reported no title, author or
    identifier and still vetoed a fabrication finding. Rank never distinguished
    those from an informative lead. A submitted identifier that resolves to the
    location is still real evidence and continues to block.
    """
    ref,trace=review_fixture();web_candidate(trace,'')
    trace['candidates'][0].update(location_rank=rank, disposition_reason_code='bounded_location_limit',
        submitted_identifier_location_match=identifier_match,
        title_absence_reason='no_title_element')
    assert bool(assess_reference_credibility(ref,None,trace)['findings']) is expected


def test_historical_scope_not_reinterpreted():
    ref,trace=review_fixture();web_candidate(trace,'Youth culture and cinematic violence in contemporary society')
    trace.pop('bounded_review_policy_version')
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('provider_index',[0,2])
def test_unknown_empty_count_is_not_a_negative_vote(provider_index):
    ref,trace=review_fixture();trace['queries'][provider_index]['result_count']=None
    if provider_index == 0:
        trace['queries'][0]['bounded_review_screen']=screen_metadata(ref.title,ref.author,[])
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_required_library_query_failure_cannot_hide_in_successful_attempt():
    ref,trace=review_fixture()
    q=deepcopy(trace['queries'][0]);q.update(query_id='library',provider='library',execution_provider='library',
        route_category='library_metadata',execution_outcome='timeout',result_count=None)
    a=deepcopy(trace['attempts'][0]);a.update(attempt_id='library',provider='library',
        route_category='library_metadata',query_ids=['library'],outcome='candidate_found')
    trace['queries'].append(q);trace['attempts'].append(a)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_filtered_matching_identifier_is_not_lost_by_title_scope():
    ref,trace=review_fixture()
    trace['queries'][1].update(result_count=1,reason_code='metadata_candidates_filtered',
        bounded_review_screen=screen_metadata(ref.title,ref.author,[RetrievalResult(
            source_name='openalex',success=True,title='Volcanic sediment dynamics',authors=['Stone'],doi=ref.doi)]))
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_web_results_without_any_dispositions_cannot_count_as_complete():
    ref,trace=review_fixture();trace['queries'][3].update(execution_outcome='results',result_count=2)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def blank_web_candidate(trace, acquisition='identity_unconfirmed', title='',
                        title_reason='no_title_element'):
    """A discovery hit that returned no usable bibliographic observation.

    `title_reason` is what the fetcher saw. Omitting it (None) reproduces an
    unexplained empty record, which must not clear the reference.
    """
    metadata = {'title_reason': title_reason} if title_reason else {}
    candidate = build_reference_discovery_candidate(attempt_id='exa', provider='exa',
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=RetrievalResult(source_name='exa', success=True, title=title, metadata=metadata),
        acquisition_outcome=acquisition, location_url='https://example.org/lead',
        discovery_provider='exa')
    assert candidate.plausible_identity_match is False
    trace['candidates'].append(candidate.model_dump(mode='json'))
    trace['attempts'][3]['outcome'] = 'candidate_found'
    trace['queries'][3].update(execution_outcome='results', result_count=1)


@pytest.mark.parametrize('acquisition', ['identity_unconfirmed', 'unavailable'])
def test_lead_with_no_observed_fields_is_absence_of_information_not_a_match(acquisition):
    """Paper-11 Baker, Cohen J/M and Sullivan were held open by empty records.

    Their observed title, authors and identifier were all missing, so every
    field comparison scored `unknown` and the review treated the lead as a
    possible match for the cited work. Nothing was observed to be possible.
    """
    ref, trace = review_fixture()
    blank_web_candidate(trace, acquisition)
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == ['potentially_fabricated_reference']


def test_rejected_candidate_identity_does_not_keep_a_reference_open():
    """Paper-11 Shin & Lee: discovery rejected the lead, the review kept it.

    `resolved_different` needs a material conflict on both title and author, so
    a rejected candidate carrying no author fell through to `unknown`.
    """
    ref, trace = review_fixture()
    blank_web_candidate(trace, 'identity_rejected', title='An unrelated study in another language',
                        title_reason=None)
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == ['potentially_fabricated_reference']


def test_lead_observing_the_cited_title_still_blocks_the_finding():
    """The Bork protection: a real work's catalog records must veto the flag."""
    ref, trace = review_fixture()
    web_candidate(trace, trace['expected']['title'], 'identity_unconfirmed')
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == []
    assert result['bounded_review_reason'] == 'material_candidate_unresolved'


def test_blocked_lead_is_not_treated_as_an_observed_absence():
    """Paper-11 Cohen M: the app never reached the page behind the paywall.

    Failing to read a lead is not the same as reading it and finding no work
    there. Only the second is a reason to tell a student their reference may be
    fabricated, so an access-restricted lead keeps the reference open.
    """
    ref, trace = review_fixture()
    blank_web_candidate(trace, 'access_restricted')
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == []
    assert result['bounded_review_reason'] == 'material_candidate_unresolved'


def test_unexplained_empty_record_leaves_the_reference_open():
    """An extraction failure looks exactly like a page naming no work.

    Until the fetcher says which happened, the app cannot treat the silence as
    evidence, so the reference stays open rather than becoming a finding.
    """
    ref, trace = review_fixture()
    blank_web_candidate(trace, 'identity_unconfirmed', title_reason=None)
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == []
    assert result['bounded_review_reason'] == 'material_candidate_unresolved'


def test_unreadable_page_is_not_an_observed_absence():
    ref, trace = review_fixture()
    blank_web_candidate(trace, 'identity_unconfirmed', title_reason='no_text_layer')
    assert kinds(assess_reference_credibility(ref, None, trace)) == []


def test_interstitial_page_is_not_an_observed_absence():
    """"Access Denied" says nothing about the work behind it."""
    ref, trace = review_fixture()
    blank_web_candidate(trace, 'identity_unconfirmed', title_reason='boilerplate_title_only')
    assert kinds(assess_reference_credibility(ref, None, trace)) == []


def positive_only_candidate(trace, title='', provider='openaire'):
    """A lead from a source allowed to confirm a work but never to impugn one.

    Bound to its own completed academic-adapter attempt, exactly as a real one
    would be; an unbound candidate fails trace completion for another reason.
    """
    attempt = deepcopy(trace['attempts'][1])
    attempt.update(attempt_id=provider, provider=provider, query_ids=[provider],
                   outcome='candidate_found', required=False)
    query = deepcopy(trace['queries'][1])
    query.update(query_id=provider, provider=provider, execution_provider=provider,
                 execution_outcome='results' if title else 'no_results',
                 result_count=1 if title else 0)
    trace['attempts'].append(attempt)
    trace['queries'].append(query)
    candidate = build_reference_discovery_candidate(attempt_id=provider, provider=provider,
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=RetrievalResult(source_name=provider, success=bool(title), title=title),
        acquisition_outcome='metadata_only', discovery_provider=provider)
    trace['candidates'].append(candidate.model_dump(mode='json'))
    return candidate


def test_positive_only_near_match_does_not_block_a_finding():
    """OpenAIRE answers a fabricated title with a different real work.

    Measured: "Competition policy and the limits of antitrust" returns "Fair
    Competition: The Law and Economics of Antitrust". Left unresolved, that
    lead would veto the finding the reference deserves.
    """
    ref, trace = review_fixture()
    positive_only_candidate(trace, 'Fair Competition: The Law and Economics of Antitrust')
    assert kinds(assess_reference_credibility(ref, None, trace)) == ['potentially_fabricated_reference']


def test_positive_only_silence_does_not_block_a_finding():
    """ERIC holds no law; DOAJ no paywalled work. Silence is not absence."""
    ref, trace = review_fixture()
    positive_only_candidate(trace, '', provider='eric')
    assert kinds(assess_reference_credibility(ref, None, trace)) == ['potentially_fabricated_reference']


def test_positive_only_agreement_still_protects_the_reference():
    """The asymmetry's whole point: it may still confirm a work exists."""
    ref, trace = review_fixture()
    positive_only_candidate(trace, trace['expected']['title'])
    result = assess_reference_credibility(ref, None, trace)
    assert kinds(result) == []
    assert result['bounded_review_reason'] == 'material_candidate_unresolved'


def test_an_ordinary_provider_is_unaffected_by_the_allowance():
    """Only named positive-only sources lose their veto."""
    ref, trace = review_fixture()
    positive_only_candidate(trace, '', provider='crossref')
    assert kinds(assess_reference_credibility(ref, None, trace)) == []


def test_positive_only_sources_are_absent_from_the_coverage_quorum():
    """Their silence must never count as completed metadata coverage."""
    import inspect
    from app.services import bounded_reference_review as module

    source = inspect.getsource(module.assess)
    quorum = source[source.index("route_category != 'academic_adapter'"):]
    named = quorum[:quorum.index('\n', quorum.index('provider not in'))]
    for provider in module.POSITIVE_ONLY_PROVIDERS:
        assert provider not in named, provider
