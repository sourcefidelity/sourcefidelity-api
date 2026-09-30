from copy import deepcopy
import hashlib
from datetime import datetime, timezone
import pytest
from app.services.schemas import ParsedReference
from app.services.reference_discovery import _value_hash
from app.services.required_reference_locator import required_doi_omissions
from app.services.required_reference_locator import LEGACY_DOI_POLICY, DOI_POLICY


def inputs():
    ref=ParsedReference(reference_id='r',raw_ref='Smith, J. (2020). A title.',author='Smith, J.',year='2020',title='A title')
    candidate={'candidate_id':'c','attempt_id':'a','provider':'publisher',
        'identity_evidence_kind':'source_representation', 'location_provenance':'independently_acquired_content',
        'validated_identity_content_sha256':'a'*64,
        'observed':{'title':ref.title,'authors':[ref.author],'year':ref.year,'doi':'10.1234/example'},
        'comparisons':[{'field_name':k,'expected_sha256':_value_hash(v),'observed_sha256':_value_hash(v),
                        'outcome':'agreement','reason_code':'exact_match'}
                       for k,v in [('title',ref.title),('author',ref.author),('year',ref.year)]]}
    record={'reference_id':'r','created_at':datetime.now(timezone.utc).isoformat(),
        'expected':{'title':ref.title,'authors':[ref.author],'year':ref.year},
        'candidates':[candidate],'outcome':'confirmed','contributes_to_neutral_pattern':False}
    return {'citation_format':'apa','references':{'r':ref},'discoveries':{'r':record},
        'inventory':{'entries':[{'reference_id':'r','status':'not_observed','rectangles':[],
                                'reference_text_sha256':hashlib.sha256(ref.raw_ref.encode()).hexdigest()}]}}


@pytest.mark.parametrize('control', ['changed_raw', 'missing_binding', 'wrong_binding'])
def test_omission_requires_current_original_reference_inventory(control):
    data = metadata_inputs()
    if control == 'changed_raw':
        data['references']['r'].raw_ref += ' https://doi.org/10.1234/example'
    elif control == 'missing_binding':
        data['inventory']['entries'][0].pop('reference_text_sha256')
    else:
        data['inventory']['entries'][0]['reference_text_sha256'] = '0' * 64
    assert required_doi_omissions(**data) == []


def metadata_inputs():
    data = inputs()
    record = data['discoveries']['r']
    candidate = record['candidates'][0]
    candidate.update(provider='crossref', identity_evidence_kind=None,
        validated_identity_content_sha256=None, location_provenance='discovery_result',
        acquisition_outcome='access_restricted')
    candidate['comparisons'].append({'field_name':'doi','outcome':'unknown',
        'reason_code':'doi_missing_on_one_side','expected_sha256':None,
        'observed_sha256':_value_hash(candidate['observed']['doi'])})
    record['attempts'] = [{'attempt_id':'a','provider':'crossref','route_category':'academic_adapter',
        'required':True,'permitted':True,'outcome':'candidate_found',
        'started_at':datetime.now(timezone.utc).isoformat(),
        'completed_at':datetime.now(timezone.utc).isoformat()}]
    return data


def test_crossref_metadata_flags_omission_without_acquisition_or_input_mutation():
    data = metadata_inputs(); before = deepcopy(data)
    result = required_doi_omissions(**data)
    assert len(result) == 1
    assert result[0]['rule_id'] == DOI_POLICY
    assert result[0]['verification_basis'] == 'crossref_registration_metadata'
    assert result[0]['identity_content_sha256'] is None
    assert result[0]['bibliographic_record_sha256']
    assert data == before
    assert required_doi_omissions(**data, policy_version=LEGACY_DOI_POLICY) == []


def test_unknown_acquisition_kind_does_not_block_doi_but_conflicting_kind_does():
    data=metadata_inputs(); c=data['discoveries']['r']['candidates'][0]
    comparison={'field_name':'source_kind','expected_sha256':_value_hash('journal_article'),
        'observed_sha256':_value_hash('unknown'),'outcome':'unknown','reason_code':'kind_unavailable'}
    c['comparisons'].append(comparison)
    assert len(required_doi_omissions(**data)) == 1
    comparison['outcome']='material_conflict'
    assert required_doi_omissions(**data) == []


@pytest.mark.parametrize('control', ['search','aggregator','missing_attempt','wrong_attempt',
    'wrong_route','unpermitted','wrong_observed_title','conflict','minor_difference','two_dois','tampered_doi'])
def test_metadata_boundary_rejects_unreliable_or_ambiguous_records(control):
    data = metadata_inputs(); record = data['discoveries']['r']; c = record['candidates'][0]
    if control == 'search': c['provider'] = 'web_search'
    elif control == 'aggregator': c['provider'] = 'openalex'
    elif control == 'missing_attempt': record['attempts'] = []
    elif control == 'wrong_attempt': c['attempt_id'] = 'other'
    elif control == 'wrong_route': record['attempts'][0]['route_category'] = 'bounded_web'
    elif control == 'unpermitted': record['attempts'][0]['permitted'] = False
    elif control == 'wrong_observed_title': c['observed']['title'] = 'Another work'
    elif control == 'tampered_doi': c['observed']['doi'] = '10.1234/tampered'
    elif control in {'conflict','minor_difference'}:
        c['comparisons'][0]['outcome'] = 'material_conflict' if control == 'conflict' else control
    elif control == 'two_dois':
        other=deepcopy(c);other['candidate_id']='c2';other['observed']['doi']='10.1234/other'
        other['comparisons'][-1]['observed_sha256']=_value_hash('10.1234/other')
        record['candidates'].append(other)
    assert required_doi_omissions(**data) == []


def test_historical_extraction_policy_does_not_silently_upgrade():
    from app.services.paper_extraction import PaperExtractionArtifact, extract_paper_evidence
    old = PaperExtractionArtifact.model_validate({'paper_version_id':'old','citation_format':'apa'})
    assert old.required_doi_policy_version == LEGACY_DOI_POLICY
    fresh = extract_paper_evidence('A paper.\n\nReferences\nSmith, J. (2020). A title.',
        paper_version_id='fresh',format_hint='apa',use_llm_boundaries=False,
        use_llm_atomizer=False,use_llm_reference_fallback=False)
    assert fresh.required_doi_policy_version == DOI_POLICY
    assert PaperExtractionArtifact.model_validate_json(fresh.model_dump_json()) == fresh


def test_verified_apa_doi_is_required_without_changing_inventory():
    data=inputs(); before=deepcopy(data['inventory'])
    result=required_doi_omissions(**data)
    assert len(result)==1 and result[0]['verified_doi']=='10.1234/example'
    assert data['inventory']==before


@pytest.mark.parametrize('control',['mla','unknown','supplied','edition','unconfirmed','unbound','conflict','other_reference','wrong_title','no_doi','multiple_dois'])
def test_uncertain_or_inapplicable_does_not_flag(control):
    data=inputs();record=data['discoveries']['r'];candidate=record['candidates'][0]
    if control=='mla':data['citation_format']='mla'
    elif control in {'unknown','supplied'}:data['inventory']['entries'][0]['status']=control
    elif control=='edition':record['expected']['edition_sensitive']=True
    elif control=='unconfirmed':record['outcome']='possible_match'
    elif control=='unbound':candidate.pop('validated_identity_content_sha256')
    elif control=='conflict':candidate['comparisons'][0]['outcome']='material_conflict'
    elif control=='other_reference':record['reference_id']='other'
    elif control=='wrong_title':data['references']['r'].title='Changed title'
    elif control=='no_doi':candidate['observed']['doi']=''
    elif control=='multiple_dois':
        other=deepcopy(candidate);other['candidate_id']='c2';other['observed']['doi']='10.1234/other';record['candidates'].append(other)
    assert required_doi_omissions(**data)==[]


def test_chip_uses_existing_panel_without_highlighting_missing_text():
    from app.services.evidence_report import render_evidence_report_html
    data=inputs(); finding=required_doi_omissions(**data)[0]
    finding['source']={'author':'Smith, J.','year':'2020','title':'A title','raw_reference':'Smith, J. (2020). A title.'}
    finding['rectangles']=[{'page_index':0,'x0':72,'y0':100,'x1':400,'y1':120}]
    view={'title':'DOI chip control','citation_format':'APA','citations':[],
        'reference_practice':[finding], 'paper_surface':{'page_dimensions':[{'page_index':0,'width':612,'height':792}], 'page_href_template':'/page-{page_index}.png'}}
    html=render_evidence_report_html(view,csp_nonce='doi-chip-test-nonce')
    assert '<title>Required DOI missing</title>' in html
    # A reference-list finding opens its reference's single Reference N window.
    assert 'data-panel-template="reference-entry-panel-1"' in html
    assert 'id="reference-panel-1"' not in html
    assert 'https://doi.org/10.1234/example' in html
    assert '<rect class="reference-hit"' not in html


def cache_inputs():
    """A source reused from the repository: its accepted record's DOI (Chalaby, 2026-09-30)."""
    data = metadata_inputs()
    record = data['discoveries']['r']; c = record['candidates'][0]
    c.update(provider='local_cache', acquisition_outcome='metadata_only')
    c['comparisons'].append({'field_name':'container_title','outcome':'unknown',
        'reason_code':'component_field_missing_or_unresolved','expected_sha256':_value_hash('A Journal'),
        'observed_sha256':None})
    record['attempts'][0].update(provider='local_cache', route_category='durable_repository', required=False)
    return data


def test_a_reused_accepted_source_record_supplies_the_doi():
    result = required_doi_omissions(**cache_inputs())
    assert len(result) == 1 and result[0]['verification_basis'] == 'accepted_repository_record'
    assert result[0]['finding'].endswith('Add https://doi.org/10.1234/example.')


@pytest.mark.parametrize('control', ['legacy', 'missing_attempt', 'failed_attempt', 'wrong_observed_title',
                                     'conflict', 'unconfirmed'])
def test_a_reused_record_still_needs_its_own_completed_lookup_and_agreeing_identity(control):
    data = cache_inputs(); record = data['discoveries']['r']; c = record['candidates'][0]
    if control == 'missing_attempt': record['attempts'] = []
    elif control == 'failed_attempt': record['attempts'][0]['outcome'] = 'operational_failure'
    elif control == 'wrong_observed_title': c['observed']['title'] = 'Another work'
    elif control == 'conflict': c['comparisons'][-1]['outcome'] = 'material_conflict'
    elif control == 'unconfirmed': record['outcome'] = 'possible_match'
    kwargs = {'policy_version': LEGACY_DOI_POLICY} if control == 'legacy' else {}
    assert required_doi_omissions(**data, **kwargs) == []
