"""Bounded APA DOI omissions from verified bibliographic records.

No search, acquisition, or assumption about the consulted medium is performed.
URL-only/MLA requirements remain unassessed until their premises are established.
"""
import re
import hashlib
from app.services.reference_discovery import ReferenceDiscoveryRecord, _value_hash

RULE_SOURCE = 'https://apastyle.apa.org/style-grammar-guidelines/references/dois-urls'
LEGACY_DOI_POLICY = 'apa7_verified_doi_required_v1'
DOI_POLICY = 'apa7_bibliographic_doi_required_v2'


def required_doi_omissions(*, citation_format, inventory, references, discoveries, policy_version=DOI_POLICY):
    if citation_format.lower() != 'apa' or not inventory or policy_version not in {LEGACY_DOI_POLICY, DOI_POLICY}:
        return []
    findings = []
    for entry in inventory.get('entries', []):
        if entry['status'] != 'not_observed':
            continue
        reference = references.get(entry['reference_id'])
        if reference is None or reference.needs_review:
            continue
        # A stable ID does not prove the original entry is unchanged. Never
        # reuse an absence observation after a link was added or text changed.
        if entry.get('reference_text_sha256') != hashlib.sha256(reference.raw_ref.encode()).hexdigest():
            continue
        try:
            record = ReferenceDiscoveryRecord.model_validate(discoveries.get(entry['reference_id']))
        except ValueError:
            continue
        if record.reference_id != reference.reference_id or record.outcome != 'confirmed' or record.expected.edition_sensitive:
            continue
        # Bind comparisons to this submitted reference, not merely a provider
        # result with a matching identifier or a reference ID reused elsewhere.
        expected = {'title':reference.title, 'year':reference.year, 'author':reference.author}
        dois = {}
        for candidate in record.candidates:
            content_bound = (
                candidate.identity_evidence_kind in {'source_representation','landing_page_metadata'}
                and candidate.validated_identity_content_sha256
                and candidate.location_provenance == 'independently_acquired_content')
            # Registration metadata verifies bibliographic facts, not source
            # content. Use the existing Crossref adapter's completed, bound
            # observation; a search result naming Crossref is not sufficient.
            registration_bound = policy_version == DOI_POLICY and candidate.provider == 'crossref' and any(
                a.attempt_id == candidate.attempt_id and a.provider == 'crossref'
                and a.route_category == 'academic_adapter' and a.permitted
                and a.outcome == 'candidate_found' and a.completed_at is not None
                for a in record.attempts)
            # A source reused from the repository carries the DOI of its
            # accepted canonical record (Chalaby, Franchise 1, 2026-09-30: a
            # re-check reused the stored copy, so no Crossref observation ran
            # and the omission went unreported). Only accepted representations
            # are reused; title, author and year must still agree below.
            cache_bound = policy_version == DOI_POLICY and candidate.provider == 'local_cache' and any(
                a.attempt_id == candidate.attempt_id and a.provider == 'local_cache'
                and a.outcome == 'candidate_found' and a.completed_at is not None
                for a in record.attempts)
            unresolved = candidate.has_unresolved_supplied_identity_fields
            if cache_bound:
                # The stored record keeps the work's identity fields only; an
                # unobserved journal, volume or page range is not a conflict.
                unresolved = any(c.outcome == 'material_conflict' for c in candidate.comparisons)
            if registration_bound:
                # A missing acquired-content kind is not a DOI identity
                # failure. Explicit type conflicts still fail below; missing
                # supplied identifiers and bibliographic fields remain closed.
                unresolved = any(c.expected_sha256 and c.outcome not in {'agreement','minor_difference'}
                    and not (c.field_name == 'source_kind' and c.outcome == 'unknown')
                    for c in candidate.comparisons)
            if (not (content_bound or registration_bound or cache_bound)
                    or candidate.has_material_conflict or candidate.has_minor_difference
                    or unresolved
                    or candidate.acquisition_outcome == 'identity_rejected'
                    or candidate.observed.edition_sensitive
                    or candidate.edition_metadata):
                continue
            comparisons = {c.field_name:c for c in candidate.comparisons}
            if registration_bound and ('doi' not in comparisons
                    or comparisons['doi'].observed_sha256 != _value_hash(candidate.observed.doi)):
                continue
            observed = {'title':candidate.observed.title, 'year':candidate.observed.year,
                        'author':' | '.join(candidate.observed.authors)}
            if any(not value or name not in comparisons
                   or comparisons[name].outcome != 'agreement'
                   or comparisons[name].expected_sha256 != _value_hash(value)
                   or ((registration_bound or cache_bound)
                       and comparisons[name].observed_sha256 != _value_hash(observed[name]))
                   for name,value in expected.items()):
                continue
            if not any(c.field_name=='author' and c.outcome=='agreement' for c in candidate.comparisons):
                continue
            doi = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', candidate.observed.doi.strip(), flags=re.I)
            if not re.fullmatch(r'10\.\d{4,9}/[^\s<>"\x00-\x1f]+',doi):
                continue
            dois[doi.casefold()] = (doi, candidate, 'crossref_registration_metadata' if registration_bound
                                    else 'accepted_repository_record' if cache_bound
                                    else 'independently_acquired_identity')
        if len(dois) != 1:
            continue
        doi,candidate,basis = next(iter(dois.values()))
        findings.append({'finding_type':'required_doi_missing','reference_id':reference.reference_id,
            'conflicting_fields':['required DOI'],
            'rule_id':policy_version,'rule_source':RULE_SOURCE,
            'finding':f'APA requires a DOI when the cited work has one. This reference contains neither a DOI nor a URL. Add https://doi.org/{doi}.',
            'verified_doi':doi,'candidate_id':candidate.candidate_id,'provider':candidate.provider,
            'identity_content_sha256':candidate.validated_identity_content_sha256,
            'bibliographic_record_sha256':hashlib.sha256(candidate.observed.model_dump_json().encode()).hexdigest(),
            'verification_basis':basis,
            'rectangles':entry.get('rectangles',[]), 'localization_status':'bound_entry' if entry.get('rectangles') else 'not_assessed'})
    return findings
