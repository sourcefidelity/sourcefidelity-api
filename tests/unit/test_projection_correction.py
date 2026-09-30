"""Immutable report correction and evidence-target reuse boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace
import uuid

import fitz
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models import Base
from app.models.job import Job
from app.models.report import Report, ReportPaperArtifactRecord
from app.services.evidence_report_refresh import refresh_evidence_report_projection, _validate_retained_claims
from app.services.paper_extraction import extract_paper_evidence


@pytest.mark.parametrize('changed',['scope','claim','binding','source','extraction','digest'])
def test_reassessment_replacement_rejects_changed_boundaries(changed):
    from app.services.evidence_report_refresh import _validate_verification_replacement
    from app.services.verification_report import _payload_digest
    payload={'claim':{'text':'Exact claim'},'source_binding':{'reference_id':'r'},
        'source_identity':{'representation_id':'rep','content_sha256':'hash','canonical_work_id':'work'},
        'authoritative_evidence_package':{'extracted_text_sha256':'text-hash','extraction_version':'v1'}}
    old=SimpleNamespace(scope_type='personal_owner',scope_id='owner',paper_version_id='p',
        report_payload=payload,evidence_sha256=_payload_digest(payload))
    new=deepcopy(old);job=SimpleNamespace(scope_type='personal_owner',scope_id='owner',paper_version_id='p')
    _validate_verification_replacement(old,new,job)
    if changed=='scope':new.scope_id='other'
    elif changed=='claim':new.report_payload['claim']['text']='Changed claim'
    elif changed=='binding':new.report_payload['source_binding']['reference_id']='other'
    elif changed=='source':new.report_payload['source_identity']['content_sha256']='other'
    elif changed=='extraction':new.report_payload['authoritative_evidence_package']['extracted_text_sha256']='other'
    else:new.evidence_sha256='tampered'
    if changed!='digest':new.evidence_sha256=_payload_digest(new.report_payload)
    with pytest.raises(ValueError):_validate_verification_replacement(old,new,job)


def test_correction_extends_latest_lineage_without_changing_shared_anchors():
    with fitz.open() as doc:
        page=doc.new_page()
        page.insert_text((72,90),'Gilda (1946) illustrates a film.')
        page.insert_text((72,200),'References')
        page.insert_text((72,230),'Vidor, C. (Director). (1946). Gilda [Film]. Columbia Pictures.')
        text=page.get_text()
        paper=doc.tobytes()
    extracted=extract_paper_evidence(text,paper_version_id='paper-v1',format_hint='apa',
        use_llm_boundaries=False,use_llm_atomizer=False,use_llm_reference_fallback=False)
    engine=create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    backend=SimpleNamespace(download=lambda key:paper)
    with Session(engine) as session:
        job=Job(filename='paper.pdf',paper_version_id='paper-v1',status='completed',stage='completed',
            scope_id='owner',scope_type='personal_owner',input_sha256=hashlib.sha256(paper).hexdigest(),
            input_media_type='application/pdf',input_byte_size=len(paper),
            input_expires_at=datetime.now(timezone.utc)+timedelta(days=1),
            extraction_payload=extracted.model_dump(mode='json'),source_results=[],verification_summary={})
        session.add(job); session.flush()
        first=Report(job_id=job.id,report_version=1,report_json={'report_ids':[],'citation_groups':[]})
        session.add(first);session.flush()
        second=Report(job_id=job.id,report_version=2,previous_report_id=first.id,report_json=deepcopy(first.report_json))
        session.add(second);session.flush()
        artifact=ReportPaperArtifactRecord(job_id=job.id,report_id=first.id,paper_version_id=job.paper_version_id,
            scope_id=job.scope_id,scope_type=job.scope_type,storage_key='paper',content_sha256=job.input_sha256,
            media_type='application/pdf',byte_size=len(paper),artifact_kind='submitted_pdf',
            presentation_status='page_faithful_ready',presentation_storage_key='paper',
            presentation_sha256=job.input_sha256,presentation_media_type='application/pdf',
            presentation_evidence={'page_dimensions':[{'page_index':0,'width':612,'height':792}]},
            sanitization_evidence={},expires_at=job.input_expires_at)
        session.add(artifact);session.commit()
        original=deepcopy(first.report_json),deepcopy(second.report_json),deepcopy(artifact.presentation_evidence)
        result=refresh_evidence_report_projection(session,backend,report_id=first.id,reextract_citations=True)
        assert result['report_version']==3 and result['previous_report_id']==str(second.id)
        successor=session.get(Report,uuid.UUID(result['report_id']))
        assert successor.report_json['evidence_report']['paper_surface']['matched_citation_anchor_count']==1
        assert successor.report_json['projection_correction']['llm_calls']==0
        assert first.report_json==original[0] and second.report_json==original[1]
        assert artifact.presentation_evidence==original[2] and artifact.report_id==first.id
        # A later ordinary refresh must retain corrected report-local anchors.
        later=refresh_evidence_report_projection(session,backend,report_id=first.id)
        assert later['report_version']==4
        view=session.get(Report,uuid.UUID(later['report_id'])).report_json['evidence_report']
        assert view['paper_surface']['matched_citation_anchor_count']==1
        assert artifact.presentation_evidence==original[2]
        from app.services.reference_layout import extract_reference_layout_from_bytes
        payload=deepcopy(job.extraction_payload)
        payload['reference_layout']=extract_reference_layout_from_bytes(
            paper,'paper.pdf',references=extracted.references,citation_format='apa').model_dump(mode='json')
        job.extraction_payload=payload;session.commit()
        styled=refresh_evidence_report_projection(session,backend,report_id=first.id,reassess_reference_formatting=True)
        successor=session.get(Report,uuid.UUID(styled['report_id']))
        assert successor.report_json['reference_formatting_correction']['version']=='reference-formatting-reassessment-v1'
        assert successor.report_json['evidence_report']['paper_surface']['matched_citation_anchor_count']==1
        assert first.report_json==original[0] and second.report_json==original[1]
        assert artifact.presentation_evidence==original[2]
        job.status='running';session.commit()
        with pytest.raises(ValueError,match='completed'):
            refresh_evidence_report_projection(session,backend,report_id=first.id)
        assert len(list(session.scalars(select(Report))))==5


@pytest.mark.parametrize('changed', [True,False])
def test_verified_claim_cannot_be_reused_after_target_changes(changed):
    def claim(text):
        return SimpleNamespace(claim_id='claim',model_dump=lambda **kwargs:{'text':text})
    old=SimpleNamespace(citation_claims=[claim('original')])
    new=SimpleNamespace(citation_claims=[claim('changed' if changed else 'original')])
    record=SimpleNamespace(report_payload={'authoritative_evidence_package':{'claim_id':'claim'}})
    if changed:
        with pytest.raises(ValueError,match='fresh evidence analysis'):
            _validate_retained_claims(old,new,[record])
    else:
        _validate_retained_claims(old,new,[record])
