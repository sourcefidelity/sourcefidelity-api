import fitz
import pytest
from pydantic import ValidationError
from app.services import identity_ocr_observation as module
from app.services.file_safety import FileSafetyReport, SafetyVerdict
from app.services.ocr_derivative import OcrDerivativeError


@pytest.mark.parametrize('size,allowed', [(20_317_468,True),(25_000_000,True),(25_000_001,False)])
def test_source_size_boundary_precedes_safety_and_rendering(monkeypatch,size,allowed):
    calls=[]
    class ReachedSafety(Exception):pass
    def safety(source):
        calls.append(len(source))
        raise ReachedSafety()
    monkeypatch.setattr(module,'inspect_uploaded_pdf',safety)
    with pytest.raises(ReachedSafety if allowed else OcrDerivativeError):
        module.build_identity_observation(b'x'*size)
    assert calls == ([size] if allowed else [])


@pytest.fixture
def setup(monkeypatch):
    document=fitz.open();document.new_page(width=300,height=400)
    source=document.tobytes();document.close()
    clean=SafetyVerdict.CLEAN
    monkeypatch.setattr(module,'inspect_uploaded_pdf',lambda _: FileSafetyReport(clean,clean,clean))
    monkeypatch.setattr(module,'_tesseract_version',lambda _: 'test-engine')
    monkeypatch.setattr(module,'_run_tesseract_page',lambda *a,**k: ('Title and author observation',90))
    return source


def test_observation_preserves_source_and_has_no_grants(setup):
    before=bytes(setup);observation=module.build_identity_observation(setup)
    module.validate_identity_observation(observation,setup)
    module.validate_identity_observation(module.IdentityOcrObservation.model_validate_json(observation.model_dump_json()),setup)
    assert setup==before
    assert not observation.grants_admission and not observation.grants_evidence_use
    assert observation.identity_status=='unverified'
    assert 'Title and author' not in repr(observation)


@pytest.mark.parametrize('field,value',[('text','changed'),('render_sha256','f'*64),('grants_admission',True),('page_index',1)])
def test_tampering_fails(setup,field,value):
    item=module.build_identity_observation(setup).model_copy(update={field:value})
    with pytest.raises((OcrDerivativeError,ValidationError)):
        module.validate_identity_observation(item,setup)


def test_other_source_fails(setup):
    item=module.build_identity_observation(setup)
    with pytest.raises(OcrDerivativeError):module.validate_identity_observation(item,setup+b'changed')


def test_safety_failure_precedes_render(setup,monkeypatch):
    rejected=SafetyVerdict.REJECTED
    monkeypatch.setattr(module,'inspect_uploaded_pdf',lambda _:FileSafetyReport(rejected,rejected,rejected))
    monkeypatch.setattr(module.fitz,'open',lambda **k:pytest.fail('unsafe source rendered'))
    with pytest.raises(OcrDerivativeError,match='clean'):module.build_identity_observation(setup)
