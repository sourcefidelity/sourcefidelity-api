"""Shared identity policy and source-bound OCR adapter regressions."""
from hashlib import sha256
from types import SimpleNamespace

import pytest

from app.services import identity_ocr_observation as ocr
from app.services import source_validator as validator
from app.services.ocr_derivative import OcrDerivativeError


TITLE = "Independent Research on Coastal Habitats"
TEXT = TITLE + "\nJane Smith\nRevised January 2020"
SOURCE = b"source-bound-private-test-bytes"


def observation(text=TEXT, top=30):
    words = []
    cursor = 0
    for line_index, line in enumerate(text.splitlines()):
        left = 10
        for token in line.split():
            start = text.index(token, cursor)
            words.append(ocr.IdentityOcrWord(start=start, end=start+len(token),
                bbox=(left, top+line_index*20, left+len(token)*4, top+line_index*20+10),
                line=(1, 1, line_index+1)))
            cursor = start+len(token)
            left += len(token)*4+5
    item = ocr.LayoutIdentityOcrObservation(source_sha256=sha256(SOURCE).hexdigest(),
        render_sha256="a"*64, text_sha256=sha256(text.encode()).hexdigest(),
        region=(0,0,400,500), engine_version="test", renderer_version="test",
        text=text, words=tuple(words), observation_sha256="0"*64)
    values = item.model_dump(); values.pop("observation_sha256")
    return item.model_copy(update={"observation_sha256":ocr._digest(values)})


@pytest.fixture
def authorized(monkeypatch):
    from app.services import verification_evidence
    calls = []
    def require(capability):
        calls.append(capability)
    principal = SimpleNamespace(require=require, scope_type="personal", scope_id="test")
    monkeypatch.setattr(verification_evidence, "authorize_representation",
                        lambda *a, **k: SimpleNamespace(content=SOURCE))
    return principal, calls


def compare(item, authorized, **kwargs):
    return ocr.compare_authorized_identity_observation(None,None,
        principal=authorized[0],representation_id="test",observation=item,
        expected_fields={"title":TITLE,"author":"Smith, J.","year":"2020"}, **kwargs)


def test_layout_recovers_identity_not_admission(authorized):
    item = observation()
    result = compare(item, authorized)
    assert result.identity_confidence == "high"
    assert not result.accept and result.completeness == "uncertain"
    assert authorized[1] and not item.grants_admission and not item.grants_evidence_use
    restored = ocr.LayoutIdentityOcrObservation.model_validate_json(item.model_dump_json())
    ocr.validate_identity_observation(restored, SOURCE)


def test_plain_occurrence_and_lower_page_mentions_are_not_prominent(authorized):
    assert compare(observation(top=300), authorized).identity_confidence == "medium"
    result = validator.validate_ocr_derivative_text(TEXT, expected_title=TITLE,
        expected_author="Smith, J.",expected_year="2020",completeness="complete")
    assert not result.accept and result.identity_confidence == "medium"


def test_shared_date_and_listing_guards(authorized):
    assert compare(observation(TEXT.replace("2020","2019")),authorized).identity_confidence == "medium"
    listing = observation("Curriculum Vitae\n"+TEXT)
    assert compare(listing,authorized).identity_confidence == "rejected"


@pytest.mark.parametrize("field,value",[("source_sha256","b"*64),("words",()),
                                        ("grants_admission",True)])
def test_adapter_rejects_altered_observation(authorized,field,value):
    with pytest.raises((OcrDerivativeError,ValueError)):
        compare(observation().model_copy(update={field:value}),authorized)


def test_current_authority_failure_precedes_observation_use(authorized,monkeypatch):
    from app.services import verification_evidence
    def denied(*a,**k):
        raise PermissionError("unavailable")
    monkeypatch.setattr(verification_evidence,"authorize_representation",denied)
    monkeypatch.setattr(ocr,"validate_identity_observation",lambda *a:pytest.fail("consumed"))
    with pytest.raises(PermissionError):compare(observation(),authorized)


@pytest.mark.parametrize("author",["Smith, J.","Jane Smith"])
def test_native_adapter_and_ocr_use_identical_signals(monkeypatch,authorized,author):
    monkeypatch.setattr(validator,"_extract_pdf_front_text",lambda _:TEXT)
    monkeypatch.setattr(validator,"_extract_pdf_identity_metadata",lambda _:"")
    monkeypatch.setattr(validator,"_has_prominent_front_title_support",lambda *a:True)
    monkeypatch.setattr(validator,"_explicit_first_page_document_years",lambda _: {"2020"})
    native = validator._check_identity(SOURCE,None,TITLE,author,"2020")
    assert native == compare(observation(),authorized).identity_confidence == "high"


def test_wrong_fields_cannot_become_high():
    for title,author in [("Unrelated Study of Lunar Craters","Smith, J."), (TITLE,"Jones, A.")]:
        result = validator._check_identity_observations(TEXT,"",None,title,author,"2020",
            prominent_title=title==TITLE,prominent_years={"2020"})
        assert result != "high"


def test_word_positions_are_bound_and_cover_all_text():
    header="level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    row="5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t90\tHello\n"
    words=ocr._bind_layout_words(header+row,"Hello",(0,0,400,500))
    assert words[0].start==0 and words[0].end==5
    assert words[0].bbox==pytest.approx((2.88,5.76,11.52,17.28))
    with pytest.raises(OcrDerivativeError):ocr._bind_layout_words(header+row,"Different",(0,0,400,500))
    with pytest.raises(OcrDerivativeError):ocr._bind_layout_words(header+row,"Hello remainder",(0,0,400,500))


def test_rehashed_invalid_layout_still_rejects():
    item=observation()
    changed=item.model_copy(update={"words":(item.words[0].model_copy(update={"bbox":(0,0,float('nan'),5)}),)+item.words[1:]})
    value=changed.model_dump();value.pop("observation_sha256")
    changed=changed.model_copy(update={"observation_sha256":ocr._digest(value)})
    with pytest.raises(OcrDerivativeError):ocr.validate_identity_observation(changed,SOURCE)


def test_v1_observations_remain_readable_and_nonadmitting(authorized):
    values=observation().model_dump();values.pop("words")
    values["version"]="identity-ocr-observation-v1"
    values.pop("observation_sha256")
    values["observation_sha256"]=ocr._digest(values)
    old=ocr.IdentityOcrObservation.model_validate(values)
    result=compare(old,authorized)
    assert result.identity_confidence=="medium" and not result.accept


def test_layout_builder_uses_engine_positions(monkeypatch):
    import fitz
    from app.services.file_safety import FileSafetyReport,SafetyVerdict
    with fitz.open() as document:
        document.new_page(width=300,height=400)
        source=document.tobytes()
    clean=SafetyVerdict.CLEAN
    monkeypatch.setattr(ocr,"inspect_uploaded_pdf",lambda _:FileSafetyReport(clean,clean,clean))
    monkeypatch.setattr(ocr,"_tesseract_version",lambda _:"test")
    def run(path,output,**kwargs):
        output.with_suffix('.tsv').write_text(
            'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
            '5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t90\tHello\n')
        return 'Hello',90
    monkeypatch.setattr(ocr,"_run_tesseract_page",run)
    item=ocr.build_identity_observation(source,include_layout=True)
    assert item.version=='identity-ocr-observation-v2' and len(item.words)==1
    ocr.validate_identity_observation(item,source)
    monkeypatch.setattr(ocr,"_run_tesseract_page",lambda *a,**k:('Hello',90))
    with pytest.raises(OcrDerivativeError,match='layout missing'):
        ocr.build_identity_observation(source,include_layout=True)


def front_page(text, index, top=30):
    values=observation(text,top).model_dump()
    values.update(version="identity-ocr-observation-v3",page_index=index)
    values.pop("observation_sha256")
    values["observation_sha256"]=ocr._digest(values)
    return ocr.FrontMatterIdentityOcrObservation.model_validate(values)


def compare_pages(pages,authorized):
    return ocr.compare_authorized_identity_observations(None,None,principal=authorized[0],
        representation_id="test",observations=pages,
        expected_fields={"title":TITLE,"author":"Smith, J.","year":"2020"})


def test_later_opening_title_and_author_use_shared_policy(authorized):
    pages=(front_page(TITLE,0,300),front_page("Jane Smith",1),front_page(TITLE,2))
    assert compare_pages(pages[:1],authorized).identity_confidence != "high"
    result=compare_pages(pages,authorized)
    assert result.identity_confidence=="high" and result.page_count==3
    assert not result.accept and result.completeness=="uncertain"


@pytest.mark.parametrize("indices",[(),(1,),(0,0),(0,2),(0,1,2,2)])
def test_opening_window_rejects_skips_duplicates_and_overflow(authorized,indices):
    with pytest.raises(OcrDerivativeError):
        compare_pages(tuple(front_page(TEXT,index) for index in indices),authorized)


def test_mixed_source_opening_pages_reject(authorized):
    wrong=front_page(TEXT,1).model_copy(update={"source_sha256":"b"*64})
    with pytest.raises(OcrDerivativeError):compare_pages((observation(),wrong),authorized)


def test_no_title_stitched_across_page_boundaries(authorized):
    pages=(front_page("Independent Research",0),
           front_page("on Coastal Habitats\nJane Smith",1))
    assert compare_pages(pages,authorized).identity_confidence != "high"


def test_first_page_date_conflict_survives_later_title(authorized):
    pages=(front_page("Revised January 2019",0),front_page(TEXT,1))
    assert compare_pages(pages,authorized).identity_confidence=="medium"


def test_legacy_first_page_retains_later_layout_limitations(authorized):
    values=observation().model_dump();values.pop("words")
    values["version"]="identity-ocr-observation-v1";values.pop("observation_sha256")
    values["observation_sha256"]=ocr._digest(values)
    legacy=ocr.IdentityOcrObservation.model_validate(values)
    result=compare_pages((legacy,front_page(TEXT,1)),authorized)
    assert result.identity_confidence=="high" and not result.accept
    assert 'positions available' in result.reason


@pytest.mark.parametrize("index",[-1,3,True,1.5])
def test_builder_rejects_unbounded_page_before_safety(monkeypatch,index):
    monkeypatch.setattr(ocr,"inspect_uploaded_pdf",lambda _:pytest.fail("unsafe page requested"))
    with pytest.raises(OcrDerivativeError):ocr.build_identity_observation(SOURCE,include_layout=True,page_index=index)
