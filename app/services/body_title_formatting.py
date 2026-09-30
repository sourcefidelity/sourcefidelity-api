"""Bounded DOCX title/year style observations; no semantic media detector."""
import hashlib
import io
import re
from collections import defaultdict
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from docx import Document
from docx.text.run import Run
from app.services.reference_layout import _docx_run_italic
from app.services.reference_formatting import _BOOK_PUBLISHER_RE


class BodyTitleObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    reference_id: str
    reference_text_sha256: str
    paragraph_index: int = Field(ge=0)
    paragraph_sha256: str
    paragraph_text: str
    paragraph_start: int = Field(ge=0)
    title_start: int = Field(ge=0)
    title_end: int = Field(gt=0)
    title_sha256: str
    status: Literal['matches_rule', 'difference', 'not_assessed']
    reason_code: Literal['title_italic', 'title_not_fully_italic', 'unknown_or_italic_surroundings']
    expected_italic: Literal[True]

    @model_validator(mode='after')
    def validate_binding(self):
        start, end = self.title_start-self.paragraph_start, self.title_end-self.paragraph_start
        if not 0 <= start < end <= len(self.paragraph_text):
            raise ValueError('Invalid title span')
        for text, digest in ((self.paragraph_text, self.paragraph_sha256),
                             (self.paragraph_text[start:end], self.title_sha256)):
            if hashlib.sha256(text.encode()).hexdigest() != digest:
                raise ValueError('Changed title observation')
        reasons = {'difference': 'title_not_fully_italic', 'matches_rule': 'title_italic',
                   'not_assessed': 'unknown_or_italic_surroundings'}
        if reasons[self.status] != self.reason_code:
            raise ValueError('Inconsistent title observation')
        return self


class BodyTitleAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    version: Literal['apa7_body_title_italics_v2']
    paper_sha256: str
    body_sha256: str
    observations: list[BodyTitleObservation]
    automatic_findings_enabled: Literal[False]


def body_title_findings(assessment, references, paper_sha256):
    """Project saved, input-bound differences; geometry must still be established."""
    if assessment is None or assessment.paper_sha256 != paper_sha256:
        return []
    findings = []
    for item in assessment.observations:
        ref = references.get(item.reference_id)
        if (item.status != 'difference' or ref is None or ref.needs_review
                or hashlib.sha256(ref.raw_ref.encode()).hexdigest() != item.reference_text_sha256):
            continue
        start, end = item.title_start-item.paragraph_start, item.title_end-item.paragraph_start
        findings.append(dict(finding_type='body_title_style', reference_id=item.reference_id,
            observation_id=f'{item.reference_id}:{item.title_start}',
            citation_text=item.paragraph_text,
            finding='Italicize this film or book title in the body of the paper.',
            field_difference={'field_name': 'body_title', 'submitted_value': item.paragraph_text[start:end]},
            rule_source='https://academicguides.waldenu.edu/writingcenter/apa/other/italics',
            rectangles=[], localization_status='not_assessed'))
    return findings


def assess_body_title_italics(*, content: bytes, body: str, references,
                             citation_format: str = 'apa') -> dict:
    """Inspect exact title + year mentions in uniquely mapped body paragraphs.

    Caller owns upload authorization/safety. Observations alone are not report flags.
    Unknown/ambiguous work identities and implicit mentions are not assessed.
    """
    digest=lambda value:hashlib.sha256(value).hexdigest()
    result=dict(version='apa7_body_title_italics_v2',paper_sha256=digest(content),
        body_sha256=digest(body.encode()),observations=[],automatic_findings_enabled=False)
    if citation_format!='apa':return result
    groups=defaultdict(list)
    for ref in references:
        title=re.sub(r'\s+\[(?:Film|Documentary)\]$', '', ref.title, flags=re.I).strip()
        if title:groups[title.casefold()].append((title,ref))
    eligible=[]
    for entries in groups.values():
        if len(entries)!=1:continue
        title,ref=entries[0]
        if ref.needs_review or ref.source_kind_confidence!='high':continue
        if any(char in title for char in '“”"[]'):continue
        film=ref.source_kind=='traditional_media' and re.search(r'\[(?:Film|Documentary)\]',ref.raw_ref,re.I)
        if not (film or ref.source_kind=='monograph') or not re.fullmatch(r'(?:19|20)\d{2}',ref.year or ''):continue
        if ref.source_kind == 'monograph':
            # Reuse the reference-title rule's independent book premise.
            if ref.raw_ref.count(title) != 1:continue
            tail = ref.raw_ref.split(title, 1)[1].lstrip(' .,*')
            if re.match(r'In\b', tail, re.I) or not _BOOK_PUBLISHER_RE.search(tail):continue
        eligible.append((title,ref))
    document=Document(io.BytesIO(content))
    for index,paragraph in enumerate(document.paragraphs):
        if paragraph._p.xpath('.//w:ins | .//w:del | .//w:moveFrom | .//w:moveTo | .//w:fldChar | .//w:fldSimple | .//w:drawing | .//w:pict'):
            continue
        runs=[run for part in paragraph.iter_inner_content() for run in ([part] if isinstance(part,Run) else part.runs)]
        text=''.join(run.text for run in runs)
        if not text.strip() or body.count(text)!=1:continue
        offset=body.index(text)
        styles=[flag for run in runs for flag in [_docx_run_italic(run)]*len(run.text)]
        for title,ref in eligible:
            pattern=r'(?<!\w)'+re.escape(title)+r'(?!\w)(?=\s*[（(]\s*'+re.escape(ref.year)+r'\s*[)）])'
            for match in re.finditer(pattern,text,re.I):
                s,e=match.span();observed=[styles[i] for i in range(s,e) if not text[i].isspace()]
                surroundings=[styles[i] for i in range(len(text)) if not s<=i<e and not text[i].isspace()]
                state='not_assessed';reason='unknown_or_italic_surroundings'
                if observed and all(v is not None for v in observed) and surroundings and all(v is False for v in surroundings):
                    state='matches_rule' if all(observed) else 'difference'
                    reason='title_italic' if all(observed) else 'title_not_fully_italic'
                result['observations'].append(dict(reference_id=ref.reference_id,
                    reference_text_sha256=digest(ref.raw_ref.encode()),paragraph_index=index,
                    paragraph_sha256=digest(text.encode()),title_start=offset+s,title_end=offset+e,
                    paragraph_text=text,paragraph_start=offset,
                    title_sha256=digest(body[offset+s:offset+e].encode()),
                    status=state,reason_code=reason,expected_italic=True))
    return result
