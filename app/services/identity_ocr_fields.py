"""Locate requested field wording in an OCR observation, without identity grants.

A first-page mention can belong to prose, a bibliography or another work.
These occurrences are NOT field agreements or bibliographic conflicts. Callers
must establish current source authority. This is a development diagnostic, not
a parallel verifier: identity decisions belong to the consolidated source
validator and its shared bibliographic safeguards.
"""
from hashlib import sha256
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from app.services.citation_extractor import _surname_key_with_offsets
from app.services.identity_ocr_observation import IdentityOcrObservation, validate_identity_observation
from app.security import REPORT_SOURCE_CAPABILITY
from app.services.verification_evidence import authorize_representation


class IdentityFieldOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: Literal["title", "author", "year", "doi"]
    expected: str = Field(max_length=2000, repr=False)
    status: Literal["observed_wording", "not_observed", "not_supplied", "ambiguous"]
    spans: tuple[tuple[int, int], ...] = ()
    bibliographic_role: Literal["unverified"] = "unverified"


class IdentityFieldComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["identity-ocr-fields-v1"] = "identity-ocr-fields-v1"
    observation_sha256: str
    expected_fields_sha256: str
    fields: tuple[IdentityFieldOccurrence, ...]
    identity_status: Literal["unverified"] = "unverified"
    grants_admission: Literal[False] = False
    grants_evidence_use: Literal[False] = False


def compare_authorized_identity_fields(session, backend, *, principal,
                                       representation_id: str,
                                       observation: IdentityOcrObservation,
                                       expected_fields: dict[str, str]) -> IdentityFieldComparison:
    """Reauthorize an admitted source before consuming its private observation.

    This does not relax access to held sources. Pending-upload consumption needs
    its own separately authorized path; no public endpoint is added here.
    """
    principal.require(REPORT_SOURCE_CAPABILITY)
    source = authorize_representation(session, backend,
        representation_id=representation_id, scope_type=principal.scope_type,
        scope_id=principal.scope_id)
    return locate_identity_fields(observation, source.content, expected_fields)


def locate_identity_fields(observation: IdentityOcrObservation, source: bytes,
                          expected_fields: dict[str, str]) -> IdentityFieldComparison:
    """Return bounded exact-normalized occurrences with original OCR offsets.

    Author values are compared as supplied, not reduced to a surname or guessed.
    Missing wording never means wrong source. Never search outside this page.
    """
    validate_identity_observation(observation, source)
    if set(expected_fields) - {"title", "author", "year", "doi"}:
        raise ValueError("Unsupported identity field")
    if any(not isinstance(v, str) or len(v) > 2000 for v in expected_fields.values()):
        raise ValueError("Identity fields exceed their bounds")
    key, offsets = _surname_key_with_offsets(observation.text)
    rows=[]
    for field in ("title", "author", "year", "doi"):
        expected=expected_fields.get(field, "")
        wanted, _ = _surname_key_with_offsets(expected)
        spans=[]
        if wanted:
            position=0
            while len(spans) < 10:
                found=key.find(wanted, position)
                if found < 0: break
                position=found+1
                start,end=offsets[found],offsets[found+len(wanted)-1]+1
                before=observation.text[start-1:start] if start else ""
                after=observation.text[end:end+1]
                if before.isalnum() or after.isalnum(): continue
                # Avoid matching fragments separated by an entire intervening
                # layout region. This is a locality bound, not a role verdict.
                if end-start > max(80, len(expected)*3): continue
                spans.append((start,end))
        status=("not_supplied" if not expected.strip() else "ambiguous" if len(spans)>1
                else "observed_wording" if spans else "not_observed")
        rows.append(IdentityFieldOccurrence(field=field, expected=expected, status=status, spans=tuple(spans)))
    digest=sha256(json.dumps(expected_fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return IdentityFieldComparison(observation_sha256=observation.observation_sha256,
                                   expected_fields_sha256=digest,fields=tuple(rows))
