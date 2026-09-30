"""Purpose-scoped edition relationship records, not source-admission grants.

These records describe independently validated evidence. Schema validity alone
does not establish work identity, edition correspondence or processing rights.
The report may display a human-reviewed relationship; this grants no resolver
or admission privilege. Attestations must come from an authorized workflow.
"""
from typing import Literal
from datetime import datetime
import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EditionEvidenceBinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    evidence_id: str = Field(min_length=1)
    representation_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    passage_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    purpose: Literal['work_identity', 'edition_relationship', 'quotation',
                     'paraphrase_scope', 'retrieved_locator', 'cross_edition_locator']


class EditionHumanReview(BaseModel):
    """Attestation populated by a trusted review workflow, never a model.

    Hash binding detects stale decisions, not reviewer authentication. The
    caller must independently authorize the reviewer and resolve evidence.
    """
    model_config = ConfigDict(extra='forbid', frozen=True)
    reviewer_id: str = Field(min_length=1, max_length=255)
    reviewed_at: datetime
    decision: Literal['confirmed', 'same_edition_later_printing', 'uncertain', 'rejected']
    record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reviewed_evidence_ids: tuple[str, ...]
    notes: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def require_auditable_review(self):
        if self.reviewed_at.tzinfo is None:
            raise ValueError('review_time_requires_timezone')
        if len(set(self.reviewed_evidence_ids)) != len(self.reviewed_evidence_ids):
            raise ValueError('duplicate_reviewed_evidence')
        if not self.reviewer_id.strip() or not self.notes.strip():
            raise ValueError('empty_review_provenance')
        return self


class AlternateEditionRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    contract_version: Literal['alternate-edition-v1', 'alternate-edition-v2'] = 'alternate-edition-v1'
    submitted_reference_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    retrieved_representation_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    work_identity: Literal['verified', 'unverified', 'conflict'] = 'unverified'
    relationship: Literal['verified_alternate_edition', 'same_edition_later_printing', 'unverified', 'conflict'] = 'unverified'
    exact_edition_match: Literal[False] = False
    claim_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    quotation_usable: bool = False
    paraphrase_usable: bool = False
    retrieved_locator_match: Literal['verified', 'unverified', 'incompatible'] = 'unverified'
    cited_edition_locator_correspondence: Literal['verified', 'unverified', 'incompatible'] = 'unverified'
    evidence: tuple[EditionEvidenceBinding, ...] = ()
    limitations: tuple[str, ...] = ()
    human_review: EditionHumanReview | None = None

    def review_input_sha256(self) -> str:
        value = self.model_dump(mode='json', exclude={'human_review'})
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False).encode()).hexdigest()

    @property
    def human_verified(self) -> bool:
        review = self.human_review
        return bool(review and review.decision == 'confirmed'
                    and self.relationship == 'verified_alternate_edition'
                    and review.record_sha256 == self.review_input_sha256()
                    and set(review.reviewed_evidence_ids) == {e.evidence_id for e in self.evidence})

    @model_validator(mode='after')
    def require_scoped_evidence(self):
        same_printing = self.relationship == 'same_edition_later_printing'
        if same_printing and self.contract_version != 'alternate-edition-v2':
            raise ValueError('later_printing_requires_v2')
        purposes = {e.purpose for e in self.evidence}
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError('duplicate_edition_evidence_id')
        verified = self.relationship == 'verified_alternate_edition'
        if (verified or same_printing) and (self.work_identity != 'verified' or
                         not {'work_identity', 'edition_relationship'} <= purposes):
            raise ValueError('alternate_relationship_requires_work_and_version_evidence')
        tasks = ((self.quotation_usable, 'quotation'),
                 (self.paraphrase_usable, 'paraphrase_scope'),
                 (self.retrieved_locator_match == 'verified', 'retrieved_locator'),
                 (self.cited_edition_locator_correspondence == 'verified', 'cross_edition_locator'))
        for active, purpose in tasks:
            if active and (not verified or not self.claim_sha256 or purpose not in purposes):
                raise ValueError('task_requires_verified_relationship_claim_and_evidence')
        for e in self.evidence:
            if e.purpose in {'quotation', 'paraphrase_scope', 'retrieved_locator'}:
                if e.representation_sha256 != self.retrieved_representation_sha256:
                    raise ValueError('task_evidence_bound_to_different_representation')
        if self.human_review:
            if (self.human_review.decision == 'same_edition_later_printing') != same_printing:
                raise ValueError('later_printing_review_relationship_mismatch')
            if self.human_review.record_sha256 != self.review_input_sha256():
                raise ValueError('stale_edition_human_review')
            if set(self.human_review.reviewed_evidence_ids) != {e.evidence_id for e in self.evidence}:
                raise ValueError('review_must_bind_all_record_evidence')
            if self.human_review.decision == 'confirmed' and not verified:
                raise ValueError('confirmed_review_requires_verified_relationship')
        return self
