"""Bounded operational outcomes; never search results or evidence admission."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_OUTCOMES = {
    'candidate_budget_exhausted': 'not_attempted',
    'access_restricted': 'access_restricted',
    'transport_failure': 'transport_failure',
    'fetch_unavailable': 'unavailable',
    'pdf_route_required': 'not_attempted',
    'cross_script_title_unresolved': 'identity_unconfirmed',
    'page_title_mismatch_unconfirmed': 'identity_unconfirmed',
    'site_homepage': 'identity_unconfirmed',
    # A log-in, visitor or bot-check page stands in for the work (2026-10-07).
    'access_wall': 'access_restricted',
    'readable_text_unavailable': 'identity_unconfirmed',
    'source_kind_unconfirmed': 'type_unconfirmed',
    'bibliographic_fields_conflict': 'identity_rejected',
    'bibliographic_identity_confirmed': 'acquired',
    'bibliographic_identity_unconfirmed': 'identity_unconfirmed',
    'diagnostic_unavailable': 'unknown',
}


class WebFetchDiagnostic(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    version: Literal['web-fetch-diagnostic-v1'] = 'web-fetch-diagnostic-v1'
    reason: Literal[
        'candidate_budget_exhausted',
        'access_restricted', 'transport_failure', 'fetch_unavailable',
        'pdf_route_required', 'cross_script_title_unresolved',
        'page_title_mismatch_unconfirmed', 'readable_text_unavailable', 'site_homepage', 'access_wall',
        'source_kind_unconfirmed', 'bibliographic_fields_conflict',
        'bibliographic_identity_confirmed', 'bibliographic_identity_unconfirmed',
        'diagnostic_unavailable',
    ]
    observed_content_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def require_binding(self):
        if self.reason in {'bibliographic_fields_conflict', 'bibliographic_identity_confirmed'} and not self.observed_content_sha256:
            raise ValueError('Identity disposition requires independently observed content binding')
        return self

    @property
    def outcome(self) -> str:
        return _OUTCOMES[self.reason]


def read_web_fetch_diagnostic(result) -> WebFetchDiagnostic:
    """Old/malformed metadata remains unknown; do not parse free-text errors."""
    try:
        return WebFetchDiagnostic.model_validate((result.metadata or {}).get('web_fetch_diagnostic'))
    except ValidationError:
        return WebFetchDiagnostic(reason='diagnostic_unavailable')
