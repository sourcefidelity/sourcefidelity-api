"""Bound metadata observations, never downloaded source representations.

Only fields already consumed by the existing adapters are retained. Replaying
their parsers verifies the projection before it can participate in a separate
cross-provider identity comparison. No network requests are made here.
"""
from copy import deepcopy
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY = 'corroborated-metadata-identity-v1'
_FIELDS = {
    'openalex': {'id', 'doi', 'title', 'display_name', 'publication_year', 'authorships'},
    'core': {'id', 'doi', 'title', 'yearPublished', 'authors'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


class MetadataIdentityObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    policy_version: Literal['corroborated-metadata-identity-v1'] = POLICY
    provider: Literal['openalex', 'core']
    record_id: str = Field(min_length=1, max_length=255)
    native_fields: dict
    record_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    observation_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def bounded_record(self):
        if set(self.native_fields) - _FIELDS[self.provider]:
            raise ValueError('Unexpected metadata identity field')
        if len(json.dumps(self.native_fields, ensure_ascii=False)) > 20000:
            raise ValueError('Metadata identity projection too large')
        if digest(self.native_fields) != self.record_sha256:
            raise ValueError('Metadata identity record hash mismatch')
        if record_id(self.provider, self.native_fields) != self.record_id:
            raise ValueError('Metadata identity record ID mismatch')
        # Enforce the same minimal projection on reload, including nested keys.
        try:
            canonical = project(self.provider, self.native_fields)
        except (TypeError, ValueError, AttributeError, KeyError, IndexError):
            raise ValueError('Malformed metadata identity projection') from None
        if canonical != self.native_fields:
            raise ValueError('Metadata identity projection is not canonical')
        return self


def record_id(provider, data):
    value = data.get('id')
    if provider == 'openalex':
        return value if isinstance(value, str) and re.fullmatch(r'https://openalex.org/W\d+', value) else None
    return str(value) if type(value) is int and value > 0 else (
        value if isinstance(value, str) and re.fullmatch(r'[1-9]\d*', value) else None)


def project(provider, data):
    """Drop abstracts, links, scores and incidental payloads, including nested ones."""
    value = {key: deepcopy(data[key]) for key in _FIELDS[provider] if key in data}
    if provider == 'openalex' and 'authorships' in value:
        value['authorships'] = [dict(author=dict(display_name=item['author']['display_name']))
                               for item in value['authorships']]
    if provider == 'core' and 'authors' in value:
        if not isinstance(value['authors'], list) or any(
                not isinstance(item, (str, dict)) or
                (isinstance(item, dict) and not isinstance(item.get('name'), str))
                for item in value['authors']):
            raise ValueError('Malformed metadata authors')
        value['authors'] = [dict(name=item['name']) if isinstance(item, dict) else item
                            for item in value['authors']]
    return value


def parse(provider, native_fields):
    # Reuse normal adapter parsing; these methods do not need credentials or I/O.
    if provider == 'openalex':
        from app.services.retrieval.openalex import OpenAlexRetriever
        return OpenAlexRetriever.__new__(OpenAlexRetriever)._parse_work(native_fields)
    from app.services.retrieval.core import CoreRetriever
    return CoreRetriever.__new__(CoreRetriever)._parse_output(native_fields)


def observed_fields(result):
    return dict(title=result.title or '', authors=list(result.authors),
                year=result.year or '', doi=result.doi or '')


def build_observation(provider, result):
    if provider not in _FIELDS or result.source_name != provider or not result.success:
        return None
    try:
        native = project(provider, result.metadata or {})
        identity = record_id(provider, native)
        parsed = parse(provider, native)
        fields = observed_fields(parsed)
        if not identity or fields != observed_fields(result):
            return None
        if (not isinstance(fields['title'], str) or not fields['title'].strip()
                or not fields['authors'] or not all(isinstance(a, str) and a.strip() for a in fields['authors'])):
            return None
        return MetadataIdentityObservation(provider=provider, record_id=identity,
            native_fields=native, record_sha256=digest(native), observation_sha256=digest(fields))
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return None  # Optional provenance never certifies malformed metadata.


def observation_is_bound(candidate):
    receipt = candidate.metadata_identity
    if receipt is None or receipt.provider != candidate.provider:
        return False
    try:
        # Also validates objects edited in memory after model construction.
        receipt = MetadataIdentityObservation.model_validate(receipt.model_dump())
        fields = observed_fields(parse(receipt.provider, receipt.native_fields))
        observed = candidate.observed
        return (fields == dict(title=observed.title, authors=observed.authors,
                               year=observed.year, doi=observed.doi)
                and digest(fields) == receipt.observation_sha256)
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return False
