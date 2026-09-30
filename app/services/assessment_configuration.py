"""Versioned, paper-snapshot assessment requirements; never global defaults."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, StrictBool


class AssessmentConfiguration(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    version: Literal['assessment-configuration-v1'] = 'assessment-configuration-v1'
    require_reference_links: StrictBool = False


def assessment_link_omissions(configuration, inventory):
    configuration = AssessmentConfiguration.model_validate(configuration or {})
    if not configuration.require_reference_links or not inventory:
        return []
    return [{
        'finding_type':'assessment_link_missing',
        'reference_id':entry['reference_id'],
        'rule_id':'assessment_all_references_link_v1',
        'conflicting_fields':['assessment-required DOI, URL or library link'],
        'finding':'This assessment requires a DOI, URL, or library link for every reference. None was found in this entry.',
        'rectangles':entry.get('rectangles', []),
        'localization_status':'bound_entry' if entry.get('rectangles') else 'not_assessed',
    } for entry in inventory['entries'] if entry['status']=='not_observed']
