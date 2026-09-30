"""Publisher is an identity field and must survive discovery record building.

The merge rule has scored publisher since the identity-combination work, and
the discovery builder emits a publisher comparison, but the comparison model's
field enum did not accept it. Every retrieval of a reference carrying a
publisher raised a validation error, which failed the whole retrieve stage.
"""
import pytest
from pydantic import ValidationError

from app.services.reference_discovery import BibliographicFieldComparison


def _comparison(field_name):
    return BibliographicFieldComparison(
        field_name=field_name,
        submitted_sha256="a" * 64,
        observed_sha256="b" * 64,
        outcome="unknown",
        reason_code="component_field_missing_or_unresolved",
    )


def test_publisher_is_an_accepted_comparison_field():
    assert _comparison("publisher").field_name == "publisher"


@pytest.mark.parametrize("field_name", [
    "title", "author", "year", "doi", "isbn",
    "container_title", "volume", "issue", "pages", "source_kind",
])
def test_established_fields_still_accepted(field_name):
    assert _comparison(field_name).field_name == field_name


def test_an_unknown_field_is_still_refused():
    with pytest.raises(ValidationError):
        _comparison("unrecognised_field")
