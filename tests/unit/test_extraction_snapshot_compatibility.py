"""Adding optional observations must not rewrite historical snapshot digests."""
import hashlib
import json

from app.services.paper_extraction import PaperExtractionArtifact


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_historical_body_title_absence_preserves_anchor_binding():
    historical = PaperExtractionArtifact(paper_version_id='old', citation_format='apa').model_dump(mode='json')
    historical.pop('body_title_formatting', None)
    expected = digest(historical)
    restored = PaperExtractionArtifact.model_validate(historical)
    assert restored.body_title_formatting is None
    assert digest(restored.model_dump(mode='json')) == expected
    assert digest(PaperExtractionArtifact.model_validate_json(restored.model_dump_json()).model_dump(mode='json')) == expected


def test_explicit_null_is_not_reinterpreted_as_historical_absence():
    current = PaperExtractionArtifact(paper_version_id='new', citation_format='apa', body_title_formatting=None)
    value = current.model_dump(mode='json')
    assert 'body_title_formatting' in value
    restored = PaperExtractionArtifact.model_validate(value)
    assert digest(restored.model_dump(mode='json')) == digest(value)
    absent = dict(value); absent.pop('body_title_formatting')
    assert digest(absent) != digest(value)


def test_changed_existing_field_still_invalidates_snapshot_digest():
    original = PaperExtractionArtifact(paper_version_id='old', citation_format='apa').model_dump(mode='json')
    original.pop('body_title_formatting', None)
    restored = PaperExtractionArtifact.model_validate(original)
    restored.body_word_count += 1
    assert digest(restored.model_dump(mode='json')) != digest(original)
