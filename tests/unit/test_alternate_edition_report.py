from app.services.evidence_report import _render_alternate_edition


def test_missing_record_keeps_historical_report_unchanged():
    assert _render_alternate_edition(None) == ''


def test_schema_valid_record_does_not_claim_verified_equivalence():
    record = dict(submitted_reference_sha256='a'*64,
                  retrieved_representation_sha256='b'*64,
                  work_identity='verified',relationship='verified_alternate_edition',
                  evidence=[dict(evidence_id=p,purpose=p,representation_sha256='b'*64,
                                 passage_sha256='c'*64)
                            for p in ('work_identity','edition_relationship')])
    html = _render_alternate_edition(record)
    assert 'still needs evidence review' in html
    assert 'attention' not in html and 'verified equivalent' not in html


def test_invalid_record_is_neutral_and_not_echoed():
    html = _render_alternate_edition({'relationship':'<script>unsafe()</script>'})
    assert 'could not be validated' in html
    assert '<script>' not in html and 'attention' not in html
