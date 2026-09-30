from app.services.source_resolver import _timed_retrieval
from app.services.retrieval import RetrievalResult
from app.services.reference_discovery import ReferenceRouteAttempt


def test_timing_binds_real_boundaries_and_preserves_result():
    original = RetrievalResult(source_name='test', success=False, error='No results', metadata={'existing': True})

    @_timed_retrieval
    def operation():
        return original

    result = operation()
    timing = result.metadata['operation_timing']
    assert timing['started_at'] <= timing['completed_at']
    assert timing['elapsed_seconds'] > 0
    assert result.metadata['existing']
    assert 'operation_timing' not in original.metadata
    assert result.error == original.error


def test_historical_route_has_no_manufactured_elapsed_time():
    row = ReferenceRouteAttempt(attempt_id='test', route_category='student_url', provider='test',
        required=False, permitted=True, query_ids=[], outcome='no_match',
        started_at='2026-09-01T00:00:00Z', completed_at='2026-09-01T00:00:00Z')
    assert row.elapsed_seconds is None and row.timing_version is None
