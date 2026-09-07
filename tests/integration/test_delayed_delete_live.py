"""Real delayed DELETE acceptance; original request and isolated resources."""
from contextlib import contextmanager
from dataclasses import replace
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import os
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError
import pytest

from test_source_generation_live import isolated

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_SOURCE_GENERATION") != "1",
    reason="requires isolated PostgreSQL/MinIO and a bounded loopback DELETE fixture")
from app.config import settings
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services import source_repository as sources
from app.services.storage.backend import S3Backend
from source_upload_recovery_live_worker import fixture_request


@contextmanager
def delayed_http_delete(backend, selected_key):
    upstream = urlsplit(settings.S3_ENDPOINT)
    assert upstream.scheme == 'http' and upstream.hostname in {'localhost', '127.0.0.1'}
    release, completed = threading.Event(), threading.Event()
    observed = {}
    exact_path = f'/{backend.storage._bucket}/{backend.prefix}{selected_key}'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_DELETE(self):
            assert self.path == exact_path
            assert int(self.headers.get('Content-Length', '0')) == 0
            observed['received'] = True
            if not release.wait(20):
                observed['expired'] = True
                completed.set()
                return
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port or 80, timeout=10)
            try:
                connection.request('DELETE', self.path, headers=dict(self.headers))
                response = connection.getresponse()
                observed['status'] = response.status
                response.read()
                try:
                    self.send_response(response.status)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                connection.close()
                completed.set()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    storage = object.__new__(S3Backend)
    storage._bucket = backend.storage._bucket
    storage._client = boto3.client('s3', endpoint_url=f'http://127.0.0.1:{server.server_port}',
        aws_access_key_id=settings.S3_ACCESS_KEY, aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION, config=Config(connect_timeout=1, read_timeout=0.25,
            retries={'total_max_attempts': 1}, s3={'addressing_style': 'path'}))
    class Delayed:
        def __getattr__(self, name):
            return getattr(backend, name)
        def delete(self, key):
            assert key == selected_key
            return storage.delete(backend.prefix + key)
        def finish(self):
            assert observed.get('received') and not completed.is_set()
            release.set()
            assert completed.wait(12)
            assert observed.get('status') == 204 and not observed.get('expired')
    try:
        yield Delayed(), observed
    finally:
        release.set()
        if observed.get('received'):
            assert completed.wait(12)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        storage._client.close()


def test_timed_out_delete_cannot_remove_later_admitted_source(isolated):
    factory, backend, engine, receipt = isolated
    request = fixture_request('delayed-delete-original-request')
    with factory() as session:
        first = sources.admit_representation(session, backend, request)
        sources.commit_source_admissions(session)
        key = session.get(ContentObjectRecord, first.content_object_id).storage_key
        assert sources.delete_representation(session, first.id)
        session.commit()
    with delayed_http_delete(backend, key) as (delayed, observed):
        with factory() as cleaner:
            with pytest.raises(ReadTimeoutError):
                sources.finalize_pending_object_deletions(cleaner, delayed)
            cleaner.rollback()
        assert backend.exists(key)
        with factory() as successor:
            record = sources.admit_representation(successor, backend,
                replace(request, scope_id='new-isolated-owner'))
            sources.commit_source_admissions(successor)
            new_id = record.id
            assert record.admission_state == 'accepted'
            new_key = successor.get(ContentObjectRecord, record.content_object_id).storage_key
            assert new_key != key
            assert backend.download(new_key) == request.representation.content
        delayed.finish()
        assert not backend.exists(key)
    with factory() as session:
        record = session.get(SourceRepresentationRecord, new_id)
        obj = session.get(ContentObjectRecord, record.content_object_id)
        assert record.admission_state == 'accepted' and not obj.deletion_pending
        assert backend.download(obj.storage_key) == request.representation.content
    receipt.update(case='real_http_delayed_delete', regression_passed=True,
        accepted_source_bytes_preserved=True, database_connection_fault=False,
        client_timeout=True, original_request_forwarded=True,
        minio_delete_status=observed['status'])
