"""Real HTTP timeouts followed by delayed MinIO PUTs, confined to test resources."""

from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit
import uuid

import boto3
from botocore.config import Config
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.services.storage.backend import S3Backend
from source_upload_recovery_live_worker import NamespacedStorage
from test_paper_dispatch_broker_live import _public_snapshot
from test_transient_verification_retry_live import _load


pytestmark = pytest.mark.skipif(os.environ.get('RUN_LIVE_UNCERTAIN_UPLOAD') != '1',
    reason='requires isolated PostgreSQL/MinIO and a bounded loopback HTTP delay fixture')


@contextmanager
def delayed_http_upload(backend, boundary):
    upstream = urlsplit(settings.S3_ENDPOINT)
    assert upstream.scheme == 'http' and upstream.hostname in {'localhost', '127.0.0.1'}
    release, completed = threading.Event(), threading.Event()
    observed = {}
    prefix = f'/{backend.storage._bucket}/{backend.prefix}'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Request paths/headers never enter test logs.
        def do_PUT(self):
            assert self.path.startswith(prefix)
            length = int(self.headers['Content-Length'])
            assert 0 < length < 2_000_000
            body = self.rfile.read(length)
            observed['received'] = True
            if not release.wait(20):
                observed['expired'] = True
                completed.set()
                return
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port or 80, timeout=10)
            try:
                # Preserve signed headers/body: this is the delayed original
                # request, not an application-level replacement upload.
                connection.request('PUT', self.path, body=body, headers=dict(self.headers))
                response = connection.getresponse()
                observed['status'] = response.status
                response.read()
                try:
                    self.send_response(response.status)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The test deliberately allowed the client to time out.
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
        pending = None
        def __getattr__(self, name):
            return getattr(backend, name)
        def upload(self, content, key):
            selected = (key.endswith('/chunks.json') if boundary == 'derivative' else
                not key.startswith(('source-upload-intents/', 'source-upload-recovery/')))
            if selected and self.pending is None:
                self.pending = (key, content)
                return storage.upload(content, backend.prefix + key)
            return backend.upload(content, key)
        def finish(self):
            assert observed.get('received') and not completed.is_set()
            release.set()
            assert completed.wait(12)
            assert observed.get('status') == 200 and not observed.get('expired')
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


@pytest.mark.parametrize('boundary', ['source', 'paper', 'transient', 'derivative', 'new_owner'])
def test_cleanup_retains_authority_after_real_http_timeout(monkeypatch, tmp_path, boundary):
    namespace = 'sf_source_upload_test_' + uuid.uuid4().hex
    before = _public_snapshot()
    isolated = create_engine(make_url(settings.DATABASE_URL).update_query_dict({'options': '-csearch_path=' + namespace}))
    factory = sessionmaker(bind=isolated, expire_on_commit=False)
    backend = NamespacedStorage(namespace)
    unit_dir = Path(__file__).parents[1] / 'unit'
    monkeypatch.syspath_prepend(str(unit_dir))
    fixture = _load('uncertain_upload_regressions', unit_dir / 'test_uncertain_upload_cleanup.py')
    receipt = {'namespace': namespace, 'case': boundary, 'synthetic_inputs': True, 'model_calls': 0}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        Base.metadata.create_all(isolated)
        if boundary == 'new_owner':
            fixture.test_new_owner_protects_bytes_without_settling_old_uncertain_upload((factory, backend))
            receipt['fault'] = 'injected_delayed_completion_with_real_storage'
        else:
            with delayed_http_upload(backend, boundary) as (delayed, observed):
                fixture.exercise_late_write(factory, backend, delayed, boundary, monkeypatch)
                receipt['http_client_timed_out'] = True
                receipt['delayed_original_put_status'] = observed['status']
                receipt['fault'] = 'controlled_loopback_request_delay'
        receipt['public_data_unchanged'] = _public_snapshot() == before
        assert receipt['public_data_unchanged']
        receipt['status'] = 'passed'
    finally:
        isolated.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        for key in backend.storage.list_keys(backend.prefix):
            assert key.startswith(namespace + '/')
            assert backend.storage.delete(key)
        assert not backend.storage.list_keys(backend.prefix)
        receipt['test_data_removed'] = True
        (tmp_path / 'uncertain-upload-receipt.json').write_text(json.dumps(receipt, indent=2))
