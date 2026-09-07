"""Process-exit fixtures confined to one test schema and object-store prefix."""

import io
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.storage.backend import S3Backend
from app.services.upload_completion import UploadReceipt, upload_confirmed


class NamespacedStorage:
    def __init__(self, namespace):
        if not re.fullmatch(r"sf_source_upload_test_[0-9a-f]{32}", namespace):
            raise ValueError("Invalid test namespace")
        self.prefix = namespace + "/"
        self.storage = S3Backend()
    def upload(self, content, key):
        receipt = self.storage.upload(content, self.prefix + key)
        return UploadReceipt(key, confirmed=upload_confirmed(receipt))
    def download(self, key):
        return self.storage.download(self.prefix + key)
    def delete(self, key):
        return self.storage.delete(self.prefix + key)
    def exists(self, key):
        return self.storage.exists(self.prefix + key)
    def list_keys(self, prefix):
        keys = self.storage.list_keys(self.prefix + prefix)
        assert all(key.startswith(self.prefix) for key in keys)
        return [key[len(self.prefix):] for key in keys]
    def list_keys_page(self, prefix, *, start_after="", limit=100):
        keys = self.storage.list_keys_page(self.prefix + prefix,
                start_after=self.prefix + start_after if start_after else "", limit=limit)
        assert all(key.startswith(self.prefix) for key in keys)
        return [key[len(self.prefix):] for key in keys]


def fixture_request(label):
    from app.services.source_repository import AdmissionRequest, WorkIdentity
    from app.services.retrieval.base import SourceRepresentation, RepresentationKind
    return AdmissionRequest(
        work=WorkIdentity(title="Synthetic storage lifecycle " + label, work_type="journal_article"),
        representation=SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT,
                    media_type="text/plain", content=("Synthetic source storage fixture: " + label).encode()),
        provenance="instructor_upload", license_class="commercial_user_upload",
        scope_type="personal_owner", scope_id="isolated-test-owner",
        identity_verdict="match", completeness_verdict="complete", cleanliness_verdict="clean",
    )


if __name__ == "__main__":
    from sqlalchemy import text
    from app.database import SessionLocal
    from app.services.source_repository import admit_representation, rollback_source_admissions
    from app.services.source_upload_recovery import INTENT_PREFIX
    namespace = os.environ["SOURCEFIDELITY_TEST_NAMESPACE"]
    backend = NamespacedStorage(namespace)
    mode, label = sys.argv[1:3]
    with SessionLocal() as session:
        assert session.scalar(text("SELECT current_schema()")) == namespace
        upload = backend.upload
        intent_puts = 0
        def interrupted(content, key):
            global intent_puts
            receipt = upload(content, key)
            if key.startswith(INTENT_PREFIX):
                intent_puts += 1
            if (mode == "before-source" and key.startswith(INTENT_PREFIX)
                    or mode == "after-source" and not key.startswith(INTENT_PREFIX)
                    or mode == "after-confirmation" and intent_puts == 2):
                os._exit(73)
            return receipt
        backend.upload = interrupted
        if mode == "paper-input":
            from docx import Document
            from app.services import paper_upload
            from app.services.file_safety import SafetyVerdict
            # Safety validation is outside this synthetic lifecycle fixture.
            paper_upload.scan_with_clamd = lambda _content: (SafetyVerdict.CLEAN, "synthetic fixture")
            paper_upload.prepare_dispatch = lambda _job: os._exit(73)
            document = Document()
            document.add_paragraph("Synthetic interrupted paper upload.")
            output = io.BytesIO()
            document.save(output)
            paper_upload.create_paper_job(session, backend, content=output.getvalue(),
                    filename="synthetic.docx", media_type=paper_upload.DOCX_MEDIA_TYPE, scope_id=namespace)
            raise AssertionError("Expected fixture process exit")
        admit_representation(session, backend, fixture_request(label))
        if mode == "after-commit":
            session.commit()
        elif mode == "rollback-delete-failed":
            backend.delete = lambda _key: False
            assert rollback_source_admissions(session, backend) == 0
        else:
            raise AssertionError("Expected earlier fixture process exit")
        os._exit(73)
