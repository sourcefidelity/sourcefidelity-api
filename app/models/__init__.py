"""SQLAlchemy models for SourceFidelity."""

from sqlalchemy.orm import declarative_base

Base = declarative_base()

from app.models.job import Job, JobStage, JobStatus  # noqa: E402, F401
from app.models.report import (  # noqa: E402, F401
    PaperAnnotationRecord,
    Report,
    ReportPaperArtifactRecord,
    VerificationReportRecord,
)
# The unused legacy StoredDocument sketch is deliberately not registered in
# metadata. CanonicalWork/ContentObject/SourceRepresentation are the accepted
# repository schema; registering both creates a phantom unmigrated table.
from app.models.source_repository import (  # noqa: E402, F401
    CanonicalWorkRecord,
    ContentObjectRecord,
    SourceRepresentationRecord,
)
from app.models.verification_run import VerificationRunRecord  # noqa: E402, F401
