"""SQLAlchemy models for SourceFidelity."""

from sqlalchemy.orm import declarative_base

Base = declarative_base()

from app.models.job import Job, JobStatus  # noqa: E402, F401
from app.models.report import Report  # noqa: E402, F401
from app.models.document import StoredDocument  # noqa: E402, F401
