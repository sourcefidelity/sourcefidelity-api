"""Report model – stores citation checking results."""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True
    )
    total_references = Column(Text, nullable=True)  # JSON string
    verified_references = Column(Text, nullable=True)  # JSON string
    summary = Column(Text, nullable=True)  # JSON or Markdown summary
    report_markdown = Column(Text, nullable=True)
    report_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return (
            f"<Report(id={self.id}, job_id={self.job_id}, "
            f"refs={self.total_references})>"
        )
