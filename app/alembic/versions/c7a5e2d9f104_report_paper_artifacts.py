"""report-scoped sanitized marking copies

Revision ID: c7a5e2d9f104
Revises: b6f4c1d8e2a9
Create Date: 2026-08-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c7a5e2d9f104"
down_revision: Union[str, None] = "b6f4c1d8e2a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "report_paper_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("paper_version_id", sa.String(255), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("artifact_kind", sa.String(40), nullable=False),
        sa.Column("presentation_status", sa.String(40), nullable=False),
        sa.Column("sanitization_evidence", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["report_id"], ["reports.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", name="uq_report_paper_artifact_job"),
    )
    op.create_index(
        "ix_report_paper_artifacts_report_id", "report_paper_artifacts", ["report_id"]
    )
    op.create_index(
        "ix_report_paper_artifacts_expires_at", "report_paper_artifacts", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_report_paper_artifacts_expires_at", table_name="report_paper_artifacts")
    op.drop_index("ix_report_paper_artifacts_report_id", table_name="report_paper_artifacts")
    op.drop_table("report_paper_artifacts")
