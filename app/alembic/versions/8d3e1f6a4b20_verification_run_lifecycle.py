"""verification run lifecycle

Revision ID: 8d3e1f6a4b20
Revises: 4a1d8b7c2e90
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "8d3e1f6a4b20"
down_revision: Union[str, None] = "4a1d8b7c2e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "verification_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("audit_version", sa.String(64), nullable=False),
        sa.Column("paper_version_id", sa.String(255), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("canonical_work_id", sa.String(255), nullable=False),
        sa.Column("representation_kind", sa.String(30), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("acquisition_route", sa.String(100), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("source_byte_size", sa.Integer(), nullable=False),
        sa.Column("transient_object_count", sa.Integer(), nullable=False),
        sa.Column("transient_byte_size", sa.Integer(), nullable=False),
        sa.Column("transient_objects", sa.JSON(), nullable=False),
        sa.Column("identity_verdict", sa.String(30), nullable=False),
        sa.Column("identity_confidence", sa.Float(), nullable=True),
        sa.Column("completeness_verdict", sa.String(30), nullable=False),
        sa.Column("cleanliness_verdict", sa.String(30), nullable=False),
        sa.Column("text_quality", sa.String(30), nullable=False),
        sa.Column("edition_or_version", sa.Text(), nullable=True),
        sa.Column("terminal_outcome", sa.String(30), nullable=True),
        sa.Column("cleanup_attempts", sa.Integer(), nullable=False),
        sa.Column("last_cleanup_error_code", sa.String(80), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cleaned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_verification_runs_paper_version_id",
        "verification_runs",
        ["paper_version_id"],
    )
    op.create_index(
        "ix_verification_runs_status", "verification_runs", ["status"]
    )
    op.create_index(
        "ix_verification_runs_content_sha256",
        "verification_runs",
        ["content_sha256"],
    )
    op.add_column(
        "verification_reports",
        sa.Column("verification_run_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_verification_reports_run",
        "verification_reports",
        "verification_runs",
        ["verification_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_verification_reports_verification_run_id",
        "verification_reports",
        ["verification_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_verification_reports_verification_run_id",
        table_name="verification_reports",
    )
    op.drop_constraint(
        "fk_verification_reports_run",
        "verification_reports",
        type_="foreignkey",
    )
    op.drop_column("verification_reports", "verification_run_id")
    op.drop_table("verification_runs")
