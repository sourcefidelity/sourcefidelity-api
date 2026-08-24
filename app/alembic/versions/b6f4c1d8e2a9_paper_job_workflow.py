"""paper upload and checkpointed job workflow

Revision ID: b6f4c1d8e2a9
Revises: 8d3e1f6a4b20
Create Date: 2026-08-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b6f4c1d8e2a9"
down_revision: Union[str, None] = "8d3e1f6a4b20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "jobs",
        "created_at",
        type_=sa.DateTime(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "jobs",
        "updated_at",
        type_=sa.DateTime(timezone=True),
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )
    op.add_column("jobs", sa.Column("stage", sa.String(30), nullable=False, server_default="uploaded"))
    op.add_column("jobs", sa.Column("paper_version_id", sa.String(255), nullable=True))
    op.add_column("jobs", sa.Column("scope_type", sa.String(30), nullable=False, server_default="personal_owner"))
    op.add_column("jobs", sa.Column("scope_id", sa.String(255), nullable=True))
    op.add_column("jobs", sa.Column("input_storage_key", sa.String(500), nullable=True))
    op.add_column("jobs", sa.Column("input_sha256", sa.String(64), nullable=True))
    op.add_column("jobs", sa.Column("input_media_type", sa.String(100), nullable=True))
    op.add_column("jobs", sa.Column("input_byte_size", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("input_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("jobs", sa.Column("input_deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("jobs", sa.Column("store_only", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("jobs", sa.Column("task_id", sa.String(255), nullable=True))
    op.add_column("jobs", sa.Column("upload_evidence", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))
    op.add_column("jobs", sa.Column("extraction_payload", sa.JSON(), nullable=True))
    op.add_column("jobs", sa.Column("source_results", sa.JSON(), nullable=True))
    op.add_column("jobs", sa.Column("verification_summary", sa.JSON(), nullable=True))
    op.execute("UPDATE jobs SET paper_version_id = CAST(id AS TEXT), scope_id = 'personal-default', input_sha256 = '', input_media_type = 'application/octet-stream', input_byte_size = 0, input_expires_at = NOW() WHERE paper_version_id IS NULL")
    op.alter_column("jobs", "paper_version_id", nullable=False)
    op.alter_column("jobs", "scope_id", nullable=False)
    op.alter_column("jobs", "input_sha256", nullable=False)
    op.alter_column("jobs", "input_media_type", nullable=False)
    op.alter_column("jobs", "input_byte_size", nullable=False)
    op.alter_column("jobs", "input_expires_at", nullable=False)
    op.create_unique_constraint("uq_jobs_paper_version_id", "jobs", ["paper_version_id"])
    op.create_index("ix_jobs_paper_version_id", "jobs", ["paper_version_id"])
    op.create_index("ix_jobs_stage", "jobs", ["stage"])
    op.create_index("ix_jobs_input_expires_at", "jobs", ["input_expires_at"])


def downgrade() -> None:
    op.drop_index("ix_jobs_input_expires_at", table_name="jobs")
    op.drop_index("ix_jobs_stage", table_name="jobs")
    op.drop_index("ix_jobs_paper_version_id", table_name="jobs")
    op.drop_constraint("uq_jobs_paper_version_id", "jobs", type_="unique")
    for name in (
        "verification_summary", "source_results", "extraction_payload", "upload_evidence", "task_id",
        "store_only", "input_deleted_at", "input_expires_at", "input_byte_size",
        "input_media_type", "input_sha256", "input_storage_key", "scope_id",
        "scope_type", "paper_version_id", "stage",
    ):
        op.drop_column("jobs", name)
    op.alter_column(
        "jobs",
        "updated_at",
        type_=sa.DateTime(timezone=False),
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "jobs",
        "created_at",
        type_=sa.DateTime(timezone=False),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
