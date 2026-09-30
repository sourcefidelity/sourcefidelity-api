"""Assessment-level "marks released" record and the job's optional assessment.

Revision ID: f3c9a1d7b852
Revises: e2b7c4d9a613
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f3c9a1d7b852"
down_revision = "e2b7c4d9a613"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("assessment_id", sa.String(255), nullable=True))
    op.create_index("ix_jobs_assessment_id", "jobs", ["assessment_id"])
    op.create_table("assessment_marks_releases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("assessment_id", sa.String(255), nullable=False),
        sa.Column("marks_released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_by", sa.String(255), nullable=True),
        sa.Column("release_source", sa.String(20), nullable=True),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cleared_by", sa.String(255), nullable=True),
        sa.Column("clear_source", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "assessment_id",
                            name="uq_assessment_marks_release_scope_assessment"))


def downgrade():
    op.drop_table("assessment_marks_releases")
    op.drop_index("ix_jobs_assessment_id", table_name="jobs")
    op.drop_column("jobs", "assessment_id")
