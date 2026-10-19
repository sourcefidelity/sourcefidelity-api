"""Drop the Institutional "marks released" record (owner decision 2026-10-19: Personal only).

The job's optional `assessment_id` stays: it is the user's own batch label.

Revision ID: c4a9e2f7b610
Revises: b8e4f6a2d031
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4a9e2f7b610"
down_revision = "b8e4f6a2d031"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("assessment_marks_releases")


def downgrade():
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
