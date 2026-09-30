"""Judgment coaching notes and the per-candidate coaching field.

Revision ID: d4e0f6ab1758
Revises: c3d9e5fa0647
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d4e0f6ab1758"
down_revision = "c3d9e5fa0647"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("judgment_candidate_results", sa.Column("coaching", sa.JSON(), nullable=True))
    op.create_table("judgment_coaching_notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "cache_key", name="uq_judgment_coaching_scope_key"))


def downgrade():
    op.drop_table("judgment_coaching_notes")
    op.drop_column("judgment_candidate_results", "coaching")
