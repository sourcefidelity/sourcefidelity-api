"""Scoped memos of completed paid web searches that found no source.

Revision ID: e2b7c4d9a613
Revises: d4e0f6ab1758
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e2b7c4d9a613"
down_revision = "d4e0f6ab1758"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("search_memos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("reference_key_sha256", sa.String(64), nullable=False),
        sa.Column("memo_policy_version", sa.String(64), nullable=False),
        sa.Column("search_policy_version", sa.String(64), nullable=False),
        sa.Column("outcome_summary", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "reference_key_sha256",
                            name="uq_search_memo_scope_reference"))


def downgrade():
    op.drop_table("search_memos")
