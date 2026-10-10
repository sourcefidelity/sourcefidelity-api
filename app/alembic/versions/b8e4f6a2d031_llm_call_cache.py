"""Remembered model answers for exact repeated requests.

Revision ID: b8e4f6a2d031
Revises: a7d3e5f1c920
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b8e4f6a2d031"
down_revision = "a7d3e5f1c920"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("llm_call_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("caller", sa.String(120), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "cache_key", name="uq_llm_call_cache_scope_key"))
    op.create_index("ix_llm_call_cache_created_at", "llm_call_cache", ["created_at"])


def downgrade():
    op.drop_index("ix_llm_call_cache_created_at", table_name="llm_call_cache")
    op.drop_table("llm_call_cache")
