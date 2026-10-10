"""A scope's storage settings for sources, papers and reports.

Revision ID: a7d3e5f1c920
Revises: f3c9a1d7b852
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a7d3e5f1c920"
down_revision = "f3c9a1d7b852"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("storage_policies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("updated_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", name="uq_storage_policy_scope"))


def downgrade():
    op.drop_table("storage_policies")
