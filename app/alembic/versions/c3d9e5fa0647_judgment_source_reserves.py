"""Judgment wider-search sentence reserves.

Revision ID: c3d9e5fa0647
Revises: b2c8d4e9f536
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c3d9e5fa0647"
down_revision = "b2c8d4e9f536"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("judgment_source_reserves",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("verification_report_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("verification_reports.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("reserve_version", sa.String(64), nullable=False),
        sa.Column("retention_policy", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))


def downgrade():
    op.drop_table("judgment_source_reserves")
