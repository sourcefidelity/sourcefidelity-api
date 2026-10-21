"""Addresses of deleted older reports (owner decision 2026-10-20: keep only the latest report).

Revision ID: d7b3e5a1c920
Revises: c4a9e2f7b610
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d7b3e5a1c920"
down_revision = "c4a9e2f7b610"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("report_redirects",
        sa.Column("old_report_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_report_redirects_report_id", "report_redirects", ["report_id"])


def downgrade():
    op.drop_index("ix_report_redirects_report_id", table_name="report_redirects")
    op.drop_table("report_redirects")
