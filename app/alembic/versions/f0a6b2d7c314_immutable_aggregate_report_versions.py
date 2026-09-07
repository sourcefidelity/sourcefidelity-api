"""immutable aggregate report versions

Revision ID: f0a6b2d7c314
Revises: e9f4a1c6b203
Create Date: 2026-09-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f0a6b2d7c314"
down_revision: Union[str, None] = "e9f4a1c6b203"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("reports", sa.Column("report_version", sa.Integer(), nullable=True))
    op.add_column(
        "reports",
        sa.Column("previous_report_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column("reports", sa.Column("amendment_reason", sa.String(100), nullable=True))
    op.execute(
        """
        WITH ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY job_id ORDER BY created_at, id
            ) AS version_number
            FROM reports
        )
        UPDATE reports
        SET report_version = ranked.version_number
        FROM ranked
        WHERE reports.id = ranked.id
        """
    )
    op.alter_column("reports", "report_version", nullable=False)
    op.create_foreign_key(
        "fk_reports_previous_report_id",
        "reports",
        "reports",
        ["previous_report_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_report_job_version", "reports", ["job_id", "report_version"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_report_job_version", "reports", type_="unique")
    op.drop_constraint("fk_reports_previous_report_id", "reports", type_="foreignkey")
    op.drop_column("reports", "amendment_reason")
    op.drop_column("reports", "previous_report_id")
    op.drop_column("reports", "report_version")
