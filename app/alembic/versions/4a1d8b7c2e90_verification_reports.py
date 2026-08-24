"""versioned verification reports

Revision ID: 4a1d8b7c2e90
Revises: 7c2a5f0b9d31
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "4a1d8b7c2e90"
down_revision: Union[str, None] = "7c2a5f0b9d31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "verification_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("verification_id", sa.String(128), nullable=False),
        sa.Column("paper_version_id", sa.String(255), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("report_version", sa.Integer(), nullable=False),
        sa.Column(
            "previous_report_id",
            UUID(as_uuid=True),
            sa.ForeignKey("verification_reports.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("artifact_version", sa.String(64), nullable=False),
        sa.Column("verdict", sa.String(40), nullable=False),
        sa.Column("evidence_sha256", sa.String(64), nullable=False),
        sa.Column("report_payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint(
            "scope_type",
            "scope_id",
            "verification_id",
            "report_version",
            name="uq_verification_report_scope_version",
        ),
    )
    op.create_index(
        "ix_verification_reports_verification_id",
        "verification_reports",
        ["verification_id"],
    )
    op.create_index(
        "ix_verification_reports_paper_version_id",
        "verification_reports",
        ["paper_version_id"],
    )
    op.create_index(
        "ix_verification_reports_evidence_sha256",
        "verification_reports",
        ["evidence_sha256"],
    )


def downgrade() -> None:
    op.drop_table("verification_reports")
