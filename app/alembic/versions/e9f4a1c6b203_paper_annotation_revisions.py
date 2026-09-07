"""append-only paper annotation revisions

Revision ID: e9f4a1c6b203
Revises: d8b6f3e0a215
Create Date: 2026-09-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e9f4a1c6b203"
down_revision: Union[str, None] = "d8b6f3e0a215"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_annotation_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("annotation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_version_id", sa.String(255), nullable=False),
        sa.Column("paper_content_sha256", sa.String(64), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("previous_revision_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("annotation_type", sa.String(20), nullable=False),
        sa.Column("anchor_kind", sa.String(30), nullable=False),
        sa.Column("anchor_sha256", sa.String(64), nullable=False),
        sa.Column("anchor_payload", sa.JSON(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("user_label", sa.String(100), nullable=True),
        sa.Column("author_provider", sa.String(60), nullable=False),
        sa.Column("author_subject", sa.String(255), nullable=False),
        sa.Column("visibility", sa.String(20), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["paper_artifact_id"], ["report_paper_artifacts.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["previous_revision_id"], ["paper_annotation_revisions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["report_id"], ["reports.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scope_type",
            "scope_id",
            "report_id",
            "annotation_id",
            "revision",
            name="uq_paper_annotation_scope_revision",
        ),
    )
    op.create_index("ix_paper_annotation_revisions_annotation_id", "paper_annotation_revisions", ["annotation_id"])
    op.create_index("ix_paper_annotation_revisions_report_id", "paper_annotation_revisions", ["report_id"])
    op.create_index("ix_paper_annotation_revisions_paper_artifact_id", "paper_annotation_revisions", ["paper_artifact_id"])
    op.create_index("ix_paper_annotation_revisions_anchor_sha256", "paper_annotation_revisions", ["anchor_sha256"])


def downgrade() -> None:
    op.drop_index("ix_paper_annotation_revisions_anchor_sha256", table_name="paper_annotation_revisions")
    op.drop_index("ix_paper_annotation_revisions_paper_artifact_id", table_name="paper_annotation_revisions")
    op.drop_index("ix_paper_annotation_revisions_report_id", table_name="paper_annotation_revisions")
    op.drop_index("ix_paper_annotation_revisions_annotation_id", table_name="paper_annotation_revisions")
    op.drop_table("paper_annotation_revisions")
