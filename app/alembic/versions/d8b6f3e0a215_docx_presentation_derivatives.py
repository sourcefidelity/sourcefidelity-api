"""docx presentation derivatives

Revision ID: d8b6f3e0a215
Revises: c7a5e2d9f104
Create Date: 2026-08-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8b6f3e0a215"
down_revision: Union[str, None] = "c7a5e2d9f104"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("report_paper_artifacts", sa.Column("presentation_storage_key", sa.String(500), nullable=True))
    op.add_column("report_paper_artifacts", sa.Column("presentation_sha256", sa.String(64), nullable=True))
    op.add_column("report_paper_artifacts", sa.Column("presentation_media_type", sa.String(100), nullable=True))
    op.add_column("report_paper_artifacts", sa.Column("presentation_byte_size", sa.Integer(), nullable=True))
    op.add_column("report_paper_artifacts", sa.Column("presentation_evidence", sa.JSON(), nullable=True))
    op.execute(
        """
        UPDATE report_paper_artifacts
        SET presentation_storage_key = storage_key,
            presentation_sha256 = content_sha256,
            presentation_media_type = media_type,
            presentation_byte_size = byte_size,
            presentation_evidence = sanitization_evidence
        WHERE presentation_status = 'page_faithful_ready'
        """
    )


def downgrade() -> None:
    op.drop_column("report_paper_artifacts", "presentation_evidence")
    op.drop_column("report_paper_artifacts", "presentation_byte_size")
    op.drop_column("report_paper_artifacts", "presentation_media_type")
    op.drop_column("report_paper_artifacts", "presentation_sha256")
    op.drop_column("report_paper_artifacts", "presentation_storage_key")
