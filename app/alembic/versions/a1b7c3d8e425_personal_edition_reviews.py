"""Personal edition review snapshots and append-only decisions.

Revision ID: a1b7c3d8e425
Revises: f0a6b2d7c314
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a1b7c3d8e425"
down_revision = "f0a6b2d7c314"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("edition_review_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("representation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("source_representations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("snapshot_sha256", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("edition_review_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("edition_review_snapshots.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("reviewer_provider", sa.String(100), nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))


def downgrade():
    op.drop_table("edition_review_decisions")
    op.drop_table("edition_review_snapshots")
