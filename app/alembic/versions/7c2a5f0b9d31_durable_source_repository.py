"""durable source repository

Revision ID: 7c2a5f0b9d31
Revises: 13ad600454b7
Create Date: 2026-08-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "7c2a5f0b9d31"
down_revision: Union[str, None] = "13ad600454b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "canonical_works",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("doi", sa.String(255), nullable=True),
        sa.Column("isbn", sa.String(32), nullable=True),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column("display_title", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("year", sa.String(10), nullable=True),
        sa.Column("work_type", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("doi", name="uq_canonical_works_doi"),
        sa.UniqueConstraint("isbn", name="uq_canonical_works_isbn"),
    )
    op.create_index("ix_canonical_works_normalized_title", "canonical_works", ["normalized_title"])

    op.create_table(
        "content_objects",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("license_class", sa.String(50), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("deletion_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("storage_key", name="uq_content_objects_storage_key"),
        sa.UniqueConstraint(
            "license_class",
            "content_sha256",
            name="uq_content_object_license_hash",
        ),
    )
    op.create_index("ix_content_objects_content_sha256", "content_objects", ["content_sha256"])
    op.create_index("ix_content_objects_license_class", "content_objects", ["license_class"])
    op.create_index("ix_content_objects_deletion_pending", "content_objects", ["deletion_pending"])

    op.create_table(
        "source_representations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "canonical_work_id",
            UUID(as_uuid=True),
            sa.ForeignKey("canonical_works.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "content_object_id",
            UUID(as_uuid=True),
            sa.ForeignKey("content_objects.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("representation_kind", sa.String(30), nullable=False),
        sa.Column("original_kind", sa.String(30), nullable=True),
        sa.Column("provenance", sa.String(50), nullable=False),
        sa.Column("admission_state", sa.String(30), nullable=False),
        sa.Column("scope_type", sa.String(30), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("edition_or_version", sa.Text(), nullable=True),
        sa.Column("identity_verdict", sa.String(30), nullable=False),
        sa.Column("identity_confidence", sa.Float(), nullable=True),
        sa.Column("completeness_verdict", sa.String(30), nullable=False),
        sa.Column("cleanliness_verdict", sa.String(30), nullable=False),
        sa.Column("text_quality", sa.String(30), nullable=True),
        sa.Column("validation_evidence", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("admitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("admitted_by", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "canonical_work_id",
            "content_object_id",
            "provenance",
            "scope_type",
            "scope_id",
            name="uq_representation_work_object_provenance_scope",
        ),
    )
    op.create_index("ix_source_representations_canonical_work_id", "source_representations", ["canonical_work_id"])
    op.create_index("ix_source_representations_content_object_id", "source_representations", ["content_object_id"])
    op.create_index("ix_source_representations_provenance", "source_representations", ["provenance"])
    op.create_index("ix_source_representations_admission_state", "source_representations", ["admission_state"])


def downgrade() -> None:
    op.drop_table("source_representations")
    op.drop_table("content_objects")
    op.drop_table("canonical_works")
