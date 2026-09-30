"""Experimental Judgment layout records.

Revision ID: b2c8d4e9f536
Revises: a1b7c3d8e425
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b2c8d4e9f536"
down_revision = "a1b7c3d8e425"
branch_labels = None
depends_on = None

_UUID = postgresql.UUID(as_uuid=True)


def upgrade():
    op.create_table("judgment_acknowledgements",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("principal_provider", sa.String(100), nullable=False),
        sa.Column("principal_subject", sa.String(255), nullable=False),
        sa.Column("notice_version", sa.String(80), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "principal_subject", "notice_version",
                            name="uq_judgment_ack_principal_notice"))
    op.create_table("judgment_runs",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("report_id", _UUID, sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("requested_by", sa.String(255), nullable=False),
        sa.Column("notice_version", sa.String(80), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("reason_code", sa.String(80), nullable=True),
        sa.Column("panel_version", sa.String(64), nullable=False),
        sa.Column("policy_hash", sa.String(64), nullable=True),
        sa.Column("policy_snapshot", sa.JSON(), nullable=True),
        sa.Column("candidate_total", sa.Integer(), nullable=False),
        sa.Column("candidate_done", sa.Integer(), nullable=False),
        sa.Column("spend_usd", sa.Float(), nullable=False),
        sa.Column("unpriced_calls", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_judgment_runs_report_id", "judgment_runs", ["report_id"])
    op.create_table("judgment_arm_results",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("scope_type", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(255), nullable=False),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("arm_id", sa.String(30), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("returned_model", sa.String(120), nullable=True),
        sa.Column("returned_provider", sa.String(80), nullable=True),
        sa.Column("contract_version", sa.String(64), nullable=False),
        sa.Column("prompt_sha256", sa.String(64), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("cost_basis", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_type", "scope_id", "cache_key", name="uq_judgment_arm_result_scope_key"))
    op.create_table("judgment_candidate_results",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("run_id", _UUID, sa.ForeignKey("judgment_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("citation_index", sa.Integer(), nullable=True),
        sa.Column("verification_report_id", _UUID,
                  sa.ForeignKey("verification_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_id", sa.String(128), nullable=False),
        sa.Column("display_state", sa.String(30), nullable=False),
        sa.Column("reason_code", sa.String(60), nullable=False),
        sa.Column("panel", sa.JSON(), nullable=False),
        sa.Column("wider_search", sa.JSON(), nullable=True),
        sa.Column("arm_result_ids", sa.JSON(), nullable=False),
        sa.Column("spend_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "verification_report_id", "candidate_id",
                            name="uq_judgment_candidate_per_run"),
        sa.UniqueConstraint("run_id", "seq", name="uq_judgment_candidate_seq"))
    op.create_index("ix_judgment_candidate_results_run_id", "judgment_candidate_results", ["run_id"])


def downgrade():
    op.drop_index("ix_judgment_candidate_results_run_id", table_name="judgment_candidate_results")
    op.drop_table("judgment_candidate_results")
    op.drop_table("judgment_arm_results")
    op.drop_index("ix_judgment_runs_report_id", table_name="judgment_runs")
    op.drop_table("judgment_runs")
    op.drop_table("judgment_acknowledgements")
