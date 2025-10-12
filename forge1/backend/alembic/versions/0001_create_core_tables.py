"""Initial schema for Phase 1 foundations"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_create_core_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS forge1_core")
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema="forge1_core",
    )


def downgrade() -> None:
    op.drop_table("tenants", schema="forge1_core")
    op.execute("DROP SCHEMA IF EXISTS forge1_core CASCADE")
