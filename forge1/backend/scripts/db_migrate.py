#!/usr/bin/env python3
"""Simple SQL migrations runner (Phase A.4)

Applies SQL files in forge1/backend/migrations in lexicographic order,
tracking progress in forge1_migrations.schema_migrations.
"""

import os
import asyncio
from glob import glob

from forge1.core.database_config import get_database_manager


async def apply_migration(conn, filename: str, sql: str) -> None:
    await conn.execute(sql)
    await conn.execute(
        "INSERT INTO forge1_migrations.schema_migrations (filename) VALUES ($1) ON CONFLICT DO NOTHING",
        os.path.basename(filename),
    )


async def main() -> int:
    db = await get_database_manager()
    path = os.path.join(os.path.dirname(__file__), "..", "migrations")
    files = sorted(glob(os.path.join(path, "*.sql")))
    async with db.postgres.acquire() as conn:
        # ensure tracking table exists (idempotent)
        await conn.execute("CREATE SCHEMA IF NOT EXISTS forge1_migrations")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forge1_migrations.schema_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
        )
        rows = await conn.fetch("SELECT filename FROM forge1_migrations.schema_migrations")
        applied = {r[0] for r in rows}
        for f in files:
            name = os.path.basename(f)
            if name in applied:
                continue
            with open(f, "r") as fh:
                sql = fh.read()
            print(f"Applying migration: {name}")
            await apply_migration(conn, f, sql)
    print("Migrations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

