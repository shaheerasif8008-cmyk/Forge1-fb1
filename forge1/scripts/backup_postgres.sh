#!/usr/bin/env bash
set -euo pipefail

# Usage: ./backup_postgres.sh [output_dir]
OUTDIR=${1:-"./backups"}
mkdir -p "$OUTDIR"
FILENAME="$OUTDIR/pg_backup_$(date +%Y%m%d_%H%M%S).sql.gz"

echo "Backing up Postgres to $FILENAME"
PGPASSWORD=${POSTGRES_PASSWORD:-forge1_db_pass} pg_dump -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-forge1_user} -d ${POSTGRES_DB:-forge1} | gzip > "$FILENAME"
echo "Done"

