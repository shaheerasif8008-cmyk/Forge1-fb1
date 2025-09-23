#!/usr/bin/env bash
set -euo pipefail

# Usage: ./restore_postgres.sh <backup_file.sql.gz>
if [ $# -lt 1 ]; then
  echo "Usage: $0 <backup_file.sql.gz>" && exit 1
fi
FILE="$1"
echo "Restoring Postgres from $FILE"
gunzip -c "$FILE" | PGPASSWORD=${POSTGRES_PASSWORD:-forge1_db_pass} psql -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-forge1_user} -d ${POSTGRES_DB:-forge1}
echo "Done"

