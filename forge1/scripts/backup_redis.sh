#!/usr/bin/env bash
set -euo pipefail

# Usage: ./backup_redis.sh [output_dir]
OUTDIR=${1:-"./backups"}
mkdir -p "$OUTDIR"
FILENAME="$OUTDIR/redis_backup_$(date +%Y%m%d_%H%M%S).rdb"

echo "Triggering Redis SAVE"
redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} SAVE

RDB_SOURCE=${REDIS_RDB_PATH:-/var/lib/redis/dump.rdb}
if [ -f "$RDB_SOURCE" ]; then
  cp "$RDB_SOURCE" "$FILENAME"
  echo "Saved to $FILENAME"
else
  echo "WARNING: Could not locate RDB at $RDB_SOURCE; adjust REDIS_RDB_PATH"
fi

