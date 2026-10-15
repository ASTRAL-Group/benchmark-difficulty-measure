#!/usr/bin/env bash
# Pack a minimal lmms-eval source tree into vendor/lmms-eval.tar.gz (~4MB).
#
# Usage:
#   LMMS_EVAL_SRC=<path to an lmms-eval checkout> bash scripts/bundle_lmms_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${LMMS_EVAL_SRC:-}"
OUT="$REPO_ROOT/vendor/lmms-eval.tar.gz"

if [[ -z "$SRC" || ! -d "$SRC/lmms_eval" ]]; then
  echo "ERROR: lmms_eval not found under $SRC" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/vendor"
tar -czf "$OUT" -C "$SRC" \
  --exclude='.venv' \
  --exclude='logs' \
  --exclude='oldlogs' \
  --exclude='old_logs' \
  --exclude='.git' \
  --exclude='lmms_eval_sync.tar.gz' \
  .

echo "[bundle] wrote $OUT ($(du -sh "$OUT" | cut -f1))"
echo "[bundle] extract with: bash scripts/setup_server.sh"
