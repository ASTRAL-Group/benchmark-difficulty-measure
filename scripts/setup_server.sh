#!/usr/bin/env bash
# Patch verifier task yamls and sanity-check a fresh clone.
#
# Usage (from repo root):
#   cp env.example env.local   # optional overrides
#   source env.local
#   bash scripts/setup_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-}"
if [[ -z "$LMMS_EVAL_REPO" || ! -d "$LMMS_EVAL_REPO/lmms_eval" ]]; then
  if [[ -d "$REPO_ROOT/vendor/lmms-eval/lmms_eval" ]]; then
    LMMS_EVAL_REPO="$REPO_ROOT/vendor/lmms-eval"
  else
    LMMS_EVAL_REPO="$REPO_ROOT/vendor/lmms-eval"
  fi
fi
export LMMS_EVAL_REPO

echo "[setup] REPO_ROOT=$REPO_ROOT"
echo "[setup] LMMS_EVAL_REPO=$LMMS_EVAL_REPO"

VENDOR_TAR="$REPO_ROOT/vendor/lmms-eval.tar.gz"
if [[ ! -d "$LMMS_EVAL_REPO/lmms_eval" ]]; then
  if [[ -f "$VENDOR_TAR" ]]; then
    echo "[setup] extracting $VENDOR_TAR → $LMMS_EVAL_REPO"
    mkdir -p "$LMMS_EVAL_REPO"
    tar -xzf "$VENDOR_TAR" -C "$LMMS_EVAL_REPO"
  else
    echo "WARN: lmms-eval not found at $LMMS_EVAL_REPO and no $VENDOR_TAR" >&2
    echo "      clone lmms-eval elsewhere or run: bash scripts/bundle_lmms_eval.sh" >&2
  fi
fi

if [[ -d "$LMMS_EVAL_REPO/lmms_eval" && ! -x "$LMMS_EVAL_REPO/.venv/bin/python" ]]; then
  echo "[setup] lmms-eval venv missing — run: cd \"$LMMS_EVAL_REPO\" && uv sync"
fi

python3 << PY
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
lmms = Path(os.environ["LMMS_EVAL_REPO"])
tasks = repo / "verifier" / "tasks"
n = 0
for path in tasks.rglob("*.yaml"):
    text = path.read_text(encoding="utf-8")
    new = text.replace("__REPO_ROOT__", str(repo)).replace("__LMMS_EVAL_REPO__", str(lmms))
    if new != text:
        path.write_text(new, encoding="utf-8")
        n += 1
print(f"[setup] patched {n} yaml file(s) under verifier/tasks/")
PY

if grep -rq '__REPO_ROOT__\|__LMMS_EVAL_REPO__' "$REPO_ROOT/verifier/tasks" 2>/dev/null; then
  echo "WARN: some yaml placeholders remain — check LMMS_EVAL_REPO" >&2
fi

for bm in mmlu_bio ifeval bbh_3obj; do
  f="$REPO_ROOT/data/pools/$bm/test.jsonl"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing pool $f" >&2
    exit 1
  fi
done
echo "[setup] pool jsonl OK (mmlu_bio, ifeval, bbh_3obj)"

mkdir -p "$REPO_ROOT/verifier/lmms_candidates"
touch "$REPO_ROOT/verifier/lmms_candidates/.gitkeep"
echo "[setup] done"
