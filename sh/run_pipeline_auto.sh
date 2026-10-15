#!/usr/bin/env bash
# Fully automatic multi-round gap-filling pipeline.
#
# Loops: generate → verify → (repeat if no acceptance) → done.
# Generator vLLM and the wave-by-wave rollout verifier never run at the same time:
#   - split_generate_with_serve.sh starts the generator, generates, then kills it.
#   - verify runs split_gpu_pipeline.py verify directly (run_rollout.sh manages GPU waves).
#
# Prerequisites:
#   1. Run init first (once):
#        bash sh/run_pipeline_auto.sh init [extra args for split_gpu_pipeline.py init]
#   2. Set the env vars below (SERVE_SCRIPT and VLLM_BASE_URL at minimum).
#
# Usage:
#   export REPO_ROOT="$PWD"   # optional, from the repo root (default: parent of sh/)
#   export GAP_ROOT=$REPO_ROOT   # optional; defaults to REPO_ROOT
#   export SPLIT_WORKDIR=$GAP_ROOT/propose/results/e2e_split
#
#   # Generator (MiniMax-M2.7, TP=4 DP=2 on GPUs 0–7):
#   export SERVE_SCRIPT=$GAP_ROOT/propose/sh/serve_minimax_m2_7_8gpu.sh
#   export VLLM_BASE_URL=http://127.0.0.1:8010/v1
#
#   bash sh/run_pipeline_auto.sh init --benchmark aime25 --model minimax-m2.7
#   bash sh/run_pipeline_auto.sh run

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GAP_ROOT="${GAP_ROOT:-$REPO_ROOT}"
WORKDIR="${SPLIT_WORKDIR:-$GAP_ROOT/propose/results/e2e_split}"
SH="$GAP_ROOT/sh"

# Generation uses CUDA devices 0–7 (MiniMax-M2.7, TP=4 DP=2); verifier/run_rollout sets its own MATH_CUDA_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Forwarded to split_generate_with_serve.sh via env.
export REPO_ROOT GAP_ROOT SPLIT_WORKDIR="$WORKDIR"
export SERVE_SCRIPT="${SERVE_SCRIPT:-$GAP_ROOT/propose/sh/serve_minimax_m2_7_8gpu.sh}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-${PROPOSE_VLLM_BASE_URL:-http://127.0.0.1:8010/v1}}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

get_status() {
  python3 -c "
import json, sys
try:
    d = json.load(open('$WORKDIR/state.json'))
    print(d.get('status', 'unknown'))
except Exception:
    print('no_state')
"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd="${1:-help}"
shift || true

case "$cmd" in
  init)
    mkdir -p "$WORKDIR"
    cd "$GAP_ROOT/propose"
    # All extra args (--benchmark, --model, --k-parents, --n-children, --memory-size, etc.)
    # are forwarded directly to split_gpu_pipeline.py init.
    PYTHONPATH=. python split_gpu_pipeline.py init \
      --workdir "$WORKDIR" \
      --workspace "$REPO_ROOT" \
      "$@"
    echo ""
    echo "Next: bash $0 run"
    ;;

  run)
    status="$(get_status)"
    if [[ "$status" == "done" ]]; then
      echo "Pipeline already done. Check $WORKDIR/state.json."
      exit 0
    fi
    if [[ "$status" == "no_state" ]]; then
      echo "state.json not found — run init first: bash $0 init ..." >&2
      exit 1
    fi

    round=0
    while true; do
      status="$(get_status)"
      echo ""
      echo "===== Round $round  status=$status ====="

      if [[ "$status" == "done" ]]; then
        echo "Pipeline finished."
        break
      fi

      if [[ "$status" == "need_generate" ]]; then
        echo "--- GENERATE ---"
        bash "$SH/split_generate_with_serve.sh"
        status="$(get_status)"
      fi

      if [[ "$status" == "need_verify" ]]; then
        echo "--- VERIFY (wave-by-wave rollout) ---"
        cd "$GAP_ROOT/propose"
        PYTHONPATH=. python split_gpu_pipeline.py verify --workdir "$WORKDIR"
        status="$(get_status)"
      fi

      if [[ "$status" == "done" ]]; then
        echo "Pipeline finished."
        break
      fi

      round=$((round + 1))
    done
    ;;

  status)
    cd "$GAP_ROOT/propose"
    PYTHONPATH=. python split_gpu_pipeline.py status --workdir "$WORKDIR"
    ;;

  help|*)
    echo "Usage: $0 {init|run|status} [extra args]"
    echo ""
    echo "  init [args]  — create workdir/state.json (pass args to split_gpu_pipeline.py init)"
    echo "  run          — start the automatic loop (generate → verify, repeat until done)"
    echo "  status       — print current pipeline state"
    echo ""
    echo "Required env vars for 'run':"
    echo "  SERVE_SCRIPT              generator vLLM serve script"
    echo "  VLLM_BASE_URL             generator API base URL (for readiness probe)"
    echo ""
    echo "Optional:"
    echo "  CUDA_VISIBLE_DEVICES      GPUs visible to generate wrapper (default: 0–7)"
    echo "  SPLIT_WORKDIR             working directory (default: \$GAP_ROOT/propose/results/e2e_split)"
    echo "  READY_MAX_WAIT            seconds to wait for generator readiness (default 3600)"
    echo "  VLLM_LOG_FILE             append vLLM server logs here"
    echo "  CONDA_SERVE_ENV           conda env for generator server"
    echo "  SKIP_SERVE                1 = skip start/stop of generator server"
    exit 0
    ;;
esac
