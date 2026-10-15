#!/usr/bin/env bash
# Start panel vLLM servers → run split_gpu_pipeline.py verify → stop all panel servers.
#
# Symmetric counterpart to sh/split_generate_with_serve.sh.
# The panel typically runs many smaller models (12+ in the aime25 setup); all of them
# must be up before verify can compute IRT-based difficulty estimates.
#
# Usage:
#   export REPO_ROOT="$PWD"   # optional, from the repo root (default: parent of sh/)
#   export GAP_ROOT=$REPO_ROOT   # optional; defaults to REPO_ROOT
#   export SPLIT_WORKDIR=$GAP_ROOT/propose/results/e2e_split
#   export PANEL_SERVE_SCRIPT=/path/to/start_panel_all.sh   # starts all panel servers
#   export PANEL_READY_URLS="http://127.0.0.1:8011/v1 http://127.0.0.1:8012/v1 ..."
#   bash sh/split_verify_with_serve.sh
#
# Env vars:
#   PANEL_SERVE_SCRIPT   — script that starts all panel vLLM servers (required unless SKIP_SERVE=1)
#   PANEL_READY_URLS     — space-separated /v1 base_urls to probe before running verify
#   READY_MAX_WAIT       — seconds to wait per endpoint (default 3600)
#   READY_INTERVAL       — seconds between readiness probes (default 10)
#   SKIP_SERVE           — if 1, assume panel is already up (skip start/stop)
#   VLLM_LOG_FILE        — if set, append panel stdout/stderr here
#   CONDA_PANEL_ENV      — if set, activate this conda env before running PANEL_SERVE_SCRIPT

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GAP_ROOT="${GAP_ROOT:-$REPO_ROOT}"
WORKDIR="${SPLIT_WORKDIR:-$GAP_ROOT/propose/results/e2e_split}"
PANEL_SERVE_SCRIPT="${PANEL_SERVE_SCRIPT:-}"
PANEL_READY_URLS="${PANEL_READY_URLS:-}"
READY_MAX_WAIT="${READY_MAX_WAIT:-3600}"
READY_INTERVAL="${READY_INTERVAL:-10}"
SKIP_SERVE="${SKIP_SERVE:-0}"

# 8× NVIDIA RTX PRO 6000 Blackwell, devices 0–7.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PANEL_PID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_conda_source() {
  local base="${CONDA_BASE:-}"
  if [[ -z "$base" ]] && command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  fi
  if [[ -z "$base" || ! -f "$base/etc/profile.d/conda.sh" ]]; then
    echo "split_verify_with_serve: need conda on PATH or CONDA_BASE for CONDA_PANEL_ENV" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
}

_start_panel_bg() {
  if [[ -n "${CONDA_PANEL_ENV:-}" ]]; then
    local log_args=()
    if [[ -n "${VLLM_LOG_FILE:-}" ]]; then
      touch "$VLLM_LOG_FILE"
      log_args=(">>" "$VLLM_LOG_FILE" "2>&1")
    fi
    # setsid creates a new process group — we kill the whole group on exit.
    if [[ ${#log_args[@]} -gt 0 ]]; then
      (
        _conda_source
        conda activate "$CONDA_PANEL_ENV"
        exec bash "$PANEL_SERVE_SCRIPT"
      ) >>"$VLLM_LOG_FILE" 2>&1 &
    else
      (
        _conda_source
        conda activate "$CONDA_PANEL_ENV"
        exec bash "$PANEL_SERVE_SCRIPT"
      ) &
    fi
  else
    if [[ -n "${VLLM_LOG_FILE:-}" ]]; then
      touch "$VLLM_LOG_FILE"
      setsid bash "$PANEL_SERVE_SCRIPT" >>"$VLLM_LOG_FILE" 2>&1 &
    else
      setsid bash "$PANEL_SERVE_SCRIPT" &
    fi
  fi
}

_wait_for_url() {
  local url="$1"
  local probe="${url%/}/models"
  local deadline=$((SECONDS + READY_MAX_WAIT))
  echo "[$(date +%H:%M:%S)] Waiting for panel endpoint: $probe"
  while (( SECONDS < deadline )); do
    if curl -sf --max-time 10 "$probe" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] Ready: $url"
      return 0
    fi
    sleep "$READY_INTERVAL"
  done
  echo "[$(date +%H:%M:%S)] Timed out after ${READY_MAX_WAIT}s waiting for $url" >&2
  return 1
}

stop_panel() {
  if [[ -z "$PANEL_PID" ]]; then return 0; fi
  if kill -0 "$PANEL_PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Stopping panel (pid=$PANEL_PID and its process group)..."
    # Kill the entire process group started by setsid.
    kill -- -"$PANEL_PID" 2>/dev/null || kill "$PANEL_PID" 2>/dev/null || true
    wait "$PANEL_PID" 2>/dev/null || true
  fi
  PANEL_PID=""
}

trap 'stop_panel' EXIT
trap 'stop_panel; exit 130' INT
trap 'stop_panel; exit 143' TERM

# ---------------------------------------------------------------------------
# Start panel
# ---------------------------------------------------------------------------

if [[ "$SKIP_SERVE" != "1" ]]; then
  if [[ -z "$PANEL_SERVE_SCRIPT" ]]; then
    echo "PANEL_SERVE_SCRIPT must be set to the script that starts all panel vLLM servers." >&2
    echo "Set SKIP_SERVE=1 to assume the panel is already up." >&2
    exit 1
  fi
  if [[ ! -f "$PANEL_SERVE_SCRIPT" ]]; then
    echo "PANEL_SERVE_SCRIPT not found: $PANEL_SERVE_SCRIPT" >&2
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] Starting panel: $PANEL_SERVE_SCRIPT"
  _start_panel_bg
  PANEL_PID=$!

  # Wait for all panel endpoints to be ready.
  if [[ -n "$PANEL_READY_URLS" ]]; then
    for url in $PANEL_READY_URLS; do
      _wait_for_url "$url"
    done
  else
    echo "[$(date +%H:%M:%S)] PANEL_READY_URLS not set — sleeping 120s for panel warmup."
    sleep 120
  fi
fi

# ---------------------------------------------------------------------------
# Run verify
# ---------------------------------------------------------------------------

cd "$GAP_ROOT/propose"
echo "[$(date +%H:%M:%S)] Running split_gpu_pipeline.py verify (workdir=$WORKDIR) ..."
PYTHONPATH=. python split_gpu_pipeline.py verify --workdir "$WORKDIR"
echo "[$(date +%H:%M:%S)] Verify finished. EXIT trap will stop panel servers."
