#!/usr/bin/env bash
# Start generator vLLM → run split_gpu_pipeline.py generate → stop vLLM (free GPUs).
#
# Wrap ``vllm serve`` in a subshell, wait for HTTP readiness,
# run generation, then kill the server PID and wait for exit.
#
# Usage (from anywhere):
#   export REPO_ROOT="$PWD"   # optional, from the repo root (default: parent of sh/)
#   export SPLIT_WORKDIR=$REPO_ROOT/propose/split_wd
#   bash "$REPO_ROOT/sh/split_generate_with_serve.sh"
#
# Extra args are forwarded to ``split_gpu_pipeline.py generate`` (e.g. --model).
# Do not pass a second ``--workdir``; set SPLIT_WORKDIR instead (this script always passes --workdir).
#
# Env:
#   SERVE_SCRIPT     — default: $REPO_ROOT/propose/sh/serve_minimax_m2_7_8gpu.sh
#   VLLM_BASE_URL    — first generator URL for probes (default http://127.0.0.1:8010/v1)
#   VLLM_BASE_URLS   — comma-separated list; if set, wait_for_vllm_ready requires all (default: 8010–8013)
#   READY_MAX_WAIT   — seconds to wait for /v1/models (default 3600)
#   READY_INTERVAL   — seconds between probes (default 5)
#   VLLM_LOG_FILE    — if set, append serve stdout/stderr here
#   SKIP_SERVE       — if 1, do not start/kill vLLM (only run generate; for debugging)
#
# Two conda envs (generation): vLLM often needs ``propose-vllm``; lightweight propose deps another env.
#   CONDA_SERVE_ENV    — if set, run SERVE_SCRIPT after ``conda activate`` in a subshell (see _conda_source).
#   CONDA_GENERATE_ENV — if set, run ``split_gpu_pipeline.py generate`` under that env.
#   CONDA_BASE         — optional; else ``conda info --base`` (requires ``conda`` on PATH).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GAP_ROOT="${GAP_ROOT:-$REPO_ROOT}"
WORKDIR="${SPLIT_WORKDIR:-$GAP_ROOT/propose/results/e2e_split}"
SERVE_SCRIPT="${SERVE_SCRIPT:-$GAP_ROOT/propose/sh/serve_minimax_m2_7_8gpu.sh}"
VLLM_BASE_URL="${VLLM_BASE_URL:-${PROPOSE_VLLM_BASE_URL:-http://127.0.0.1:8010/v1}}"
# Must list every generator instance started by SERVE_SCRIPT (MiniMax-M2.7, one server on :8010).
export VLLM_BASE_URLS="${VLLM_BASE_URLS:-http://127.0.0.1:8010/v1}"
READY_MAX_WAIT="${READY_MAX_WAIT:-3600}"
READY_INTERVAL="${READY_INTERVAL:-5}"
SKIP_SERVE="${SKIP_SERVE:-0}"

# Generation uses CUDA devices 0–7 (single Qwen3.5-397B × TP=8).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

VLLM_PID=""

_conda_source() {
  local base="${CONDA_BASE:-}"
  if [[ -z "$base" ]] && command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  fi
  if [[ -z "$base" || ! -f "$base/etc/profile.d/conda.sh" ]]; then
    echo "split_generate_with_serve: need conda (PATH) or CONDA_BASE for CONDA_*_ENV" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
}

_start_serve_bg() {
  # Launch under setsid so the child becomes its own session/pgid leader.
  # Then $! == pgid, and stop_vllm's `kill -- -$VLLM_PID` reaches all vLLM
  # workers spawned by the serve script (otherwise workers escape and hold VRAM).
  local log="${VLLM_LOG_FILE:-/dev/null}"
  [[ -n "${VLLM_LOG_FILE:-}" ]] && touch "$VLLM_LOG_FILE"
  if [[ -n "${CONDA_SERVE_ENV:-}" ]]; then
    setsid bash -c '
      base="${CONDA_BASE:-}"
      [[ -z "$base" ]] && base="$(conda info --base 2>/dev/null || true)"
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      conda activate "'"$CONDA_SERVE_ENV"'"
      exec bash "'"$SERVE_SCRIPT"'"
    ' </dev/null >>"$log" 2>&1 &
  else
    setsid bash "$SERVE_SCRIPT" </dev/null >>"$log" 2>&1 &
  fi
}

_run_generate() {
  local log_file="${GENERATE_LOG_FILE:-}"
  if [[ -n "$log_file" ]]; then
    touch "$log_file"
    echo "[$(date +%H:%M:%S)] Logging generation output to $log_file"
    _run_generate_inner "$@" 2>&1 | tee -a "$log_file"
  else
    _run_generate_inner "$@"
  fi
}

_run_generate_inner() {
  if [[ -n "${CONDA_GENERATE_ENV:-}" ]]; then
    (
      _conda_source
      conda activate "$CONDA_GENERATE_ENV"
      cd "$GAP_ROOT/propose"
      export PYTHONPATH=.
      exec python split_gpu_pipeline.py generate --workdir "$WORKDIR" "$@"
    )
  else
    cd "$GAP_ROOT/propose"
    PYTHONPATH=. python split_gpu_pipeline.py generate --workdir "$WORKDIR" "$@"
  fi
}

_models_url() {
  local b="${VLLM_BASE_URL%/}"
  echo "${b}/models"
}

wait_for_vllm_ready() {
  # Collect all base URLs to probe: VLLM_BASE_URLS (comma-separated) takes
  # precedence over the single VLLM_BASE_URL, so all 4 servers are checked.
  local raw="${VLLM_BASE_URLS:-${VLLM_BASE_URL:-http://127.0.0.1:8010/v1}}"
  local -a urls=()
  IFS=',' read -ra urls <<< "$raw"

  local deadline=$((SECONDS + READY_MAX_WAIT))
  echo "[$(date +%H:%M:%S)] Waiting for ${#urls[@]} vLLM endpoint(s) to be ready..."
  while (( SECONDS < deadline )); do
    local all_up=1
    for base in "${urls[@]}"; do
      local probe="${base%/}/models"
      if ! curl -sf --max-time 10 "$probe" >/dev/null 2>&1; then
        all_up=0
        break
      fi
    done
    if (( all_up )); then
      echo "[$(date +%H:%M:%S)] All ${#urls[@]} vLLM endpoint(s) are up."
      return 0
    fi
    sleep "$READY_INTERVAL"
  done
  echo "Timed out after ${READY_MAX_WAIT}s waiting for vLLM endpoints." >&2
  return 1
}

stop_vllm() {
  if [[ -z "${VLLM_PID}" ]]; then
    return 0
  fi
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Stopping vLLM process group (pid ${VLLM_PID})..."
    # Kill the entire process group so vLLM worker subprocesses are also terminated.
    kill -- -"$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    # Belt-and-suspenders: kill any surviving vLLM worker processes.
    pkill -f "VLLM::Worker" 2>/dev/null || true
  fi
  VLLM_PID=""
}

trap 'stop_vllm' EXIT
trap 'stop_vllm; exit 130' INT
trap 'stop_vllm; exit 143' TERM

if [[ ! -f "$SERVE_SCRIPT" ]]; then
  echo "SERVE_SCRIPT not found: $SERVE_SCRIPT" >&2
  exit 1
fi

if [[ "$SKIP_SERVE" != "1" ]]; then
  if [[ -n "${CONDA_SERVE_ENV:-}" ]]; then
    echo "[$(date +%H:%M:%S)] Starting vLLM (conda env: ${CONDA_SERVE_ENV}): ${SERVE_SCRIPT}"
  else
    echo "[$(date +%H:%M:%S)] Starting: bash ${SERVE_SCRIPT}"
  fi
  _start_serve_bg
  VLLM_PID=$!
  wait_for_vllm_ready
fi

if [[ -n "${CONDA_GENERATE_ENV:-}" ]]; then
  echo "[$(date +%H:%M:%S)] Running generate in conda env: ${CONDA_GENERATE_ENV} (workdir=${WORKDIR}) ..."
else
  echo "[$(date +%H:%M:%S)] Running split_gpu_pipeline.py generate (workdir=${WORKDIR}) ..."
fi
_run_generate "$@"

echo "[$(date +%H:%M:%S)] Generate finished."
if [[ "$SKIP_SERVE" != "1" ]]; then
  echo "Next: bash sh/run_pipeline_auto.sh run  (or manually run split_verify_with_serve.sh)"
fi
echo "[$(date +%H:%M:%S)] Exiting; EXIT trap stops vLLM (GPUs free for verifier)."
