#!/usr/bin/env bash
# Orchestrate budget-matched uniform-parent rewriting baseline.
#
# Generator + verifier settings locked to MiniMax BENCHFILL method
# (TP=4 DP=2 PP=1 EP on 8× RTX PRO 6000 Blackwell).
#
# Default: ``generate`` starts MiniMax vLLM, runs rewrite, then kills vLLM
# on exit / Ctrl-C / disconnect (no orphaned GPU occupancy).
#
# Usage:
#   bash random_baseline/scripts/run_budget_match.sh all          # generate → verify → summary
#   bash random_baseline/scripts/run_budget_match.sh generate
#   SKIP_VLLM=1 ... generate   # use an already-running endpoint
#   bash .../run_budget_match.sh verify-all
#   bash .../run_budget_match.sh summary all
#
# Resume: re-run ``all`` anytime — finished gaps/models are skipped; partial
# generated.jsonl slots are kept. Log: random_baseline/results/_pipeline.log

set -euo pipefail

GAP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$GAP/random_baseline/scripts"
RESULTS="$GAP/random_baseline/results"
SPEC="$RESULTS/_budget_spec.json"
SETTINGS_JSON="$RESULTS/_method_settings.json"
export PYTHONPATH="${GAP}/propose${PYTHONPATH:+:$PYTHONPATH}"

# Optional machine overrides
if [[ -f "$GAP/env.local" ]]; then
    # shellcheck disable=SC1091
    source "$GAP/env.local"
fi

# ---------------------------------------------------------------------------
# Locked to BENCHFILL method parallel layout (8× RTX PRO 6000 Blackwell)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SERVE_SCRIPT="${SERVE_SCRIPT:-$GAP/propose/sh/serve_minimax_m2_7_8gpu.sh}"
export CONDA_SERVE_ENV="${CONDA_SERVE_ENV:-diff_minimax}"
export TP="${TP:-4}"
export DP="${DP:-2}"
export PP="${PP:-1}"
export ENABLE_EP="${ENABLE_EP:-1}"
export PORT="${PORT:-8010}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
export SERVED_NAME="${SERVED_NAME:-minimax-m2.7}"
export VLLM_MODEL="${VLLM_MODEL:-MiniMaxAI/MiniMax-M2.7}"

export MODEL="${MODEL:-minimax-m2.7}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export VLLM_BASE_URLS="${VLLM_BASE_URLS:-$VLLM_BASE_URL}"

export GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
export GEN_TOP_P="${GEN_TOP_P:-0.95}"
export GEN_TOP_K="${GEN_TOP_K:-40}"
export GEN_SYSTEM_PROMPT="${GEN_SYSTEM_PROMPT:-You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.}"
export PROPOSE_MAX_COMPLETION_TOKENS="${PROPOSE_MAX_COMPLETION_TOKENS:-30000}"
export BASELINE_GEN_WORKERS="${BASELINE_GEN_WORKERS:-16}"

# Auto vLLM lifecycle for ``generate`` / ``bench`` (set SKIP_VLLM=1 to disable)
export SKIP_VLLM="${SKIP_VLLM:-0}"
export READY_MAX_WAIT="${READY_MAX_WAIT:-3600}"
export READY_INTERVAL="${READY_INTERVAL:-5}"
export VLLM_LOG_FILE="${VLLM_LOG_FILE:-$RESULTS/_vllm_generate.log}"
export PIPELINE_LOG="${PIPELINE_LOG:-$RESULTS/_pipeline.log}"

VLLM_PGID=""

_log() {
    mkdir -p "$RESULTS"
    local msg="[$(date '+%F %T')] $*"
    echo "$msg" | tee -a "$PIPELINE_LOG"
}

_write_settings_json() {
    mkdir -p "$RESULTS"
    python3 - <<PY
import json, os
from pathlib import Path
out = Path(os.environ["SETTINGS_JSON"])
payload = {
  "locked_to": "method MiniMax vLLM (TP=4 DP=2 PP=1 EP); auto start/stop on generate",
  "hardware_note": "Runs used 8x NVIDIA RTX PRO 6000 Blackwell (~96GB each). Parallelism unchanged vs method.",
  "repo_root": os.environ["GAP"],
  "generator_vllm": {
    "script": "propose/sh/serve_minimax_m2_7_8gpu.sh",
    "conda_env": os.environ["CONDA_SERVE_ENV"],
    "lifecycle": "auto on generate (SKIP_VLLM=1 to reuse external)",
    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
    "TP": int(os.environ["TP"]),
    "DP": int(os.environ["DP"]),
    "PP": int(os.environ["PP"]),
    "ENABLE_EP": int(os.environ["ENABLE_EP"]),
    "PORT": int(os.environ["PORT"]),
    "MAX_MODEL_LEN": int(os.environ["MAX_MODEL_LEN"]),
    "GPU_MEM_UTIL": float(os.environ["GPU_MEM_UTIL"]),
    "VLLM_MODEL": os.environ["VLLM_MODEL"],
    "SERVED_NAME": os.environ["SERVED_NAME"],
  },
  "generator_client": {
    "model": os.environ["MODEL"],
    "base_urls": os.environ["VLLM_BASE_URLS"],
    "temperature": float(os.environ["GEN_TEMPERATURE"]),
    "top_p": float(os.environ["GEN_TOP_P"]),
    "top_k": int(os.environ["GEN_TOP_K"]),
    "system_prompt": os.environ["GEN_SYSTEM_PROMPT"],
    "max_tokens": int(os.environ["PROPOSE_MAX_COMPLETION_TOKENS"]),
    "max_workers": int(os.environ["BASELINE_GEN_WORKERS"]),
  },
  "verifier": {
    "script": "verifier/run_rollout.sh",
    "note": "Same wave/TP layout as method; n_sampling math=32 lmms=4",
  },
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[settings] wrote {out}")
PY
}

_print_settings() {
    echo "======== baseline = method settings ========"
    echo "  repo: $GAP"
    echo "  vLLM: $SERVE_SCRIPT  (auto start/stop unless SKIP_VLLM=1)"
    echo "  GPUs=$CUDA_VISIBLE_DEVICES  TP=$TP DP=$DP PP=$PP EP=$ENABLE_EP"
    echo "  model=$MODEL  url=$VLLM_BASE_URLS"
    echo "  temp=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K max_tokens=$PROPOSE_MAX_COMPLETION_TOKENS"
    echo "  verify: $GAP/verifier/run_rollout.sh"
    echo "============================================"
}

_conda_source() {
    local base="${CONDA_BASE:-}"
    if [[ -z "$base" ]] && command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
    fi
    if [[ -z "$base" || ! -f "$base/etc/profile.d/conda.sh" ]]; then
        echo "[vllm] need conda on PATH (or CONDA_BASE) for env $CONDA_SERVE_ENV" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh"
}

_models_url() {
    local b="${VLLM_BASE_URL%/}"
    echo "${b}/models"
}

_wait_vllm_ready() {
    local url deadline
    url="$(_models_url)"
    deadline=$((SECONDS + READY_MAX_WAIT))
    echo "[$(date +%H:%M:%S)] waiting for vLLM at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -sf --max-time 10 "$url" >/dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] vLLM ready."
            return 0
        fi
        # If the process group died early, fail fast
        if [[ -n "$VLLM_PGID" ]] && ! kill -0 "-$VLLM_PGID" 2>/dev/null; then
            echo "[vllm] process group $VLLM_PGID exited before ready; see $VLLM_LOG_FILE" >&2
            return 1
        fi
        sleep "$READY_INTERVAL"
    done
    echo "[vllm] timed out after ${READY_MAX_WAIT}s waiting for ${url}" >&2
    return 1
}

_stop_vllm() {
    if [[ -z "${VLLM_PGID}" ]]; then
        return 0
    fi
    echo "[$(date +%H:%M:%S)] stopping vLLM (pgid ${VLLM_PGID}) ..."
    # Kill whole process group (conda + vllm workers)
    kill -TERM "-$VLLM_PGID" 2>/dev/null || true
    local i=0
    while (( i < 60 )); do
        if ! kill -0 "-$VLLM_PGID" 2>/dev/null; then
            break
        fi
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "-$VLLM_PGID" 2>/dev/null; then
        echo "[vllm] still alive; kill -9"
        kill -KILL "-$VLLM_PGID" 2>/dev/null || true
    fi
    VLLM_PGID=""
    echo "[$(date +%H:%M:%S)] vLLM stopped (GPUs should be free)."
}

_start_vllm() {
    mkdir -p "$RESULTS"
    touch "$VLLM_LOG_FILE"
    echo "[$(date +%H:%M:%S)] starting MiniMax vLLM → log $VLLM_LOG_FILE"
    # New session so we can kill the whole group on exit / Ctrl-C / SIGHUP
    setsid bash -c '
        set -euo pipefail
        '"$(declare -f _conda_source)"'
        _conda_source
        conda activate "'"$CONDA_SERVE_ENV"'"
        exec bash "'"$SERVE_SCRIPT"'"
    ' >>"$VLLM_LOG_FILE" 2>&1 &
    VLLM_PGID=$!
    # setsid: $! is the session leader PID (== PGID)
    echo "[vllm] started pgid=$VLLM_PGID"
    _wait_vllm_ready
}

# Run generate client; optionally wrap with auto vLLM lifecycle.
_with_vllm() {
    if [[ "$SKIP_VLLM" == "1" ]]; then
        echo "[vllm] SKIP_VLLM=1 — expecting endpoint already up at $VLLM_BASE_URL"
        "$@"
        return
    fi
    # If something is already answering on the port, reuse it (don't double-load).
    if curl -sf --max-time 3 "$(_models_url)" >/dev/null 2>&1; then
        echo "[vllm] endpoint already up at $VLLM_BASE_URL — reusing (will NOT kill on exit)"
        "$@"
        return
    fi
    trap '_stop_vllm' EXIT
    trap '_stop_vllm; exit 130' INT
    trap '_stop_vllm; exit 143' TERM
    trap '_stop_vllm; exit 129' HUP
    _start_vllm
    local rc=0
    "$@" || rc=$?
    _stop_vllm
    trap - EXIT INT TERM HUP
    return "$rc"
}

export GAP SETTINGS_JSON

CMD="${1:?Usage: $0 all|settings|generate|verify|verify-all|summary|bench|serve ...}"
shift || true

_py() {
    if command -v conda >/dev/null 2>&1; then
        conda run -n diff --no-capture-output python "$@"
    else
        python3 "$@"
    fi
}

_run_generate_client() {
    export PROPOSE_MAX_COMPLETION_TOKENS
    # Array keeps spaces inside --system-prompt intact (do NOT expand via $()).
    local -a gen_args=(
        --base-url "$VLLM_BASE_URLS"
        --model "$MODEL"
        --temperature "$GEN_TEMPERATURE"
        --top-p "$GEN_TOP_P"
        --top-k "$GEN_TOP_K"
        --system-prompt "$GEN_SYSTEM_PROMPT"
        --max-tokens "$PROPOSE_MAX_COMPLETION_TOKENS"
        --max-workers "$BASELINE_GEN_WORKERS"
    )
    if [[ $# -ge 2 && "$1" != --* ]]; then
        _py "$SCRIPTS/generate_uniform_rewrite.py" --spec "$SPEC" \
            --benchmark "$1" --gap-rank "$2" "${gen_args[@]}" "${@:3}"
    else
        _py "$SCRIPTS/generate_uniform_rewrite.py" --spec "$SPEC" --all "${gen_args[@]}" "$@"
    fi
}

case "$CMD" in
    all|run)
        # One-shot: generate (auto vLLM) → verify-all → summary. Safe to re-run.
        _write_settings_json
        _print_settings
        _log "========== PIPELINE START (all) =========="
        _log "phase=generate"
        if [[ ! -f "$SPEC" ]]; then
            _py "$SCRIPTS/build_budget_spec.py" --out "$SPEC"
        fi
        _with_vllm _run_generate_client "$@"
        _log "phase=generate done; stopping vLLM before verify"
        # EXIT trap from _with_vllm already stopped vLLM when that function returned.
        _log "phase=verify-all"
        bash "$0" verify-all
        _log "phase=summary"
        bash "$0" summary all
        _log "========== PIPELINE DONE =========="
        echo
        echo "Artifacts:"
        echo "  generated:  random_baseline/results/<bench>/gap_rank_*/generated.jsonl"
        echo "  difficulty: random_baseline/results/<bench>/difficulty/*_difficulty.csv"
        echo "  summary:    random_baseline/results/<bench>/*_baseline_summary.csv"
        echo "  logs:       $PIPELINE_LOG  |  $VLLM_LOG_FILE"
        ;;
    settings)
        _write_settings_json
        _print_settings
        ;;
    serve)
        # Optional debug only — prefer ``generate`` / ``all`` (auto start/stop).
        _write_settings_json
        _print_settings
        echo "[serve] DEBUG: long-lived vLLM. Prefer: bash $0 all"
        echo "[serve] launching $SERVE_SCRIPT (conda=$CONDA_SERVE_ENV)…"
        if command -v conda >/dev/null 2>&1; then
            exec conda run -n "$CONDA_SERVE_ENV" --no-capture-output bash "$SERVE_SCRIPT"
        else
            exec bash "$SERVE_SCRIPT"
        fi
        ;;
    spec)
        _write_settings_json
        _py "$SCRIPTS/build_budget_spec.py" --out "$SPEC"
        ;;
    generate)
        _write_settings_json
        _print_settings
        if [[ ! -f "$SPEC" ]]; then
            echo "[run] building spec first…"
            _py "$SCRIPTS/build_budget_spec.py" --out "$SPEC"
        fi
        _log "phase=generate start"
        _with_vllm _run_generate_client "$@"
        _log "phase=generate done"
        ;;
    verify)
        BENCH="${1:?verify <benchmark> [gap_list]}"
        ONLY="${2:-}"
        _print_settings
        if [[ -n "$ONLY" ]]; then
            bash "$SCRIPTS/verify_random_batch.sh" "$BENCH" "$ONLY"
        else
            GAPS="$(cd "$RESULTS/$BENCH" 2>/dev/null && for d in gap_rank_*/; do
                [[ -f "$d/generated.jsonl" ]] && echo "${d#gap_rank_}" | tr -d '/'
            done | sort -n | paste -sd, -)"
            if [[ -z "$GAPS" ]]; then
                echo "no generated.jsonl under $RESULTS/$BENCH"
                exit 1
            fi
            bash "$SCRIPTS/verify_random_batch.sh" "$BENCH" "$GAPS"
        fi
        ;;
    verify-all)
        for b in aime24 aime25 aime26 amc23 gaokao2023en mmlu_bio ifeval bbh_3obj; do
            if [[ -d "$RESULTS/$b" ]] && ls "$RESULTS/$b"/gap_rank_*/generated.jsonl >/dev/null 2>&1; then
                echo "======== verify $b ========"
                _log "verify $b"
                bash "$0" verify "$b"
            else
                echo "[skip] $b — no generated.jsonl yet"
            fi
        done
        ;;
    summary)
        BENCH="${1:?summary <benchmark>|all}"
        if [[ "$BENCH" == "all" ]]; then
            _py "$SCRIPTS/build_summary.py" \
                aime24 aime25 aime26 amc23 gaokao2023en mmlu_bio ifeval bbh_3obj
        else
            _py "$SCRIPTS/build_summary.py" "$BENCH"
        fi
        ;;
    bench)
        BENCH="${1:?bench <benchmark>}"
        _write_settings_json
        if [[ ! -f "$SPEC" ]]; then
            _py "$SCRIPTS/build_budget_spec.py" --out "$SPEC"
        fi
        mapfile -t GAP_LIST < <(_py -c "
import json
spec=json.load(open('$SPEC'))
print('\\n'.join(str(g['gap_rank']) for g in spec['gaps'] if g['benchmark']=='$BENCH'))
")
        if [[ ${#GAP_LIST[@]} -eq 0 ]]; then
            echo "no gaps for $BENCH in spec"; exit 1
        fi
        _log "bench=$BENCH generate+verify"
        # One vLLM lifecycle for the whole bench, then verify (needs free GPUs).
        _with_vllm bash -c '
            set -euo pipefail
            for g in '"${GAP_LIST[*]}"'; do
                echo "======== generate '"$BENCH"' gap $g ========"
                SKIP_VLLM=1 bash "'"$0"'" generate "'"$BENCH"'" "$g"
            done
        '
        echo "======== verify $BENCH ========"
        bash "$0" verify "$BENCH"
        bash "$0" summary "$BENCH"
        _log "bench=$BENCH done"
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: $0 all|generate|verify|verify-all|summary|bench|settings [serve]"
        echo "  all       — generate → verify-all → summary (one command; resume-safe)"
        echo "  generate  — start MiniMax vLLM, rewrite, stop vLLM"
        echo "  SKIP_VLLM=1 generate — only client (endpoint already up)"
        exit 1
        ;;
esac
