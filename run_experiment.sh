#!/usr/bin/env bash
# Gap-filling experiment: generate → verify (wave-by-wave rollout) → repeat.
#
# Scope: **math** benchmarks (diff math_eval / verifier math framework). lmms-eval
# benchmarks are not exercised by this driver; quantized vLLM settings (GPUs 4–7,
# max_model_len, etc.) are left for a later pass.
#
# Multi-gap: by default runs gap_rank 1..NUM_GAPS (widest → N-th widest). Checkpoint
# files under $SPLIT_WORKDIR: gap_run_progress.json (checkpoint) and
# gap_fill_outcomes.csv (one row per finished gap: gap widths, target, new item 0–10, …).
# Override gap list with GAP_RANKS="1 2 4".
#
# Copy-paste this block into a shell where conda is on PATH and the `diff` env is active:
#
#   conda activate diff
#   bash run_experiment.sh
#
# Or run init and run separately (e.g. to inspect state between steps):
#
#   bash run_experiment.sh init      # fresh state only (new run_id)
#   bash run_experiment.sh run       # resume in-progress; skips if already done
#   bash run_experiment.sh all       # init then run (fresh experiment)
#   bash run_experiment.sh restart   # same as all (re-run after a finished gap)
#
# Self-contained repo: clone anywhere, then:
#   cp env.example env.local && source env.local
#   bash scripts/setup_server.sh
#   vLLM server:  $GAP_ROOT/propose/results/e2e_split/vllm_serve.log
#   Generation:   $GAP_ROOT/propose/results/e2e_split/generate.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

BENCHMARK="mmlu_bio"
# Proposal/generation model only (verifier rollouts unchanged — still use diff / run_rollout.sh).
MODEL="minimax-m2.7"             # --served-model-name in propose/sh/serve_minimax_m2_7.sh
# MiniMax-M2.7 card inference defaults (passed to split_gpu_pipeline init → state.json → API)
GEN_TEMPERATURE=1.0
GEN_TOP_P=0.95
GEN_TOP_K=40
GEN_SYSTEM_PROMPT='You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.'
MAX_ROUNDS=8
# Rollout repeats per candidate (passed to verifier as --n). Match IRT calibration n_trials (32 for math, 4 for lmms MCQ/ifeval).
N_SAMPLING=4
K_PARENTS=4                       # K: parent nodes selected per round (after round 0)
N_CHILDREN=4                      # N: child items generated per parent
MEMORY_SIZE=8                     # M: memory list size (M/2 per side of target difficulty)
EARLY_STOP_TOLERANCE=0.15        # round-best |Δ| < this triggers early stop (12-model 0–10). n=4 (lmms) needs ≳0.25; n=32 (math) can use 0.1–0.15.

# Gaps to fill (1 = widest). Default: gap ranks 1 .. NUM_GAPS. Set GAP_RANKS explicitly
# to override (e.g. GAP_RANKS="1 2 4"). After each gap completes (state status=done),
# propose/scripts/record_gap_progress.py updates gap_run_progress.json under SPLIT_WORKDIR.
NUM_GAPS="${NUM_GAPS:-3}"
GAP_RANKS="${GAP_RANKS:-1 2 3}"   # mmlu_bio: gap1 q93↔q157 (≈0.22), gap2 q315↔q705 (≈0.67), gap3 q160↔q342 (≈1.08)
if [[ -z "$GAP_RANKS" ]]; then
  GAP_RANKS="$(seq 1 "$NUM_GAPS")"
fi

# Two-stage pipeline (scratch → modify). Set PIPELINE_MODE=direct for legacy behaviour.
PIPELINE_MODE="direct"         # "direct" | "two_stage"
STAGE1_TOLERANCE=2.0              # stage-1 scratch-generation tolerance (0–10 scale)
STAGE1_MIN_PASS=4                 # cumulative stage-1 passes required to advance to stage 2
STAGE1_MAX_ATTEMPTS=3             # stage-1 attempt cap before hard-failing

# After top-32 collected, auto-run 12-model full verifier and write
# final_top32_full.jsonl + final_accepted.jsonl. Set 0 to skip and verify manually later.
export VERIFY_TOP32_WITH_FULL=1

# ---------------------------------------------------------------------------
# Paths (repo root = this directory; override with REPO_ROOT env)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
export GAP_ROOT="${GAP_ROOT:-$REPO_ROOT}"
export SPLIT_WORKDIR=$GAP_ROOT/propose/results/e2e_split_${BENCHMARK}
# Guardian + generate wrapper. MiniMax-M2.7: single vLLM DP=2 × TP=4 × EP on GPUs 0–7 @8010.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SERVE_SCRIPT=$GAP_ROOT/propose/sh/serve_minimax_m2_7_8gpu.sh
export VLLM_BASE_URL=http://127.0.0.1:8010/v1
export VLLM_LOG_FILE=$SPLIT_WORKDIR/vllm_serve.log
export GENERATE_LOG_FILE=$SPLIT_WORKDIR/generate.log
export CONDA_SERVE_ENV=diff_minimax
export VLLM_BASE_URLS="http://127.0.0.1:8010/v1"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

mkdir -p "$SPLIT_WORKDIR"

echo "[run_experiment] benchmark=$BENCHMARK  NUM_GAPS=$NUM_GAPS  GAP_RANKS=$GAP_RANKS"

cmd="${1:-all}"

_init_one_gap() {
    local rank="$1"
    echo "===== INIT (gap_rank=$rank) ====="
    bash $GAP_ROOT/sh/run_pipeline_auto.sh init \
        --benchmark "$BENCHMARK" \
        --model     "$MODEL" \
        --temperature "$GEN_TEMPERATURE" \
        --top-p "$GEN_TOP_P" \
        --top-k "$GEN_TOP_K" \
        --system-prompt "$GEN_SYSTEM_PROMPT" \
        --base-url  "$VLLM_BASE_URLS" \
        --max-rounds "$MAX_ROUNDS" \
        --n-sampling "$N_SAMPLING" \
        --k-parents  "$K_PARENTS" \
        --n-children "$N_CHILDREN" \
        --memory-size "$MEMORY_SIZE" \
        --early-stop-tolerance "$EARLY_STOP_TOLERANCE" \
        --pipeline-mode "$PIPELINE_MODE" \
        --stage1-tolerance "$STAGE1_TOLERANCE" \
        --stage1-min-pass "$STAGE1_MIN_PASS" \
        --stage1-max-attempts "$STAGE1_MAX_ATTEMPTS" \
        --gap-rank "$rank"
}

_sync_state_from_script() {
    # Keep state.json aligned with run_experiment.sh knobs without a full re-init.
    local st="$SPLIT_WORKDIR/state.json"
    [[ -f "$st" ]] || return 0
    python3 - "$st" "$N_SAMPLING" "$GEN_TEMPERATURE" "$GEN_TOP_P" "$GEN_TOP_K" "$GEN_SYSTEM_PROMPT" <<'PY'
import json, sys
path = sys.argv[1]
want_n = int(sys.argv[2])
want_temp = float(sys.argv[3])
want_top_p = float(sys.argv[4])
want_top_k = int(sys.argv[5])
want_sys = sys.argv[6]
data = json.load(open(path, encoding="utf-8"))
changed = []
if int(data.get("n_sampling", -1)) != want_n:
    data["n_sampling"] = want_n
    changed.append(f"n_sampling -> {want_n}")
gen = data.setdefault("generator", {})
for key, val in (
    ("temperature", want_temp),
    ("top_p", want_top_p),
    ("top_k", want_top_k),
    ("system_prompt", want_sys),
):
    if gen.get(key) != val:
        gen[key] = val
        changed.append(f"generator.{key}")
if changed:
    open(path, "w", encoding="utf-8").write(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[run_experiment] state.json synced: {', '.join(changed)}", flush=True)
PY
}

_run_one_gap() {
    local rank="$1"
    echo "===== RUN (gap_rank=$rank) ====="
    _sync_state_from_script
    bash $GAP_ROOT/sh/run_pipeline_auto.sh run
    # Checkpoint (no-op if state missing or not done)
    python3 "$GAP_ROOT/propose/scripts/record_gap_progress.py" \
        --workdir "$SPLIT_WORKDIR" --benchmark "$BENCHMARK"
}

if [[ "$cmd" == "init" || "$cmd" == "all" || "$cmd" == "restart" ]]; then
    # Prepare state for every rank in GAP_RANKS (widest → N-th). Each init overwrites
    # state.json, so we snapshot per rank, then restore the first rank for `run`.
    _init_gap_need_sleep=""
    for rank in $GAP_RANKS; do
        # split_gpu_pipeline run_id uses second-resolution time; stagger so each
        # gap gets its own run_dir and anchor files.
        if [[ -n "$_init_gap_need_sleep" ]]; then sleep 1; fi
        _init_gap_need_sleep=1
        _init_one_gap "$rank"
        cp -f "$SPLIT_WORKDIR/state.json" "$SPLIT_WORKDIR/state.prepared.gap_rank_${rank}.json"
        echo "[run_experiment] snapshot → state.prepared.gap_rank_${rank}.json"
    done
    # First rank = first word of GAP_RANKS (not head -n1: "1 2 3" is one line).
    read -r first_rank _ <<< "$GAP_RANKS"
    first_rank="${first_rank:-1}"
    cp -f "$SPLIT_WORKDIR/state.prepared.gap_rank_${first_rank}.json" "$SPLIT_WORKDIR/state.json"
    echo "[run_experiment] active state.json restored to gap_rank=${first_rank} (run starts here)"
fi

if [[ "$cmd" == "run" || "$cmd" == "all" || "$cmd" == "restart" ]]; then
    # Resuming an in-progress run: state.json may already exist. Detect status=done
    # → bump to next rank. Status not-done → continue current rank.
    state_file="$SPLIT_WORKDIR/state.json"
    cur_rank=""
    cur_status="no_state"
    if [[ -f "$state_file" ]]; then
        cur_rank="$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('gap_rank',1))" 2>/dev/null || echo "")"
        cur_status="$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('status','unknown'))" 2>/dev/null || echo "no_state")"
    fi

    for rank in $GAP_RANKS; do
        # Skip ranks already finished. We treat status=done as "this rank is finished";
        # if cur_rank > rank already, we passed this one in a previous invocation.
        if [[ "$cur_status" == "done" && -n "$cur_rank" && "$cur_rank" -gt "$rank" ]]; then
            echo "===== Skipping gap_rank=$rank (state shows we are past it) ====="
            continue
        fi
        if [[ "$cur_status" == "done" && "$cur_rank" == "$rank" ]]; then
            echo "===== gap_rank=$rank already done, advancing to next ====="
            continue
        fi

        # Need a fresh state for this rank if state.json is missing or done.
        if [[ "$cur_status" == "no_state" || "$cur_status" == "done" || "$cur_rank" != "$rank" ]]; then
            _init_one_gap "$rank"
        else
            echo "===== Resuming gap_rank=$rank from status=$cur_status ====="
        fi
        _run_one_gap "$rank"

        # After this rank finishes, refresh status for the loop's next iteration
        # so the "skip if done" branch doesn't re-init.
        if [[ -f "$state_file" ]]; then
            cur_rank="$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('gap_rank',1))" 2>/dev/null || echo "")"
            cur_status="$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('status','unknown'))" 2>/dev/null || echo "no_state")"
        fi
    done

    echo ""
    echo "===== ALL GAPS DONE ====="
    echo "Results: $SPLIT_WORKDIR/state.json"
fi

if [[ "$cmd" == "status" ]]; then
    bash $GAP_ROOT/sh/run_pipeline_auto.sh status
fi
