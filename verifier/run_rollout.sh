#!/bin/bash
# Math / lmms verifier: vLLM rollout → verifier.main (IRT difficulty).
#
# Usage (from benchfill repo root):
#   bash verifier/run_rollout.sh --benchmark aime24 --n 32 --framework math
#
# GPUs: 8× NVIDIA RTX PRO 6000 Blackwell (~96GB each), devices 0–7 via MATH_CUDA_DEVICES.
#   Dense 8-GPU layout; Waves 3/4 run Qwen3.5-397B-A17B / GLM-4.6 at TP=8.
#
# Candidates:
#   verifier/math_candidates/<benchmark>/test.jsonl
#   verifier/lmms_candidates/<benchmark>.jsonl
#
# --framework lmms: same 12-model panel as math (mmlu_bio / ifeval / bbh_3obj).

set -e

export NCCL_NET_PLUGIN=none      # do not load any net plugin
export NCCL_NET=Socket           # force socket transport (enough for single-node TP=8)
export NCCL_DEBUG=WARN           # warnings/errors only (set back to INFO when debugging to see channel setup details)
export NCCL_P2P_DISABLE=0        # keep intra-node P2P

# Persistent JIT / compile caches: on later launches of Qwen3-Next, glm-4.5-air, olmo-32b, qwq-32b,
# Triton kernels (DeltaNet / FlashAttn / FusedMoE) and vLLM tuning results hit the cache,
# skipping 5-15 min of kernel compilation. The first run fills the cache; later runs benefit. Delete it to re-warm.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Script dir = <repo root>/verifier/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/env.local" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env.local"
fi

BENCHMARK=""
# Rollouts per item per model (--n). Default 8; math benchmarks use n_trials=32 in verifier/config.py (IRT) — pass --n 32 to match irt_results/.
N_SAMPLING=16
RESULTS_BASE="$SCRIPT_DIR/results"
RESULTS_DIR=""  # set below once we know the benchmark + run id
FRAMEWORK="math"   # "math" or "lmms"
ANCHORS=""
DIFFICULTY_OUT=""   # if set, pass --out <path> to verifier.main
MATH_CANDIDATES_DIR="$SCRIPT_DIR/math_candidates"
DIFF_EVAL_DIR="$SCRIPT_DIR/diff_eval"
# All rollout models except Qwen3-Next-80B use this env (unchanged from your original setup).
MATH_CONDA_ENV="diff"
# Qwen3-Next-80B wave only: separate conda env from ``diff`` (see requirements_diff_qwen_next.txt).
# Default ``diff_v``. Override: MATH_CONDA_ENV_QWEN3_NEXT=diff bash ...
MATH_CONDA_ENV_QWEN3_NEXT="${MATH_CONDA_ENV_QWEN3_NEXT:-diff_v}"
IRT_CONDA_ENV="diff"  # scipy/numpy/pandas for verifier.main
# lmms-eval lives in vendor/ after setup_server.sh, or a machine-local clone.
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-}"
LMMS_TASKS_DIR=""  # runtime-patched copy of verifier/tasks (lmms only)

# Math rollout GPU pool (physical indices). Default: the eight RTX PRO 6000 Blackwell cards, 0–7.
# Math GPU aliases (physical ids from MATH_CUDA_DEVICES)
MATH_CUDA_DEVICES="${MATH_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _MATH_GPU_LIST <<< "$MATH_CUDA_DEVICES"
MATH_G0="${_MATH_GPU_LIST[0]:-0}"
MATH_G1="${_MATH_GPU_LIST[1]:-1}"
MATH_G2="${_MATH_GPU_LIST[2]:-2}"
MATH_G3="${_MATH_GPU_LIST[3]:-3}"
MATH_G4="${_MATH_GPU_LIST[4]:-4}"
MATH_G5="${_MATH_GPU_LIST[5]:-5}"
MATH_G6="${_MATH_GPU_LIST[6]:-6}"
MATH_G7="${_MATH_GPU_LIST[7]:-7}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark) BENCHMARK="$2"; shift 2;;
        --n) N_SAMPLING="$2"; shift 2;;
        --out) RESULTS_DIR="$2"; shift 2;;
        --difficulty-out) DIFFICULTY_OUT="$2"; shift 2;;
        --framework) FRAMEWORK="$2"; shift 2;;
        --anchors) ANCHORS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$BENCHMARK" ]]; then
    echo "Usage: from the repo root: bash verifier/run_rollout.sh --benchmark <name> [--n N] [--framework math]"
    exit 1
fi

# Each invocation gets its own results + logs directory so concurrent /
# successive runs never mix. Override with --out to pin a specific path.
RUN_ID="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="$RESULTS_BASE/${BENCHMARK}_${RUN_ID}"
fi

# Map benchmark → data/task name passed to the runner
if [[ "$FRAMEWORK" == "lmms" ]]; then
    DATA_NAMES="${BENCHMARK}_verifier"
else
    DATA_NAMES="$BENCHMARK"
fi

_resolve_lmms_eval_repo() {
    local c
    for c in \
        "${LMMS_EVAL_REPO:-}" \
        "$REPO_ROOT/vendor/lmms-eval" \
        "${HOME}/lmms-eval"
    do
        [[ -z "$c" ]] && continue
        if [[ -d "$c/lmms_eval" ]]; then
            LMMS_EVAL_REPO="$c"
            return 0
        fi
    done
    echo "ERROR: lmms-eval not found. Set LMMS_EVAL_REPO or run bash scripts/setup_server.sh" >&2
    exit 1
}

# verifier/tasks/*.yaml keep git-friendly placeholders (__REPO_ROOT__, __LMMS_EVAL_REPO__).
# lmms-eval TaskManager scans the whole include_path and follows every include:, so
# an unpatched bbh yaml crashes mmlu_bio / ifeval too. Materialize a patched copy
# per run; never rewrite the tracked yaml files.
_prepare_lmms_tasks() {
    _resolve_lmms_eval_repo
    if [[ ! -x "$LMMS_EVAL_REPO/.venv/bin/python" ]]; then
        echo "ERROR: missing $LMMS_EVAL_REPO/.venv/bin/python (cd there && uv sync)" >&2
        exit 1
    fi
    LMMS_TASKS_DIR="$RESULTS_DIR/_tasks_patched"
    rm -rf "$LMMS_TASKS_DIR"
    mkdir -p "$LMMS_TASKS_DIR"
    cp -a "$SCRIPT_DIR/tasks/." "$LMMS_TASKS_DIR/"
    REPO_ROOT="$REPO_ROOT" LMMS_EVAL_REPO="$LMMS_EVAL_REPO" LMMS_TASKS_DIR="$LMMS_TASKS_DIR" python3 <<'PY'
import os
from pathlib import Path
repo = os.environ["REPO_ROOT"]
lmms = os.environ["LMMS_EVAL_REPO"]
root = Path(os.environ["LMMS_TASKS_DIR"])
n = 0
for path in root.rglob("*.yaml"):
    text = path.read_text(encoding="utf-8")
    new = text.replace("__REPO_ROOT__", repo).replace("__LMMS_EVAL_REPO__", lmms)
    if new != text:
        path.write_text(new, encoding="utf-8")
        n += 1
print(f"[lmms] patched {n} yaml(s) → {root}")
print(f"[lmms] LMMS_EVAL_REPO={lmms}")
PY
    if grep -rqE '__REPO_ROOT__|__LMMS_EVAL_REPO__' "$LMMS_TASKS_DIR" 2>/dev/null; then
        echo "ERROR: yaml placeholders remain under $LMMS_TASKS_DIR" >&2
        grep -RnE '__REPO_ROOT__|__LMMS_EVAL_REPO__' "$LMMS_TASKS_DIR" >&2 || true
        exit 1
    fi
}

if [[ "$FRAMEWORK" == "lmms" ]]; then
    _prepare_lmms_tasks
fi

LOGDIR="$RESULTS_DIR/logs"
mkdir -p "$LOGDIR" "$RESULTS_DIR"
echo "Run ID: $RUN_ID"
echo "Results dir: $RESULTS_DIR"
echo "Logs dir:    $LOGDIR"

LAST_PID=""
PIDS=()

wait_all() {
    echo "[$(date +%H:%M:%S)] Waiting for wave to finish..."
    wait "${PIDS[@]}"
    echo "[$(date +%H:%M:%S)] Wave done."
    PIDS=()
}

# Skip models whose rollout jsonl already exists (resume after a hung wave).
_math_rollout_done() {
    local name="$1"
    compgen -G "$RESULTS_DIR/$name/$DATA_NAMES/*.jsonl" > /dev/null
}

run_math_model_in_env() {
    local conda_env="$1"; shift
    local name="$1" model_path="$2" gpus="$3" prompt_type="$4" temp="$5" top_p="$6" max_len="$7" max_tokens="$8" extra="${9:-}" extra_env="${10:-}"
    if _math_rollout_done "$name"; then
        echo "[$(date +%H:%M:%S)] Skip $name — output exists: $RESULTS_DIR/$name/$DATA_NAMES/*.jsonl"
        ( : ) &
        LAST_PID=$!
        return 0
    fi
    local log="$LOGDIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date +%H:%M:%S)] Starting $name on GPUs=$gpus conda=$conda_env (max_model_len=$max_len, max_tokens=$max_tokens)"
    cd "$DIFF_EVAL_DIR"
    local -a _model_env=(CUDA_VISIBLE_DEVICES="$gpus" TOKENIZERS_PARALLELISM=false)
    if [[ -n "$extra_env" ]]; then
        _model_env+=("$extra_env")
        echo "  extra env: $extra_env"
    fi
    env "${_model_env[@]}" \
        conda run --no-capture-output -n "$conda_env" \
        python3 -u math_eval_general.py \
        --model_name_or_path "$model_path" \
        --data_names "$DATA_NAMES" \
        --data_dir "$MATH_CANDIDATES_DIR" \
        --n_sampling "$N_SAMPLING" \
        --output_dir "$RESULTS_DIR/$name" \
        --split test --prompt_type "$prompt_type" \
        --num_test_sample -1 --seed 42 \
        --temperature "$temp" --top_p "$top_p" \
        --max_tokens_per_call "$max_tokens" --max_model_len "$max_len" \
        --start 0 --end -1 \
        --use_vllm --save_outputs --overwrite --num_shots 0 \
        $extra \
        > "$log" 2>&1 &
    LAST_PID=$!
    echo "  PID=$LAST_PID → $log"
    cd "$REPO_ROOT"
}

run_math_model() {
    run_math_model_in_env "$MATH_CONDA_ENV" "$@"
}


run_math() {
    # Output dir names are the IRT model names (see irt_results/{bm}_ability.csv)
    # so verifier.main can match rollout dirs ↔ theta rows directly.
    #
    # --- dense 8-GPU layout (8× RTX PRO 6000 Blackwell, ~96GB each) ---
    # Wave 1 = old 1A+1B (8/8). Wave 2 = old 1C+glm-air TP=4 (8/8).
    # Waves 3–4 stay solo TP=8 (397B / GLM-4.6).
    echo "[run_rollout] dense 8-GPU layout: merge 1A+1B and 1C+glm-air onto 8 GPUs"
    echo "========== Wave 1: 4 small + phi + gpt-oss + next-80B (8/8) =========="
    run_math_model "smollm2-1.7b-instruct"  "HuggingFaceTB/SmolLM2-1.7B-Instruct" "$MATH_G0" "qwen25-math-cot"  0.7 0.8 8192 8192 "--apply_chat_template"; PIDS+=("$LAST_PID")
    run_math_model "vicuna-7b-v1.5"         "lmsys/vicuna-7b-v1.5"                "$MATH_G1" "llama31-math-cot" 0.6 0.9 4096 4096; PIDS+=("$LAST_PID")
    run_math_model "llama-3.2-3b-instruct"  "meta-llama/Llama-3.2-3B-Instruct"    "$MATH_G2" "llama31-math-cot" 0.6 0.9 8192 8192 "--apply_chat_template"; PIDS+=("$LAST_PID")
    run_math_model "llama-3.1-8b"           "meta-llama/Llama-3.1-8B-Instruct"    "$MATH_G3" "llama31-math-cot" 0.6 0.9 8192 8192 "--apply_chat_template"; PIDS+=("$LAST_PID")
    run_math_model "phi-4"       "microsoft/phi-4"        "$MATH_G4" "phi4-math-cot"     0.7 0.8 16384 10240 "--apply_chat_template"; PIDS+=("$LAST_PID")
    run_math_model "gpt-oss-20b" "openai/gpt-oss-20b"     "$MATH_G5" "gpt-oss-math-cot" 0.7 0.8 16384 16384 "--apply_chat_template"; PIDS+=("$LAST_PID")
    run_math_model "qwen3-next-80b-a3b-instruct" "Qwen/Qwen3-Next-80B-A3B-Instruct" "${MATH_G6},${MATH_G7}" "qwen25-math-cot" 0.7 0.8 16384 10240 "--apply_chat_template --tensor_parallel_size 2"; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 2: olmo TP=2 + qwq TP=2 + glm-4.5-air TP=4 (8/8) =========="
    run_math_model "olmo-3.1-32b-think" "allenai/Olmo-3.1-32B-Think" "${MATH_G0},${MATH_G1}" "olmo-math-cot"   0.6 0.95 32768 10240 "--apply_chat_template --tensor_parallel_size 2"; PIDS+=("$LAST_PID")
    run_math_model "qwq-32b" "Qwen/QwQ-32B" "${MATH_G2},${MATH_G3}" "qwen25-math-cot" 0.6 0.95 32768 10240 "--apply_chat_template --tensor_parallel_size 2"; PIDS+=("$LAST_PID")
    run_math_model "glm-4.5-air" "zai-org/GLM-4.5-Air" "${MATH_G4},${MATH_G5},${MATH_G6},${MATH_G7}" "glm-math-cot" 0.6 0.95 32768 10240 "--apply_chat_template --tensor_parallel_size 4"; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 3: Qwen3.5-397B-A17B FP8 (TP=8) =========="
    run_math_model "qwen3.5-397b-a17b" "Qwen/Qwen3.5-397B-A17B-FP8" "$MATH_CUDA_DEVICES" "qwen25-math-cot" 0.7 0.8 32768 10240 "--apply_chat_template --quantization fp8 --tensor_parallel_size 8 --attention_backend TRITON_ATTN --no-enable_prefix_caching"; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 4: glm-4.6 AWQ (TP=8) =========="
    run_math_model "glm-4.6" "bullpoint/GLM-4.6-AWQ" "$MATH_CUDA_DEVICES" "glm-math-cot" 0.6 0.95 65536 32768 "--apply_chat_template --tensor_parallel_size 8 --enable_expert_parallel"; PIDS+=("$LAST_PID")
    wait_all
}

# =====================================================================
# --framework lmms: same 12-model panel as run_math / irt_results ability CSVs.
# =====================================================================
_lmms_rollout_done() {
    # lmms-eval writes under HF-style dirs: org__ModelName/*samples*.jsonl
    local model_path="$1"
    local hf_dir="${model_path//\//__}"
    compgen -G "$RESULTS_DIR/$hf_dir/*samples*.jsonl" > /dev/null
}

run_lmms_model() {
    # Signature: name, model_path, gpus, tp, max_model_len, max_num_seqs,
    #            max_new_tokens, gpu_mem_util, extra_model_args(csv), system_instruction
    local name="$1" model_path="$2" gpus="$3" tp="$4" max_len="$5"
    local max_num_seqs="$6" max_new_tokens="$7" gpu_mem="${8:-0.85}"
    local extra_model_args="${9:-}" system_instruction="${10:-}"
    if _lmms_rollout_done "$model_path"; then
        echo "[$(date +%H:%M:%S)] Skip $name — output exists: $RESULTS_DIR/${model_path//\//__}/*samples*.jsonl"
        ( : ) &
        LAST_PID=$!
        return 0
    fi
    local log="$LOGDIR/lmms_${name}_$(date +%Y%m%d_%H%M%S).log"
    local model_args="model=${model_path},tensor_parallel_size=${tp},gpu_memory_utilization=${gpu_mem},max_model_len=${max_len},max_num_seqs=${max_num_seqs},enforce_eager=True"
    if [[ -n "$extra_model_args" ]]; then
        model_args="${model_args},${extra_model_args}"
    fi
    local cli_args=(
        --model vllm
        --model_args "$model_args"
        --tasks "$DATA_NAMES"
        --include_path "${LMMS_TASKS_DIR:-$SCRIPT_DIR/tasks}"
        --batch_size 256
        --output_path "$RESULTS_DIR"
        --log_samples
        --gen_kwargs "temperature=0.6,top_k=20,top_p=0.95,do_sample=True,max_new_tokens=${max_new_tokens}"
        --repeats "$N_SAMPLING"
        --seed 42
        --verbosity INFO
    )
    if [[ -n "$system_instruction" ]]; then
        cli_args+=(--system_instruction "$system_instruction")
    fi
    echo "[$(date +%H:%M:%S)] Starting $name on GPUs=$gpus (TP=$tp, max_model_len=$max_len, max_new_tokens=$max_new_tokens, repeats=$N_SAMPLING)"
    cd "$LMMS_EVAL_REPO"
    env CUDA_VISIBLE_DEVICES="$gpus" \
        "$LMMS_EVAL_REPO/.venv/bin/python" -m lmms_eval "${cli_args[@]}" \
        > "$log" 2>&1 &
    LAST_PID=$!
    echo "  PID=$LAST_PID → $log"
}

run_lmms() {
    # Same 12-model panel as run_math. Dense 8-GPU pack (mirrors math dense layout).
    # Wave 1 = old 1A+1B. Wave 2 = old 1C + glm-air TP=4. Waves 3–4 solo big models.
    local big_tp=8
    local big_gpus="$MATH_CUDA_DEVICES"
    echo "[run_lmms] dense 8-GPU layout (397B/GLM-4.6 TP=8)"

    echo "========== Wave 1: 4 small + phi + gpt-oss + next-80B (8/8) =========="
    run_lmms_model "smollm2-1.7b-instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct" "$MATH_G0" 1 8192  8 8192 0.9; PIDS+=("$LAST_PID")
    run_lmms_model "vicuna-7b-v1.5"        "lmsys/vicuna-7b-v1.5"                "$MATH_G1" 1 4096  8 4096 0.9 "chat_template=$REPO_ROOT/scripts/vicuna.jinja"; PIDS+=("$LAST_PID")
    run_lmms_model "llama-3.2-3b-instruct" "meta-llama/Llama-3.2-3B-Instruct"    "$MATH_G2" 1 10240 8 10240 0.9; PIDS+=("$LAST_PID")
    run_lmms_model "llama-3.1-8b"          "meta-llama/Llama-3.1-8B-Instruct"    "$MATH_G3" 1 10240 8 10240 0.9; PIDS+=("$LAST_PID")
    run_lmms_model "phi-4"       "microsoft/phi-4"        "$MATH_G4" 1 10240 8 10240 0.9; PIDS+=("$LAST_PID")
    run_lmms_model "gpt-oss-20b" "openai/gpt-oss-20b"     "$MATH_G5" 1 10240 8 10240 0.9; PIDS+=("$LAST_PID")
    run_lmms_model "qwen3-next-80b-a3b-instruct" "Qwen/Qwen3-Next-80B-A3B-Instruct" "${MATH_G6},${MATH_G7}" 2 10240 8 10240 0.85; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 2: olmo TP=2 + qwq TP=2 + glm-4.5-air TP=4 (8/8) =========="
    run_lmms_model "olmo-3.1-32b-think" "allenai/Olmo-3.1-32B-Think" "${MATH_G0},${MATH_G1}" 2 20480 128 10240 0.8; PIDS+=("$LAST_PID")
    run_lmms_model "qwq-32b" "Qwen/QwQ-32B" "${MATH_G2},${MATH_G3}" 2 20480 64 10240 0.8; PIDS+=("$LAST_PID")
    run_lmms_model "glm-4.5-air" "zai-org/GLM-4.5-Air" "${MATH_G4},${MATH_G5},${MATH_G6},${MATH_G7}" 4 32768 64 20480 0.85; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 3: qwen3.5-397b-a17b FP8 (TP=${big_tp}) =========="
    run_lmms_model "qwen3.5-397b-a17b" "Qwen/Qwen3.5-397B-A17B-FP8" "$big_gpus" "$big_tp" 32768 64 10240 0.85 "quantization=fp8,enable_prefix_caching=False"; PIDS+=("$LAST_PID")
    wait_all

    echo "========== Wave 4: glm-4.6 AWQ (TP=${big_tp}) =========="
    run_lmms_model "glm-4.6" "bullpoint/GLM-4.6-AWQ" "$big_gpus" "$big_tp" 32768 64 20480 0.85 "enable_expert_parallel=True"; PIDS+=("$LAST_PID")
    wait_all
}

# =====================================================================
# Dispatch rollout
# =====================================================================
echo "Framework: $FRAMEWORK | Benchmark: $BENCHMARK | Task: $DATA_NAMES | N=$N_SAMPLING"
if [ "$FRAMEWORK" = "math" ]; then
    echo "Math GPU pool (physical): $MATH_CUDA_DEVICES — run_math sets TP/PP to match visible GPU count"
    run_math
elif [ "$FRAMEWORK" = "lmms" ]; then
    run_lmms
else
    echo "Unknown framework: $FRAMEWORK (use math or lmms)"
    exit 1
fi

echo ""
echo "========== Rollout done. Estimating difficulty =========="

cd "$SCRIPT_DIR/.."
ANCHORS_ARG=""
if [[ -n "$ANCHORS" ]]; then
    ANCHORS_ARG="--anchors $ANCHORS"
fi
OUT_ARG=""
if [[ -n "$DIFFICULTY_OUT" ]]; then
    OUT_ARG="--out $DIFFICULTY_OUT"
fi
# 12-model θ panel from irt_results/<bm>_ability.csv. Rollout must include every panel
# model; verifier.main estimates b̂ with the full θ vector and normalizes difficulty to
# 0–10 via pool min/max + synthetic all-pass/all-fail anchors on that same panel.
conda run --no-capture-output -n "$IRT_CONDA_ENV" python -m verifier.main \
    --results-dir "$RESULTS_DIR" \
    --benchmark "$BENCHMARK" \
    --framework "$FRAMEWORK" \
    --irt-results-dir "$SCRIPT_DIR/../irt_results" \
    $ANCHORS_ARG \
    $OUT_ARG
