#!/usr/bin/env bash
# Run verification sequentially across multiple benchmarks; auto-dispatch math vs lmms.
# Usage:
#   bash verify_random_seq.sh ifeval mmlu_bio aime25
#   bash verify_random_seq.sh ifeval:2,4,5 mmlu_bio aime25:1,3   # per-bench gap subset
#
# Logs are appended to <repo>/random_baseline/results/_logs/
set -euo pipefail

(($# > 0)) || { echo "Usage: $0 <bench>[:gaps] [<bench>[:gaps] ...]"; exit 1; }

GAP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS=$GAP/random_baseline/scripts
LOGDIR=$GAP/random_baseline/results/_logs
mkdir -p "$LOGDIR"

declare -A FRAMEWORK
for b in aime24 aime25 aime26 amc23 gaokao2023en math-500 gsm8k; do FRAMEWORK[$b]=math; done
for b in ifeval mmlu_bio bbh_3obj; do FRAMEWORK[$b]=lmms; done

for arg in "$@"; do
    bench="${arg%%:*}"
    only="${arg#*:}"
    [[ "$only" == "$arg" ]] && only=""

    fw="${FRAMEWORK[$bench]:-}"
    if [[ -z "$fw" ]]; then
        echo "[seq] ERROR: unknown framework for $bench — skipping"
        continue
    fi
    log=$LOGDIR/seq_${bench}_$(date +%Y%m%d_%H%M%S).log
    echo
    echo "########################################################"
    echo "##  $(date +%H:%M:%S)  $bench  ($fw)  gaps=${only:-all}"
    echo "##  log: $log"
    echo "########################################################"

    if [[ "$fw" == "math" ]]; then
        bash "$SCRIPTS/verify_math_random.sh" "$bench" "$only" < /dev/null 2>&1 | tee -a "$log"
    else
        bash "$SCRIPTS/verify_lmms_random.sh" "$bench" "$only" < /dev/null 2>&1 | tee -a "$log"
    fi

    # After all gaps for this bench finish, generate the per-bench summary CSV
    # (closest-to-target item per gap with delta).
    echo
    echo "[seq] $(date +%H:%M:%S) building summary for $bench"
    python3 "$SCRIPTS/build_summary.py" "$bench" 2>&1 | tee -a "$log"
done

echo
echo "[seq] $(date +%H:%M:%S) ALL DONE"
