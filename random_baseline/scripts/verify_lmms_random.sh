#!/usr/bin/env bash
# Wrapper: for a given lmms-style benchmark, loop over all generated gap_rank_* dirs
# under random_baseline/results/<bench>/, convert generated.jsonl -> lmms candidate
# schema, swap into verifier/lmms_candidates/<bench>.jsonl, run rollout, restore.
#
# Usage:
#   bash verify_lmms_random.sh ifeval
#   bash verify_lmms_random.sh ifeval 2,4,5

set -euo pipefail

BENCH="${1:?Usage: $0 <benchmark> [gap1,gap2,...]}"
ONLY="${2:-}"

GAP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RB=$GAP/random_baseline/results/$BENCH
LMMS=$GAP/verifier/lmms_candidates
SCRIPTS=$GAP/random_baseline/scripts
N_SAMPLING="${N_SAMPLING:-4}"

case "$BENCH" in
    ifeval|mmlu_bio|bbh_3obj) ;;
    *) echo "ERROR: unsupported benchmark $BENCH"; exit 1;;
esac

[[ -d "$RB" ]] || { echo "no random_baseline results for $BENCH at $RB"; exit 1; }

if [[ -n "$ONLY" ]]; then
    GAPS="${ONLY//,/ }"
else
    GAPS="$(cd "$RB" && for d in gap_rank_*/; do
        [[ -f "$d/generated.jsonl" ]] && echo "${d#gap_rank_}" | tr -d '/'
    done | sort -n | xargs)"
fi
[[ -z "$GAPS" ]] && { echo "no gaps to run for $BENCH"; exit 0; }

echo "== $BENCH: gaps=$GAPS =="

ORIG=$LMMS/$BENCH.jsonl
BACKUP=$LMMS/$BENCH.jsonl.MAIN_BACKUP
if [[ -f "$ORIG" && ! -f "$BACKUP" ]]; then
    cp "$ORIG" "$BACKUP"
    echo "[backup] $ORIG -> $BACKUP"
fi

restore() {
    if [[ -f "$BACKUP" ]]; then
        mv "$BACKUP" "$ORIG"
        echo "[restore] $BACKUP -> $ORIG"
    fi
}
trap restore EXIT INT TERM

cd "$GAP"

for g in $GAPS; do
    echo
    echo "===================================================================="
    echo "  $BENCH gap_rank=$g — convert + rollout"
    echo "===================================================================="

    python3 "$SCRIPTS/build_lmms_candidates.py" --benchmark "$BENCH" --gap-rank "$g"
    cp "$LMMS/${BENCH}_random_gap${g}.jsonl" "$ORIG"

    OUT="$RB/gap_rank_$g/rollout_results"
    DIFF_OUT="$RB/difficulty/${BENCH}_gap${g}_difficulty.csv"
    rm -rf "$OUT"
    mkdir -p "$OUT" "$RB/difficulty"

    bash "$GAP/verifier/run_rollout.sh" \
        --benchmark "$BENCH" \
        --framework lmms \
        --n "$N_SAMPLING" \
        --out "$OUT" \
        --difficulty-out "$DIFF_OUT" < /dev/null

    echo "[done] $BENCH gap_rank=$g -> $DIFF_OUT"
done

echo
echo "ALL DONE for $BENCH (gaps: $GAPS)"
