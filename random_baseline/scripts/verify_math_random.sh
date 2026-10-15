#!/usr/bin/env bash
# Math-framework wrapper: convert random gap items, swap into math_candidates/<bench>/test.jsonl,
# run rollout, restore. Same backup-and-restore flow as verify_lmms_random.sh.
set -euo pipefail

BENCH="${1:?Usage: $0 <benchmark> [gap1,gap2,...]}"
ONLY="${2:-}"

GAP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RB=$GAP/random_baseline/results/$BENCH
MATH=$GAP/verifier/math_candidates
SCRIPTS=$GAP/random_baseline/scripts
N_SAMPLING="${N_SAMPLING:-32}"

case "$BENCH" in
    aime24|aime25|aime26|amc23|gaokao2023en|math-500|gsm8k) ;;
    *) echo "ERROR: unsupported math benchmark $BENCH"; exit 1;;
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

echo "== $BENCH (math): gaps=$GAPS =="

ORIG=$MATH/$BENCH/test.jsonl
BACKUP=$MATH/$BENCH/test.jsonl.MAIN_BACKUP
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
    echo "========================================================"
    echo "  $BENCH gap_rank=$g (math) — convert + rollout"
    echo "========================================================"

    python3 "$SCRIPTS/build_lmms_candidates.py" --benchmark "$BENCH" --gap-rank "$g"

    SRC_JSONL=$MATH/${BENCH}_random_gap${g}.jsonl
    [[ -f "$SRC_JSONL" ]] || { echo "ERROR: $SRC_JSONL not produced"; exit 1; }
    mkdir -p "$MATH/$BENCH"
    cp "$SRC_JSONL" "$ORIG"

    OUT="$RB/gap_rank_$g/rollout_results"
    DIFF_OUT="$RB/difficulty/${BENCH}_gap${g}_difficulty.csv"
    rm -rf "$OUT"
    mkdir -p "$OUT" "$RB/difficulty"

    bash "$GAP/verifier/run_rollout.sh" \
        --benchmark "$BENCH" \
        --framework math \
        --n "$N_SAMPLING" \
        --out "$OUT" \
        --difficulty-out "$DIFF_OUT" < /dev/null

    echo "[done] $BENCH gap_rank=$g -> $DIFF_OUT"
done

echo
echo "ALL DONE for $BENCH (gaps: $GAPS)"
