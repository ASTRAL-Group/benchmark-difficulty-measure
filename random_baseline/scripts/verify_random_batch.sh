#!/usr/bin/env bash
# Batch-mode verifier: combine multiple gaps of one benchmark into a single rollout.
# Saves N-1 model-load cycles per benchmark (huge with TP=4 80B+ models).
#
# Resume-safe:
#   - skips gaps that already have difficulty/<bench>_gap<r>_difficulty.csv
#     (set FORCE_VERIFY=1 to redo)
#   - keeps rollout_results/ across interrupt if gap list stamp matches
#     (run_rollout.sh itself skips finished models)
#
# Usage:
#   bash verify_random_batch.sh aime26 2,4,5
#   FORCE_VERIFY=1 bash verify_random_batch.sh ifeval 1,2,4

set -euo pipefail

BENCH="${1:?Usage: $0 <benchmark> <gap_list>}"
GAPS_RAW="${2:?Usage: $0 <benchmark> <gap_list>}"
GAPS="${GAPS_RAW//,/ }"
FORCE_VERIFY="${FORCE_VERIFY:-0}"

GAP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RB=$GAP/random_baseline/results/$BENCH
MATH=$GAP/verifier/math_candidates
LMMS=$GAP/verifier/lmms_candidates
SCRIPTS=$GAP/random_baseline/scripts
BATCH=$RB/_batch

case "$BENCH" in
    aime24|aime25|aime26|amc23|gaokao2023en|math-500|gsm8k)
        FW=math
        N_SAMPLING="${N_SAMPLING:-32}"
        ;;
    ifeval|mmlu_bio|bbh_3obj)
        FW=lmms
        N_SAMPLING="${N_SAMPLING:-4}"
        ;;
    *) echo "ERROR: unknown benchmark $BENCH"; exit 1 ;;
esac

NEED_GAPS=()
for g in $GAPS; do
    [[ -f $RB/gap_rank_$g/generated.jsonl ]] || { echo "missing gap_rank_$g/generated.jsonl"; exit 1; }
    DIFF=$RB/difficulty/${BENCH}_gap${g}_difficulty.csv
    if [[ -f "$DIFF" && "$FORCE_VERIFY" != "1" ]]; then
        echo "[skip] $BENCH gap$g — difficulty exists: $DIFF"
    else
        NEED_GAPS+=("$g")
    fi
done

mkdir -p $BATCH $RB/difficulty

if [[ ${#NEED_GAPS[@]} -eq 0 ]]; then
    echo "[batch] $BENCH — all requested gaps already verified; refreshing summary"
    python3 $SCRIPTS/build_summary.py $BENCH
    echo "[batch] $BENCH done (nothing to roll)"
    exit 0
fi

GAPS="${NEED_GAPS[*]}"
echo "== batch verify: $BENCH ($FW) gaps=$GAPS (FORCE_VERIFY=$FORCE_VERIFY) =="

# 1. Convert each gap separately
for g in $GAPS; do
    python3 $SCRIPTS/build_lmms_candidates.py --benchmark $BENCH --gap-rank $g
done

# 2. Concatenate; for lmms renumber `key` to keep them globally unique
GAPS_CSV="${GAPS// /,}"
python3 - <<PYEOF
import csv, json
from pathlib import Path
gaps = [int(g) for g in "$GAPS_CSV".split(",")]
fw = "$FW"
bench = "$BENCH"
src_dir = Path("$MATH" if fw == "math" else "$LMMS")
combined = Path("$BATCH/combined.jsonl")
mapping = []  # for lmms: list of (gap, new_key)

with combined.open("w", encoding="utf-8") as fout:
    next_key = 0
    total = 0
    for g in gaps:
        fp = src_dir / f"{bench}_random_gap{g}.jsonl"
        if not fp.is_file():
            raise SystemExit(f"missing {fp}")
        n = 0
        with fp.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                if fw == "lmms":
                    row["key"] = next_key
                    mapping.append({"gap": g, "key": next_key})
                    next_key += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        total += n
        print(f"  gap {g}: {n} items")
    print(f"combined: {total} items -> {combined}")

with (Path("$BATCH") / "key_mapping.json").open("w") as f:
    json.dump({"framework": fw, "bench": bench, "rows": mapping}, f)
PYEOF

# 3. Backup + swap into bench's main candidate path
if [ "$FW" = "math" ]; then
    ORIG=$MATH/$BENCH/test.jsonl
else
    ORIG=$LMMS/$BENCH.jsonl
fi
BAK=$ORIG.MAIN_BACKUP_BATCH
[[ -f $ORIG && ! -f $BAK ]] && cp $ORIG $BAK
mkdir -p $(dirname $ORIG)
cp $BATCH/combined.jsonl $ORIG

restore() {
    [[ -f $BAK ]] && mv $BAK $ORIG
}
trap restore EXIT INT TERM

# 4. Single rollout (resume model waves if stamp matches)
OUT_DIR=$BATCH/rollout_results
DIFF_OUT=$BATCH/difficulty.csv
STAMP=$BATCH/gaps_stamp.txt
mkdir -p $OUT_DIR
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$GAPS_CSV" && "$FORCE_VERIFY" != "1" ]]; then
    echo "== resume rollout in $OUT_DIR (stamp=$GAPS_CSV) =="
else
    echo "== fresh rollout (stamp was '$(cat "$STAMP" 2>/dev/null || echo none)' → $GAPS_CSV) =="
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    echo "$GAPS_CSV" > "$STAMP"
fi
echo
cd "$GAP"
bash $GAP/verifier/run_rollout.sh \
    --benchmark $BENCH \
    --framework $FW \
    --n "$N_SAMPLING" \
    --out $OUT_DIR \
    --difficulty-out $DIFF_OUT < /dev/null

# 5. Split combined difficulty.csv by gap
echo
echo "== split combined difficulty.csv -> per-gap CSVs =="
python3 - <<PYEOF
import csv, json, re
from pathlib import Path
combined = Path("$DIFF_OUT")
fw = "$FW"
bench = "$BENCH"
out_dir = Path("$RB/difficulty")
mapping = json.loads(Path("$BATCH/key_mapping.json").read_text())["rows"]
key_to_gap = {row["key"]: row["gap"] for row in mapping}

with combined.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    all_rows = list(reader)

grouped = {}
for r in all_rows:
    item = str(r["item"])
    if fw == "math":
        m = re.match(r"rand_g(\d+)_", item)
        if not m: continue
        g = int(m.group(1))
    else:
        try: k = int(item)
        except: continue
        g = key_to_gap.get(k)
        if g is None: continue
    grouped.setdefault(g, []).append(r)

for g, rows in sorted(grouped.items()):
    out_path = out_dir / f"{bench}_gap{g}_difficulty.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  gap_{g}: {len(rows)} rows -> {out_path}")
PYEOF

# 6. Build summary
echo
python3 $SCRIPTS/build_summary.py $BENCH

echo
echo "[batch] $BENCH done; gaps=$GAPS"
