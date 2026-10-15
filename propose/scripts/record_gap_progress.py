#!/usr/bin/env python3
"""Checkpoint completed gaps + append one summary row to gap_fill_outcomes.csv.

Writes:
  - gap_run_progress.json  — list of completed gap_rank entries (JSON).
  - gap_fill_outcomes.csv — one row per completed run (status=done), all difficulties
    on the pipeline **0–10** scale (same as init target / gap_full_left.right).

CSV columns (all difficulties on pipeline 0–10 scale):
  gap_rank, gap_before, gap_after, target_difficulty, new_item_difficulty_0_10, delta,
  id, item, bracket_low_0_10, bracket_high_0_10, in_gap,
  benchmark, run_id, done_reason, succeeded

gap_before: width of the bracket [bracket_low, bracket_high] before inserting the new item.
gap_after: if the new item lies inside the bracket, max(new-lo, hi-new); else equal to gap_before.
new_item_difficulty_0_10: prefers full-panel verify (accepted_items / full rollout); else best_item.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_FLOAT_TOL = 1e-9


def _best_record_for_output(st: dict) -> dict | None:
    """Prefer full-verify accepted row, else state best_item."""
    acc = st.get("accepted_items") or []
    if acc:
        return min(acc, key=lambda x: float(x.get("abs_delta_full", 1e9)))
    bi = st.get("best_item")
    return bi if isinstance(bi, dict) else None


def _extract_new_difficulty_0_10(st: dict) -> float | None:
    """Best available 0–10 difficulty for the chosen item (full verify if present)."""
    rec = _best_record_for_output(st)
    if not rec:
        return None
    ed = rec.get("estimated_difficulty") or {}
    if isinstance(ed, dict) and ed.get("estimate") is not None:
        return float(ed["estimate"])
    if isinstance(ed, (int, float)):
        return float(ed)
    return None


def _bracket_0_10(st: dict) -> tuple[float | None, float | None]:
    gl = st.get("gap_full_left") or {}
    gr = st.get("gap_full_right") or {}
    try:
        dl = float(gl["difficulty_full"])
        dr = float(gr["difficulty_full"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return min(dl, dr), max(dl, dr)


def _gap_before_after_in(
    lo: float | None,
    hi: float | None,
    new_d: float | None,
) -> tuple[float | None, float | None, bool | None]:
    if lo is None or hi is None:
        return None, None, None
    gap_before = hi - lo
    if new_d is None:
        return gap_before, None, None
    in_gap = (lo - _FLOAT_TOL) <= new_d <= (hi + _FLOAT_TOL)
    if not in_gap:
        return gap_before, gap_before, False
    gap_after = max(new_d - lo, hi - new_d)
    return gap_before, gap_after, True


def _append_outcomes_csv(workdir: Path, benchmark: str, st: dict) -> Path:
    csv_path = workdir / "gap_fill_outcomes.csv"
    lo, hi = _bracket_0_10(st)
    target = float(st.get("full_target_difficulty", st.get("target_difficulty", 0.0)))
    new_d = _extract_new_difficulty_0_10(st)
    gap_before, gap_after, in_gap = _gap_before_after_in(lo, hi, new_d)
    delta = (new_d - target) if new_d is not None else None

    rec = _best_record_for_output(st) or {}
    row_id = str(rec.get("id", ""))
    item_text = str(rec.get("question") or "").replace("\r", " ").replace("\n", " ")

    row = {
        "gap_rank": int(st.get("gap_rank", 0)),
        "gap_before": gap_before if gap_before is not None else "",
        "gap_after": gap_after if gap_after is not None else "",
        "target_difficulty": target,
        "new_item_difficulty_0_10": new_d if new_d is not None else "",
        "delta": delta if delta is not None else "",
        "id": row_id,
        "item": item_text,
        "bracket_low_0_10": lo if lo is not None else "",
        "bracket_high_0_10": hi if hi is not None else "",
        "in_gap": "" if in_gap is None else ("true" if in_gap else "false"),
        "run_id": st.get("run_id", ""),
        "done_reason": st.get("done_reason", ""),
        "succeeded": st.get("succeeded", ""),
        "benchmark": benchmark,
    }

    fieldnames = [
        "gap_rank",
        "gap_before",
        "gap_after",
        "target_difficulty",
        "new_item_difficulty_0_10",
        "delta",
        "id",
        "item",
        "bracket_low_0_10",
        "bracket_high_0_10",
        "in_gap",
        "benchmark",
        "run_id",
        "done_reason",
        "succeeded",
    ]

    new_file = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, required=True, help="e2e_split_<benchmark> directory")
    ap.add_argument("--benchmark", type=str, required=True)
    args = ap.parse_args()
    workdir: Path = args.workdir.resolve()
    state_path = workdir / "state.json"
    prog_path = workdir / "gap_run_progress.json"
    if not state_path.is_file():
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    if st.get("status") != "done":
        return
    entry = {
        "gap_rank": int(st.get("gap_rank", 0)),
        "run_id": st.get("run_id"),
        "status": st.get("status"),
        "done_reason": st.get("done_reason"),
        "succeeded": st.get("succeeded"),
        "best_item_id": (st.get("best_item") or {}).get("id") if isinstance(st.get("best_item"), dict) else None,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    data: dict = {}
    if prog_path.is_file():
        data = json.loads(prog_path.read_text(encoding="utf-8"))
    data["benchmark"] = args.benchmark
    comp = [x for x in data.get("completed", []) if x.get("gap_rank") != entry["gap_rank"]]
    comp.append(entry)
    data["completed"] = sorted(comp, key=lambda x: int(x.get("gap_rank", 0)))
    data["last_completed_gap_rank"] = entry["gap_rank"]
    prog_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[record_gap_progress] wrote {prog_path} (gap_rank={entry['gap_rank']})")

    csv_path = _append_outcomes_csv(workdir, args.benchmark, st)
    print(f"[record_gap_progress] appended row → {csv_path}")


if __name__ == "__main__":
    main()
