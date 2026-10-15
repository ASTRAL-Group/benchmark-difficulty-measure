"""OCR-friendly Benchmark Difficulty sheet export for mass-fill.

Matches the paper Experiments sheet layout:

    ,gap size,target,difficulty,id,Parent id,item
    AIME24,,,,,,
    gap1,1.786,8.730,8.788,1:gen_...,q62,"{...json...}"
    ...
    remain1,0.637,1.768,,,,

For machines without an easy download path: dump → photograph → OCR.

  bash propose/mass_fill/run_mass_fill.sh export aime24
  bash propose/mass_fill/run_mass_fill.sh export          # all benches that have state
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from mass_fill.distribution import MassFillState, load_seed_records, load_state, results_dir

DISPLAY = {
    "aime24": "AIME24",
    "aime25": "AIME25",
    "aime26": "AIME26",
    "amc23": "AMC23",
}

SHEET_FIELDS = ["", "gap size", "target", "difficulty", "id", "Parent id", "item"]
NUM_FIELDS = [
    "benchmark",
    "gap",
    "status",
    "gap_size",
    "target",
    "difficulty",
    "abs_delta",
    "id",
    "parent_id",
]
DEFAULT_BENCHES = ("aime24", "aime25", "aime26", "amc23")


def _fmt3(x: float | None) -> str:
    if x is None:
        return ""
    return f"{float(x):.3f}"


def _parent_id(p) -> str:
    meta = p.meta or {}
    v = meta.get("parent_id") or meta.get("Parent id") or ""
    return str(v) if v not in (None, "") else ""


def _seed_gap_size(benchmark: str, item_id: str, root: Path | None) -> float | None:
    for row in load_seed_records(benchmark, root):
        if str(row.get("id") or "") == str(item_id):
            raw = row.get("gap_size")
            if raw not in (None, ""):
                return float(raw)
    return None


def _gap_size(p, benchmark: str, root: Path | None) -> float | None:
    if p.gap_low is not None and p.gap_high is not None:
        return float(p.gap_high) - float(p.gap_low)
    if p.status == "seed":
        return _seed_gap_size(benchmark, p.item_id, root)
    return None


def _sheet_id(ord_i: int, item_id: str) -> str:
    s = str(item_id or "").strip()
    if not s:
        return f"{ord_i}:"
    if s.split(":", 1)[0].isdigit():
        return s
    return f"{ord_i}:{s}"


def _item_json(p, *, benchmark: str, gap_label: str, gap_size: float | None) -> str:
    meta = dict(p.meta or {})
    parent = _parent_id(p)
    payload: dict[str, Any] = {
        "id": p.item_id,
        "question": p.question or "",
        "answer": p.answer or "",
        "parent_id": parent,
        "target": p.target,
        "difficulty": p.difficulty_0_10,
        "status": p.status,
        "benchmark": benchmark,
        "gap": gap_label,
    }
    if gap_size is not None:
        payload["gap_size"] = round(float(gap_size), 6)
    if p.abs_delta is not None:
        payload["abs_delta"] = p.abs_delta
    if p.gap_low is not None:
        payload["gap_low"] = p.gap_low
    if p.gap_high is not None:
        payload["gap_high"] = p.gap_high
    for k in ("top_gap", "done_reason", "run_id", "round"):
        if meta.get(k) not in (None, ""):
            payload[k] = meta[k]
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


def remaining_gaps(state: MassFillState, k: int = 5) -> list[tuple[float, float, float, float]]:
    """Largest leftover adjacent gaps on the virtual 0–10 distribution."""
    xs = sorted(float(x) for x in state.virtual_difficulties())
    recs: list[tuple[float, float, float, float]] = []
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        recs.append((hi - lo, 0.5 * (lo + hi), lo, hi))
    recs.sort(key=lambda t: t[0], reverse=True)
    return recs[: max(0, k)]


def _fill_rows(state: MassFillState, *, remain_k: int, root: Path | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, p in enumerate(state.fills, start=1):
        label = f"gap{i}"
        gs = _gap_size(p, state.benchmark, root)
        rows.append(
            {
                "label": label,
                "status": p.status,
                "gap_size": _fmt3(gs),
                "target": _fmt3(p.target),
                "difficulty": _fmt3(p.difficulty_0_10),
                "abs_delta": _fmt3(p.abs_delta),
                "id": _sheet_id(i, p.item_id),
                "parent_id": _parent_id(p),
                "item": _item_json(p, benchmark=state.benchmark, gap_label=label, gap_size=gs),
                "question": p.question or "",
                "answer": p.answer or "",
            }
        )
    for j, (w, mid, lo, hi) in enumerate(remaining_gaps(state, remain_k), start=1):
        rows.append(
            {
                "label": f"remain{j}",
                "status": "",
                "gap_size": _fmt3(w),
                "target": _fmt3(mid),
                "difficulty": "",
                "abs_delta": "",
                "id": "",
                "parent_id": "",
                "item": "",
                "question": "",
                "answer": "",
            }
        )
    return rows


def export_dir(benchmark: str, root: Path | None = None) -> Path:
    d = results_dir(benchmark, root) / "export"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_export(
    state: MassFillState,
    *,
    remain_k: int = 5,
    wrap: int = 96,
    root: Path | None = None,
) -> dict[str, Path]:
    """Write sheet.csv / numbers.tsv / items.txt / SUMMARY.txt under export/."""
    out = export_dir(state.benchmark, root)
    disp = DISPLAY.get(state.benchmark, state.benchmark.upper())
    rows = _fill_rows(state, remain_k=remain_k, root=root)

    sheet_path = out / "benchmark_difficulty_sheet.csv"
    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_FIELDS)
        w.writerow([disp, "", "", "", "", "", ""])
        for r in rows:
            if r["label"].startswith("remain1"):
                w.writerow(["", "", "", "", "", "", ""])
            w.writerow(
                [
                    r["label"],
                    r["gap_size"],
                    r["target"],
                    r["difficulty"],
                    r["id"],
                    r["parent_id"],
                    r["item"],
                ]
            )

    num_path = out / "numbers.tsv"
    with num_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NUM_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "benchmark": disp,
                    "gap": r["label"],
                    "status": r["status"],
                    "gap_size": r["gap_size"],
                    "target": r["target"],
                    "difficulty": r["difficulty"],
                    "abs_delta": r["abs_delta"],
                    "id": r["id"],
                    "parent_id": r["parent_id"],
                }
            )

    items_path = out / "items.txt"
    blocks: list[str] = []
    blocks.append(f"# {disp}  filled={state.filled_count()}/{state.target_count}  "
                  f"seed={state.seed_count()} real={state.real_count()} pretend={state.pretend_count()}")
    blocks.append("")
    for r in rows:
        if not r["item"]:
            blocks.append(f"========== {disp} {r['label']} (unfilled leftover) ==========")
            blocks.append(f"gap_size: {r['gap_size']}")
            blocks.append(f"target:   {r['target']}")
            blocks.append("")
            continue
        blocks.append(f"========== {disp} {r['label']} ({r['status']}) ==========")
        blocks.append(f"gap_size:   {r['gap_size']}")
        blocks.append(f"target:     {r['target']}")
        blocks.append(f"difficulty: {r['difficulty']}")
        blocks.append(f"abs_delta:  {r['abs_delta']}")
        blocks.append(f"id:         {r['id']}")
        blocks.append(f"parent_id:  {r['parent_id']}")
        if r["answer"]:
            blocks.append(f"answer:     {r['answer']}")
        if r["question"]:
            blocks.append("question:")
            blocks.extend(textwrap.wrap(r["question"], width=wrap) or [r["question"]])
        blocks.append("item_json:")
        pretty = json.dumps(json.loads(r["item"]), ensure_ascii=False, indent=2)
        for line in pretty.splitlines():
            if len(line) <= wrap:
                blocks.append(line)
            else:
                blocks.extend(textwrap.wrap(line, width=wrap, replace_whitespace=False))
        blocks.append("")
    items_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")

    rel = f"mass_fill_results/{state.benchmark}/export"
    summary_path = out / "SUMMARY.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"MASS FILL EXPORT  {disp}",
                f"filled {state.filled_count()}/{state.target_count}  "
                f"seed={state.seed_count()} real={state.real_count()} pretend={state.pretend_count()}",
                f"completed={state.completed}",
                "",
                "Photograph these (OCR → file):",
                f"  1) {summary_path.name}          # this page",
                f"  2) {num_path.name}              # compact numbers (best for OCR)",
                f"  3) {items_path.name}            # questions, page with less",
                f"  4) {sheet_path.name}            # full Experiments-sheet CSV",
                "",
                f"less -S {rel}/{num_path.name}",
                f"less    {rel}/{items_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "summary": summary_path,
        "numbers": num_path,
        "items": items_path,
        "sheet": sheet_path,
    }


def write_combined_sheet(
    states: list[MassFillState],
    dest: Path,
    *,
    remain_k: int = 5,
    root: Path | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_FIELDS)
        for si, state in enumerate(states):
            if si:
                w.writerow(["", "", "", "", "", "", ""])
                w.writerow(["", "", "", "", "", "", ""])
            disp = DISPLAY.get(state.benchmark, state.benchmark.upper())
            w.writerow([disp, "", "", "", "", "", ""])
            for r in _fill_rows(state, remain_k=remain_k, root=root):
                if r["label"].startswith("remain1"):
                    w.writerow(["", "", "", "", "", "", ""])
                w.writerow(
                    [
                        r["label"],
                        r["gap_size"],
                        r["target"],
                        r["difficulty"],
                        r["id"],
                        r["parent_id"],
                        r["item"],
                    ]
                )
    return dest


def print_numbers_table(state: MassFillState, *, remain_k: int = 5, root: Path | None = None) -> None:
    disp = DISPLAY.get(state.benchmark, state.benchmark.upper())
    rows = _fill_rows(state, remain_k=remain_k, root=root)
    print(
        f"{disp}  filled={state.filled_count()}/{state.target_count}  "
        f"seed={state.seed_count()} real={state.real_count()} pretend={state.pretend_count()}",
        flush=True,
    )
    hdr = f"{'gap':<10} {'status':<8} {'size':>8} {'target':>8} {'diff':>8} {'|d-t|':>8}  id"
    print(hdr, flush=True)
    print("-" * min(96, len(hdr) + 40), flush=True)
    for r in rows:
        print(
            f"{r['label']:<10} {r['status']:<8} {r['gap_size']:>8} {r['target']:>8} "
            f"{r['difficulty']:>8} {r['abs_delta']:>8}  {r['id']}",
            flush=True,
        )


def export_benchmarks(
    benchmarks: list[str],
    *,
    remain_k: int = 5,
    wrap: int = 96,
    root: Path | None = None,
) -> list[tuple[str, dict[str, Path] | None]]:
    from pathlib import Path as _P

    root_p = root or _P(__file__).resolve().parents[2]
    done: list[tuple[str, dict[str, Path] | None]] = []
    states: list[MassFillState] = []
    for bm in benchmarks:
        st = load_state(bm, root_p)
        if st is None:
            done.append((bm, None))
            continue
        paths = write_export(st, remain_k=remain_k, wrap=wrap, root=root_p)
        print_numbers_table(st, remain_k=remain_k, root=root_p)
        print(f"wrote {paths['sheet']}", flush=True)
        print(f"     {paths['numbers']}", flush=True)
        print(f"     {paths['items']}", flush=True)
        print(f"     {paths['summary']}", flush=True)
        print(flush=True)
        done.append((bm, paths))
        states.append(st)
    if states:
        comb = write_combined_sheet(
            states,
            root_p / "mass_fill_results" / "BENCHMARK_DIFFICULTY_export.csv",
            remain_k=remain_k,
            root=root_p,
        )
        print(f"combined sheet → {comb}", flush=True)
    return done
