"""Virtual difficulty distribution for iterative max-gap mass fill.

Never writes to ``data/pools/`` or ``irt_results/``. All state lives under
``mass_fill_results/<benchmark>/``.

Virtual list used for max-gap = original pool 0–10 difficulties + fill points
(seed / real / pretend). ``target_count`` counts **fills only** (e.g. 30 for AIME
including the 3 method seeds).
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from utils.difficulty_scale import b_range_from_values, to_difficulty_0_10
from utils.max_gap import gap_midpoint_target, largest_normalized_gap_bracket_b

FillStatus = Literal["seed", "real", "pretend"]


@dataclass
class VirtualPoint:
    """One **fill** difficulty on the pipeline 0–10 scale (not an original pool row)."""

    difficulty_0_10: float
    status: FillStatus
    item_id: str = ""
    source: str = ""
    target: float | None = None
    abs_delta: float | None = None
    gap_low: float | None = None
    gap_high: float | None = None
    question: str = ""
    answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MassFillState:
    benchmark: str
    target_count: int
    b_min: float
    b_max: float
    pool_0_10: list[float]  # deep-copied original pool difficulties (0–10)
    fills: list[VirtualPoint]  # seed + real + pretend only
    step: int = 0  # completed post-seed fill attempts
    completed: bool = False

    def filled_count(self) -> int:
        return len(self.fills)

    def remaining(self) -> int:
        return max(0, self.target_count - self.filled_count())

    def virtual_difficulties(self) -> list[float]:
        return list(self.pool_0_10) + [float(p.difficulty_0_10) for p in self.fills]

    def real_count(self) -> int:
        return sum(1 for p in self.fills if p.status == "real")

    def pretend_count(self) -> int:
        return sum(1 for p in self.fills if p.status == "pretend")

    def seed_count(self) -> int:
        return sum(1 for p in self.fills if p.status == "seed")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir(benchmark: str, root: Path | None = None) -> Path:
    root = root or _repo_root()
    return root / "mass_fill_results" / benchmark


def default_target_count(benchmark: str) -> int:
    if benchmark == "amc23":
        return 40
    if benchmark in {"aime24", "aime25", "aime26"}:
        return 30
    raise ValueError(f"unsupported mass-fill benchmark: {benchmark}")


def load_original_pool_0_10(benchmark: str, root: Path | None = None) -> tuple[list[float], float, float]:
    root = root or _repo_root()
    irt = root / "irt_results" / f"{benchmark}_difficulty.csv"
    if not irt.is_file():
        raise FileNotFoundError(irt)
    bs: list[float] = []
    with irt.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bs.append(float(row["b_mean"]))
    b_min, b_max = b_range_from_values(bs)
    d10 = [to_difficulty_0_10(b, b_min, b_max) for b in bs]
    return d10, b_min, b_max


def load_seed_records(benchmark: str, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or _repo_root()
    path = root / "new_item_records.csv"
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != benchmark:
                continue
            out.append(row)
    out.sort(key=lambda r: r.get("top_gap", ""))
    return out


def init_state(
    benchmark: str,
    *,
    target_count: int | None = None,
    seed_top3: bool = True,
    root: Path | None = None,
) -> MassFillState:
    root = root or _repo_root()
    d10, b_min, b_max = load_original_pool_0_10(benchmark, root)
    fills: list[VirtualPoint] = []
    if seed_top3:
        for row in load_seed_records(benchmark, root):
            item: dict[str, Any] = {}
            try:
                item = json.loads(row.get("item") or "{}")
            except json.JSONDecodeError:
                item = {}
            tgt = float(row["target"]) if row.get("target") not in (None, "") else None
            dif = float(row["difficulty"])
            fills.append(
                VirtualPoint(
                    difficulty_0_10=dif,
                    status="seed",
                    item_id=str(row.get("id") or item.get("id") or ""),
                    source="new_item_records",
                    target=tgt,
                    abs_delta=(abs(dif - tgt) if tgt is not None else None),
                    question=str(item.get("question") or ""),
                    answer=str(item.get("answer") or ""),
                    meta={"top_gap": row.get("top_gap"), "parent_id": row.get("parent_id")},
                )
            )
    n = target_count if target_count is not None else default_target_count(benchmark)
    return MassFillState(
        benchmark=benchmark,
        target_count=n,
        b_min=b_min,
        b_max=b_max,
        pool_0_10=deepcopy(d10),
        fills=fills,
        step=0,
        completed=len(fills) >= n,
    )


def pick_max_gap(state: MassFillState) -> tuple[float, float, float, dict[str, float | int]]:
    """Return ``(gap_low, gap_high, target, meta)`` on 0–10 scale."""
    xs = state.virtual_difficulties()
    lo, hi, meta = largest_normalized_gap_bracket_b(xs)
    target = gap_midpoint_target(lo, hi)
    return lo, hi, target, meta


def append_fill(state: MassFillState, point: VirtualPoint) -> None:
    state.fills.append(point)
    state.step += 1
    if state.filled_count() >= state.target_count:
        state.completed = True


def state_to_dict(state: MassFillState) -> dict[str, Any]:
    return {
        "benchmark": state.benchmark,
        "target_count": state.target_count,
        "b_min": state.b_min,
        "b_max": state.b_max,
        "pool_0_10": list(state.pool_0_10),
        "fills": [asdict(p) for p in state.fills],
        "step": state.step,
        "completed": state.completed,
        "counts": {
            "filled": state.filled_count(),
            "remaining": state.remaining(),
            "seed": state.seed_count(),
            "real": state.real_count(),
            "pretend": state.pretend_count(),
        },
    }


def state_from_dict(d: dict[str, Any]) -> MassFillState:
    fills = [VirtualPoint(**p) for p in d.get("fills", [])]
    return MassFillState(
        benchmark=str(d["benchmark"]),
        target_count=int(d["target_count"]),
        b_min=float(d["b_min"]),
        b_max=float(d["b_max"]),
        pool_0_10=[float(x) for x in d["pool_0_10"]],
        fills=fills,
        step=int(d.get("step", 0)),
        completed=bool(d.get("completed", False)),
    )


def save_state(state: MassFillState, root: Path | None = None) -> Path:
    out = results_dir(state.benchmark, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    path = out / "state.json"
    path.write_text(json.dumps(state_to_dict(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # companion views (never touch original IRT/pool)
    _write_distribution_virtual(state, out)
    _write_seed_and_filled(state, out)
    _write_progress_csv(state, out)
    try:
        from mass_fill.export_sheet import write_export

        write_export(state, root=root)
    except Exception:
        pass
    return path


def load_state(benchmark: str, root: Path | None = None) -> MassFillState | None:
    path = results_dir(benchmark, root) / "state.json"
    if not path.is_file():
        return None
    return state_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_distribution_virtual(state: MassFillState, out: Path) -> None:
    rows = []
    for i, d in enumerate(state.pool_0_10):
        rows.append({"kind": "pool", "index": i, "difficulty_0_10": d, "status": "pool"})
    for i, p in enumerate(state.fills):
        rows.append(
            {
                "kind": "fill",
                "index": i,
                "difficulty_0_10": p.difficulty_0_10,
                "status": p.status,
                "item_id": p.item_id,
            }
        )
    path = out / "distribution_virtual.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_seed_and_filled(state: MassFillState, out: Path) -> None:
    seed_path = out / "seed_items.jsonl"
    filled_path = out / "filled_items.jsonl"
    with seed_path.open("w", encoding="utf-8") as fs, filled_path.open("w", encoding="utf-8") as ff:
        for p in state.fills:
            row = asdict(p)
            line = json.dumps(row, ensure_ascii=False) + "\n"
            if p.status == "seed":
                fs.write(line)
            else:
                ff.write(line)


def _write_progress_csv(state: MassFillState, out: Path) -> None:
    path = out / "progress.csv"
    fields = [
        "ord",
        "status",
        "item_id",
        "target",
        "difficulty_0_10",
        "abs_delta",
        "gap_low",
        "gap_high",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(state.fills, start=1):
            w.writerow(
                {
                    "ord": i,
                    "status": p.status,
                    "item_id": p.item_id,
                    "target": p.target if p.target is not None else "",
                    "difficulty_0_10": p.difficulty_0_10,
                    "abs_delta": p.abs_delta if p.abs_delta is not None else "",
                    "gap_low": p.gap_low if p.gap_low is not None else "",
                    "gap_high": p.gap_high if p.gap_high is not None else "",
                    "source": p.source,
                }
            )
