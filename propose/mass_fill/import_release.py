"""Rebuild mass_fill_results/<bench>/state.json from mass_result/release CSVs.

Does not replace pretend fills. target_count stays at the AIME default (30).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from copy import deepcopy
from pathlib import Path

from mass_fill.distribution import (
    MassFillState,
    VirtualPoint,
    default_target_count,
    load_original_pool_0_10,
    results_dir,
    save_state,
)

_AIME = ("aime24", "aime25", "aime26")


def _f(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    return float(s)


def _i(v: str | None) -> int | None:
    x = _f(v)
    return int(x) if x is not None else None


def _source(status: str) -> str:
    if status == "seed":
        return "new_item_records"
    if status == "pretend":
        return "mass_fill_pretend"
    return "mass_fill"


def fills_from_csv(path: Path) -> list[VirtualPoint]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.sort(key=lambda r: int(r["gap_rank"]))
    out: list[VirtualPoint] = []
    for r in rows:
        status = (r.get("status") or "real").strip()
        if status not in {"seed", "real", "pretend"}:
            raise ValueError(f"unknown status {status!r} in {path}")
        tgt = _f(r.get("target_difficulty"))
        dif = _f(r.get("item_difficulty"))
        if dif is None:
            raise ValueError(f"missing item_difficulty for {r.get('item_id')} in {path}")
        meta: dict = {
            "parent_id": r.get("parent_id") or "",
            "gap_label": r.get("gap_label") or "",
            "gap_rank": _i(r.get("gap_rank")),
            "round": _i(r.get("round")),
            "done_reason": r.get("done_reason") or "",
        }
        gs = _f(r.get("gap_size"))
        if gs is not None:
            meta["gap_size"] = gs
        out.append(
            VirtualPoint(
                difficulty_0_10=float(dif),
                status=status,  # type: ignore[arg-type]
                item_id=str(r.get("item_id") or ""),
                source=_source(status),
                target=tgt,
                abs_delta=_f(r.get("abs_delta")),
                gap_low=_f(r.get("gap_low")),
                gap_high=_f(r.get("gap_high")),
                question=str(r.get("question") or ""),
                answer=str(r.get("answer") or ""),
                meta=meta,
            )
        )
    return out


def rebuild_state(
    benchmark: str,
    *,
    csv_path: Path,
    root: Path,
    target_count: int | None = None,
) -> MassFillState:
    pool, b_min, b_max = load_original_pool_0_10(benchmark, root)
    fills = fills_from_csv(csv_path)
    n = target_count if target_count is not None else default_target_count(benchmark)
    n_post_seed = sum(1 for p in fills if p.status != "seed")
    return MassFillState(
        benchmark=benchmark,
        target_count=n,
        b_min=b_min,
        b_max=b_max,
        pool_0_10=deepcopy(pool),
        fills=fills,
        step=n_post_seed,
        completed=len(fills) >= n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="Directory with <bench>_fills.csv (default: <repo>/mass_result/release)",
    )
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(_AIME),
        choices=["aime24", "aime25", "aime26", "amc23"],
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    rel = args.release_dir or (root / "mass_result" / "release")

    for bm in args.benchmarks:
        src = rel / f"{bm}_fills.csv"
        if not src.is_file():
            print(f"skip {bm}: missing {src}")
            continue
        dest_dir = results_dir(bm, root)
        dest_dir.mkdir(parents=True, exist_ok=True)
        old = dest_dir / "state.json"
        if old.is_file():
            bak = dest_dir / "state.json.bak_before_import"
            shutil.copy2(old, bak)
            print(f"backed up {old} → {bak}")
        st = rebuild_state(bm, csv_path=src, root=root)
        path = save_state(st, root)
        print(
            f"{bm}: filled={st.filled_count()}/{st.target_count} "
            f"seed={st.seed_count()} real={st.real_count()} pretend={st.pretend_count()} "
            f"remaining={st.remaining()} usable={st.seed_count()+st.real_count()} → {path}"
        )


if __name__ == "__main__":
    main()
