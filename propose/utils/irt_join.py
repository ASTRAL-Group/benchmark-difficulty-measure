"""Join IRT `*_difficulty.csv` (q0..q{n-1}) with `data/pools/*/test.jsonl`.

``q_k`` is the same index as column ``q_k`` in `data/matrix/*_response_matrix.csv`
(k-th line of the pool jsonl = k-th question),
when evaluation used the same ordering as ``load_data`` (no ``--shuffle``).

Indexing matches ``verifier/diff_eval/data_loader.load_data`` for local jsonl:

- If rows have no ``idx`` field, evaluation assigns ``idx = 0..n-1`` in **file line order**,
  then sorts by ``idx`` (order unchanged).
- IRT rows named ``q0``..``q{n-1}`` correspond to those same positions: **``q_k`` = k-th
  example after that load** (not necessarily AoPS ``Problem_1``, ``Problem_2``, … order
  in the file — see each row's ``url`` / ``id`` if you need contest numbering).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from utils.schemas import ProblemItem


def _parse_q_index(item: str) -> int | None:
    m = re.match(r"^q(\d+)$", item.strip(), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_eval_jsonl_like_data_loader(path: str | Path) -> list[dict]:
    """Same ordering as ``evaluation/data_loader.load_data`` for a jsonl file."""
    p = Path(path)
    raw: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))
    if not raw:
        return []
    if "idx" not in raw[0]:
        raw = [{"idx": i, **ex} for i, ex in enumerate(raw)]
    return sorted(raw, key=lambda x: int(x["idx"]))


def merge_q_indexed_irt_csv_with_eval_jsonl(
    difficulty_csv: str | Path,
    eval_jsonl: str | Path,
    *,
    id_key: str = "id",
) -> list[ProblemItem]:
    """Merge ``item`` = ``q0``..``q{n-1}`` with one jsonl row per index.

    ``id_key``: use this field from each json row as ``ProblemItem.id`` (falls back to
    ``q{k}`` if missing).
    """
    csv_path = Path(difficulty_csv)
    by_index: dict[int, tuple[float, float]] = {}
    with csv_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            item = (row.get("item") or "").strip()
            qi = _parse_q_index(item)
            if qi is None:
                raise ValueError(f"Expected item like q0, got {item!r} in {csv_path}")
            b_raw = row.get("b_mean")
            if b_raw is None or b_raw == "":
                raise ValueError(f"Missing b_mean for {item} in {csv_path}")
            b_mean = float(b_raw)
            b_sd = float(row["b_sd"]) if row.get("b_sd") not in (None, "") else 0.0
            if qi in by_index:
                raise ValueError(f"Duplicate item index q{qi} in {csv_path}")
            by_index[qi] = (b_mean, b_sd)

    examples = load_eval_jsonl_like_data_loader(eval_jsonl)
    n = len(examples)
    expected = set(range(n))
    got = set(by_index.keys())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"IRT indices {sorted(got)} do not match jsonl length {n}: "
            f"missing {missing}, extra {extra}"
        )

    out: list[ProblemItem] = []
    for k in range(n):
        row = examples[k]
        b_mean, _ = by_index[k]
        qtext = row.get("problem") or row.get("question") or row.get("stem") or ""
        ans = str(row.get("answer") or row.get("reference") or "")
        rid = str(row[id_key]) if id_key in row and row[id_key] is not None else f"q{k}"
        raw_opts = row.get("options")
        opts: list[str] | None = None
        if isinstance(raw_opts, list) and raw_opts:
            opts = [str(o) for o in raw_opts]
        raw_il = row.get("instruction_id_list")
        il: list[str] | None = [str(x) for x in raw_il] if isinstance(raw_il, list) and raw_il else None
        raw_kw = row.get("kwargs")
        kw: list[dict] | None = [dict(x) if isinstance(x, dict) else {} for x in raw_kw] if isinstance(raw_kw, list) and raw_kw else None
        out.append(
            ProblemItem(
                id=rid,
                question=str(qtext),
                reference_answer=ans,
                difficulty=b_mean,
                options=opts,
                instruction_id_list=il,
                kwargs=kw,
            )
        )
    return out
