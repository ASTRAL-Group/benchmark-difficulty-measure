from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.difficulty_scale import b_range_from_pool, normalize_items_to_0_10
from utils.schemas import ProblemItem


def load_anchors_from_jsonl(
    path: str | Path,
    *,
    id_key: str = "id",
    question_key: str = "problem",
    answer_key: str = "answer",
    difficulty_key: str = "difficulty",
    max_items: int | None = None,
    normalize_difficulty_0_10: bool = False,
) -> list[ProblemItem]:
    """Load anchors from jsonl. Question: question_key or fallbacks. Difficulty: difficulty_key, b, or irt_b.

    If ``normalize_difficulty_0_10`` is True, replace each finite difficulty with a 0–10 min–max
    score over all loaded rows (IRT *b* or any common scale).
    """
    p = Path(path)
    out: list[ProblemItem] = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and len(out) >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            rid = str(row.get(id_key, f"line_{i}"))
            qtext = (
                row.get(question_key)
                or row.get("question")
                or row.get("stem")
                or row.get("problem")
                or row.get("input")
                or ""
            )
            ans = str(row.get(answer_key) or row.get("reference") or row.get("output") or "")
            raw_d = row.get(difficulty_key) or row.get("b") or row.get("irt_b")
            diff: float | None = float(raw_d) if isinstance(raw_d, (int, float)) else None
            out.append(
                ProblemItem(
                    id=rid,
                    question=str(qtext),
                    reference_answer=ans,
                    difficulty=diff,
                )
            )
    if normalize_difficulty_0_10 and out:
        lo, hi = b_range_from_pool(out)
        out = normalize_items_to_0_10(out, lo, hi)
    return out
