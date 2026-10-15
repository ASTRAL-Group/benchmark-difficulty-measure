#!/usr/bin/env python3
"""Convert random_baseline/results/<bench>/gap_rank_<r>/generated.jsonl into the
candidate schema expected by run_rollout.sh.

  ifeval     (lmms) : parent's instruction_id_list + kwargs (program-checked constraints)
  mmlu_bio   (lmms) : parent's options + answer
  bbh_3obj   (lmms) : parent's target letter
  aime*/amc23(math) : random's own claimed answer (parent's answer doesn't apply
                      since the rewrite changes the problem entirely)

Output files:
  lmms : verifier/lmms_candidates/<bench>_random_gap<r>.jsonl
  math : verifier/math_candidates/<bench>_random_gap<r>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_GAP = Path(__file__).resolve().parents[1].parent  # repo root
_RESULTS = _GAP / "random_baseline" / "results"
_LMMS = _GAP / "verifier" / "lmms_candidates"
_MATH = _GAP / "verifier" / "math_candidates"
_EVAL = _GAP / "data" / "pools"

_INLINE_MCQ_RE = re.compile(r"\n[A-Z]\.\s")


def _load_orig(benchmark: str) -> dict[str, dict]:
    path = _EVAL / benchmark / "test.jsonl"
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            ex = json.loads(line)
            keys = {f"q{i}"}
            if "idx" in ex: keys.add(f"q{int(ex['idx'])}")
            if "id" in ex and ex["id"] is not None: keys.add(str(ex["id"]))
            for k in keys: out[k] = ex
    return out


def _strip_inline_mcq(question: str) -> str:
    return _INLINE_MCQ_RE.split(question, maxsplit=1)[0].rstrip()


def _convert(bench: str, style: str, gap: int) -> None:
    src = _RESULTS / bench / f"gap_rank_{gap}" / "generated.jsonl"
    if style == "math":
        dst = _MATH / f"{bench}_random_gap{gap}.jsonl"
    else:
        dst = _LMMS / f"{bench}_random_gap{gap}.jsonl"
    if not src.is_file():
        print(f"[skip] {src} not found"); return

    orig = _load_orig(bench) if style != "math" else {}
    n_in = n_out = n_miss = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for row_idx, line in enumerate(fin):
            line = line.strip()
            if not line: continue
            n_in += 1
            item = json.loads(line)

            row: dict | None = None
            if style == "math":
                ans = str(item.get("answer") or "").strip()
                if not ans:
                    n_miss += 1; continue
                row = {
                    "idx": item.get("id") or f"rand_g{gap}_r{row_idx}",
                    "question": item.get("question") or "",
                    "gt_cot": "",
                    "gt": ans,
                    "answer": ans,
                }
            else:
                pid = item.get("parent_id")
                parent = orig.get(str(pid)) if pid is not None else None
                if parent is None:
                    n_miss += 1
                    print(f"[warn] parent_id={pid!r} not found in {bench}/test.jsonl")
                    continue

                if style == "ifeval":
                    il = parent.get("instruction_id_list") or []
                    kw = parent.get("kwargs") or []
                    if not il:
                        n_miss += 1; continue
                    row = {
                        "key": row_idx,
                        "prompt": item.get("question") or "",
                        "instruction_id_list": list(il),
                        "kwargs": list(kw),
                    }
                elif style == "mmlu_pro":
                    opts = parent.get("options") or []
                    ans = str(parent.get("answer") or "").strip()
                    if ans.startswith("(") and ans.endswith(")"):
                        ans = ans[1:-1]
                    if not opts or not ans:
                        n_miss += 1; continue
                    qtext = _strip_inline_mcq(item.get("question") or "")
                    row = {
                        "question_id": item.get("id") or f"rand_g{gap}_r{row_idx}",
                        "question": qtext,
                        "options": list(opts),
                        "answer": ans,
                        "answer_index": (ord(ans.upper()) - 65) if ans else 0,
                        "cot_content": "",
                        "category": parent.get("category", ""),
                        "src": "random_baseline",
                    }
                elif style == "bbh":
                    tgt = str(item.get("answer") or "").strip()
                    if not tgt:
                        n_miss += 1; continue
                    row = {
                        "input": item.get("question") or "",
                        "target": tgt,
                    }
                else:
                    raise ValueError(f"unknown style: {style!r}")

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[{bench} gap{gap}] in={n_in} out={n_out} missing={n_miss} -> {dst}")


_STYLES = {
    "ifeval": "ifeval",
    "mmlu_bio": "mmlu_pro",
    "bbh_3obj": "bbh",
    "aime24": "math", "aime25": "math", "aime26": "math",
    "amc23": "math", "gaokao2023en": "math",
    "math-500": "math", "gsm8k": "math",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=sorted(_STYLES.keys()))
    ap.add_argument("--gap-rank", required=True, help="int or 'all'")
    args = ap.parse_args()
    style = _STYLES[args.benchmark]

    if args.gap_rank == "all":
        bench_dir = _RESULTS / args.benchmark
        if not bench_dir.is_dir():
            raise SystemExit(f"no random_baseline results for {args.benchmark}")
        gaps = sorted(int(p.name.removeprefix("gap_rank_"))
                      for p in bench_dir.iterdir()
                      if p.is_dir() and p.name.startswith("gap_rank_"))
    else:
        gaps = [int(args.gap_rank)]

    for g in gaps:
        _convert(args.benchmark, style, g)


if __name__ == "__main__":
    main()
