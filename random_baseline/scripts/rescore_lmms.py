#!/usr/bin/env python3
"""Re-score lmms-eval rollout sample jsonls against a corrected target rubric.

Use case: random_baseline candidates may be converted with the WRONG target (e.g.,
parent's answer when LLM's own answer was correct). The rollout responses are
still valid; only the grader's target needs updating. This script:

  1. Reads the corrected candidate jsonl (doc_id == row index)
  2. For each model's *samples*.jsonl in rollout_results/, re-grades by comparing
     filtered_resps against the new target (case-insensitive after stripping the
     yaml's regexes_to_ignore)
  3. Writes the updated sample jsonl back, with corrected `target` and `exact_match`

After running, re-run verifier.main on the same rollout-dir to produce a corrected
difficulty.csv. Lossless: `resps` and `filtered_resps` are not modified.

Usage:
  python rescore_lmms.py --candidate <new_candidate.jsonl> --rollout-dir <rollout_results>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Mirrors the bbh / mmlu_pro / ifeval-style yaml `regexes_to_ignore` for exact_match.
_REGEXES_TO_IGNORE = [r"\.$", r",", r"\\", r"\n", r'"']


def _normalize(s: str) -> str:
    s = s.lower()
    for pat in _REGEXES_TO_IGNORE:
        s = re.sub(pat, "", s)
    return s.strip()


def _grade_one(filtered_resp: str, target: str) -> int:
    return int(_normalize(filtered_resp) == _normalize(target))


def _rescore_sample(sample: dict, new_target: str) -> dict:
    sample["target"] = new_target
    resps = sample.get("filtered_resps") or []
    if not resps:
        return sample
    grades = [_grade_one(str(r), new_target) for r in resps]
    sample["exact_match"] = sum(grades) / len(grades)
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="Path to corrected candidate jsonl (doc_id == row index).")
    ap.add_argument("--rollout-dir", required=True,
                    help="Path to rollout_results/ containing per-model subdirectories.")
    ap.add_argument("--target-field", default="target",
                    help="Field name in candidate jsonl to read target from (default: target).")
    args = ap.parse_args()

    cand = Path(args.candidate)
    targets: dict[int, str] = {}
    with cand.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get(args.target_field)
            if t is None:
                continue
            targets[i] = str(t)
    print(f"loaded {len(targets)} new targets from {cand}")

    rollout = Path(args.rollout_dir)
    if not rollout.is_dir():
        raise SystemExit(f"rollout-dir not found: {rollout}")

    total_files = total_samples = total_changed = 0
    for model_dir in sorted(rollout.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "logs":
            continue
        for jsonl in model_dir.rglob("*samples*.jsonl"):
            total_files += 1
            n_in_file = 0
            n_changed = 0
            updated_lines = []
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    n_in_file += 1
                    doc_id = sample.get("doc_id")
                    if doc_id is None:
                        updated_lines.append(line)
                        continue
                    new_t = targets.get(int(doc_id))
                    if new_t is None:
                        updated_lines.append(line)
                        continue
                    old_target = sample.get("target")
                    sample = _rescore_sample(sample, new_t)
                    if str(old_target) != str(new_t):
                        n_changed += 1
                    updated_lines.append(json.dumps(sample, ensure_ascii=False))
            with jsonl.open("w", encoding="utf-8") as f:
                for l in updated_lines:
                    f.write(l + "\n")
            total_samples += n_in_file
            total_changed += n_changed
            print(f"  {model_dir.name}/{jsonl.name}: {n_in_file} samples ({n_changed} target changed)")
    print(f"total: {total_files} files, {total_samples} samples, {total_changed} target updated")


if __name__ == "__main__":
    main()
