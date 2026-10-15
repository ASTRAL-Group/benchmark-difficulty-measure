#!/usr/bin/env python3
"""
8-GPU split workflow: GENERATE (generator vLLM) and VERIFY (wave-by-wave local rollout) never run at once.

Parent-child tree algorithm
---------------------------
Round 0: anchor items are the parents.  Generates n_children items per anchor = A*n_children total.
Round r >= 1: selects k_parents items from all scored items (greedy: closest to target difficulty);
              generates n_children per parent = k_parents*n_children total.

Memory list (M items, shown to the LLM alongside each parent during generation):
  M/2 scored items closest below target difficulty + M/2 closest above.

Commands
--------
  1) init     — create workdir/state.json (anchors, target, generator settings, initial parents).
  2) generate — generator server UP; for each parent launch n_children tasks in parallel;
                writes round_<n>/generated.jsonl; then STOP the generator.
  3) verify   — all GPU memory free; runs verifier/run_rollout.sh wave-by-wave (full panel),
                then IRT estimation; updates all_scored_items, memory, parents; writes
                round_<n>/difficulty.csv.

Repeat generate/verify until ``done`` or max_rounds exhausted.

Run from ``propose/`` with ``PYTHONPATH=.``::

    python split_gpu_pipeline.py init --workdir ./split_wd ...
    python split_gpu_pipeline.py generate --workdir ./split_wd
    python split_gpu_pipeline.py verify --workdir ./split_wd
    python split_gpu_pipeline.py status --workdir ./split_wd
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _progress(iterable, total: int, desc: str):
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, unit="task", dynamic_ncols=True)
    return iterable

from correctness import NonEmptyFieldsVerifier
from correctness.difficulty import within_tolerance
from generate.vllm_generators import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    VllmOpenAICompatGenerator,
    VllmOpenAICompatSettings,
)
from run.pipeline import (
    append_round_jsonl,
    build_memory_list,
    select_parents,
    summarize_failures_for_next_round,
    write_anchor_dataset_ids_txt,
)
from utils.anchor_select import select_gap_end_anchors
from utils.config import default_run_config
from utils.difficulty_scale import b_range_from_values, normalize_items_to_0_10, to_difficulty_0_10
from utils.irt_join import merge_q_indexed_irt_csv_with_eval_jsonl
from utils.max_gap import (
    gap_midpoint_target,
    largest_normalized_gap_bracket_b,
    top_k_largest_gap_brackets_b,
)
from utils.schemas import (
    DifficultyResult,
    ProblemItem,
    RunConfig,
    ScoredCandidate,
    problem_item_to_json_dict,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GAP_FILLING_DIR = Path(__file__).resolve().parent.parent
_ROLLOUT_SCRIPT = _GAP_FILLING_DIR / "verifier" / "run_rollout.sh"
_MATH_CANDIDATES_DIR = _GAP_FILLING_DIR / "verifier" / "math_candidates"
_LMMS_CANDIDATES_DIR = _GAP_FILLING_DIR / "verifier" / "lmms_candidates"
# Bayesian IRT item difficulties (same tree as verifier.config.IRT_RESULTS_DIR).
# 12-model panel — used for finding gaps, anchors, target, and prompt-side difficulty annotations.
# Verifier rollout + IRT also use the same full panel (see verifier/run_rollout.sh).
_FULL_IRT_DIR = _GAP_FILLING_DIR / "irt_results"

# Benchmark → rollout framework. Math benchmarks go through verifier/diff_eval/math_eval_general.py;
# everything else goes through lmms-eval. Mirrors verifier/main.py LMMS_TASK_NAMES.
_LMMS_BENCHMARKS: set[str] = {
    "gpqa", "mmlu_bio", "mmlu_chem", "mmlu_phys", "medqa", "scienceqa",
    "bbh_3obj", "bbh_5obj", "bbh_7obj", "ifeval", "zebra",
}

# Free-response benchmarks where the lmms-eval task uses program-checked scoring
# (no letter/string match, no filter regex) — the candidate jsonl must carry the
# program-check metadata (e.g. ifeval's instruction_id_list + kwargs).
_IFEVAL_BENCHMARKS: set[str] = {"ifeval"}

# Per-benchmark MCQ specs. The generator prompt and the lmms candidate writer both
# branch on these. Style:
#   "bbh"      — options baked into ``question``, answer formatted as "(X)".
#                Candidate jsonl row: {input, target}.
#   "mmlu_pro" — separate ``options`` list, answer is a bare letter (e.g. "B").
#                Candidate jsonl row: {question, options, answer, category, ...}.
_MCQ_SPECS: dict[str, dict] = {
    "bbh_3obj":   {"style": "bbh",      "num_options": 3},
    "bbh_5obj":   {"style": "bbh",      "num_options": 5},
    "bbh_7obj":   {"style": "bbh",      "num_options": 7},
    "gpqa":       {"style": "bbh",      "num_options": 4},
    "scienceqa":  {"style": "bbh",      "num_options": 4},
    "medqa":      {"style": "bbh",      "num_options": 5},
    "mmlu_bio":   {"style": "mmlu_pro", "num_options": 10, "category": "biology"},
    "mmlu_chem":  {"style": "mmlu_pro", "num_options": 10, "category": "chemistry"},
    "mmlu_phys":  {"style": "mmlu_pro", "num_options": 10, "category": "physics"},
}


def _benchmark_framework(benchmark: str) -> str:
    return "lmms" if benchmark in _LMMS_BENCHMARKS else "math"

# ---------------------------------------------------------------------------
# Verifier helpers
# ---------------------------------------------------------------------------
def _write_math_candidates(items: list[ProblemItem], benchmark: str) -> Path:
    out_dir = _MATH_CANDIDATES_DIR / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "idx": it.id,
                "question": it.question,
                "gt_cot": "",
                "gt": it.reference_answer,
                "answer": it.reference_answer,
            }, ensure_ascii=False) + "\n")
    print(f"[verify] Wrote {len(items)} candidates → {out_path}", flush=True)
    return out_path


def _write_lmms_candidates(items: list[ProblemItem], benchmark: str) -> Path:
    """Write candidates for the lmms-eval framework. The verifier yamls under
    ``verifier/tasks/**`` point at this exact path, so the file
    location is part of the contract.

    Schema is benchmark-specific:
      - bbh-style ({input, target}): options are already baked into ``question``
        and answer is "(X)". Yaml's ``doc_to_text`` reads ``{{input}}``.
      - mmlu_pro-style ({question, options, answer, category, ...}): yaml's
        ``format_cot_example`` template renders Question:/Options:/A./B./...,
        matching the prompt format the calibration response_matrix was built with.
    """
    _LMMS_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _LMMS_CANDIDATES_DIR / f"{benchmark}.jsonl"
    spec = _MCQ_SPECS.get(benchmark)
    is_ifeval = benchmark in _IFEVAL_BENCHMARKS
    style = "ifeval" if is_ifeval else (spec or {}).get("style", "bbh")
    with out_path.open("w", encoding="utf-8") as f:
        for idx, it in enumerate(items):
            if is_ifeval:
                if not it.instruction_id_list:
                    raise ValueError(
                        f"item {it.id!r}: ifeval candidate requires inherited "
                        f"instruction_id_list (got {it.instruction_id_list!r})"
                    )
                # process_results uses doc["key"] as the lookup key. Try to recover
                # an int key from the id ("q123" → 123); else fall back to row index.
                kfx = it.id.lstrip("q")
                key_val = int(kfx) if kfx.isdigit() else idx
                f.write(json.dumps({
                    "key": key_val,
                    "prompt": it.question,
                    "instruction_id_list": list(it.instruction_id_list),
                    "kwargs": list(it.kwargs or []),
                }, ensure_ascii=False) + "\n")
                continue
            if style == "mmlu_pro":
                if not it.options:
                    raise ValueError(
                        f"item {it.id!r}: mmlu_pro candidate requires 'options' list "
                        f"(got {it.options!r})"
                    )
                # answer must be a bare letter to match doc_to_target: answer
                ans = it.reference_answer.strip()
                if ans.startswith("(") and ans.endswith(")"):
                    ans = ans[1:-1]
                row: dict[str, Any] = {
                    "question_id": it.id,
                    "question": it.question,
                    "options": list(it.options),
                    "answer": ans,
                    "answer_index": (ord(ans.upper()) - 65) if ans else 0,
                    "cot_content": "",
                    "category": spec.get("category", ""),
                    "src": "gap_filling",
                }
            else:
                row = {"input": it.question, "target": it.reference_answer}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[verify] Wrote {len(items)} lmms candidates ({style}) → {out_path}", flush=True)
    return out_path


def _write_candidates(items: list[ProblemItem], benchmark: str) -> Path:
    if _benchmark_framework(benchmark) == "lmms":
        return _write_lmms_candidates(items, benchmark)
    return _write_math_candidates(items, benchmark)


def _run_rollout(
    benchmark: str,
    results_dir: Path,
    difficulty_out: Path,
    n_sampling: int = 8,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    difficulty_out.parent.mkdir(parents=True, exist_ok=True)
    framework = _benchmark_framework(benchmark)
    cmd = [
        "bash", str(_ROLLOUT_SCRIPT),
        "--benchmark", benchmark,
        "--n", str(n_sampling),
        "--framework", framework,
        "--out", str(results_dir),
        "--difficulty-out", str(difficulty_out),
    ]
    repo_root = _GAP_FILLING_DIR
    print(f"[verify] Running rollout (framework={framework}, cwd={repo_root}): {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    print("[verify] Rollout complete.", flush=True)


def _read_difficulty_map(diff_csv: Path) -> dict[str, float]:
    """Read per-item difficulty on the 0–10 (12-model) scale from verifier's difficulty.csv."""
    import csv
    out: dict[str, float] = {}
    with diff_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row["item"])
            raw = row.get("difficulty")
            if raw in (None, "", "nan"):
                continue
            out[item_id] = float(raw)
    return out


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _pool_b_values(pool: list[ProblemItem]) -> list[float]:
    out = [float(p.difficulty) for p in pool if p.difficulty is not None]
    if len(out) != len(pool):
        raise ValueError("pool items must all have difficulty for max-gap + anchors")
    return out


def _benchmark_difficulty_csv(benchmark: str) -> Path:
    """Bayesian IRT reference difficulties for ``benchmark``."""
    return _FULL_IRT_DIR / f"{benchmark}_difficulty.csv"


def _load_anchors_and_target(
    workspace: Path,
    benchmark: str,
    per_side: int,
    gap_rank: int = 1,
) -> tuple[list[ProblemItem], float, float, float, list[ProblemItem], ProblemItem, ProblemItem]:
    """Load IRT pool and pick anchors around the ``gap_rank``-th largest gap
    (``gap_rank=1`` is the widest gap, ``=2`` is the second-widest, etc.).

    The 0–10 scale is the 12-model pool min/max normalization
    (``b_range_from_values`` over the 12-model Bayesian IRT b_mean values from
    ``irt_results/<bm>_difficulty.csv``). Pool's hardest item → 10, easiest → 0.
    Gap targets are always within [0, 10].

    Returns:
      anchors, target, b_min, b_max, pool, gap_left_item, gap_right_item
    """
    irt_csv = _benchmark_difficulty_csv(benchmark)
    eval_jsonl = workspace / "data" / "pools" / benchmark / "test.jsonl"
    if not irt_csv.is_file():
        raise SystemExit(f"missing difficulty csv: {irt_csv}")
    if not eval_jsonl.is_file():
        raise SystemExit(f"missing eval jsonl: {eval_jsonl}")
    pool_raw = merge_q_indexed_irt_csv_with_eval_jsonl(irt_csv, eval_jsonl)
    bs = _pool_b_values(pool_raw)
    b_min, b_max = b_range_from_values(bs)
    if gap_rank <= 1:
        gap_low_b, gap_high_b, gap_meta = largest_normalized_gap_bracket_b(bs)
    else:
        ranked = top_k_largest_gap_brackets_b(bs, gap_rank)
        if len(ranked) < gap_rank:
            raise SystemExit(
                f"benchmark {benchmark!r} has only {len(ranked)} adjacent gaps; "
                f"cannot pick gap_rank={gap_rank}"
            )
        gap_low_b, gap_high_b, gap_meta = ranked[gap_rank - 1]
    gap_low = to_difficulty_0_10(gap_low_b, b_min, b_max)
    gap_high = to_difficulty_0_10(gap_high_b, b_min, b_max)
    target = gap_midpoint_target(gap_low, gap_high)
    pool = normalize_items_to_0_10(pool_raw, b_min, b_max)
    anchors = select_gap_end_anchors(pool, gap_low, gap_high, per_side=per_side)
    # Resolve the two gap-bracket items via sorted index from gap_meta.
    pool_sorted_by_b = sorted(pool, key=lambda p: float(p.difficulty))
    gi = int(gap_meta["gap_index"])
    gap_left_item = pool_sorted_by_b[gi]
    gap_right_item = pool_sorted_by_b[gi + 1]
    return anchors, target, b_min, b_max, pool, gap_left_item, gap_right_item


def _closest_to_target(pool: list[ProblemItem], target: float, k: int) -> list[ProblemItem]:
    """Pick the k items from ``pool`` closest to ``target`` (0–10 difficulty)."""
    ranked = sorted(
        (p for p in pool if p.difficulty is not None),
        key=lambda p: abs(float(p.difficulty) - target),
    )
    return ranked[:k]


def _state_path(workdir: Path) -> Path:
    return workdir / "state.json"


def _load_state(workdir: Path) -> dict[str, Any]:
    p = _state_path(workdir)
    if not p.is_file():
        raise SystemExit(f"missing state file; run init first: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(workdir: Path, state: dict[str, Any]) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    _state_path(workdir).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _row_to_item(d: dict[str, Any]) -> ProblemItem:
    ans = d.get("answer") if d.get("answer") is not None else d.get("reference_answer")
    diff = d.get("difficulty")
    diff_f = float(diff) if isinstance(diff, (int, float)) else None
    mp = d.get("modification_policy")
    mp_s = str(mp).strip() if mp is not None and str(mp).strip() else None
    parent_id = d.get("parent_id")
    parent_id_s = str(parent_id).strip() if parent_id is not None and str(parent_id).strip() else None
    raw_opts = d.get("options")
    opts: list[str] | None = None
    if isinstance(raw_opts, list) and raw_opts:
        opts = [str(o) for o in raw_opts]
    raw_il = d.get("instruction_id_list")
    il: list[str] | None = [str(x) for x in raw_il] if isinstance(raw_il, list) and raw_il else None
    raw_kw = d.get("kwargs")
    kw: list[dict] | None = [dict(x) if isinstance(x, dict) else {} for x in raw_kw] if isinstance(raw_kw, list) and raw_kw else None
    return ProblemItem(
        id=str(d.get("id") or "gen"),
        question=str(d.get("question") or ""),
        reference_answer=str(ans or ""),
        difficulty=diff_f,
        modification_policy=mp_s,
        parent_id=parent_id_s,
        options=opts,
        instruction_id_list=il,
        kwargs=kw,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    workspace = Path(args.workspace).resolve()
    max_rounds = max(1, int(args.max_rounds))

    gap_rank = max(1, int(getattr(args, "gap_rank", 1)))
    gap_low_ov = getattr(args, "gap_low", None)
    gap_high_ov = getattr(args, "gap_high", None)
    if gap_low_ov is not None and gap_high_ov is not None:
        # Mass-fill / custom bracket on 0–10: keep original pool for anchors only.
        irt_csv = _benchmark_difficulty_csv(args.benchmark)
        eval_jsonl = workspace / "data" / "pools" / args.benchmark / "test.jsonl"
        if not irt_csv.is_file():
            raise SystemExit(f"missing difficulty csv: {irt_csv}")
        if not eval_jsonl.is_file():
            raise SystemExit(f"missing eval jsonl: {eval_jsonl}")
        pool_raw = merge_q_indexed_irt_csv_with_eval_jsonl(irt_csv, eval_jsonl)
        bs = _pool_b_values(pool_raw)
        b_min, b_max = b_range_from_values(bs)
        pool = normalize_items_to_0_10(pool_raw, b_min, b_max)
        gap_low = float(gap_low_ov)
        gap_high = float(gap_high_ov)
        if gap_low > gap_high:
            gap_low, gap_high = gap_high, gap_low
        target = float(args.target_difficulty) if getattr(args, "target_difficulty", None) is not None else gap_midpoint_target(gap_low, gap_high)
        anchors = select_gap_end_anchors(pool, gap_low, gap_high, per_side=args.per_side)
        if not anchors:
            anchors = _closest_to_target(pool, target, max(2, 2 * int(args.per_side)))
        gap_left_full = ProblemItem(id="virtual_gap_lo", question="", reference_answer="", difficulty=gap_low)
        gap_right_full = ProblemItem(id="virtual_gap_hi", question="", reference_answer="", difficulty=gap_high)
        gap_rank = 0  # custom bracket (not static IRT rank)
    else:
        anchors, target, b_min, b_max, pool, gap_left_full, gap_right_full = _load_anchors_and_target(
            workspace, args.benchmark, args.per_side, gap_rank=gap_rank
        )
        if getattr(args, "target_difficulty", None) is not None:
            target = float(args.target_difficulty)
    gap_full_left = {
        "id": gap_left_full.id,
        "difficulty_full": float(gap_left_full.difficulty),
    }
    gap_full_right = {
        "id": gap_right_full.id,
        "difficulty_full": float(gap_right_full.difficulty),
    }

    run_id = f"split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = workdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_anchor_dataset_ids_txt(run_dir, anchors)

    max_tok = args.max_tokens
    if max_tok is None:
        v = os.environ.get("PROPOSE_MAX_COMPLETION_TOKENS", "").strip()
        max_tok = int(v) if v.isdigit() else DEFAULT_MAX_COMPLETION_TOKENS

    pipeline_mode = str(args.pipeline_mode)
    if pipeline_mode not in ("direct", "two_stage"):
        raise SystemExit(f"--pipeline-mode must be 'direct' or 'two_stage', got {pipeline_mode!r}")

    state: dict[str, Any] = {
        "version": 5,
        "workspace": str(workspace),
        "benchmark": args.benchmark,
        "per_side": args.per_side,
        "target_difficulty": target,  # 12-model 0–10 scale (gap midpoint)
        "b_min": b_min,
        "b_max": b_max,
        "gap_full_left": gap_full_left,
        "gap_full_right": gap_full_right,
        "pipeline_mode": pipeline_mode,
        # Parent-child algorithm parameters
        "k_parents": args.k_parents,          # K: parents per round (after round 0)
        "n_children": args.n_children,        # N: children per parent
        "memory_size": args.memory_size,      # M: memory list size
        "parent_selection": args.parent_selection,
        # Bookkeeping
        "max_rounds": max_rounds,
        "early_stop_tolerance": float(args.early_stop_tolerance),
        "n_sampling": int(args.n_sampling),
        "feedback_context": "",
        "current_round": 0,
        "gap_rank": gap_rank,
        "run_id": run_id,
        "anchors": [problem_item_to_json_dict(a) for a in anchors],
        "all_scored_items": [],
        "generator": {
            "base_urls": [u.strip() for u in str(args.base_url).split(",") if u.strip()],
            "model": str(args.model),
            "temperature": float(args.temperature),
            "top_p": args.top_p,
            "top_k": args.top_k,
            "system_prompt": args.system_prompt,
            "max_tokens": int(max_tok),
        },
    }

    if pipeline_mode == "two_stage":
        # Stage 1: scratch generation using the memory_size closest benchmark items
        # to target as style references. Anchors are the 4 closest.
        # ifeval inherits the donor's instruction_id_list/kwargs unchanged (strategy c),
        # so each task that reuses a donor produces an item with an identical constraint
        # structure. Expand the donor pool to k_parents*n_children for ifeval so every
        # task in an attempt gets a unique structure (preserving 1 unique structure per
        # generated item). Non-ifeval benchmarks keep the configured memory_size.
        is_ifeval_init = str(args.benchmark) in _IFEVAL_BENCHMARKS
        stage1_memory_size = int(args.memory_size)
        if is_ifeval_init:
            stage1_memory_size = max(stage1_memory_size, int(args.k_parents) * int(args.n_children))
        stage1_memory = _closest_to_target(pool, target, stage1_memory_size)
        state.update({
            "stage": "stage1",
            "stage1_tolerance": float(args.stage1_tolerance),
            "stage1_min_pass": int(args.stage1_min_pass),
            "stage1_max_attempts": int(args.stage1_max_attempts),
            "stage1_attempts_used": 0,
            "stage1_passing_items": [],
            "current_parents": [],
            "memory_items": [problem_item_to_json_dict(m) for m in stage1_memory],
            "status": "need_generate",
        })
    else:
        # Legacy direct mode: anchors are round-0 parents, memory fills up as rounds run.
        state.update({
            "stage": "stage2",
            "current_parents": [problem_item_to_json_dict(a) for a in anchors],
            "memory_items": [],
            "status": "need_generate",
        })

    _save_state(workdir, state)

    # ECDF chart of full-scale difficulty + top-5 gap markers, saved next to state.json.
    try:
        ecdf_path = _write_ecdf_plot(workspace, args.benchmark, run_dir, top_k=5)
        print(f"  ecdf plot → {ecdf_path}")
    except Exception as e:
        print(f"  WARNING: ECDF plot failed ({e!r}); continuing without it.")

    n_servers = len(state["generator"]["base_urls"])
    print(
        f"init: workdir={workdir} run_id={run_id!r} "
        f"target={target:.4f} (IRT 0–10, gap_rank={gap_rank}) servers={n_servers}"
    )
    print(f"  pipeline_mode={pipeline_mode}")
    print(f"  anchors ({len(anchors)}): {', '.join(a.id for a in anchors)}")
    print(f"  max_rounds={max_rounds}")
    if pipeline_mode == "two_stage":
        stage1_n = int(args.k_parents) * int(args.n_children)
        print(
            f"  stage1: {stage1_n} scratch items per attempt, tolerance={args.stage1_tolerance}, "
            f"min_pass={args.stage1_min_pass}, max_attempts={args.stage1_max_attempts}, "
            f"memory={len(state['memory_items'])} benchmark items"
        )
        print(f"  stage2: modification rounds (full verifier) after stage1 succeeds")
    else:
        print(f"  round 0 will generate {len(anchors)} × {args.n_children} = {len(anchors) * args.n_children} items")
        print(f"  subsequent rounds: {args.k_parents} parents × {args.n_children} = {args.k_parents * args.n_children} items")
    print(f"  memory_size={args.memory_size}, parent_selection={args.parent_selection!r}")
    print(f"  run_dir={run_dir}")
    print(f"Next: start generator vLLM, then: python split_gpu_pipeline.py generate --workdir {workdir}")


def cmd_generate(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    state = _load_state(workdir)
    if state["status"] not in ("need_generate",):
        raise SystemExit(f"generate expects status=need_generate, got {state['status']!r}")

    stage = str(state.get("stage", "stage2"))
    is_stage1 = stage == "stage1"

    memory_items = [_row_to_item(d) for d in state.get("memory_items", [])]
    r = int(state["current_round"])
    run_dir = workdir / state["run_id"]

    if is_stage1:
        attempt_idx = int(state.get("stage1_attempts_used", 0))
        round_dir = run_dir / f"stage1_attempt_{attempt_idx}"
        # Stage 1 synthesizes N_scratch = k_parents * n_children items from scratch.
        # We reuse "current_parents" as placeholders (None) so task fan-out stays the same.
        n_scratch = int(state["k_parents"]) * int(state["n_children"])
        current_parents: list[ProblemItem | None] = [None] * n_scratch
    else:
        round_dir = run_dir / f"round_{r}"
        current_parents = [_row_to_item(d) for d in state["current_parents"]]
    round_dir.mkdir(parents=True, exist_ok=True)

    g = state["generator"]
    # Support both old single base_url and new comma-separated / list base_urls.
    raw_urls = args.base_url or ",".join(g.get("base_urls") or [g.get("base_url", "")])
    base_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not base_urls:
        raise SystemExit("No base_url configured. Pass --base-url or set it during init.")
    model = args.model or g["model"]
    if not str(model).strip():
        raise SystemExit("--model required in state or pass --model")
    temperature = float(args.temperature) if args.temperature is not None else float(g["temperature"])
    max_tokens = int(args.max_tokens) if args.max_tokens is not None else int(g["max_tokens"])
    top_p = float(args.top_p) if args.top_p is not None else g.get("top_p")
    top_k = int(args.top_k) if args.top_k is not None else g.get("top_k")
    system_prompt = (
        args.system_prompt if args.system_prompt is not None else g.get("system_prompt")
    )

    # One generator per server — tasks are distributed round-robin across them.
    generators = [
        VllmOpenAICompatGenerator(
            VllmOpenAICompatSettings(
                base_url=url,
                model=str(model).strip(),
                temperature=temperature,
                top_p=float(top_p) if top_p is not None else None,
                top_k=int(top_k) if top_k is not None else None,
                system_prompt=str(system_prompt) if system_prompt else None,
                max_tokens=max_tokens,
            )
        )
        for url in base_urls
    ]

    cfg: RunConfig = default_run_config()
    cfg.target_difficulty = float(state["target_difficulty"])
    cfg.irt_b_min = float(state["b_min"])
    cfg.irt_b_max = float(state["b_max"])
    cfg.n_children = int(state["n_children"])
    cfg.memory_size = int(state["memory_size"])
    cfg.mcq_spec = _MCQ_SPECS.get(state["benchmark"])
    cfg.ifeval_mode = state["benchmark"] in _IFEVAL_BENCHMARKS
    cfg.artifact_dir = str(run_dir)
    if is_stage1:
        cfg.run_id_for_artifacts = f"propose_gen_{state['benchmark']}_stage1_a{attempt_idx}"
    else:
        cfg.run_id_for_artifacts = f"propose_gen_{state['benchmark']}_round{r}"
    cfg.current_round_index = 0

    target = float(state["target_difficulty"])
    fb = str(state.get("feedback_context") or "")
    n_children = int(state["n_children"])

    # Build tasks: (task_index, parent_or_none, child_index)
    tasks: list[tuple[int, ProblemItem | None, int]]
    if is_stage1:
        # One task per scratch item; no parent to fan out over.
        tasks = [(i, None, i) for i in range(len(current_parents))]
    else:
        tasks = [
            (task_idx, parent, child_idx)
            for task_idx, (parent, child_idx) in enumerate(
                (parent, child_idx)
                for parent in current_parents
                for child_idx in range(n_children)
            )
        ]

    if is_stage1:
        print(
            f"[generate] Stage1 attempt {attempt_idx}: {len(tasks)} scratch tasks across "
            f"{len(generators)} server(s), memory={len(memory_items)} items",
            flush=True,
        )
    else:
        print(
            f"[generate] Round {r}: {len(current_parents)} parents × {n_children} children "
            f"= {len(tasks)} tasks across {len(generators)} server(s), memory={len(memory_items)} items",
            flush=True,
        )

    def _gen_one(task_idx: int, parent: ProblemItem | None, child_idx: int) -> ProblemItem:
        # Round-robin server assignment.
        gen = generators[task_idx % len(generators)]
        # Unique, deterministic seed per task so each child differs.
        seed_salt = attempt_idx if is_stage1 else r
        task_seed = (seed_salt * 100003 + task_idx * 9973 + random.randint(0, 9999)) & 0x7FFF_FFFF
        task_cfg = dataclasses.replace(cfg, seed=task_seed)
        if parent is None:
            tag = f"scratch_a{attempt_idx}_t{task_idx}"
        else:
            tag = f"r{r}_p{parent.id}_c{child_idx}"
        # Strategy (c) for ifeval: pre-select the donor so the SAME donor drives
        # both (a) the propose prompt the LLM sees (constraint_donor below) and
        # (b) the instruction_id_list / kwargs we inherit afterward. Selecting
        # them independently — as a previous version did — let the LLM write a
        # prompt for one constraint while the grader checked a different one,
        # producing prompts that "answered" the wrong question.
        donor: ProblemItem | None = None
        if cfg.ifeval_mode:
            if parent is not None:
                donor = parent
            elif memory_items:
                donor = memory_items[task_idx % len(memory_items)]
        item = gen.generate_one(
            parent,
            memory=memory_items,
            target_difficulty=target,
            feedback_context=fb,
            config=task_cfg,
            artifact_tag=tag,
            constraint_donor=donor,
        )
        # Guarantee a unique ID regardless of what the model returned.
        if parent is None:
            item.id = f"gen_scratch_a{attempt_idx}_t{task_idx}"
        else:
            item.id = f"gen_r{r}_{parent.id}_c{child_idx}"
        if cfg.ifeval_mode and donor is not None:
            item.instruction_id_list = list(donor.instruction_id_list or [])
            item.kwargs = list(donor.kwargs or [])
        return item

    items: list[ProblemItem] = []
    slot_retries = max(1, int(os.environ.get("PROPOSE_GEN_SLOT_RETRIES", "3")))
    pending: list[tuple[int, ProblemItem | None, int]] = list(tasks)
    attempt_by_slot: dict[tuple[str, int], int] = {}

    def _slot_key(parent: ProblemItem | None, child_idx: int) -> tuple[str, int]:
        return (parent.id if parent is not None else "scratch", child_idx)

    while pending:
        batch = pending
        pending = []
        n_workers = max(1, len(batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_to_info = {
                ex.submit(_gen_one, task_idx, parent, child_idx): (
                    parent,
                    child_idx,
                )
                for task_idx, parent, child_idx in batch
            }
            desc = f"Stage1 a{attempt_idx} generate" if is_stage1 else f"Round {r} generate"
            bar = _progress(
                concurrent.futures.as_completed(fut_to_info),
                total=len(fut_to_info),
                desc=desc,
            )
            for fut in bar:
                parent, child_idx = fut_to_info[fut]
                parent_id = parent.id if parent is not None else "scratch"
                sk = _slot_key(parent, child_idx)
                try:
                    items.append(fut.result())
                    print(f"[generate] ✓ parent={parent_id} child={child_idx}", flush=True)
                except Exception as e:
                    n_try = attempt_by_slot.get(sk, 0) + 1
                    attempt_by_slot[sk] = n_try
                    print(
                        f"[generate] ✗ parent={parent_id} child={child_idx} "
                        f"(attempt {n_try}/{slot_retries}): {e}",
                        flush=True,
                    )
                    if n_try < slot_retries:
                        task_idx = next(
                            ti for ti, p, ci in tasks if _slot_key(p, ci) == sk
                        )
                        pending.append((task_idx, parent, child_idx))
                    else:
                        print(
                            f"[generate] giving up on parent={parent_id} child={child_idx} "
                            f"after {slot_retries} attempts",
                            flush=True,
                        )

        if pending:
            print(
                f"[generate] Retrying {len(pending)} failed slot(s) "
                f"(target {len(tasks)} items, have {len(items)} so far)…",
                flush=True,
            )

    out_path = round_dir / "generated.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(problem_item_to_json_dict(it), ensure_ascii=False) + "\n")

    state["status"] = "need_verify"
    if is_stage1:
        state["last_generate_dir"] = str(round_dir)
    else:
        state["last_generate_round"] = r
    _save_state(workdir, state)
    print(f"[generate] Wrote {len(items)} items → {out_path}")
    print("STOP the generator vLLM to free GPUs, then:")
    print(f"  python split_gpu_pipeline.py verify --workdir {workdir}")


def _item_dict_with_est_diff(it: ProblemItem, est_by_id: dict[str, float]) -> dict[str, Any]:
    """Serialize a ProblemItem, injecting its estimated difficulty into the
    ``difficulty`` field so downstream _row_to_item/prompt code sees the
    measured value rather than whatever the generator originally claimed."""
    d = problem_item_to_json_dict(it)
    if it.id in est_by_id:
        d["difficulty"] = float(est_by_id[it.id])
    return d


def _write_ecdf_plot(
    workspace: Path,
    benchmark: str,
    run_dir: Path,
    *,
    top_k: int = 5,
) -> Path:
    """Write a PNG of the 12-model 0–10 difficulty ECDF (pool min/max normalization)
    with the top-``top_k`` gaps highlighted.

    Output: ``run_dir / "ecdf_full_top_gaps.png"``.
    """
    irt_csv = _benchmark_difficulty_csv(benchmark)
    eval_jsonl = workspace / "data" / "pools" / benchmark / "test.jsonl"
    pool_raw = merge_q_indexed_irt_csv_with_eval_jsonl(irt_csv, eval_jsonl)
    bs = _pool_b_values(pool_raw)
    b_min, b_max = b_range_from_values(bs)
    items_by_b = sorted(pool_raw, key=lambda p: float(p.difficulty))
    diffs_010 = sorted(to_difficulty_0_10(b, b_min, b_max) for b in bs)

    ranked = top_k_largest_gap_brackets_b(bs, k=top_k)
    gaps: list[dict[str, Any]] = []
    for rk, (b_lo, b_hi, meta) in enumerate(ranked, start=1):
        gi = int(meta["gap_index"])
        lo_010 = to_difficulty_0_10(b_lo, b_min, b_max)
        hi_010 = to_difficulty_0_10(b_hi, b_min, b_max)
        gaps.append({
            "rank": rk,
            "lo_id": items_by_b[gi].id,
            "hi_id": items_by_b[gi + 1].id,
            "lo_010": lo_010,
            "hi_010": hi_010,
            "target": gap_midpoint_target(lo_010, hi_010),
            "norm_w": float(meta["normalized_gap_width"]),
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    n = len(diffs_010)
    ys = [(i + 1) / n for i in range(n)]
    ax.step(diffs_010, ys, where="post", color="black", linewidth=1.4,
            label=f"ECDF (n={n} items)")

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]  # red/orange/green/blue/purple
    for g in gaps:
        c = colors[(g["rank"] - 1) % len(colors)]
        ax.axvspan(g["lo_010"], g["hi_010"], alpha=0.18, color=c, zorder=1)
        ax.axvline(g["target"], color=c, linestyle="--", linewidth=1.0, alpha=0.85, zorder=2)
        # Stagger labels vertically so they don't overlap.
        y_pos = 0.97 - 0.07 * (g["rank"] - 1)
        ax.annotate(
            f"#{g['rank']}: [{g['lo_id']}→{g['hi_id']}]\n"
            f"t={g['target']:.2f}  w={g['norm_w']:.3f}",
            xy=(g["target"], y_pos),
            ha="center", va="top", fontsize=8, color=c,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=c, linewidth=0.8, alpha=0.95),
        )

    ax.set_xlabel("Difficulty (0-10, full 12-model Bayesian IRT)")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{benchmark}: full-scale difficulty ECDF with top-{top_k} gaps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = run_dir / "ecdf_full_top_gaps.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _gap_context_for_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable gap context for the run, written into every candidate row.

    Schema:
      gap_full: {left: {id, difficulty_full}, right: {id, difficulty_full}} —
                12-model items bracketing the gap (0–10 scale).
      gap_rank: int — which gap (1=widest).

    The 12-model target is recorded per-candidate inside ``estimated_difficulty.target_12m``.
    """
    return {
        "gap_full": {
            "left": dict(state.get("gap_full_left") or {}),
            "right": dict(state.get("gap_full_right") or {}),
        },
        "gap_rank": int(state.get("gap_rank", 1)),
    }


def _est_diff_field(b: float | None, target: float) -> dict[str, Any] | None:
    """Per-candidate 12-model 0–10 difficulty record. Returns ``None`` if estimate is missing."""
    if b is None:
        return None
    return {
        "estimate": float(b),
        "target": float(target),
    }


def _verify_stage1(workdir: Path, state: dict[str, Any]) -> None:
    """Stage-1 verification (two-stage pipeline only).

    Runs a full-panel rollout over the latest scratch-attempt batch, counts how many
    items fall within ``stage1_tolerance`` of the target, and transitions to
    stage2 once cumulative passing items ≥ ``stage1_min_pass``. Otherwise
    retries up to ``stage1_max_attempts`` before failing with
    ``done_reason = "stage1_failed"``.
    """
    run_dir = workdir / state["run_id"]
    attempt_idx = int(state.get("stage1_attempts_used", 0))
    last_dir = Path(state.get("last_generate_dir") or (run_dir / f"stage1_attempt_{attempt_idx}"))
    gen_path = last_dir / "generated.jsonl"
    if not gen_path.is_file():
        raise SystemExit(f"missing {gen_path}")

    items: list[ProblemItem] = []
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(_row_to_item(json.loads(line)))

    benchmark = state["benchmark"]
    n_sampling = int(state.get("n_sampling", 8))
    target = float(state["target_difficulty"])
    tol = float(state["stage1_tolerance"])
    min_pass = int(state["stage1_min_pass"])
    max_attempts = int(state["stage1_max_attempts"])

    _write_candidates(items, benchmark)
    rollout_results_dir = last_dir / "rollout_results"
    diff_csv = last_dir / "difficulty.csv"
    _run_rollout(benchmark, rollout_results_dir, diff_csv, n_sampling=n_sampling)

    difficulty_map = _read_difficulty_map(diff_csv)
    print(
        f"[verify] stage1 attempt {attempt_idx}: difficulty_map: {len(difficulty_map)} items",
        flush=True,
    )

    # math framework writes ProblemItem.id into the csv item column; lmms writes the
    # lmms-eval doc_id (row index). Try id first, fall back to row index.
    estimated_by_id: dict[str, float] = {}
    for i, it in enumerate(items):
        if it.id in difficulty_map:
            key = it.id
        elif str(i) in difficulty_map:
            key = str(i)
        else:
            continue
        estimated_by_id[it.id] = difficulty_map[key]

    correctness = NonEmptyFieldsVerifier()
    scored: list[ScoredCandidate] = []
    jsonl_rows: list[dict[str, Any]] = []
    new_passing: list[tuple[ProblemItem, float]] = []
    gap_ctx = _gap_context_for_state(state)
    for it in _progress(items, total=len(items), desc="Scoring stage1 items"):
        c_res = correctness.verify(it)
        b = estimated_by_id.get(it.id)
        d_res: DifficultyResult | None = None
        if c_res.ok and b is not None:
            d_res = DifficultyResult(estimate=b, target=target)
        accepted = False
        if d_res is not None:
            accepted = c_res.ok and within_tolerance(d_res.estimate, target, tol)
        sc = ScoredCandidate(candidate=it, correctness=c_res, difficulty=d_res, accepted=accepted)
        scored.append(sc)
        if accepted and d_res is not None:
            new_passing.append((it, float(d_res.estimate)))
        jsonl_rows.append(
            {
                "stage": "stage1",
                "attempt": attempt_idx,
                "tolerance": tol,
                "id": it.id,
                "question": it.question,
                "options": it.options,
                "answer": it.reference_answer,
                "correctness_ok": c_res.ok,
                "estimated_difficulty": _est_diff_field(b, target),
                "accepted": accepted,
                **gap_ctx,
            }
        )

    stage1_log = last_dir / "candidates.jsonl"
    with stage1_log.open("a", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prev_passing_dicts: list[dict[str, Any]] = state.get("stage1_passing_items", [])
    new_passing_dicts = [
        {
            **problem_item_to_json_dict(it),
            "estimated_difficulty": _est_diff_field(d_b, target),
        }
        for it, d_b in new_passing
    ]
    passing_dicts = prev_passing_dicts + new_passing_dicts
    state["stage1_passing_items"] = passing_dicts

    print(
        f"[verify] stage1 attempt {attempt_idx}: {len(new_passing)} new passing (tol={tol}, "
        f"cumulative {len(passing_dicts)}/{min_pass})",
        flush=True,
    )

    if len(passing_dicts) >= min_pass:
        # Transition to stage 2: anchors become parents AND memory for round 0.
        anchors = [_row_to_item(d) for d in state["anchors"]]
        state["stage"] = "stage2"
        state["current_round"] = 0
        state["status"] = "need_generate"
        state["feedback_context"] = ""
        state["current_parents"] = [problem_item_to_json_dict(a) for a in anchors]
        state["memory_items"] = [problem_item_to_json_dict(a) for a in anchors]
        _save_state(workdir, state)
        print(
            f"[verify] stage1 → stage2: {len(passing_dicts)} passing items collected; "
            f"starting modification rounds with {len(anchors)} anchors as parents."
        )
        print(f"  python split_gpu_pipeline.py generate --workdir {workdir}")
        return

    if attempt_idx + 1 >= max_attempts:
        state["status"] = "done"
        state["done_reason"] = "stage1_failed"
        state["succeeded"] = False
        state["best_item"] = None
        _save_state(workdir, state)
        print(
            f"[verify] stage1 FAILED after {max_attempts} attempts "
            f"({len(passing_dicts)}/{min_pass} passing). Stopping."
        )
        return

    state["stage1_attempts_used"] = attempt_idx + 1
    state["status"] = "need_generate"
    _save_state(workdir, state)
    print(
        f"[verify] stage1 attempt {attempt_idx} insufficient; retrying "
        f"(attempt {attempt_idx + 1}/{max_attempts})."
    )
    print(f"  python split_gpu_pipeline.py generate --workdir {workdir}")


def cmd_verify(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    state = _load_state(workdir)
    if state["status"] != "need_verify":
        raise SystemExit(f"verify expects status=need_verify, got {state['status']!r}")

    if str(state.get("stage", "stage2")) == "stage1":
        _verify_stage1(workdir, state)
        return

    r = int(state["last_generate_round"])
    max_rounds = int(state["max_rounds"])

    run_dir = workdir / state["run_id"]
    gen_path = run_dir / f"round_{r}" / "generated.jsonl"
    if not gen_path.is_file():
        raise SystemExit(f"missing {gen_path}")

    items: list[ProblemItem] = []
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(_row_to_item(json.loads(line)))

    benchmark = state["benchmark"]
    n_sampling = int(state.get("n_sampling", 8))
    target = float(state["target_difficulty"])
    early_stop_tol = float(state.get("early_stop_tolerance", 0.15))
    k_parents = int(state["k_parents"])
    memory_size = int(state["memory_size"])
    parent_selection = str(state.get("parent_selection", "greedy"))
    gap_ctx = _gap_context_for_state(state)

    # 1. Write candidates
    _write_candidates(items, benchmark)

    # 2. Wave-by-wave rollout + IRT estimation (12-model 0–10 in difficulty.csv)
    round_dir = run_dir / f"round_{r}"
    rollout_results_dir = round_dir / "rollout_results"
    diff_csv = round_dir / "difficulty.csv"
    _run_rollout(benchmark, rollout_results_dir, diff_csv, n_sampling=n_sampling)

    difficulty_map = _read_difficulty_map(diff_csv)
    print(f"[verify] round={r} difficulty_map: {len(difficulty_map)} items", flush=True)

    # math framework writes ProblemItem.id into the csv item column; lmms writes the
    # lmms-eval doc_id (row index). Try id first, fall back to row index.
    estimated_by_id: dict[str, float] = {}
    for i, it in enumerate(items):
        if it.id in difficulty_map:
            key = it.id
        elif str(i) in difficulty_map:
            key = str(i)
        else:
            continue
        estimated_by_id[it.id] = difficulty_map[key]

    correctness = NonEmptyFieldsVerifier()
    scored: list[ScoredCandidate] = []
    jsonl_rows: list[dict[str, Any]] = []

    for it in _progress(items, total=len(items), desc="Scoring items"):
        c_res = correctness.verify(it)
        b = estimated_by_id.get(it.id)
        d_res: DifficultyResult | None = None
        if c_res.ok and b is not None:
            d_res = DifficultyResult(estimate=b, target=target)
        sc = ScoredCandidate(candidate=it, correctness=c_res, difficulty=d_res, accepted=False)
        scored.append(sc)
        jsonl_rows.append(
            {
                "round": r,
                "id": it.id,
                "question": it.question,
                "options": it.options,
                "answer": it.reference_answer,
                "parent_id": it.parent_id,
                "modification_policy": it.modification_policy,
                "item_difficulty": it.difficulty,
                "correctness_ok": c_res.ok,
                "estimated_difficulty": _est_diff_field(b, target),
                **gap_ctx,
            }
        )

    append_round_jsonl(run_dir, r, jsonl_rows)

    # Merge scored items into all_scored_items.
    prev_scored_dicts: list[dict[str, Any]] = state.get("all_scored_items", [])

    def _row_b(d: dict[str, Any]) -> float | None:
        ed = d.get("estimated_difficulty")
        if isinstance(ed, dict) and ed.get("estimate") is not None:
            return float(ed["estimate"])
        return None

    prev_scored_items: list[tuple[ProblemItem, float]] = []
    for d in prev_scored_dicts:
        b_prev = _row_b(d)
        if b_prev is None:
            continue
        prev_scored_items.append((_row_to_item(d), b_prev))

    new_scored_items: list[tuple[ProblemItem, float]] = [
        (s.candidate, s.difficulty.estimate)
        for s in scored
        if s.difficulty is not None
    ]
    all_scored_items = prev_scored_items + new_scored_items

    all_scored_dicts: list[dict[str, Any]] = [
        {
            **problem_item_to_json_dict(it),
            "estimated_difficulty": _est_diff_field(float(d_b), target),
        }
        for it, d_b in all_scored_items
    ]

    # Early stop: this round's best (closest to target) within ±early_stop_tol of target.
    round_scored = [s for s in scored if s.difficulty is not None]
    print(
        f"[verify] round={r} items={len(scored)} scored={len(difficulty_map)} "
        f"valid={len(round_scored)}"
    )

    if round_scored:
        best = min(round_scored, key=lambda s: abs(s.difficulty.estimate - target))
        delta = abs(best.difficulty.estimate - target)
        if delta < early_stop_tol:
            state["status"] = "done"
            state["done_reason"] = "best_within_tol"
            state["succeeded"] = True
            state["best_item"] = {
                **problem_item_to_json_dict(best.candidate),
                "estimated_difficulty": _est_diff_field(
                    float(best.difficulty.estimate), target
                ),
                **gap_ctx,
            }
            state["all_scored_items"] = all_scored_dicts
            _save_state(workdir, state)
            print(
                f"[verify] Early stop — best item id={best.candidate.id!r} "
                f"difficulty={best.difficulty.estimate:.3f} target={target:.3f} "
                f"|Δ|={delta:.3f} < {early_stop_tol}"
            )
            return

    if r + 1 >= max_rounds:
        state["status"] = "done"
        state["done_reason"] = "max_rounds_exhausted"
        if round_scored:
            best = min(round_scored, key=lambda s: abs(s.difficulty.estimate - target))
            state["succeeded"] = False
            state["best_item"] = {
                **problem_item_to_json_dict(best.candidate),
                "estimated_difficulty": _est_diff_field(
                    float(best.difficulty.estimate), target
                ),
                **gap_ctx,
            }
        else:
            state["succeeded"] = False
            state["best_item"] = None
        state["all_scored_items"] = all_scored_dicts
        _save_state(workdir, state)
        print(f"[verify] Max rounds reached without |Δ|<{early_stop_tol}.")
        return

    memory_items = build_memory_list(all_scored_items, target, memory_size)
    next_parents = select_parents(all_scored_items, target, k_parents, parent_selection)

    est_by_id = {it.id: d for it, d in all_scored_items}
    failures = [s for s in scored if s.difficulty is None or abs(s.difficulty.estimate - target) >= early_stop_tol]
    fb = summarize_failures_for_next_round(failures, target, early_stop_tol).to_prompt_bullets()

    print(
        f"[verify] memory={len(memory_items)} parents={len(next_parents)} for round {r + 1}",
        flush=True,
    )
    if next_parents:
        diffs = [f"{it.id}(d={d:.2f})" for it, d in all_scored_items
                 if it.id in {p.id for p in next_parents}]
        print(f"[verify] Next parents: {diffs}", flush=True)

    state["current_round"] = r + 1
    state["status"] = "need_generate"
    state["feedback_context"] = fb
    state["all_scored_items"] = all_scored_dicts
    state["memory_items"] = [_item_dict_with_est_diff(it, est_by_id) for it in memory_items]
    state["current_parents"] = [_item_dict_with_est_diff(it, est_by_id) for it in next_parents]
    _save_state(workdir, state)
    print(f"[verify] Next round prepared (round {r + 1}).")
    print(f"  python split_gpu_pipeline.py generate --workdir {workdir}")


def cmd_status(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.workdir).resolve())
    skip = {"anchors", "current_parents", "memory_items", "all_scored_items"}
    print(json.dumps({k: v for k, v in state.items() if k not in skip}, indent=2, ensure_ascii=False))
    print(f"anchors: {len(state.get('anchors', []))} items")
    print(f"current_parents: {len(state.get('current_parents', []))} items")
    print(f"memory_items: {len(state.get('memory_items', []))} items")
    print(f"all_scored_items: {len(state.get('all_scored_items', []))} items")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split GPU pipeline: parent-child tree generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create workdir/state.json")
    p_init.add_argument("--workdir", type=Path, required=True)
    p_init.add_argument("--workspace", type=Path, default=Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent))))
    p_init.add_argument("--benchmark", default="aime25")
    p_init.add_argument("--per-side", type=int, default=2,
                        help="Anchor items per gap side (total anchors = 2 * per_side)")
    p_init.add_argument("--gap-rank", type=int, default=1, dest="gap_rank",
                        help="Which gap to target: 1=widest (default), 2=second-widest, ...")
    p_init.add_argument("--gap-low", type=float, default=None, dest="gap_low",
                        help="Override gap bracket low end on 0–10 (with --gap-high; mass-fill).")
    p_init.add_argument("--gap-high", type=float, default=None, dest="gap_high",
                        help="Override gap bracket high end on 0–10 (with --gap-low; mass-fill).")
    p_init.add_argument("--target-difficulty", type=float, default=None, dest="target_difficulty",
                        help="Override target on 0–10 (default: midpoint of gap bracket).")
    p_init.add_argument("--k-parents", type=int, default=4, dest="k_parents",
                        help="K: parents selected per round after round 0 (default 4)")
    p_init.add_argument("--n-children", type=int, default=4, dest="n_children",
                        help="N: children generated per parent (default 4)")
    p_init.add_argument("--memory-size", type=int, default=8, dest="memory_size",
                        help="M: memory list size (M/2 per side of target, default 8)")
    p_init.add_argument("--parent-selection", default="greedy", dest="parent_selection",
                        choices=["greedy"],
                        help="Parent selection policy (default: greedy)")
    p_init.add_argument("--max-rounds", type=int, default=3)
    p_init.add_argument("--early-stop-tolerance", type=float, default=0.15, dest="early_stop_tolerance",
                        help="Stop when this round's best item has |estimate − target| < this value (12-model 0–10 scale). Default 0.15.")
    p_init.add_argument("--n-sampling", type=int, default=8,
                        help="Rollout samples per item for verifier (default 8; use 32 to match IRT calibration)")
    p_init.add_argument("--pipeline-mode", default="direct", dest="pipeline_mode",
                        choices=["direct", "two_stage"],
                        help="'direct' (legacy): round-0 parents = anchors. "
                             "'two_stage': first scratch-generate items, then modify. Default: direct.")
    p_init.add_argument("--stage1-tolerance", type=float, default=2.0, dest="stage1_tolerance",
                        help="Stage-1 difficulty tolerance on 0–10 scale (wider than stage-2). Default 2.0.")
    p_init.add_argument("--stage1-min-pass", type=int, default=4, dest="stage1_min_pass",
                        help="Cumulative passing-item count required to exit stage 1. Default 4.")
    p_init.add_argument("--stage1-max-attempts", type=int, default=3, dest="stage1_max_attempts",
                        help="Maximum stage-1 scratch attempts before giving up. Default 3.")
    p_init.add_argument("--base-url", default=os.environ.get("PROPOSE_VLLM_BASE_URL", "http://127.0.0.1:8010/v1"))
    p_init.add_argument("--model", default=os.environ.get("PROPOSE_VLLM_MODEL", "qwen3.5-397b"))
    p_init.add_argument("--temperature", type=float, default=0.7)
    p_init.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        help="Sampling top_p (MiniMax-M2.7 card: 0.95)",
    )
    p_init.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Sampling top_k (MiniMax-M2.7 card: 40)",
    )
    p_init.add_argument(
        "--system-prompt",
        default=None,
        dest="system_prompt",
        help="Optional system message prepended to every generation request",
    )
    p_init.add_argument("--max-tokens", type=int, default=None)

    p_gen = sub.add_parser("generate", help="Run generator (generator vLLM must be up)")
    p_gen.add_argument("--workdir", type=Path, required=True)
    p_gen.add_argument("--base-url", default=None)
    p_gen.add_argument("--model", default=None)
    p_gen.add_argument("--temperature", type=float, default=None)
    p_gen.add_argument("--top-p", type=float, default=None, dest="top_p")
    p_gen.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_gen.add_argument("--system-prompt", default=None, dest="system_prompt")
    p_gen.add_argument("--max-tokens", type=int, default=None)

    p_ver = sub.add_parser("verify", help="Score items via local wave-by-wave rollout (all GPUs free)")
    p_ver.add_argument("--workdir", type=Path, required=True)

    p_st = sub.add_parser("status", help="Print state summary (without item lists)")
    p_st.add_argument("--workdir", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
