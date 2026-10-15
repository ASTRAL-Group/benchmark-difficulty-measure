"""End-to-end parent-child tree proposal loop.

Algorithm overview
------------------
Round 0 (initial):
  - Parents = anchor items (A items, selected around the difficulty gap).
  - Generate N children per anchor → A*N items total.

Round r >= 1:
  - Select K parents from *all* scored items so far (default: greedy — K closest to target).
  - Generate N children per parent → K*N items total.

Between rounds:
  - Memory list (M items shown to the LLM with each generation prompt):
      M/2 closest scored items from below target difficulty,
      M/2 closest scored items from above target difficulty.
  - Anchor items are treated like any other generated items for memory/parent selection.

Each child is generated in a separate API call (one parent → one child), parallelised with threads.

How to read this file:
  1. ``build_memory_list`` / ``select_parents`` — pure utilities, no I/O.
  2. ``run_pipeline`` — the main loop; calls generator, scores items, updates state.
  3. ``append_round_jsonl`` / ``save_run_state`` — persistence helpers.
  4. ``summarize_failures_for_next_round`` — optional feedback text for the generator.
"""

from __future__ import annotations

import concurrent.futures
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from correctness.difficulty import (
    DifficultyEstimator,
    edit_hints_for_difficulty,
    within_tolerance,
)
from correctness.verifier import CorrectnessVerifier
from generate.generator import ItemGenerator
from utils.schemas import (
    CorrectnessResult,
    DifficultyResult,
    ProblemItem,
    RoundRecord,
    RunConfig,
    RunState,
    ScoredCandidate,
    VerifierFeedback,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def write_anchor_dataset_ids_txt(run_dir: Path, anchors: Sequence[ProblemItem]) -> Path:
    """Write each anchor's ``ProblemItem.id`` from the source pool (one id per line)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "anchor_dataset_ids.txt"
    header = (
        "# Anchor ids (same order as initial parent items in round 0).\n"
        "# One dataset id per line (ProblemItem.id from the merged pool).\n"
    )
    body = "\n".join(a.id for a in anchors)
    path.write_text(header + body + "\n", encoding="utf-8")
    return path


def append_round_jsonl(run_dir: Path, round_index: int, rows: list[dict[str, Any]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"candidates_round_{round_index}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_memory_list(
    all_scored: list[tuple[ProblemItem, float]],
    target: float,
    memory_size: int,
) -> list[ProblemItem]:
    """Build the memory list: up to ``memory_size // 2`` items from each side of ``target``.

    Left side (difficulty <= target): sorted descending (closest to target first).
    Right side (difficulty > target): sorted ascending (closest to target first).
    The actual number added to each side is min(memory_size // 2, items available on that side).
    """
    half = max(1, memory_size // 2)
    left = [(it, d) for it, d in all_scored if d <= target]
    right = [(it, d) for it, d in all_scored if d > target]
    left.sort(key=lambda x: x[1], reverse=True)   # descending: closest below target first
    right.sort(key=lambda x: x[1])                 # ascending: closest above target first
    n_left = min(half, len(left))
    n_right = min(half, len(right))
    return [it for it, _ in left[:n_left]] + [it for it, _ in right[:n_right]]


def select_parents(
    all_scored: list[tuple[ProblemItem, float]],
    target: float,
    k: int,
    policy: str = "greedy",
) -> list[ProblemItem]:
    """Select ``k`` parent items for the next generation round.

    Policies:
      - ``"greedy"``: pick the K items with difficulty closest to ``target`` (by absolute distance).
    """
    if not all_scored:
        return []
    if policy == "greedy":
        sorted_items = sorted(all_scored, key=lambda x: abs(x[1] - target))
        return [it for it, _ in sorted_items[:k]]
    raise ValueError(f"Unknown parent_selection policy: {policy!r}. Supported: 'greedy'")


def summarize_failures_for_next_round(
    failures: list[ScoredCandidate],
    target: float,
    tol: float,
) -> VerifierFeedback:
    """Turn rejected candidates into short text for the next generator call."""
    hints: list[str] = []
    codes: list[str] = []
    last_diff: DifficultyResult | None = None
    for s in failures:
        if not s.correctness.ok:
            codes.append(str(s.correctness.failure_code))
            if s.correctness.evidence:
                hints.append(f"Correctness: {s.correctness.evidence}")
            continue
        if s.difficulty is None:
            continue
        last_diff = s.difficulty
        if not within_tolerance(s.difficulty.estimate, target, tol):
            hints.extend(edit_hints_for_difficulty(s.difficulty.estimate, target))
    seen: set[str] = set()
    uniq = [h for h in hints if h not in seen and not seen.add(h)]
    return VerifierFeedback(
        validity=False,
        difficulty=last_diff,
        failure_codes=list(dict.fromkeys(codes)),
        edit_hints=uniq[:8],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _run_state_to_json(state: RunState) -> dict[str, Any]:
    def item_dict(it: ProblemItem) -> dict[str, Any]:
        return asdict(it)

    def corr(c: CorrectnessResult) -> dict[str, Any]:
        return asdict(c)

    def diff(d: DifficultyResult | None) -> dict[str, Any] | None:
        return None if d is None else asdict(d)

    def pack_scored(s: ScoredCandidate) -> dict[str, Any]:
        return {
            "candidate": item_dict(s.candidate),
            "correctness": corr(s.correctness),
            "difficulty": diff(s.difficulty),
            "accepted": s.accepted,
        }

    rounds_out: list[dict[str, Any]] = []
    for rr in state.rounds:
        rounds_out.append(
            {
                "round_index": rr.round_index,
                "feedback_context": rr.feedback_context,
                "scored": [pack_scored(x) for x in rr.scored],
            }
        )

    return {
        "run_id": state.run_id,
        "config": asdict(state.config),
        "anchors": [item_dict(a) for a in state.anchors],
        "rounds": rounds_out,
        "accepted": [pack_scored(x) for x in state.accepted],
    }


def save_run_state(base_dir: Path, state: RunState) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _run_state_to_json(state)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    path = run_dir / "run_state.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_pipeline(
    anchors: Sequence[ProblemItem],
    generator: ItemGenerator,
    correctness: CorrectnessVerifier,
    difficulty: DifficultyEstimator,
    config: RunConfig,
    *,
    run_id: str | None = None,
) -> RunState:
    """Parent-child tree generation loop.

    Round 0: ``anchors`` are the parents; generates ``len(anchors) * config.n_children`` items.
    Round r >= 1: selects ``config.k_per_round`` parents from all scored items (greedy by default);
                  generates ``config.k_per_round * config.n_children`` items.

    Each item is generated by a separate ``generator.generate_one`` call (one parent → one child).
    Calls within a round are parallelised with ``ThreadPoolExecutor``.
    """
    rid = run_id or new_run_id()
    state = RunState(run_id=rid, config=config, anchors=list(anchors))
    base = Path(config.artifact_dir)
    run_dir = base / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    write_anchor_dataset_ids_txt(run_dir, anchors)
    config.run_id_for_artifacts = rid

    # Accumulate all scored (item, estimated_difficulty) pairs across rounds for parent selection
    # and memory list construction.
    all_scored: list[tuple[ProblemItem, float]] = []
    memory: list[ProblemItem] = []           # current memory list shown to the LLM
    current_parents: list[ProblemItem] = list(anchors)
    feedback_context = ""
    max_rounds = 1 if config.single_round_mode else config.max_rounds

    for round_idx in range(max_rounds):
        config.current_round_index = round_idx
        n_children = config.n_children

        # Build list of (parent, child_index) tasks for this round.
        tasks: list[tuple[ProblemItem, int]] = [
            (parent, child_idx)
            for parent in current_parents
            for child_idx in range(n_children)
        ]

        def _make_generate(parent: ProblemItem, child_idx: int, mem: list[ProblemItem]) -> ProblemItem:
            tag = f"parent_{parent.id}_child_{child_idx}"
            return generator.generate_one(
                parent,
                memory=mem,
                target_difficulty=config.target_difficulty,
                feedback_context=feedback_context,
                config=config,
                artifact_tag=tag,
            )

        # Capture current memory snapshot for this round (avoid closure-capture issues).
        _memory_snapshot = list(memory)
        items: list[ProblemItem] = []
        n_workers = max(1, len(tasks))
        print(
            f"[pipeline] Round {round_idx}: {len(current_parents)} parents × {n_children} children "
            f"= {len(tasks)} tasks (workers={n_workers})",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_to_task = {
                ex.submit(_make_generate, parent, child_idx, _memory_snapshot): (parent.id, child_idx)
                for parent, child_idx in tasks
            }
            for fut in concurrent.futures.as_completed(fut_to_task):
                parent_id, child_idx = fut_to_task[fut]
                try:
                    items.append(fut.result())
                except Exception as e:
                    print(
                        f"[pipeline] Child generation failed (parent={parent_id}, child={child_idx}): {e}",
                        flush=True,
                    )

        # Score all generated items.
        scored: list[ScoredCandidate] = []
        jsonl_rows: list[dict[str, Any]] = []

        for it in items:
            c_res = correctness.verify(it)
            d_res: DifficultyResult | None = None
            if c_res.ok:
                d_res = difficulty.estimate(
                    it,
                    target=config.target_difficulty,
                    anchors=anchors,
                )
            ok_diff = (
                d_res is not None
                and within_tolerance(d_res.estimate, config.target_difficulty, config.difficulty_tolerance)
            )
            accepted = c_res.ok and ok_diff
            sc = ScoredCandidate(candidate=it, correctness=c_res, difficulty=d_res, accepted=accepted)
            scored.append(sc)
            if c_res.ok and d_res is not None:
                all_scored.append((it, d_res.estimate))

            jsonl_rows.append(
                {
                    "round": round_idx,
                    "id": it.id,
                    "question": it.question,
                    "answer": it.reference_answer,
                    "parent_id": it.parent_id,
                    "modification_policy": it.modification_policy,
                    "item_difficulty": it.difficulty,
                    "correctness_ok": c_res.ok,
                    "estimated_difficulty": None
                    if d_res is None
                    else {"estimate": d_res.estimate, "target": d_res.target},
                    "accepted": accepted,
                }
            )

        append_round_jsonl(run_dir, round_idx, jsonl_rows)
        state.rounds.append(
            RoundRecord(
                round_index=round_idx,
                feedback_context=feedback_context,
                scored=scored,
            )
        )

        # Condition 1: any item within tolerance → keep only the single closest to target.
        round_accepted = [s for s in scored if s.accepted and s.difficulty is not None]
        if round_accepted:
            best = min(round_accepted, key=lambda s: abs(s.difficulty.estimate - config.target_difficulty))
            state.accepted = [best]
            break

        if config.single_round_mode:
            break

        # Update memory and select parents for the next round.
        memory = build_memory_list(all_scored, config.target_difficulty, config.memory_size)
        current_parents = select_parents(
            all_scored, config.target_difficulty, config.k_per_round, config.parent_selection
        )
        if not current_parents:
            print("[pipeline] No scored items available for parent selection; stopping.", flush=True)
            break

        failures = [s for s in scored if not s.accepted]
        feedback_context = summarize_failures_for_next_round(
            failures, config.target_difficulty, config.difficulty_tolerance
        ).to_prompt_bullets()

        print(
            f"[pipeline] Round {round_idx} done: {len(scored)} items scored, "
            f"{len(all_scored)} total scored, memory={len(memory)}, "
            f"next parents={len(current_parents)}",
            flush=True,
        )

    save_run_state(base, state)
    config.run_id_for_artifacts = None
    config.current_round_index = 0
    return state
