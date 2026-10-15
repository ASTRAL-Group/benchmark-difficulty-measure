#!/usr/bin/env python3
"""Budget-matched best-of-N uniform-parent rewriting baseline — generation.

Paper:
  For each gap g with T_g BENCHFILL rounds, sample 4 original benchmark items
  uniformly as parents; rewrite 4*T_g children per parent (total 16*T_g),
  conditioned on parent + target difficulty only (no memory / iterative retrieval).

Writes:
  random_baseline/results/<bench>/gap_rank_<r>/parents.json
  random_baseline/results/<bench>/gap_rank_<r>/generated.jsonl

Usage (same MiniMax serve as BENCHFILL method — locked):
  bash random_baseline/scripts/run_budget_match.sh serve
  # → propose/sh/serve_minimax_m2_7_8gpu.sh  (TP=4 DP=2 PP=1 EP=1, 8 GPUs @8010)

  bash random_baseline/scripts/run_budget_match.sh generate aime24 2
  bash random_baseline/scripts/run_budget_match.sh generate
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

# Repo root, resolved from this file's location (portable; no hardcoded absolute paths)
_SCRIPTS = Path(__file__).resolve().parent
_GAP = _SCRIPTS.parents[1]  # repo root
_RESULTS = _GAP / "random_baseline" / "results"
_POOLS = _GAP / "data" / "pools"  # original benchmark items
_DEFAULT_SPEC = _RESULTS / "_budget_spec.json"

# Ensure propose/ is importable
_PROPOSE = _GAP / "propose"
if str(_PROPOSE) not in sys.path:
    sys.path.insert(0, str(_PROPOSE))

from generate.generator import build_baseline_child_prompt  # noqa: E402
from generate.json_items import parse_single_item_from_model_text  # noqa: E402
from generate.vllm_generators import (  # noqa: E402
    DEFAULT_MAX_COMPLETION_TOKENS,
    MINIMAX_M2_7_SYSTEM_PROMPT,
    MINIMAX_M2_7_TEMPERATURE,
    MINIMAX_M2_7_TOP_K,
    MINIMAX_M2_7_TOP_P,
)
from utils.schemas import ProblemItem  # noqa: E402

_MCQ_SPECS: dict[str, dict] = {
    "bbh_3obj": {"style": "bbh", "num_options": 3},
    "mmlu_bio": {"style": "mmlu_pro", "num_options": 10, "category": "biology"},
}
_IFEVAL = {"ifeval"}

_N_PARENTS = 4
_N_CHILDREN_PER_CHUNK = 4  # per fake-round chunk; chunks = T_g


def _gap_seed(global_seed: int, benchmark: str, gap_rank: int) -> int:
    h = hashlib.sha256(f"{global_seed}:{benchmark}:{gap_rank}".encode()).hexdigest()
    return int(h[:8], 16)


def _load_pool(benchmark: str) -> list[ProblemItem]:
    path = _POOLS / benchmark / "test.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing pool: {path}")
    items: list[ProblemItem] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "id" in ex and ex["id"] is not None:
                pid = str(ex["id"])
            elif "idx" in ex:
                pid = str(ex["idx"])
            else:
                pid = f"q{i}"
            q = ex.get("question") or ex.get("problem") or ex.get("prompt") or ex.get("input") or ""
            ans = ex.get("answer") or ex.get("target") or ex.get("gt") or ""
            opts = ex.get("options")
            if opts is not None:
                opts = [str(o) for o in opts]
            iid_list = ex.get("instruction_id_list")
            kwargs = ex.get("kwargs")
            items.append(
                ProblemItem(
                    id=pid,
                    question=str(q),
                    reference_answer=str(ans),
                    options=opts,
                    instruction_id_list=list(iid_list) if iid_list else None,
                    kwargs=list(kwargs) if kwargs else None,
                )
            )
    if len(items) < _N_PARENTS:
        raise SystemExit(f"{benchmark}: pool size {len(items)} < {_N_PARENTS}")
    return items


def _sample_parents(pool: list[ProblemItem], rng: random.Random) -> list[ProblemItem]:
    return rng.sample(pool, _N_PARENTS)


def _parse_base_urls(raw: str) -> list[str]:
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise SystemExit("no base URL configured")
    return urls


def _call_llm(
    *,
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    system_prompt: str,
    max_tokens: int,
    seed: int | None,
    parse_retries: int,
) -> ProblemItem:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    create_kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "extra_body": {"top_k": top_k},
    }
    if seed is not None:
        create_kwargs["seed"] = seed

    last_err: Exception | None = None
    for attempt in range(max(1, parse_retries)):
        resp = client.chat.completions.create(**create_kwargs)
        msg = resp.choices[0].message
        c = (getattr(msg, "content", None) or "").strip()
        rc = str(getattr(msg, "reasoning_content", None) or "").strip()
        text = "\n\n".join(x for x in (rc, c) if x) if (rc or c) else ""
        finish = str(getattr(resp.choices[0], "finish_reason", None) or "")
        try:
            return parse_single_item_from_model_text(text)
        except ValueError as e:
            last_err = e
            if attempt + 1 >= max(1, parse_retries):
                break
            create_kwargs["seed"] = (int(create_kwargs.get("seed") or 0) + 1) % (2**31 - 1)
            if finish == "length":
                recovery = (
                    "Your previous response was cut off before the JSON was delivered. "
                    "Output **only** one ```json code block containing a single-element array. "
                    "No analysis.\n\n"
                    f"Original task:\n{prompt}"
                )
                create_kwargs["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": recovery},
                ]
    raise ValueError(f"parse failed after {parse_retries} attempts: {last_err}")


def _row_from_item(
    *,
    item: ProblemItem,
    item_id: str,
    parent: ProblemItem,
    target: float,
    benchmark: str,
) -> dict:
    # mmlu: inherit parent options + answer (converter / paper tooling convention)
    if benchmark == "mmlu_bio":
        ans = str(parent.reference_answer or "").strip()
        if ans.startswith("(") and ans.endswith(")"):
            ans = ans[1:-1]
        return {
            "id": item_id,
            "parent_id": parent.id,
            "question": item.question,
            "answer": ans,
            "options": list(parent.options or []),
            "target_difficulty": target,
        }
    if benchmark in _IFEVAL:
        return {
            "id": item_id,
            "parent_id": parent.id,
            "question": item.question,  # parse maps prompt → question
            "answer": "",
            "target_difficulty": target,
            "instruction_id_list": list(parent.instruction_id_list or []),
            "kwargs": list(parent.kwargs or []),
        }
    return {
        "id": item_id,
        "parent_id": parent.id,
        "question": item.question,
        "answer": item.reference_answer,
        "target_difficulty": target,
    }


def _save_prompt_artifact(art_dir: Path, tag: str, prompt: str) -> None:
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")


def _item_id(gap_rank: int, p_idx: int, r: int, c: int) -> str:
    return f"rand_g{gap_rank}_p{p_idx}_r{r}_c{c}"


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def generate_gap(
    gap: dict,
    *,
    base_urls: list[str],
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    system_prompt: str,
    max_tokens: int,
    global_seed: int,
    force: bool,
    max_workers: int,
    slot_retries: int,
    parse_retries: int,
) -> Path:
    bm = gap["benchmark"]
    g = int(gap["gap_rank"])
    t_g = int(gap["T_g"])
    target = float(gap["target_difficulty"])
    budget = int(gap["budget"])
    out_dir = _RESULTS / bm / f"gap_rank_{g}"
    gen_path = out_dir / "generated.jsonl"
    parents_path = out_dir / "parents.json"

    # tasks: (p_idx, r, c)
    tasks: list[tuple[int, int, int]] = [
        (p, r, c)
        for p in range(_N_PARENTS)
        for r in range(t_g)
        for c in range(_N_CHILDREN_PER_CHUNK)
    ]
    assert len(tasks) == budget, (len(tasks), budget)

    results: dict[tuple[int, int, int], dict] = {}
    if gen_path.is_file() and not force:
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = re.fullmatch(r"rand_g(\d+)_p(\d+)_r(\d+)_c(\d+)", str(row.get("id", "")))
            if not m or int(m.group(1)) != g:
                continue
            results[(int(m.group(2)), int(m.group(3)), int(m.group(4)))] = row
        if len(results) >= budget:
            print(f"[skip] {bm} gap{g}: {gen_path} already has {len(results)}>={budget} lines")
            return gen_path
        if results:
            print(
                f"[resume] {bm} gap{g}: keeping {len(results)}/{budget} existing slots",
                flush=True,
            )

    rng = random.Random(_gap_seed(global_seed, bm, g))
    pool = _load_pool(bm)
    parents = _sample_parents(pool, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    if force or not parents_path.is_file():
        parents_path.write_text(
            json.dumps(
                {
                    "benchmark": bm,
                    "gap_rank": g,
                    "T_g": t_g,
                    "target_difficulty": target,
                    "seed": _gap_seed(global_seed, bm, g),
                    "method_locked_client": {
                        "model": model,
                        "base_urls": base_urls,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "max_tokens": max_tokens,
                        "system_prompt": system_prompt,
                    },
                    "parents": [
                        {"index": i, "id": p.id, "question": p.question, "answer": p.reference_answer}
                        for i, p in enumerate(parents)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    mcq_spec = _MCQ_SPECS.get(bm)
    ifeval_mode = bm in _IFEVAL
    art_dir = out_dir / "gen_artifacts"

    pending = [t for t in tasks if t not in results]
    print(
        f"[gen] {bm} gap{g}: T_g={t_g} budget={budget} target={target:.4f} "
        f"parents={[p.id for p in parents]} pending={len(pending)}/{budget} "
        f"workers={max_workers} endpoints={len(base_urls)}",
        flush=True,
    )

    write_lock = threading.Lock()

    def _checkpoint() -> None:
        ordered = [results[t] for t in tasks if t in results]
        with write_lock:
            _atomic_write_jsonl(gen_path, ordered)

    def _one(task: tuple[int, int, int]) -> tuple[tuple[int, int, int], dict | None, str | None]:
        p_idx, r, c = task
        parent = parents[p_idx]
        item_id = _item_id(g, p_idx, r, c)
        prompt = build_baseline_child_prompt(
            parent,
            target_difficulty=target,
            mcq_spec=mcq_spec,
            ifeval_mode=ifeval_mode,
        )
        _save_prompt_artifact(art_dir, item_id, prompt)
        # Round-robin across endpoints (mirrors split_gpu_pipeline multi-URL).
        # With serve_minimax_m2_7_8gpu.sh there is one URL; vLLM DP=2 load-balances.
        base_url = base_urls[(p_idx * t_g * _N_CHILDREN_PER_CHUNK + r * _N_CHILDREN_PER_CHUNK + c) % len(base_urls)]
        last_err = None
        for attempt in range(max(1, slot_retries)):
            try:
                item = _call_llm(
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    seed=_gap_seed(global_seed, bm, g) + p_idx * 1000 + r * 10 + c + attempt,
                    parse_retries=parse_retries,
                )
                item.id = item_id
                item.parent_id = parent.id
                row = _row_from_item(
                    item=item, item_id=item_id, parent=parent, target=target, benchmark=bm
                )
                return task, row, None
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                print(f"[gen] ✗ {item_id} attempt {attempt + 1}/{slot_retries}: {e}", flush=True)
                time.sleep(0.5)
        return task, None, last_err

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, t) for t in pending]
            for fut in concurrent.futures.as_completed(futs):
                task, row, err = fut.result()
                if row is not None:
                    results[task] = row
                    _checkpoint()
                    print(f"[gen] ✓ {row['id']} ({len(results)}/{budget})", flush=True)
                else:
                    print(f"[gen] FAIL {task}: {err}", flush=True)

    ordered = [results[t] for t in tasks if t in results]
    _atomic_write_jsonl(gen_path, ordered)

    print(f"[gen] wrote {len(ordered)}/{budget} → {gen_path}", flush=True)
    if len(ordered) < budget:
        print(f"[warn] shortfall {budget - len(ordered)} items for {bm} gap{g}", flush=True)
    return gen_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--gap-rank", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Regenerate even if complete.")
    ap.add_argument(
        "--base-url",
        default=os.environ.get(
            "VLLM_BASE_URLS",
            os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1"),
        ),
        help="Must match method serve (serve_minimax_m2_7_8gpu.sh → :8010/v1).",
    )
    ap.add_argument("--model", default=os.environ.get("MODEL", "minimax-m2.7"))
    ap.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("GEN_TEMPERATURE", MINIMAX_M2_7_TEMPERATURE)),
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("GEN_TOP_P", MINIMAX_M2_7_TOP_P)),
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("GEN_TOP_K", MINIMAX_M2_7_TOP_K)),
    )
    ap.add_argument(
        "--system-prompt",
        default=os.environ.get("GEN_SYSTEM_PROMPT", MINIMAX_M2_7_SYSTEM_PROMPT),
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("PROPOSE_MAX_COMPLETION_TOKENS", DEFAULT_MAX_COMPLETION_TOKENS)),
        help="Match method state.json generator.max_tokens (30000).",
    )
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("BASELINE_GEN_WORKERS", "16")))
    ap.add_argument("--slot-retries", type=int, default=3)
    ap.add_argument("--parse-retries", type=int, default=5)
    args = ap.parse_args()

    if not args.spec.is_file():
        raise SystemExit(
            f"missing {args.spec}; run: python random_baseline/scripts/build_budget_spec.py"
        )
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    gaps = spec["gaps"]
    global_seed = int(spec.get("global_seed", 42))
    base_urls = _parse_base_urls(args.base_url)
    print(
        f"[gen] method-locked: model={args.model!r} urls={base_urls} "
        f"temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"max_tokens={args.max_tokens} workers={args.max_workers}",
        flush=True,
    )

    if args.all:
        selected = gaps
    else:
        if not args.benchmark or args.gap_rank is None:
            raise SystemExit("pass --all or both --benchmark and --gap-rank")
        selected = [
            g for g in gaps if g["benchmark"] == args.benchmark and g["gap_rank"] == args.gap_rank
        ]
        if not selected:
            raise SystemExit(f"no gap {args.benchmark} gap_rank={args.gap_rank} in spec")

    for gap in selected:
        generate_gap(
            gap,
            base_urls=base_urls,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            global_seed=global_seed,
            force=args.force,
            max_workers=args.max_workers,
            slot_retries=args.slot_retries,
            parse_retries=args.parse_retries,
        )


if __name__ == "__main__":
    main()
