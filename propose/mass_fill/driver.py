"""Mass-fill driver: iterative max-gap midpoint fill (MiniMax generate + 12-model verify).

Usage:
  PYTHONPATH=propose python -m mass_fill.driver --benchmark aime24
  PYTHONPATH=propose python -m mass_fill.driver status --benchmark aime24
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow ``python -m mass_fill.driver`` from repo with PYTHONPATH=propose
_PROPOSE = Path(__file__).resolve().parents[1]
if str(_PROPOSE) not in sys.path:
    sys.path.insert(0, str(_PROPOSE))

from mass_fill.distribution import (  # noqa: E402
    MassFillState,
    VirtualPoint,
    append_fill,
    default_target_count,
    init_state,
    load_state,
    pick_max_gap,
    results_dir,
    save_state,
)
from mass_fill.health import append_log, print_status, write_heartbeat  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tolerance_for_gap(
    gap_width: float,
    *,
    frac: float = 0.08,
    floor: float = 0.05,
    cap: float = 0.25,
) -> float:
    """Relative early-stop tolerance on the 0–10 scale.

    Default: ``8%`` of current max-gap width, clipped to ``[floor, cap]``.
    Matches the old absolute 0.15 on AIME24's initial max gap (~1.79 → 8%≈0.14);
    later narrower gaps tighten acceptance automatically.
    Floor avoids impossible precision given n=32 IRT noise.
    """
    w = max(0.0, float(gap_width))
    return float(min(cap, max(floor, frac * w)))


def _best_record_for_output(st: dict[str, Any]) -> dict[str, Any] | None:
    acc = st.get("accepted_items") or []
    if acc:
        return min(acc, key=lambda x: float(x.get("abs_delta_full", 1e9)))
    bi = st.get("best_item")
    return bi if isinstance(bi, dict) else None


def _extract_new_difficulty_0_10(st: dict[str, Any]) -> float | None:
    rec = _best_record_for_output(st)
    if not rec:
        return None
    ed = rec.get("estimated_difficulty") or {}
    if isinstance(ed, dict) and ed.get("estimate") is not None:
        return float(ed["estimate"])
    if isinstance(ed, (int, float)):
        return float(ed)
    return None


def _extract_from_workdir(workdir: Path) -> tuple[dict[str, Any] | None, float | None, float, str]:
    """Return best item dict, difficulty, |delta|, done_reason from state.json."""
    st_path = workdir / "state.json"
    if not st_path.is_file():
        return None, None, float("inf"), "no_state"
    st = json.loads(st_path.read_text(encoding="utf-8"))
    reason = str(st.get("done_reason") or "")
    target = float(st.get("target_difficulty", 0.0))
    rec = _best_record_for_output(st)
    dif = _extract_new_difficulty_0_10(st)
    if dif is None:
        return rec, None, float("inf"), reason
    return rec, float(dif), abs(float(dif) - target), reason


def _run_one_gap(
    *,
    root: Path,
    benchmark: str,
    workdir: Path,
    gap_low: float,
    gap_high: float,
    target: float,
    max_rounds: int,
    early_stop_tol: float,
    n_sampling: int,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["REPO_ROOT"] = str(root)
    env["GAP_ROOT"] = str(root)
    env["SPLIT_WORKDIR"] = str(workdir)
    env["SERVE_SCRIPT"] = str(root / "propose/sh/serve_minimax_m2_7_8gpu.sh")
    env["VLLM_BASE_URL"] = env.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
    env["VLLM_BASE_URLS"] = env.get("VLLM_BASE_URLS", env["VLLM_BASE_URL"])
    env["VLLM_LOG_FILE"] = str(workdir / "vllm_serve.log")
    env["GENERATE_LOG_FILE"] = str(workdir / "generate.log")
    env["CONDA_SERVE_ENV"] = env.get("CONDA_SERVE_ENV", "diff_minimax")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    model = env.get("MODEL", "minimax-m2.7")
    gen_temp = env.get("GEN_TEMPERATURE", "1.0")
    gen_top_p = env.get("GEN_TOP_P", "0.95")
    gen_top_k = env.get("GEN_TOP_K", "40")
    gen_sys = env.get(
        "GEN_SYSTEM_PROMPT",
        "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.",
    )

    init_cmd = [
        "bash",
        str(root / "sh/run_pipeline_auto.sh"),
        "init",
        "--benchmark",
        benchmark,
        "--model",
        model,
        "--temperature",
        gen_temp,
        "--top-p",
        gen_top_p,
        "--top-k",
        gen_top_k,
        "--system-prompt",
        gen_sys,
        "--base-url",
        env["VLLM_BASE_URLS"],
        "--max-rounds",
        str(max_rounds),
        "--n-sampling",
        str(n_sampling),
        "--k-parents",
        "4",
        "--n-children",
        "4",
        "--memory-size",
        "8",
        "--early-stop-tolerance",
        str(early_stop_tol),
        "--pipeline-mode",
        "direct",
        "--gap-low",
        str(gap_low),
        "--gap-high",
        str(gap_high),
        "--target-difficulty",
        str(target),
    ]
    subprocess.run(init_cmd, check=True, env=env, cwd=str(root))
    run_cmd = ["bash", str(root / "sh/run_pipeline_auto.sh"), "run"]
    subprocess.run(run_cmd, check=True, env=env, cwd=str(root))


def run_mass_fill(
    benchmark: str,
    *,
    target_count: int | None = None,
    seed_top3: bool = True,
    max_rounds: int = 8,
    tol_frac: float = 0.08,
    tol_floor: float = 0.05,
    tol_cap: float = 0.25,
    n_sampling: int = 32,
    root: Path | None = None,
    force_reinit: bool = False,
) -> MassFillState:
    root = root or _repo_root()
    out = results_dir(benchmark, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "steps").mkdir(exist_ok=True)

    state = None if force_reinit else load_state(benchmark, root)
    if state is None:
        state = init_state(
            benchmark,
            target_count=target_count or default_target_count(benchmark),
            seed_top3=seed_top3,
            root=root,
        )
        save_state(state, root)
        append_log(
            benchmark,
            f"INIT target={state.target_count} seed_fills={state.seed_count()} "
            f"pool={len(state.pool_0_10)} remaining={state.remaining()} "
            f"tol=max({tol_floor}, {tol_frac:.0%}×gap_width) cap={tol_cap}",
            root,
        )
    else:
        append_log(
            benchmark,
            f"RESUME filled={state.filled_count()}/{state.target_count} "
            f"real={state.real_count()} pretend={state.pretend_count()}",
            root,
        )

    write_heartbeat(state, status="running", phase_detail="start", root=root)

    while not state.completed and state.remaining() > 0:
        gap_low, gap_high, target, meta = pick_max_gap(state)
        step_i = state.filled_count() + 1
        width = gap_high - gap_low
        early_stop_tol = tolerance_for_gap(width, frac=tol_frac, floor=tol_floor, cap=tol_cap)
        append_log(
            benchmark,
            f"[STEP {step_i}/{state.target_count}] max-gap width={width:.4f} "
            f"lo={gap_low:.4f} hi={gap_high:.4f} target={target:.4f} "
            f"tol={early_stop_tol:.4f} ({tol_frac:.0%}×width, floor={tol_floor}, cap={tol_cap}) "
            f"norm_w={float(meta.get('normalized_gap_width', 0)):.4f} → GENERATE+VERIFY",
            root,
        )
        write_heartbeat(
            state,
            status="generating",
            phase_detail=f"step {step_i}: target={target:.4f} tol={early_stop_tol:.4f}",
            extra={
                "current_target": target,
                "current_gap_width": width,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "early_stop_tol": early_stop_tol,
            },
            root=root,
        )

        step_dir = out / "steps" / f"step_{step_i:03d}"
        if step_dir.exists():
            # resume mid-step: if previous crashed after done, prefer reuse
            st_path = step_dir / "state.json"
            if st_path.is_file():
                st = json.loads(st_path.read_text(encoding="utf-8"))
                if st.get("status") == "done":
                    append_log(benchmark, f"[STEP {step_i}] reusing finished step dir", root)
                else:
                    shutil.rmtree(step_dir)
                    step_dir.mkdir(parents=True)
            else:
                shutil.rmtree(step_dir)
                step_dir.mkdir(parents=True)
        else:
            step_dir.mkdir(parents=True)

        try:
            write_heartbeat(
                state,
                status="verifying",
                phase_detail=f"step {step_i}: pipeline run tol={early_stop_tol:.4f}",
                extra={
                    "current_target": target,
                    "current_gap_width": width,
                    "early_stop_tol": early_stop_tol,
                },
                root=root,
            )
            if not (step_dir / "state.json").is_file() or json.loads((step_dir / "state.json").read_text()).get("status") != "done":
                _run_one_gap(
                    root=root,
                    benchmark=benchmark,
                    workdir=step_dir,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    target=target,
                    max_rounds=max_rounds,
                    early_stop_tol=early_stop_tol,
                    n_sampling=n_sampling,
                )
        except Exception as e:  # noqa: BLE001
            append_log(benchmark, f"[STEP {step_i}] PIPELINE ERROR: {e}", root)
            write_heartbeat(state, status="failed", phase_detail=str(e), last_error=str(e), root=root)
            raise

        rec, dif, abs_delta, reason = _extract_from_workdir(step_dir)
        within = dif is not None and abs_delta < early_stop_tol and reason == "best_within_tol"
        # Also accept if within tol even if reason differs (e.g. edge cases)
        if dif is not None and abs_delta < early_stop_tol:
            within = True

        if within and dif is not None and rec is not None:
            point = VirtualPoint(
                difficulty_0_10=dif,
                status="real",
                item_id=str(rec.get("id") or ""),
                source="mass_fill",
                target=target,
                abs_delta=abs_delta,
                gap_low=gap_low,
                gap_high=gap_high,
                question=str(rec.get("question") or ""),
                answer=str(rec.get("answer") or rec.get("reference_answer") or ""),
                meta={
                    "done_reason": reason,
                    "step_dir": str(step_dir),
                    "parent_id": rec.get("parent_id") or "",
                },
            )
            append_fill(state, point)
            append_log(
                benchmark,
                f"[STEP {step_i}] ACCEPT real id={point.item_id!r} b={dif:.4f} |Δ|={abs_delta:.4f}",
                root,
            )
        else:
            # Pretend: inject target midpoint
            point = VirtualPoint(
                difficulty_0_10=target,
                status="pretend",
                item_id=f"pretend_step_{step_i}",
                source="mass_fill_pretend",
                target=target,
                abs_delta=abs_delta if dif is not None else None,
                gap_low=gap_low,
                gap_high=gap_high,
                question="",
                answer="",
                meta={
                    "done_reason": reason,
                    "measured_b": dif,
                    "step_dir": str(step_dir),
                },
            )
            append_fill(state, point)
            append_log(
                benchmark,
                f"[STEP {step_i}] PRETEND injected target={target:.4f} "
                f"(reason={reason!r} measured={dif} |Δ|={abs_delta})",
                root,
            )

        save_state(state, root)
        write_heartbeat(
            state,
            status="running" if not state.completed else "done",
            phase_detail=f"after step {step_i}",
            root=root,
        )

    append_log(
        benchmark,
        f"DONE filled={state.filled_count()}/{state.target_count} "
        f"real={state.real_count()} pretend={state.pretend_count()} seed={state.seed_count()}",
        root,
    )
    write_heartbeat(state, status="done", phase_detail="complete", root=root)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative max-gap mass fill (MiniMax + verifier)")
    sub = ap.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run / resume mass fill (default)")
    p_run.add_argument("--benchmark", required=True, choices=["aime24", "aime25", "aime26", "amc23"])
    p_run.add_argument("--target-count", type=int, default=None)
    p_run.add_argument("--no-seed-top3", action="store_true")
    p_run.add_argument("--max-rounds", type=int, default=8)
    p_run.add_argument(
        "--tol-frac",
        type=float,
        default=0.08,
        help="Early-stop tol = tol_frac × current max-gap width (default 0.08 = 8%%).",
    )
    p_run.add_argument("--tol-floor", type=float, default=0.05, help="Minimum tol on 0–10 scale.")
    p_run.add_argument("--tol-cap", type=float, default=0.25, help="Maximum tol on 0–10 scale.")
    p_run.add_argument("--n-sampling", type=int, default=32)
    p_run.add_argument("--force-reinit", action="store_true")

    p_st = sub.add_parser("status", help="Print heartbeat + recent log")
    p_st.add_argument("--benchmark", required=True)

    p_init = sub.add_parser("init", help="Create state + seed only (no GPU)")
    p_init.add_argument("--benchmark", required=True, choices=["aime24", "aime25", "aime26", "amc23"])
    p_init.add_argument("--target-count", type=int, default=None)
    p_init.add_argument("--no-seed-top3", action="store_true")

    p_ex = sub.add_parser("export", help="OCR-friendly Benchmark Difficulty sheet (photo → OCR)")
    p_ex.add_argument(
        "--benchmark",
        default=None,
        help="One bench, or omit to export all that have state.json",
    )
    p_ex.add_argument("--remain-k", type=int, default=5, help="Leftover max-gaps to list (size+target only).")
    p_ex.add_argument("--wrap", type=int, default=96, help="items.txt wrap width for photographing.")

    args = ap.parse_args()
    cmd = args.cmd or "run"

    if cmd == "status":
        print_status(args.benchmark)
        return

    if cmd == "export":
        from mass_fill.export_sheet import DEFAULT_BENCHES, export_benchmarks

        bms = [args.benchmark] if args.benchmark else list(DEFAULT_BENCHES)
        results = export_benchmarks(bms, remain_k=args.remain_k, wrap=args.wrap)
        missing = [bm for bm, p in results if p is None]
        if missing:
            print(f"(no state.json yet: {', '.join(missing)})", flush=True)
        return

    if cmd == "init":
        st = init_state(
            args.benchmark,
            target_count=args.target_count,
            seed_top3=not args.no_seed_top3,
        )
        path = save_state(st)
        write_heartbeat(st, status="idle", phase_detail="initialized")
        append_log(args.benchmark, f"init-only wrote {path}")
        print(
            f"init ok: filled={st.filled_count()}/{st.target_count} "
            f"seed={st.seed_count()} remaining={st.remaining()} → {path}"
        )
        return

    # default run
    if cmd != "run":
        # allow ``python -m mass_fill.driver --benchmark X`` without subcommand
        pass

    if not hasattr(args, "benchmark") or args.benchmark is None:
        ap.error("need --benchmark")

    run_mass_fill(
        args.benchmark,
        target_count=getattr(args, "target_count", None),
        seed_top3=not getattr(args, "no_seed_top3", False),
        max_rounds=getattr(args, "max_rounds", 8),
        tol_frac=getattr(args, "tol_frac", 0.08),
        tol_floor=getattr(args, "tol_floor", 0.05),
        tol_cap=getattr(args, "tol_cap", 0.25),
        n_sampling=getattr(args, "n_sampling", 32),
        force_reinit=getattr(args, "force_reinit", False),
    )


if __name__ == "__main__":
    # Support both ``python -m mass_fill.driver run --benchmark`` and bare flags
    if len(sys.argv) > 1 and sys.argv[1] not in {"run", "status", "init", "export", "-h", "--help"}:
        sys.argv.insert(1, "run")
    main()
