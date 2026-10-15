"""Verifier entry point: estimate difficulty of candidate items.

Math: reads ``<results_dir>/<model>/<benchmark>/*.jsonl`` from ``math_eval_general`` rollouts.

Usage (working directory = repo root, conda env ``diff`` with numpy/scipy/pandas):

    conda run --no-capture-output -n diff python -m verifier.main \\
        --results-dir verifier/results/aime24_<run_id> \\
        --benchmark aime24 --framework math

``verifier/run_rollout.sh`` runs rollouts then invokes this module from the repo root.
The ``lmms`` framework path remains for legacy logs only; this repo uses ``math`` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import irt
from .config import BENCHMARK_META, MATRIX_DIR, VERIFIER_RESULTS_DIR, VerifierConfig

# ---------------------------------------------------------------------------
# Benchmark name → lmms-eval task name mapping
# ---------------------------------------------------------------------------

LMMS_TASK_NAMES: dict[str, list[str]] = {
    # benchmark → [original_task, verifier_task]
    # load_lmms_results tries both names
    "gpqa": ["gpqa_diamond_cot_n_shot", "gpqa_verifier"],
    "mmlu_bio": ["mmlu_pro_biology", "mmlu_bio_verifier", "mmlu_pro_biology_verifier"],
    "mmlu_chem": ["mmlu_pro_chemistry", "mmlu_chem_verifier", "mmlu_pro_chemistry_verifier"],
    "mmlu_phys": ["mmlu_pro_physics", "mmlu_phys_verifier", "mmlu_pro_physics_verifier"],
    "medqa": ["medqa", "medqa_verifier"],
    "scienceqa": ["scienceqa", "scienceqa_verifier"],
    "bbh_3obj": ["bbh_cot_zeroshot_logical_deduction_three_objects", "bbh_3obj_verifier"],
    "bbh_5obj": ["bbh_cot_zeroshot_logical_deduction_five_objects", "bbh_5obj_verifier"],
    "bbh_7obj": ["bbh_cot_zeroshot_logical_deduction_seven_objects", "bbh_7obj_verifier"],
    "ifeval": ["ifeval", "ifeval_verifier"],
    "zebra": ["zebralogic_grid", "zebralogic_verifier"],
}

# lmms-eval task → correctness field name
LMMS_SCORE_FIELD: dict[str, str] = {
    "gpqa_diamond_cot_n_shot": "exact_match",
    "gpqa_verifier": "exact_match",
    "mmlu_pro_biology": "exact_match",
    "mmlu_bio_verifier": "exact_match",
    "mmlu_pro_biology_verifier": "exact_match",
    "mmlu_pro_chemistry": "exact_match",
    "mmlu_chem_verifier": "exact_match",
    "mmlu_pro_chemistry_verifier": "exact_match",
    "mmlu_pro_physics": "exact_match",
    "mmlu_phys_verifier": "exact_match",
    "mmlu_pro_physics_verifier": "exact_match",
    "medqa": "accuracy",
    "medqa_verifier": "accuracy",
    "scienceqa": "exact_match",
    "scienceqa_verifier": "exact_match",
    "bbh_cot_zeroshot_logical_deduction_three_objects": "exact_match",
    "bbh_3obj_verifier": "exact_match",
    "bbh_cot_zeroshot_logical_deduction_five_objects": "exact_match",
    "bbh_5obj_verifier": "exact_match",
    "bbh_cot_zeroshot_logical_deduction_seven_objects": "exact_match",
    "bbh_7obj_verifier": "exact_match",
    "ifeval": "prompt_level_strict_acc",
    "ifeval_verifier": "prompt_level_strict_acc",
    "zebralogic_grid": "acc",
    "zebralogic_verifier": "acc",
}

# lmms-eval model dir name → IRT model name
LMMS_MODEL_NAMES: dict[str, str] = {
    "allenai__Olmo-3.1-32B-Think": "olmo-3.1-32b-think",
    "HuggingFaceTB__SmolLM2-1.7B-Instruct": "smollm2-1.7b-instruct",
    "lmsys__vicuna-7b-v1.5": "vicuna-7b-v1.5",
    "meta-llama__Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "meta-llama__Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
    "microsoft__phi-4": "phi-4",
    "openai__gpt-oss-20b": "gpt-oss-20b",
    "Qwen__Qwen3-Next-80B-A3B-Instruct": "qwen3-next-80b-a3b-instruct",
    "Qwen__Qwen3.5-397B-A17B-FP8": "qwen3.5-397b-a17b",
    "Qwen__QwQ-32B": "qwq-32b",
    "zai-org__GLM-4.5-Air": "glm-4.5-air",
    "zai-org__GLM-4.6-FP8": "glm-4.6",
    "bullpoint__GLM-4.6-AWQ": "glm-4.6",
}


# =====================================================================
# Math framework (verifier/diff_eval) loader
# =====================================================================

def load_math_results(
    results_dir: Path,
    data_name: str,
    n_trials: int,
) -> tuple[np.ndarray, list[str], list[str], int]:
    """Load verifier/diff_eval rollout results and build response matrix.

    Returns (matrix, model_names, item_ids, n_trials).
    """
    from .math_eval.parser import strip_string
    from .math_eval.grader import math_equal

    model_results: dict[str, list[dict]] = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        data_dir = model_dir / data_name
        if not data_dir.exists():
            continue
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        jsonl_file = max(jsonl_files, key=lambda f: f.stat().st_mtime)
        samples = []
        with open(jsonl_file, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        if samples:
            model_results[model_dir.name] = samples

    if not model_results:
        raise FileNotFoundError(f"No results found in {results_dir} for {data_name}")

    # item ids from first model
    first_model = next(iter(model_results.values()))
    item_ids = [str(s["idx"]) for s in first_model]
    item_idx_map = {iid: j for j, iid in enumerate(item_ids)}

    model_names = sorted(model_results.keys())
    matrix = np.zeros((len(model_names), len(item_ids)), dtype=int)

    all_pred_counts: list[int] = []
    for m, model_name in enumerate(model_names):
        for s in model_results[model_name]:
            idx = str(s["idx"])
            if idx not in item_idx_map:
                continue
            j = item_idx_map[idx]

            gt = str(s.get("gt", ""))
            preds = s.get("pred", [])
            if isinstance(preds, str):
                preds = [preds]

            actual_n = min(len(preds), n_trials)
            all_pred_counts.append(actual_n)

            n_correct = 0
            for pred in preds[:actual_n]:
                try:
                    if math_equal(strip_string(str(pred)), strip_string(gt), timeout=True):
                        n_correct += 1
                except Exception:
                    pass
            matrix[m, j] = n_correct

    actual_n_trials = min(all_pred_counts) if all_pred_counts else n_trials
    return matrix, model_names, item_ids, actual_n_trials


# =====================================================================
# lmms-eval framework loader
# =====================================================================

def _normalize_answer(s) -> str:
    """Lowercase + strip for letter-answer style benchmarks (bbh, gpqa, mmlu, scienceqa)."""
    if s is None:
        return ""
    return str(s).strip().lower().rstrip(".").rstrip(",")


def _score_sample(resp, target, score_field: str) -> int:
    """Compute 0/1 score for a single sample by comparing response vs target."""
    return int(_normalize_answer(resp) == _normalize_answer(target))


def load_lmms_results(
    logs_dir: Path,
    benchmark: str,
    n_trials: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load lmms-eval log results and build response matrix.

    Supports both file layouts:
      - Legacy: {model_dir}/run{seed}_samples_{task}.jsonl (one file per seed, 1 score/doc/file)
      - Repeats: {model_dir}/{timestamp}_samples_{task}.jsonl (one file total, filtered_resps
        is a list of N strings per doc → N scores computed inline)

    Returns (matrix, irt_model_names, item_ids).
    """
    task_names = LMMS_TASK_NAMES.get(benchmark)
    if task_names is None:
        raise ValueError(f"No lmms-eval task mapping for benchmark '{benchmark}'")

    model_data: dict[str, dict[int, list[int]]] = {}  # irt_name -> {doc_id -> [scores]}

    for model_dir in sorted(logs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        irt_name = LMMS_MODEL_NAMES.get(model_dir.name)
        if irt_name is None:
            continue

        # find all sample files for this task (any prefix, not just "run*")
        run_files = []
        task_name_used = None
        for tn in task_names:
            run_files = sorted(model_dir.glob(f"*_samples_{tn}.jsonl"))
            if run_files:
                task_name_used = tn
                break
        if not run_files:
            continue
        score_field = LMMS_SCORE_FIELD[task_name_used]

        # Free-response benchmarks (ifeval) score via lmms-eval's process_results,
        # which writes per-sample fields like prompt_level_strict_acc directly on
        # each row — there's no letter target to regex-match. Use that field instead
        # of comparing filtered_resps strings.
        program_scored = score_field in {
            "prompt_level_strict_acc", "prompt_level_loose_acc",
            "inst_level_strict_acc", "inst_level_loose_acc",
        }

        doc_scores: dict[int, list[int]] = {}
        for rf in run_files:
            with open(rf, "r") as f:
                for line in f:
                    s = json.loads(line)
                    doc_id = s["doc_id"]

                    if program_scored:
                        # lmms-eval --repeats N writes a list of N scores per row
                        # (one per sample) when using process_results.
                        v = s.get(score_field, 0)
                        scores = []
                        if isinstance(v, list):
                            for x in v:
                                scores.append(int(bool(x)) if isinstance(x, bool) else int(float(x)))
                        else:
                            scores.append(int(bool(v)) if isinstance(v, bool) else int(float(v)))
                        doc_scores.setdefault(doc_id, []).extend(scores)
                        continue

                    target = s.get("target", "")
                    filtered = s.get("filtered_resps", [])

                    # filtered_resps shapes we've seen:
                    #   - str                          (legacy single-sample, take_first applied)
                    #   - list[str], len == N          (repeats > 1, one req per doc)
                    #   - list[list[str]], 1 × N       (repeats > 1, wrapped per req) → flatten
                    # Flatten one level so we always get a list of N response strings.
                    if isinstance(filtered, str):
                        resps = [filtered]
                    elif isinstance(filtered, list):
                        resps = []
                        for item in filtered:
                            if isinstance(item, list):
                                resps.extend(item)
                            else:
                                resps.append(item)
                    else:
                        resps = []

                    if not resps:
                        # fall back to aggregated score_field
                        score = s.get(score_field, 0)
                        score = int(bool(score)) if isinstance(score, bool) else int(float(score))
                        doc_scores.setdefault(doc_id, []).append(score)
                    else:
                        for r in resps:
                            doc_scores.setdefault(doc_id, []).append(_score_sample(r, target, score_field))

        model_data[irt_name] = doc_scores

    if not model_data:
        raise FileNotFoundError(f"No lmms-eval results found in {logs_dir} for tasks {task_names}")

    # Build matrix
    # item ids from first model (sorted doc_ids)
    first_model_scores = next(iter(model_data.values()))
    item_ids = [str(d) for d in sorted(first_model_scores.keys())]
    doc_id_to_j = {int(iid): j for j, iid in enumerate(item_ids)}

    model_names = sorted(model_data.keys())
    matrix = np.zeros((len(model_names), len(item_ids)), dtype=int)

    # derive actual n_trials from data (min over all (model, doc) pairs that have data)
    all_counts: list[int] = []
    for m, model_name in enumerate(model_names):
        for doc_id, scores in model_data[model_name].items():
            if doc_id not in doc_id_to_j:
                continue
            j = doc_id_to_j[doc_id]
            matrix[m, j] = sum(scores)
            all_counts.append(len(scores))

    actual_n_trials = min(all_counts) if all_counts else n_trials
    return matrix, model_names, item_ids, actual_n_trials


# =====================================================================
# Anchor item loader
# =====================================================================

def load_anchor_results(
    benchmark: str,
    anchor_ids: list[str],
    model_names: list[str],
    n_trials: int,
    matrix_n_trials: int = 32,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load anchor item responses from the pre-computed response matrix.

    Only uses models that were also used in the rollout, and scales counts
    from matrix_n_trials down to n_trials so calibration conditions match
    the new candidate items exactly.

    Returns (matrix, common_model_names, anchor_ids).
    matrix shape: (n_common_models, n_anchors), values = n_correct out of n_trials.
    """
    matrix_path = MATRIX_DIR / f"{benchmark}_response_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Response matrix not found: {matrix_path}")

    mat_df = pd.read_csv(matrix_path, index_col=0)

    missing = [a for a in anchor_ids if a not in mat_df.columns]
    if missing:
        raise ValueError(f"Anchor items not found in matrix columns: {missing}")

    common_models = [m for m in model_names if m in mat_df.index]
    if not common_models:
        raise ValueError(
            f"No overlapping models between rollout ({model_names}) "
            f"and response matrix ({list(mat_df.index)})"
        )

    sub = mat_df.loc[common_models, anchor_ids].values.astype(float)
    # Scale counts from matrix_n_trials → n_trials
    if n_trials != matrix_n_trials:
        sub = np.round(sub * n_trials / matrix_n_trials).astype(int)
    else:
        sub = sub.astype(int)

    return sub, common_models, anchor_ids


# =====================================================================
# IRT estimation (shared by both frameworks)
# =====================================================================

def _resolve_theta_subset(
    cfg: VerifierConfig,
    model_names: list[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (theta_sub, common_models, matrix_idx).

    Loads the panel θ priors from cfg.irt_results_dir and aligns them with the rollout
    models. Rollout must include every model in ``<benchmark>_ability.csv`` (see
    ``verifier/run_rollout.sh``).
    """
    theta, theta_model_names = irt.load_theta(cfg)
    common_models = [m for m in theta_model_names if m in model_names]
    if not common_models:
        raise ValueError(
            f"No overlapping models between rollout and θ panel.\n"
            f"  Panel: {theta_model_names}\n"
            f"  Rollout: {model_names}"
        )
    theta_idx = [theta_model_names.index(m) for m in common_models]
    matrix_idx = [model_names.index(m) for m in common_models]
    return theta[theta_idx], common_models, matrix_idx


def estimate_difficulties(
    matrix: np.ndarray,
    model_names: list[str],
    item_ids: list[str],
    cfg: VerifierConfig,
    n_trials: int,
    allowed_models: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate b for each item using grid posterior.

    If ``allowed_models`` is given, only models in that list are used for theta
    (and consequently only rollout results from those models).
    """
    theta_sub, common_models, matrix_idx = _resolve_theta_subset(cfg, model_names)
    if allowed_models is not None:
        allowed = set(allowed_models)
        keep = [i for i, m in enumerate(common_models) if m in allowed]
        theta_sub = theta_sub[keep]
        matrix_idx = [matrix_idx[i] for i in keep]
        common_models = [common_models[i] for i in keep]
        if not common_models:
            raise ValueError(f"allowed_models filter left no models. Allowed={allowed_models}")
    matrix_sub = matrix[matrix_idx, :]

    meta = cfg.meta
    rows = []
    for j, item_id in enumerate(item_ids):
        y_m = matrix_sub[:, j]
        b_hat, b_sd = irt.estimate_b_grid(
            theta_sub, y_m, n_trials, meta.c,
            grid_lo=cfg.grid_lo, grid_hi=cfg.grid_hi, grid_n=cfg.grid_n,
            prior_mu=cfg.prior_mu, prior_sd=cfg.prior_sd,
        )
        rows.append({
            "item": item_id,
            "b_hat": b_hat,
            "b_sd": b_sd,
            "y_m": y_m.tolist(),
            "n_trials": n_trials,
            "n_models": len(common_models),
        })

    return pd.DataFrame(rows)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Verifier: estimate item difficulty")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Math: path to evaluate_outputs_test/. Lmms: path to logs/.")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Benchmark name (must be in BENCHMARK_META)")
    parser.add_argument("--framework", type=str, default="math", choices=["math", "lmms"],
                        help="Result format: 'math' (diff) or 'lmms' (lmms-eval)")
    parser.add_argument("--data-name", type=str, default=None,
                        help="Dataset folder name in results (math only, defaults to benchmark)")
    parser.add_argument("--anchors", type=str, default=None,
                        help="Comma-separated anchor item IDs from the original benchmark "
                             "(e.g. q0,q5,q12). Their pre-computed response matrix is loaded "
                             f"from {MATRIX_DIR}/{{benchmark}}_response_matrix.csv.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--irt-results-dir", type=str, default=None,
                        help="Directory of <bm>_ability.csv used as θ priors. Defaults to "
                             "irt_results/. Created by verifier.fit_ability when missing. "
                             "Rollout must cover every model listed there.")
    args = parser.parse_args()

    benchmark = args.benchmark
    if benchmark not in BENCHMARK_META:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_META.keys())}")

    cfg = VerifierConfig(benchmark=benchmark)
    if args.irt_results_dir:
        cfg.irt_results_dir = Path(args.irt_results_dir).resolve()
    results_dir = Path(args.results_dir)
    default_n_trials = cfg.meta.n_trials

    allowed_models = None

    print(f"Loading {args.framework} results for {benchmark}...")

    if args.framework == "math":
        data_name = args.data_name or benchmark
        matrix, model_names, item_ids, n_trials = load_math_results(results_dir, data_name, default_n_trials)
    else:
        matrix, model_names, item_ids, n_trials = load_lmms_results(results_dir, benchmark, default_n_trials)

    print(f"Found {len(model_names)} models: {model_names}")
    print(f"Matrix shape: {matrix.shape} ({len(model_names)} models × {len(item_ids)} items)")
    print(f"Actual n_trials from data: {n_trials}")

    _, theta_model_names = irt.load_theta(cfg)
    rollout_set = set(model_names)
    expected_set = set(theta_model_names)
    extra = sorted(rollout_set - expected_set)
    if extra:
        raise ValueError(
            f"Rollout has models not in θ panel ({cfg.irt_results_dir / (benchmark + '_ability.csv')}): {extra}. "
            f"Panel={sorted(expected_set)}, rollout={sorted(rollout_set)}."
        )
    missing = sorted(expected_set - rollout_set)
    if missing:
        raise ValueError(
            f"Rollout is missing {len(missing)} panel model(s) required for full 12-model IRT: {missing}. "
            f"Enable every model in verifier/run_rollout.sh (including qwen3.5-397b-a17b and glm-4.6). "
            f"Panel={sorted(expected_set)}, rollout={sorted(rollout_set)}."
        )
    if len(rollout_set) != len(expected_set):
        raise ValueError(
            f"Rollout model count mismatch: expected {len(expected_set)} panel models, "
            f"got {len(rollout_set)} in results."
        )
    print(f"[verifier] Full panel rollout: {len(rollout_set)} models match θ priors.")

    # Auto-tune the IRT grid bounds so that synth-all-fail MEAN equals pool_b_max
    # and synth-all-pass MEAN equals pool_b_min. With pool min/max as the 0–10
    # anchors, this makes "all panel models score 0/n_trials" map to
    # difficulty = 10 exactly (and "all pass" to 0 exactly), continuous in
    # between, no clipping. Bisect on grid_hi / grid_lo with a near-uniform
    # prior (the prior just guards against numerical edge cases).
    ref_difficulty_path = cfg.irt_results_dir / f"{benchmark}_difficulty.csv"
    if not ref_difficulty_path.is_file():
        raise FileNotFoundError(
            f"missing 12-model pool difficulty csv: {ref_difficulty_path}"
        )
    ref_df = pd.read_csv(ref_difficulty_path)
    b_col = "b_mean" if "b_mean" in ref_df.columns else "b_hat"
    pool_b_min = float(ref_df[b_col].min())
    pool_b_max = float(ref_df[b_col].max())

    cfg.prior_sd = 100.0  # near-uniform within grid

    # Build θ vector for synthetic all-pass/all-fail grid calibration (full panel).
    theta_sub_for_anchors, _, _ = _resolve_theta_subset(cfg, model_names)

    def _synth_mean(grid_lo: float, grid_hi: float, all_fail: bool) -> float:
        y_vec = (np.zeros(len(theta_sub_for_anchors), dtype=int) if all_fail
                 else np.full(len(theta_sub_for_anchors), n_trials, dtype=int))
        bh, _ = irt.estimate_b_grid(
            theta_sub_for_anchors, y_vec, n_trials, cfg.meta.c,
            grid_lo=grid_lo, grid_hi=grid_hi, grid_n=8000,
            prior_mu=cfg.prior_mu, prior_sd=cfg.prior_sd,
        )
        return float(bh)

    # Bisect grid_hi so synth-all-fail MEAN = pool_b_max
    lo_hi, hi_hi = pool_b_max, pool_b_max + 50.0
    grid_lo_init = pool_b_min - 5.0
    for _ in range(60):
        mid = 0.5 * (lo_hi + hi_hi)
        if _synth_mean(grid_lo_init, mid, all_fail=True) < pool_b_max:
            lo_hi = mid
        else:
            hi_hi = mid
    cfg.grid_hi = 0.5 * (lo_hi + hi_hi)

    # Bisect grid_lo so synth-all-pass MEAN = pool_b_min (search downward from pool_b_min)
    lo_lo, hi_lo = pool_b_min - 50.0, pool_b_min
    for _ in range(60):
        mid = 0.5 * (lo_lo + hi_lo)
        if _synth_mean(mid, cfg.grid_hi, all_fail=False) > pool_b_min:
            hi_lo = mid
        else:
            lo_lo = mid
    cfg.grid_lo = 0.5 * (lo_lo + hi_lo)

    # Verify (should be very close to pool extremes)
    sf = _synth_mean(cfg.grid_lo, cfg.grid_hi, all_fail=True)
    sp = _synth_mean(cfg.grid_lo, cfg.grid_hi, all_fail=False)
    print(
        f"IRT grid auto-tuned for n_trials={n_trials}: "
        f"grid=[{cfg.grid_lo:+.4f}, {cfg.grid_hi:+.4f}], prior_sd={cfg.prior_sd}; "
        f"synth-all-fail b̂={sf:+.4f} (pool_b_max={pool_b_max:+.4f}), "
        f"synth-all-pass b̂={sp:+.4f} (pool_b_min={pool_b_min:+.4f})"
    )

    print("Estimating difficulties...")
    df = estimate_difficulties(matrix, model_names, item_ids, cfg, n_trials, allowed_models=allowed_models)
    # Clamp b̂ to the calibration pool span so 0–10 normalization never exceeds 10
    # (discrete all-fail vectors can sit slightly above grid_hi / pool_b_max).
    df["b_hat"] = df["b_hat"].clip(lower=pool_b_min, upper=pool_b_max)
    df["is_anchor"] = False
    df["b_ref"] = float("nan")

    # Anchor items: load from pre-computed response matrix and calibrate separately.
    if args.anchors:
        anchor_ids = [a.strip() for a in args.anchors.split(",")]
        print(f"Loading anchor items: {anchor_ids}")
        anc_matrix, anc_models, anc_ids = load_anchor_results(benchmark, anchor_ids, model_names, n_trials)
        anc_df = estimate_difficulties(anc_matrix, anc_models, anc_ids, cfg, n_trials, allowed_models=allowed_models)
        anc_df["b_hat"] = anc_df["b_hat"].clip(lower=pool_b_min, upper=pool_b_max)
        anc_df["is_anchor"] = True

        # attach ground-truth b from irt_results
        ref_difficulty_path_anc = cfg.irt_results_dir / f"{benchmark}_difficulty.csv"
        if ref_difficulty_path_anc.exists():
            ref_df_anc = pd.read_csv(ref_difficulty_path_anc)
            b_col_anc = "b_mean" if "b_mean" in ref_df_anc.columns else "b_hat"
            ref_df_anc = ref_df_anc.set_index("item")
            anc_df["b_ref"] = anc_df["item"].map(
                lambda i: ref_df_anc[b_col_anc].get(i, float("nan"))
            )
        else:
            anc_df["b_ref"] = float("nan")

        df = pd.concat([df, anc_df], ignore_index=True)
        print(f"Added {len(anc_ids)} anchor items (n_trials=32).")

    # Normalize b̂ to the 12-model 0–10 scale (same pool min/max anchors used to
    # bind the IRT grid above — so b̂ for synth-all-fail at the full panel
    # saturates at pool_b_max → difficulty=10 exactly, and the 0–10 scale matches
    # what propose uses for target / gap brackets).
    span = pool_b_max - pool_b_min if pool_b_max - pool_b_min > 1e-12 else 1e-12
    df["difficulty"] = df["b_hat"].apply(lambda b: 10.0 * (b - pool_b_min) / span)
    df["difficulty_sd"] = df["b_sd"].apply(lambda s: s * 10.0 / span)
    if "b_ref" in df.columns:
        df["difficulty_ref"] = df["b_ref"].apply(
            lambda b: 10.0 * (b - pool_b_min) / span if pd.notna(b) else float("nan")
        )
    print(
        f"12-model 0–10 scale (pool min/max, {len(ref_df)} items): "
        f"b_min={pool_b_min:+.4f} → 0, b_max={pool_b_max:+.4f} → 10"
    )

    out_path = Path(args.out) if args.out else (VERIFIER_RESULTS_DIR / f"{benchmark}_difficulty.csv")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)

    candidates = df[~df["is_anchor"]]
    anchors = df[df["is_anchor"]]
    print(f"\nResults: {len(candidates)} candidates + {len(anchors)} anchors")
    print(f"  b_hat range (candidates): [{candidates['b_hat'].min():.2f}, {candidates['b_hat'].max():.2f}]")
    if "difficulty" in df.columns:
        print(f"  difficulty (candidates):  [{candidates['difficulty'].min():.2f}, {candidates['difficulty'].max():.2f}] (0–10)")
    if len(anchors) > 0 and "b_ref" in anchors.columns:
        valid = anchors.dropna(subset=["b_ref"])
        mae = (valid["b_hat"] - valid["b_ref"]).abs().mean()
        print(f"  anchor MAE (b scale):     {mae:.3f} ({len(valid)} anchors with ground truth)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
