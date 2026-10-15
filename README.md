# Your Benchmark Is an Empirical Measure Over Difficulty

<p align="center">
  <b>Yifan Sun</b><sup>1,*</sup> &nbsp;
  <b>Naicheng Yu</b><sup>2,*</sup> &nbsp;
  <b>Jiawen Gong</b><sup>1,*</sup> &nbsp;
  <b>Jingyan Shen</b><sup>3</sup> &nbsp;
  <b>Huan Zhang</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>University of Illinois Urbana-Champaign &nbsp;&nbsp;
  <sup>2</sup>University of California San Diego &nbsp;&nbsp;
  <sup>3</sup>New York University
</p>

<p align="center">
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="https://naicheng-yu.github.io/Benchfill-Website/">Project page</a> &nbsp;·&nbsp;
  <a href="https://github.com/ASTRAL-Group/benchmark-difficulty-measure">Code</a>
</p>

This repository releases **BENCHFILL**, a rewrite-and-annotate search that inserts new benchmark items into the largest gaps of an estimated difficulty distribution. Each round proposes child items from nearby parents, scores them with a fixed 12-model panel, and keeps the valid item whose annotated difficulty is closest to the midpoint of the gap.

The release contains the method code, the original item pools and difficulty tables used by that search, and the 24 gap-filling items reported in the paper (8 benchmarks × 3 gaps).

## What is in this repo

| Path | Contents |
|---|---|
| `new_item_records.csv` | The 24 released items: question, answer, parent, target difficulty, and annotated difficulty (0–10) |
| `paper_table_top3_mixed_drop_second.csv` | Table 2: Top-3 gap-sum reduction and target deviation |
| `irt_results/*_difficulty.csv` | Original-item IRT difficulties (`b_mean`, `b_sd`) for 13 of the paper's 25 benchmarks |
| `data/pools/` | Item pools for the eight BENCHFILL benchmarks |
| `data/matrix/` | 12-model response matrices for those eight benchmarks |
| `propose/` | Rewriter, gap selection, and the search driver |
| `verifier/` | 12-model rollout and online difficulty annotation |
| `random_baseline/scripts/` | Budget-matched uniform-parent baseline |
| `run_experiment.sh` | Entry point for a gap-filling run |
| `scripts/check_paper_metrics.py` | Recomputes Table 1 metrics and the BENCHFILL / Oracle columns of Table 2 from the files above |

Model ability tables are not part of the release. The first time a new item is scored, `verifier/fit_ability.py` estimates one ability per model from `data/matrix/` and the released item difficulties, and writes `irt_results/<benchmark>_ability.csv` on your machine. That file is gitignored. It is a conditional MAP on the same 1PL scale as the released difficulties, so your runs can score new items. It is not the posterior used for the 24 published difficulties. Baseline run logs stay under `random_baseline/results/` and are gitignored too.

Difficulties of the 24 new items are the `difficulty` column of `new_item_records.csv`, on a 0–10 scale. Target Dev. in the table is the mean of `|difficulty − target| / 10` over the three BENCHFILL items in each benchmark. Original-item `b_mean` values are on the IRT scale and are not 0–10 numbers.

## Difficulty metrics

Difficulties are `b_mean` from a 1PL IRT fit, min–max normalized to [0, 1]. DDM is the 1-Wasserstein distance to a same-size uniform reference, (1/n) Σ |z<sub>(i)</sub> − (i − ½)/n|. Max Gap is the largest spacing of the sorted difficulties with z<sub>(0)</sub> = 0 and z<sub>(n+1)</sub> = 1, so the boundary intervals are included; after min–max normalization they have length 0. Top-5 Sum adds the five largest spacings. These are the definitions of the paper's Section 3.1 and App. C.4.

```bash
python scripts/check_paper_metrics.py
```

prints each recomputed value next to the published one. It needs only numpy.

## Scope and data provenance

The final paper is the reference for every published number. This section lists where the released files differ from it.

- **Benchmarks covered.** `irt_results/` has item difficulties for AIME24/25/26, AMC23, MATH-500, Minerva-Math, OlympiadBench, Gaokao2023en, HumanEval, MBPP+, IFEval, BBH-3 objects and MMLU-Pro-Biology. Difficulties for the other 12 benchmarks of Table 1 (GAIA, CORE-Bench Hard, SWE-bench-Mini, USACO, MMLU-Pro-Chemistry, MMLU-Pro-Physics, GPQA-Diamond, ScienceQA, BBH-5/7 objects, ZebraLogic, MedQA) are not in this release.
- **Not included.** The Bayesian (PyMC/NUTS) IRT fitting, the response-matrix and MCMC diagnostics of App. C.3, the model-comparison and posterior-uncertainty analyses, and the calibration experiment of App. D are not part of this repository. `verifier/` contains only the online calibration that scores new items against fixed abilities.
- **Different difficulty version for three benchmarks.** For AMC23, BBH-3 objects and MMLU-Pro-Biology, the released difficulties reproduce the BENCHFILL column of Table 2 but not their Table 1 rows: recomputed DDM / Max Gap / Top-5 Sum are 0.089 / 0.217 / 0.539 (AMC23), 0.090 / 0.075 / 0.294 (BBH-3 objects) and 0.125 / 0.045 / 0.151 (MMLU-Pro-Biology). Table 1 and Fig. 5 appear to come from a different difficulty estimate for these three benchmarks, which is not released.
- **AMC23 Oracle.** The released AMC23 difficulties give an Oracle Top-3 reduction of 35.19%; Table 2 reports 35.78%.
- **Rounding.** IFEval Top-5 Sum recomputes to 0.1894; Table 1 reports 0.190.
- **Item text.** `new_item_records.csv` holds the items exactly as generated and annotated. Table 13 of the paper typesets them; the AMC23 item there is worded slightly differently ("Find the number of integer values of n such that …" for "For how many integer values of n does …").
- **Decoding settings.** `verifier/run_rollout.sh` holds the sampling settings used by the annotation runs. On the math path they differ from Table 4 for Qwen3.5-397B-A17B (T 0.7, top-p 0.8), GLM-4.6 (T 0.6, top-p 0.95), gpt-oss-20b (T 0.7, top-p 0.8), phi-4 (T 0.7, top-p 0.8) and vicuna-7b-v1.5 (T 0.6, top-p 0.9). The lmms-eval path uses T 0.6, top-p 0.95, top-k 20 for every model.
- **Baseline.** Baseline run outputs are not released, so the Baseline column of Table 2 cannot be recomputed from this repository.

## Data sources

`data/pools/` contains copies of public benchmarks: AIME 2024–2026 and AMC 2023 (Mathematical Association of America), Gaokao2023en, BIG-Bench Hard logical deduction (three objects), IFEval and MMLU-Pro (biology). They remain under the terms of their original sources. `vendor/lmms-eval.tar.gz` is a modified copy of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and `verifier/*/latex2sympy/` is a copy of latex2sympy2; both keep their own licenses.

## Hardware

The scripts assume **8× NVIDIA RTX PRO 6000 Blackwell (~96 GB each)**, CUDA devices `0,1,2,3,4,5,6,7`.

- Proposal model: MiniMax-M2.7, TP=4 and DP=2 across all eight GPUs.
- Verifier: the same 12 models, in waves. The two largest models do not fit in BF16 on this machine, so the release uses `Qwen/Qwen3.5-397B-A17B-FP8` and `bullpoint/GLM-4.6-AWQ`, each at TP=8. The other ten verifier models are full precision.

Model ids and wave layout are in `verifier/run_rollout.sh`.

## Setup

```bash
cp env.example env.local && source env.local
bash scripts/setup_server.sh
cd vendor/lmms-eval && uv sync && cd ../..
bash scripts/setup_diff_minimax.sh
```

Three conda environments are used:

| Env | Role |
|---|---|
| `diff_minimax` | MiniMax-M2.7 vLLM server |
| `diff` | Search driver, IRT, and most verifier models |
| `diff_v` | Qwen3-Next-80B verifier wave |

Install `diff` and `diff_v` from `requirements_diff_qwen.txt` and `requirements_diff_qwen_next.txt`. Non-math benchmarks (`mmlu_bio`, `ifeval`, `bbh_3obj`) also need the lmms-eval environment created above. More detail is in [SETUP.md](SETUP.md).

## Run BENCHFILL

`run_experiment.sh` is set to `mmlu_bio`, three widest gaps, MiniMax-M2.7 at temperature 1.0, at most 8 rounds, 4 parents × 4 children per round. Change `BENCHMARK` at the top of that file for another pool (`aime24`, `aime25`, `aime26`, `amc23`, `gaokao2023en`, `bbh_3obj`, `ifeval`). Math benchmarks use `N_SAMPLING=32`; the lmms benchmarks above use `N_SAMPLING=4`.

```bash
conda activate diff
bash run_experiment.sh init      # new run
bash run_experiment.sh run       # resume
bash run_experiment.sh restart   # init, then run
bash run_experiment.sh status
```

The driver serves MiniMax, generates candidates, and calls `verifier/run_rollout.sh` for the 12-model annotation. Outputs land in `propose/results/e2e_split_<benchmark>/`, which is not shipped with this repository.

## Baseline

The budget-matched baseline code is `random_baseline/scripts/run_budget_match.sh`. It samples parents uniformly, matches the rewrite budget of a BENCHFILL run, and selects the annotated item closest to the target. Its outputs stay under `random_baseline/results/` and are gitignored.

```bash
bash random_baseline/scripts/run_budget_match.sh serve
bash random_baseline/scripts/run_budget_match.sh generate
bash random_baseline/scripts/run_budget_match.sh verify-all
bash random_baseline/scripts/run_budget_match.sh summary all
```
