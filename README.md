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
  <a href="https://naicheng-yu.github.io/Benchfill-Website/">Project Page</a>
</p>

This repository is the official implementation and artifact release of our NeurIPS 2026 paper, *Your Benchmark Is an Empirical Measure Over Difficulty*. The paper treats a benchmark as an empirical measure over item difficulty, diagnoses it with Difficulty Distribution Mismatch (DDM) and Maximum Gap, and proposes BenchFill, which writes new items into the largest difficulty gaps. The release has three components:

| Component | Contents | Location |
|---|---|---|
| **1. [Experimental Code](#1-experimental-code)** | Difficulty diagnostics, the BenchFill algorithm and its baseline | [`propose/`](propose/), [`verifier/`](verifier/), [`random_baseline/`](random_baseline/), [`scripts/`](scripts/) |
| **2. [Difficulty Artifact](#2-difficulty-artifact)** | Original unnormalized IRT item difficulties (`b_mean`), with original items where redistribution is permitted | [`data/benchmarks/`](data/benchmarks/) |
| **3. [BenchFill Artifact](#3-benchfill-artifact)** | 106 curated BenchFill-generated items for AIME24, AIME25, AIME26 and AMC23 | [`data/filled_benchs/`](data/filled_benchs/) |

## 1. Experimental Code

| Path | Contents |
|---|---|
| [`propose/`](propose/) | BenchFill generation: rewriter, gap and parent selection, search driver |
| [`verifier/`](verifier/) | Difficulty annotation: 12-model rollouts and IRT scoring of new items against fixed model abilities |
| [`random_baseline/scripts/`](random_baseline/scripts/) | Budget-matched uniform-parent baseline |
| [`scripts/check_paper_metrics.py`](scripts/check_paper_metrics.py) | DDM, Max Gap and Top-k gap sums; Top-3 gap-sum reduction and target deviation for inserted items |
| [`run_experiment.sh`](run_experiment.sh) | Entry point for a BenchFill run |
| [`irt_results/`](irt_results/), [`data/pools/`](data/pools/), [`data/matrix/`](data/matrix/) | Runtime inputs for the eight BenchFill benchmarks: IRT difficulties (`b_mean`, `b_sd`), item pools, and 12-model response matrices |
| [`new_item_records.csv`](new_item_records.csv) | The 24 BenchFill items of Table 2 (8 benchmarks × 3 gaps) with target and annotated difficulty (0–10) |

**Diagnostics.** For difficulties z<sub>(1)</sub> ≤ … ≤ z<sub>(n)</sub> in [0, 1], DDM is the 1-Wasserstein distance to a same-size uniform reference, (1/n) Σ |z<sub>(i)</sub> − (i − ½)/n|, and Max Gap is the largest spacing with z<sub>(0)</sub> = 0 and z<sub>(n+1)</sub> = 1 (Section 3.1 and App. C.4). An example on the Difficulty Artifact is given in [Section 2](#2-difficulty-artifact-25-benchmarks).

**Setup.** The scripts assume 8× NVIDIA RTX PRO 6000 Blackwell GPUs (~96 GB each). MiniMax-M2.7 serves as the rewriter; the 12 verifier models and their sampling settings are listed in [`verifier/run_rollout.sh`](verifier/run_rollout.sh). See [SETUP.md](SETUP.md) for environments and the bundled lmms-eval.

```bash
cp env.example env.local && source env.local
bash scripts/setup_server.sh
cd vendor/lmms-eval && uv sync && cd ../..
bash scripts/setup_diff_minimax.sh
```

**Run BenchFill.** Set `BENCHMARK` at the top of `run_experiment.sh` (`aime24`, `aime25`, `aime26`, `amc23`, `gaokao2023en`, `bbh_3obj`, `mmlu_bio`, `ifeval`). By default it fills the three widest gaps with at most 8 rounds of 4 parents × 4 children.

```bash
conda activate diff
bash run_experiment.sh init      # new run
bash run_experiment.sh run       # resume
bash run_experiment.sh status
```

Model abilities are fit locally from `data/matrix/` on first use (`verifier/fit_ability.py`). Outputs are written to `propose/results/`.

**Run the baseline.**

```bash
bash random_baseline/scripts/run_budget_match.sh serve
bash random_baseline/scripts/run_budget_match.sh generate
bash random_baseline/scripts/run_budget_match.sh verify-all
bash random_baseline/scripts/run_budget_match.sh summary all
```

The Bayesian IRT fitting of the original benchmarks (PyMC/NUTS) is not included.

## 2. Difficulty Artifact

[`data/benchmarks/<benchmark>/`](data/benchmarks/) holds one directory per benchmark in Table 1 of the paper:

- **`difficulty.csv`** — `item_id,b_mean`. `b_mean` is the original unnormalized IRT item difficulty, copied directly from the selected source estimates without refitting or rounding.
- **`test.jsonl`** — the evaluated items with their original fields, or only `item_id` and `source_item_id` where questions are not redistributed.

Both files use the same integer `item_id`, numbered `0` to `N−1`; join on it. Original identifiers are kept as provenance in the field listed below. Paper diagnostics use per-benchmark min–max normalization, `z = (b_mean - min(b_mean)) / (max(b_mean) - min(b_mean))`, as shown below.

```python
import csv
import numpy as np

b = np.array([float(r["b_mean"])
              for r in csv.DictReader(open("data/benchmarks/AIME24/difficulty.csv"))])
z = np.sort((b - b.min()) / (b.max() - b.min()))
n = len(z)
ddm = np.abs(z - (np.arange(1, n + 1) - 0.5) / n).mean()
gaps = np.diff(np.concatenate([[0.0], z, [1.0]]))
print(f"DDM {ddm:.3f}  Max Gap {gaps.max():.3f}")
```

| Benchmark | Items | `test.jsonl` | Original identifier |
|---|---:|---|---|
| `MATH-500` | 500 | Questions | `unique_id` |
| `Minerva-Math` | 272 | Questions | `idx` |
| `OlympiadBench` | 675 | Questions | `id` |
| `Gaokao2023en` | 385 | Questions | `source_item_id` (`qN`) |
| `AMC23` | 40 | Questions | original problem `url` |
| `AIME24` | 30 | Questions | original problem `url` |
| `AIME25` | 30 | Questions | evaluated input `id` |
| `AIME26` | 30 | Questions | evaluated input `id` |
| `MMLU-Pro-Biology` | 717 | Questions | `id` (`qN`) |
| `MMLU-Pro-Chemistry` | 1,132 | Questions | `question_id` |
| `MMLU-Pro-Physics` | 1,299 | Questions | `question_id` |
| `GPQA-Diamond` | 198 | IDs only | `source_item_id` (original `Record ID`) |
| `ScienceQA` | 4,241 | Questions | `source_item_id` (original ScienceQA problem ID) |
| `HumanEval` | 164 | Questions | `task_id` |
| `MBPP+` | 378 | Questions | `task_id` |
| `GAIA` | 165 | IDs only | `source_item_id` (response-matrix position) |
| `CORE-Bench Hard` | 45 | IDs only | `source_item_id` (native capsule ID) |
| `SWE-bench-Mini` | 50 | IDs only | `source_item_id` (native instance ID) |
| `USACO` | 307 | IDs only | `source_item_id` (native problem ID) |
| `IFEval` | 541 | Questions | `key` |
| `BBH-3 objects` | 250 | Questions | `id` (`qN`) |
| `BBH-5 objects` | 250 | Questions | `source_item_id` (`qN`) |
| `BBH-7 objects` | 250 | Questions | `source_item_id` (`qN`) |
| `ZebraLogic` | 1,000 | Questions | `id` |
| `MedQA` | 1,273 | Questions | `source_item_id` (`qN`) |

ScienceQA is the text-only evaluation used in the paper: image filenames are kept as metadata, but no images are included. GPQA-Diamond questions and answers are withheld under the dataset's access terms. The four agent benchmarks release task IDs only; GAIA's `source_item_id` values are positions in our response matrix, not official GAIA task IDs. Per-source versions, licenses and ID provenance are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 3. BenchFill Artifact

[`data/filled_benchs/`](data/filled_benchs/) contains **106 generated benchmark items** from iterative gap filling with [`propose/mass_fill/`](propose/mass_fill/). These are generated questions, distinct from the original benchmark questions in the Difficulty Artifact. The collection includes all 12 Table 2 seed items for these four benchmarks and 94 subsequent mass-filling items.

| Benchmark | Items | JSON array |
|---|---:|---|
| AIME24 | 21 | [`AIME24.json`](data/filled_benchs/AIME24.json) |
| AIME25 | 22 | [`AIME25.json`](data/filled_benchs/AIME25.json) |
| AIME26 | 23 | [`AIME26.json`](data/filled_benchs/AIME26.json) |
| AMC23 | 40 | [`AMC23.json`](data/filled_benchs/AMC23.json) |
| **Total** | **106** | |

Each object contains:

- `id`, `benchmark`: generated-item ID and benchmark name. IDs use the format `<benchmark>-fill-<number>`, for example `AIME25-fill-004`. Original numeric suffixes and item order are preserved; gaps reflect curation.
- `question`, `answer`: complete generated question and reference answer, stored as strings.
- `target_difficulty`: the difficulty the search aimed to fill; `estimated_difficulty`: the selected item's recorded difficulty annotation.
- `source_id`, `parent_source_id`, and optional `run_id`: preserved generation identifiers for tracing provenance. Source identifiers are local to their generation context; use `id` as the release key.

**Difficulty scale.** Both difficulty fields use the benchmark-specific normalized **[0, 1]** scale of the gap-filling experiments, obtained by dividing the recorded Sheet values on the 0–10 scale by 10 without rounding. The separate Difficulty Artifact supplies original-item **raw IRT `b_mean`** values; normalize those per benchmark when comparing scales. No raw IRT parameters are reconstructed for generated items.

This is the available curated mass-filling output, not a claim that every planned generation completed successfully. The 12 Table 2 seeds have documented human review; human review is not documented for the remaining 94 items.

## Citation

```bibtex
@inproceedings{sun2026benchmark,
  title     = {Your Benchmark Is an Empirical Measure Over Difficulty},
  author    = {Sun, Yifan and Yu, Naicheng and Gong, Jiawen and Shen, Jingyan and Zhang, Huan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License and acknowledgments

Benchmark items in `data/` remain under the terms of their original sources; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for per-source licenses, attributions and the required MIT and Apache-2.0 notices. `vendor/lmms-eval.tar.gz` is a modified copy of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and `verifier/*/latex2sympy/` is a copy of latex2sympy2; both keep their own licenses.
