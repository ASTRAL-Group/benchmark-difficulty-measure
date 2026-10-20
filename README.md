# [NeurIPS 2026] Your Benchmark Is an Empirical Measure Over Difficulty

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
  <a href="https://naicheng-yu.github.io/Benchfill-Website/"><img src="https://naicheng-yu.github.io/Benchfill-Website/static/images/favicon-32.png" height="18" alt="" align="top"> Project Page</a>
</p>

This repository is the official implementation and artifact release of our NeurIPS 2026 paper, *Your Benchmark Is an Empirical Measure Over Difficulty*. The paper treats a benchmark as an empirical measure over item difficulty, diagnoses it with Difficulty Distribution Mismatch (DDM) and Maximum Gap, and proposes BenchFill, which writes new items into the largest difficulty gaps. The release has three components:

| Component | Contents | Location |
|---|---|---|
| 🧪 **1. [Experimental Code](#-1-experimental-code)** | Difficulty diagnostics, the BenchFill algorithm and its baseline | [`propose/`](propose/), [`verifier/`](verifier/), [`random_baseline/`](random_baseline/), [`scripts/`](scripts/) |
| 📊 **2. [Difficulty Artifact](#-2-difficulty-artifact)** | Original unnormalized IRT item difficulties (`b_mean`), with original items where redistribution is permitted | [`data/benchmarks/`](data/benchmarks/) |
| 🧩 **3. [BenchFill Artifact](#-3-benchfill-artifact)** | Augmented AIME24, AIME25, AIME26 and AMC23: original items plus 105 curated BenchFill-generated items | [`data/filled_benchs/`](data/filled_benchs/) |

## 🧪 1. Experimental Code

| Path | Contents |
|---|---|
| [`propose/`](propose/) | BenchFill generation: rewriter, gap and parent selection, search driver |
| [`verifier/`](verifier/) | Difficulty annotation: 12-model rollouts and IRT scoring of new items against fixed model abilities |
| [`scripts/check_paper_metrics.py`](scripts/check_paper_metrics.py) | DDM, Max Gap and Top-k gap sums; Top-3 gap-sum reduction and target deviation for inserted items |
| [`run_experiment.sh`](run_experiment.sh) | Entry point for a BenchFill run |

**Diagnostics.** For difficulties z<sub>(1)</sub> ≤ … ≤ z<sub>(n)</sub> in [0, 1], DDM is the 1-Wasserstein distance to a same-size uniform reference, (1/n) Σ |z<sub>(i)</sub> − (i − ½)/n|, and Max Gap is the largest spacing with z<sub>(0)</sub> = 0 and z<sub>(n+1)</sub> = 1 (Section 3.1 and App. C.4).

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

Supply the runtime IRT calibration tables (`b_mean`, `b_sd`), item pools, and 12-model response matrices locally before running BenchFill. Model abilities are fit from the supplied matrices on first use (`verifier/fit_ability.py`). Outputs are written to `propose/results/`.

**Run the baseline.**

```bash
bash random_baseline/scripts/run_budget_match.sh serve
bash random_baseline/scripts/run_budget_match.sh generate
bash random_baseline/scripts/run_budget_match.sh verify-all
bash random_baseline/scripts/run_budget_match.sh summary all
```

## 📊 2. Difficulty Artifact

[`data/benchmarks/<benchmark>/`](data/benchmarks/) holds one directory per benchmark in Table 1 of the paper:

- **`difficulty.csv`** — `item_id,b_mean`. `b_mean` is the unnormalized IRT item difficulty.
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

The four agent benchmarks release task IDs only; GAIA's `source_item_id` values are positions in our response matrix, not official GAIA task IDs. Per-source versions, licenses and ID provenance are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 🧩 3. BenchFill Artifact

[`data/filled_benchs/`](data/filled_benchs/) contains four augmented evaluation benchmarks, ready to use as is. Each JSON array is the **original benchmark followed by our curated generated items** from iterative gap filling with [`propose/mass_fill/`](propose/mass_fill/). The original items come first, in the `item_id` order of the Difficulty Artifact. The generated items follow in generation order.

| Benchmark | Original | Generated | Total | JSON array |
|---|---:|---:|---:|---|
| AIME24 | 30 | 20 | 50 | [`AIME24.json`](data/filled_benchs/AIME24.json) |
| AIME25 | 30 | 22 | 52 | [`AIME25.json`](data/filled_benchs/AIME25.json) |
| AIME26 | 30 | 23 | 53 | [`AIME26.json`](data/filled_benchs/AIME26.json) |
| AMC23 | 40 | 40 | 80 | [`AMC23.json`](data/filled_benchs/AMC23.json) |

Every record contains exactly three fields, in order: `id`, `question` and `answer`. Original items have IDs `<benchmark>-origin-<NNN>`, where `NNN` is the zero-based Difficulty Artifact `item_id` (for example `AIME26-origin-000`). Generated items have IDs `<benchmark>-fill-<NNN>`, with stable published numeric suffixes (for example `AIME26-fill-001`). IDs are not renumbered after curation; the missing `AIME24-fill-005` is an intentional gap. Original questions and answers are copied unchanged from the Difficulty Artifact (AMC23 answers keep their original numeric type). Original-item IRT difficulty estimates are in the [Difficulty Artifact](#-2-difficulty-artifact).

## 📖 Citation

```bibtex
@inproceedings{sun2026benchmark,
  title     = {Your Benchmark Is an Empirical Measure Over Difficulty},
  author    = {Sun, Yifan and Yu, Naicheng and Gong, Jiawen and Shen, Jingyan and Zhang, Huan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## ⚖️ License and acknowledgments

Benchmark items in `data/` remain under the terms of their original sources; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for per-source licenses, attributions and the required MIT and Apache-2.0 notices. `vendor/lmms-eval.tar.gz` is a modified copy of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and `verifier/*/latex2sympy/` is a copy of latex2sympy2; both keep their own licenses.
