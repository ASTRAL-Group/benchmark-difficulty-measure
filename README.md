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

This repository is the official implementation and artifact release of our NeurIPS 2026 paper, *Your Benchmark Is an Empirical Measure Over Difficulty*. We treat a benchmark as an empirical measure over item difficulty and formalize two problems in benchmark evaluation:

- **Measure mismatch:** the benchmark's difficulty distribution differs from the intended evaluation distribution, which can distort model comparisons.
- **Coverage gaps:** some difficulty intervals contain no benchmark items, so improvements within those intervals can go undetected.

We diagnose these problems with **Difficulty Distribution Mismatch (DDM)** and **Maximum Gap**, respectively, and introduce **BenchFill**, which generates new items to fill the largest difficulty gaps. The release has three components:

| Component | Contents | Location |
|---|---|---|
| 💻 **1. [Experimental Code](#-1-experimental-code)** | Difficulty diagnostics and the BenchFill algorithm | [`scripts/`](scripts/), [`propose/`](propose/), [`verifier/`](verifier/) |
| 📊 **2. [Difficulty Artifact](#-2-difficulty-artifact)** | Original unnormalized IRT item difficulties | [`data/benchmarks/`](data/benchmarks/) |
| 🧩 **3. [BenchFill Artifact](#-3-benchfill-artifact)** | Augmented AIME24, AIME25, AIME26 and AMC23: original items plus 105 curated BenchFill-generated items | [`data/filled_benchs/`](data/filled_benchs/) |

## 💻 1. Experimental Code

| Path | Contents |
|---|---|
| [`scripts/diagnose_benchmark.py`](scripts/diagnose_benchmark.py) | Diagnose any benchmark from its item difficulties: DDM, Max Gap and the largest uncovered interval |
| [`propose/`](propose/) | BenchFill generation: rewriter, gap and parent selection, search driver |
| [`verifier/`](verifier/) | Difficulty annotation: 12-model rollouts and IRT scoring of new items against fixed model abilities |
| [`run_experiment.sh`](run_experiment.sh) | Initialize or resume BenchFill's generate–verify loop and inspect run status |

### Diagnose a benchmark

Start with a CSV containing **one estimated difficulty per item**. For the 25 released benchmarks, use `data/benchmarks/<benchmark>/difficulty.csv`. For your own benchmark, first estimate item difficulties with IRT from model responses, then save them in a `b_mean` column. The command below reads those estimates and computes both diagnostics; it requires only Python 3.9+ and no GPUs or model setup.

Run from the repository root:

```bash
python3 scripts/diagnose_benchmark.py data/benchmarks/AIME24/difficulty.csv
```

```text
Items: 30
Normalization: per-benchmark min-max to [0, 1]
DDM: 0.077077
Max Gap: 0.178623
Largest gap endpoints: [0.783708, 0.962332]
```

For your own difficulty file:

```bash
# Read a b_mean column and apply per-benchmark min-max normalization.
python3 scripts/diagnose_benchmark.py my_benchmark/difficulty.csv

# Use a different column name; add --json for machine-readable output.
python3 scripts/diagnose_benchmark.py my_benchmark/difficulty.csv --column difficulty --json

# If difficulties are already on the intended [0, 1] scale, preserve that scale.
python3 scripts/diagnose_benchmark.py my_benchmark/normalized.csv --column z --normalized
```

**How to read the results.** DDM measures how far the difficulty distribution is from the uniform reference used in our paper; a larger value means a greater mismatch. Max Gap measures the widest difficulty interval with no items; a larger value means a wider region where localized improvements may be missed. In the AIME24 example, the largest uncovered interval spans about 17.9% of the normalized difficulty range. Its endpoints locate a region to target with new items.

**Calculation.** By default, the script normalizes each benchmark's IRT difficulties as `z = (b_mean - min(b_mean)) / (max(b_mean) - min(b_mean))`, then sorts them as z<sub>(1)</sub> ≤ … ≤ z<sub>(n)</sub>. Following the paper's reported diagnostics:

- **DDM** is the 1-Wasserstein distance to a same-size empirical uniform reference at `(i − 0.5) / n`: `DDM = mean(|z_(i) − (i − 0.5) / n|)`.
- **Max Gap** is `max(z_(i+1) − z_(i))`, including the boundary intervals defined by z<sub>(0)</sub> = 0 and z<sub>(n+1)</sub> = 1.

Per-benchmark min–max normalization describes coverage within each benchmark's observed difficulty range. With `--normalized`, the supplied scale is used unchanged, including any gaps next to 0 and 1. The script rejects empty inputs, missing or nonfinite difficulties, and constant raw difficulties for which min–max normalization is undefined.

To reproduce the paper's Table 1 and Table 2 comparisons, see [`scripts/check_paper_metrics.py`](scripts/check_paper_metrics.py).

### Run BenchFill

**Setup.** The BenchFill generation and verification scripts assume 8× NVIDIA RTX PRO 6000 Blackwell GPUs (~96 GB each). MiniMax-M2.7 serves as the rewriter; the 12 verifier models and their sampling settings are listed in [`verifier/run_rollout.sh`](verifier/run_rollout.sh). See [SETUP.md](SETUP.md) for environments and the bundled lmms-eval.

```bash
cp env.example env.local && source env.local
bash scripts/setup_server.sh
cd vendor/lmms-eval && uv sync && cd ../..
bash scripts/setup_diff_minimax.sh
```

**Configure the run.** Set `BENCHMARK` at the top of `run_experiment.sh` (`aime24`, `aime25`, `aime26`, `amc23`, `gaokao2023en`, `bbh_3obj`, `mmlu_bio`, `ifeval`). By default it targets the three widest gaps with at most 8 rounds of 4 parents × 4 children. Set `N_SAMPLING` to match your IRT calibration (32 for math benchmarks; 4 for the supported lmms-eval tasks).

```bash
conda activate diff
bash run_experiment.sh init      # initialize a new run
bash run_experiment.sh run       # start or resume generation and verification
bash run_experiment.sh status    # inspect progress
```

Supply the runtime IRT calibration tables (`b_mean`, `b_sd`), item pools, and 12-model response matrices locally before running BenchFill. Model abilities are fit from the supplied matrices on first use (`verifier/fit_ability.py`). Outputs are written to `propose/results/`.

## 📊 2. Difficulty Artifact

[`data/benchmarks/<benchmark>/`](data/benchmarks/) holds one directory per benchmark in Table 1 of the paper:

- **`difficulty.csv`** — `item_id,b_mean`. `b_mean` is the unnormalized IRT item difficulty.
- **`test.jsonl`** — the evaluated items with their original fields. For GPQA-Diamond, GAIA, CORE-Bench Hard, SWE-bench-Mini and USACO, this file contains only `item_id` and `source_item_id`; question and answer content is not included.

Use `item_id` to match each difficulty estimate to its entry in `test.jsonl`. IDs run from `0` to `N−1` within each benchmark. The table below lists the original identifier retained for each source. To compute DDM and Max Gap from a `difficulty.csv`, follow [Diagnose a benchmark](#diagnose-a-benchmark).

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

For the ID-only datasets, obtain the underlying tasks from their original sources listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). **GAIA is an exception:** its `source_item_id` values are positions in our archived response matrix, and the mapping to official GAIA task IDs is currently unavailable. These values cannot be used to retrieve the original GAIA tasks. The notices also document source versions, licenses and identifier provenance.

## 🧩 3. BenchFill Artifact

[`data/filled_benchs/`](data/filled_benchs/) contains four augmented evaluation benchmarks. Each JSON array is the **original benchmark followed by our curated generated items** from iterative gap filling with [`propose/mass_fill/`](propose/mass_fill/). The original items come first, in the `item_id` order of the Difficulty Artifact.

| Benchmark | Original | Generated | Total | JSON array |
|---|---:|---:|---:|---|
| AIME24 | 30 | 20 | 50 | [`AIME24.json`](data/filled_benchs/AIME24.json) |
| AIME25 | 30 | 22 | 52 | [`AIME25.json`](data/filled_benchs/AIME25.json) |
| AIME26 | 30 | 23 | 53 | [`AIME26.json`](data/filled_benchs/AIME26.json) |
| AMC23 | 40 | 40 | 80 | [`AMC23.json`](data/filled_benchs/AMC23.json) |

Each item has three fields: `id`, `question` and `answer`. The ID tells you where the item came from:

- `AIME26-origin-000` is an original AIME26 item; `000` corresponds to `item_id = 0` in the Difficulty Artifact.
- `AIME26-fill-001` is a new item generated by BenchFill.

Original questions and answers are unchanged. Their IRT difficulty estimates are available in the [Difficulty Artifact](#-2-difficulty-artifact).

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
