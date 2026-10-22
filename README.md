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
  <a href="https://arxiv.org/abs/2610.29927">📄 arXiv</a> &nbsp; | &nbsp;
  <a href="https://benchmark-difficulty-measure.github.io/"><img src="https://benchmark-difficulty-measure.github.io/static/images/favicon-32.png" height="18" alt="" align="top"> Project Page</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/Yifan99/BenchFill">🤗 Hugging Face Dataset</a>
</p>

This repository is the official implementation and artifact release of our NeurIPS 2026 paper, *Your Benchmark Is an Empirical Measure Over Difficulty*. We treat a benchmark as an empirical measure over item difficulty and formalize two problems in benchmark evaluation:

- **Measure mismatch:** the benchmark's difficulty distribution differs from the intended evaluation distribution, which can distort model comparisons.
- **Coverage gaps:** some difficulty intervals contain no benchmark items, so improvements within those intervals can go undetected.

<p align="center">
  <img src="https://naicheng-yu.github.io/Benchfill-Website/static/images/fig1_teaser.png" width="100%" alt="Figure 1: measure mismatch obscures model differences, while coverage gaps hide localized training improvements.">
</p>

We diagnose these problems with **Difficulty Distribution Mismatch (DDM)** and **Maximum Gap**, respectively, and introduce **BenchFill**, which generates new items to fill the largest difficulty gaps. The release has three components:

| Component | Contents | Location |
|---|---|---|
| 💻 **1. [Experimental Code](#-1-experimental-code)** | Difficulty diagnostics and the BenchFill algorithm | [`scripts/`](scripts/), [`propose/`](propose/), [`verifier/`](verifier/) |
| 📊 **2. [Difficulty Artifact](#-2-difficulty-artifact)** | Original unnormalized IRT item difficulties | [`data/benchmarks/`](data/benchmarks/) |
| 🧩 **3. [BenchFill Artifact](#-3-benchfill-artifact)** | Augmented AIME24, AIME25, AIME26 and AMC23: original items plus 105 curated BenchFill-generated items | [`data/filled_benchs/`](data/filled_benchs/) |

## 💻 1. Experimental Code

| Path | Contents |
|---|---|
| [`scripts/prepare_response_matrix.py`](scripts/prepare_response_matrix.py) | Aggregate graded model responses into a model-by-item count matrix for IRT |
| [`scripts/fit_irt.py`](scripts/fit_irt.py) | Jointly estimate item difficulties and model abilities with Bayesian 1PL IRT |
| [`scripts/diagnose_benchmark.py`](scripts/diagnose_benchmark.py) | Diagnose any benchmark from its item difficulties: DDM and Max Gap |
| [`propose/`](propose/) | BenchFill generation: gap selection, parent selection, and iterative item rewriting |
| [`verifier/`](verifier/) | Difficulty annotation: 12-model rollouts and IRT scoring of new items against fixed model abilities |
| [`run_experiment.sh`](run_experiment.sh) | Initialize or resume BenchFill's generate–verify loop and inspect run status |

### Prepare a response matrix

Collect and grade the same number of responses for every model–item pair. Save one response per row in `responses.csv`, with `correct` set to `1` or `0` and a unique `trial` ID within each pair. For example, two trials per pair:

```csv
model,item_id,trial,correct
model-a,q0,0,1
model-a,q0,1,0
model-a,q1,0,1
model-a,q1,1,1
model-b,q0,0,0
model-b,q0,1,0
model-b,q1,0,1
model-b,q1,1,0
```

```bash
python3 scripts/prepare_response_matrix.py responses.csv --n-trials 2 \
  --output my_benchmark/response_matrix.csv
```

The output has models as rows, items as columns, and **correct counts** as values:

```csv
model,q0,q1
model-a,1,2
model-b,0,1
```

The script preserves IDs and their first-seen order and checks for missing or duplicate trials. For BenchFill, save the matrix to `data/matrix/<benchmark>_response_matrix.csv`; model names, item IDs, and trial counts must match your calibration. Its `q0`, `q1`, … IDs refer to positions in the evaluated item pool.

### Fit IRT

Use Python 3.12+ to fit **item difficulties and model abilities jointly** from the count matrix. Continuing the two-trial example above:

```bash
python3 -m venv .venv-irt
source .venv-irt/bin/activate
python -m pip install -r requirements_irt.txt

python scripts/fit_irt.py my_benchmark/response_matrix.csv --n-trials 2 \
  --output-dir my_benchmark
python scripts/diagnose_benchmark.py my_benchmark/difficulty.csv
```

The fit follows [Appendix C.3](https://naicheng-yu.github.io/Benchfill-Website/static/pdf/benchfill_paper.pdf#page=23): `P(correct) = g + (1 − g) × sigmoid(theta − b)`, a binomial likelihood, and independent `Normal(0, 1)` priors on model abilities `theta` and item difficulties `b`. Defaults are 4 NUTS chains, 1,000 warmup steps and 1,000 posterior draws per chain, with `target_accept = 0.95`. Set `--n-trials` to the number of responses collected per model–item pair.

Guessing defaults to `g = 0`. For four-choice questions, add `--guessing 0.25`. For varying choice counts, use `--guessing-csv guessing.csv` with columns `item_id,guessing` and one row per item.

The output directory contains:

- **`difficulty.csv`** — `item,b_mean,b_sd,r_hat,ess_bulk`; original matrix item IDs and unnormalized posterior difficulty estimates, ready for the diagnostic script.
- **`ability.csv`** — `model,theta_mean,theta_sd,r_hat,ess_bulk`; posterior model ability estimates.
- **`diagnostics.json`** — maximum R-hat, minimum bulk ESS, divergent transitions, sampling settings, and input identifiers. Check these before interpreting the estimates.

### Diagnose a benchmark

Start with a CSV containing **one estimated difficulty per item** in a `b_mean` column. Use the `difficulty.csv` produced above, or `data/benchmarks/<benchmark>/difficulty.csv` for the 25 released benchmarks. The diagnostic script requires only Python 3.9+.

Run from the repository root:

```bash
python3 scripts/diagnose_benchmark.py data/benchmarks/AIME24/difficulty.csv
```

```text
Items: 30
Normalization: per-benchmark min-max to [0, 1]
DDM: 0.077077
Max Gap: 0.178623
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

**How to read the results.** DDM measures how far the difficulty distribution is from the uniform reference used in our paper; a larger value means a greater mismatch. Max Gap measures the widest difficulty interval with no items; a larger value means a wider region where localized improvements may be missed. In the AIME24 example, the largest uncovered interval spans about 17.9% of the normalized difficulty range.

**Calculation.** By default, the script normalizes each benchmark's IRT difficulties as `z = (b_mean - min(b_mean)) / (max(b_mean) - min(b_mean))`, then sorts them as z<sub>(1)</sub> ≤ … ≤ z<sub>(n)</sub>. Following the paper's reported diagnostics:

- **DDM** is the 1-Wasserstein distance to a same-size empirical uniform reference at `(i − 0.5) / n`: `DDM = mean(|z_(i) − (i − 0.5) / n|)`.
- **Max Gap** is `max(z_(i+1) − z_(i))`, including the boundary intervals defined by z<sub>(0)</sub> = 0 and z<sub>(n+1)</sub> = 1.

### Run BenchFill

**Setup.** The BenchFill generation and verification scripts assume 8× NVIDIA RTX PRO 6000 Blackwell GPUs (~96 GB each). MiniMax-M2.7 serves as the rewriter; the 12 verifier models and their sampling settings are listed in [`verifier/run_rollout.sh`](verifier/run_rollout.sh). See [SETUP.md](SETUP.md) for environments and the bundled lmms-eval.

```bash
cp env.example env.local && source env.local
bash scripts/setup_server.sh
cd vendor/lmms-eval && uv sync && cd ../..
bash scripts/setup_diff_minimax.sh
```

**Configure the run.** Set `BENCHMARK` at the top of `run_experiment.sh` (`aime24`, `aime25`, `aime26`, `amc23`, `gaokao2023en`, `bbh_3obj`, `mmlu_bio`, `ifeval`). By default it targets the three widest gaps with at most 8 rounds of 4 parents × 4 children.

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

Use `item_id` to match each difficulty estimate to its entry in `test.jsonl`. IDs run from `0` to `N−1` within each benchmark. The table below lists the original identifier retained for each source.

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

Original questions and answers are unchanged.

## 📖 Citation

```bibtex
@misc{sun2026benchmarkempiricalmeasuredifficulty,
  title = {Your Benchmark Is an Empirical Measure Over Difficulty},
  author = {Yifan Sun and Naicheng Yu and Jiawen Gong and Jingyan Shen and Huan Zhang},
  year = {2026},
  eprint = {2610.29927},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.29927}
}
```

## ⚖️ License and acknowledgments

Benchmark items in `data/` remain under the terms of their original sources; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for per-source licenses, attributions and the required MIT and Apache-2.0 notices. `vendor/lmms-eval.tar.gz` is a modified copy of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and `verifier/*/latex2sympy/` is a copy of latex2sympy2; both keep their own licenses.

