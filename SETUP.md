# Server setup

> Supported hardware: **8× NVIDIA RTX PRO 6000 Blackwell (~96GB each)**, CUDA devices 0–7. See section 4.

## 1. Clone and paths

```bash
git clone https://github.com/ASTRAL-Group/benchmark-difficulty-measure.git
cd benchmark-difficulty-measure
cp env.example env.local && source env.local
bash scripts/setup_server.sh   # unpacks vendor/lmms-eval.tar.gz → vendor/lmms-eval/
```

## 2. Conda environments

| Env | Purpose | Install |
|-----|---------|---------|
| `diff_minimax` | MiniMax-M2.7 proposal vLLM | `bash scripts/setup_diff_minimax.sh` or `requirements_diff_minimax.txt` |
| `diff` | Pipeline / IRT / most verify models | `requirements_diff_qwen.txt` + numpy/pandas/scipy |
| `diff_v` | The Qwen3-Next-80B wave | `requirements_diff_qwen_next.txt` |

Create `diff_minimax` with the setup script; create `diff` and `diff_v` from the requirements files above:

```bash
bash scripts/setup_diff_minimax.sh
```

## 3. lmms-eval (non-math verify)

Required for **mmlu_bio**, **ifeval**, **bbh_3obj**. Bundled as `vendor/lmms-eval.tar.gz`.

```bash
cd vendor/lmms-eval && uv sync
```

Override `LMMS_EVAL_REPO` in `env.local` if you prefer an external clone. Re-run `bash scripts/setup_server.sh` after changing it.

## 4. Hardware notes

Supported machine: **8× NVIDIA RTX PRO 6000 Blackwell (~96GB each)**, CUDA devices `0,1,2,3,4,5,6,7`.

- MiniMax-M2.7 proposal: **TP=4 DP=2** on all 8 GPUs.
- Verifier: dense 8-GPU waves; Qwen3.5-397B-A17B (FP8) and GLM-4.6 (AWQ) run at **TP=8** (see `verifier/run_rollout.sh`).

## 5. HuggingFace models

- **Proposal**: MiniMax-M2.7 (`propose/sh/serve_minimax_m2_7_8gpu.sh`)
- **Verify**: 12 models in `verifier/run_rollout.sh`

## 6. Run budget-match baseline

```bash
bash random_baseline/scripts/run_budget_match.sh serve      # tmux A
bash random_baseline/scripts/run_budget_match.sh generate   # tmux B
# stop serve, then:
bash random_baseline/scripts/run_budget_match.sh verify-all
bash random_baseline/scripts/run_budget_match.sh summary all
```

Baseline code: `random_baseline/scripts/`. Run outputs are written under `random_baseline/results/` and are not released.

## 7. Run method (gap filling)

```bash
conda activate diff
python -m verifier.fit_ability    # local model abilities; skipped if already written
bash run_experiment.sh restart
```

`fit_ability` reads `data/matrix/` and `irt_results/*_difficulty.csv`. It writes `irt_results/<benchmark>_ability.csv`, which is gitignored. `verifier.main` runs the same step if that file is missing.

Results: `propose/results/e2e_split_<benchmark>/`

## 8. Checklist

- [ ] 8× RTX PRO 6000 Blackwell visible as CUDA devices 0–7 (`nvidia-smi`)
- [ ] `bash scripts/setup_server.sh` finished
- [ ] conda envs `diff`, `diff_minimax`, `diff_v` created
- [ ] `data/pools/<bm>/test.jsonl` present
- [ ] `random_baseline/scripts/run_budget_match.sh` present
