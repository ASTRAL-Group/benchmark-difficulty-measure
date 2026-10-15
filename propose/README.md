# propose — difficulty-targeted item generation

**Contracts (field names, method signatures, extension rules):** [INTERFACES.md](INTERFACES.md).

---

## How to read this codebase

Read in **dependency order**: the pipeline calls everything else; types and loaders have no business logic.

| Step | Where | What to focus on |
|------|--------|-------------------|
| **1. Main loop** | `run/pipeline.py` | `run_pipeline()`: each round generates `k` items, runs correctness then difficulty, appends JSONL, stops when any item is `accepted` or `max_rounds` / `single_round_mode`. Follow `feedback_context` and `summarize_failures_for_next_round` to see how failures become the next prompt. |
| **2. Generation** | `generate/generator.py` | `ItemGenerator.generate(...)` contract; `build_prompt()` — what the model is asked to output. |
| | `generate/json_items.py` | `parse_problem_items_from_model_text` — JSON array → `ProblemItem`. |
| | `generate/vllm_generators.py` | Two backends: OpenAI-compatible HTTP vs in-process `vllm.LLM`; both end with the same parser. |
| **3. Checks** | `correctness/verifier.py` | `CorrectnessVerifier` + optional `NonEmptyFieldsVerifier`. |
| | `correctness/difficulty.py` | `DifficultyEstimator` + `within_tolerance` / `edit_hints_for_difficulty` used by the pipeline. |
| **4. Data & inputs** | `utils/schemas.py` | `ProblemItem`, `RunConfig`, `RunState`, scored types. |
| | `utils/anchors.py` | Loading anchors from jsonl (`difficulty` / `b` / `irt_b`); optional 0–10 normalization. |
| | `utils/difficulty_scale.py` | IRT *b* ↔ **0–10** min–max over a pool (`normalize_items_to_0_10`, etc.). |
| | `utils/anchor_select.py` | Default policy: **2 anchors per gap end** (`select_gap_end_anchors`). |
| | `utils/config.py` | `default_run_config()` for `RunConfig` defaults. |
| **5. Verify understanding** | `tests/` | Small `unittest` cases (anchor selection, JSON parse, failure summarization). Run them while reading the matching module. |

**Tips**

- **`pipeline.py` has no `__main__`** — it is a library; you run the full flow by importing `run_pipeline` from your own script.
- **Do not expect a runnable demo inside the repo** without wiring real `ItemGenerator` / `DifficultyEstimator`. To **inspect** the generator prompt, call `build_prompt(anchors, target_difficulty=..., feedback_context="", k=6)` from a short script or REPL.
- When in doubt, cross-check **INTERFACES.md** for the intended public surface; implementation details may grow, but the ABCs and `ProblemItem` shape are the integration contract.

---

## Concepts (short)

- **Item shape:** in code, **`ProblemItem`** uses `reference_answer`; exported JSONL uses **`answer`**, optional **`modification_policy`** (model-chosen: `small_edit` / `medium_edit` / `large_edit`), and optional **`difficulty`** (**0–10** for anchors / after estimation; the generator prompt does not ask the model to output `difficulty`).
- **Default anchors:** for a gap `[gap_low, gap_high]` on the **same scale as pool difficulties** (typically 0–10), use **`select_gap_end_anchors`** — **2 items per side** (closest from below `gap_low`, closest from above `gap_high`). Order: low group first, then high group → pass into `run_pipeline` / `build_prompt`.

---

## Layout

| Path | Role |
|------|------|
| `run/pipeline.py` | `run_pipeline()` |
| `generate/generator.py` | `ItemGenerator`, `build_prompt` |
| `generate/vllm_generators.py` | `VllmOpenAICompatGenerator` (HTTP API), `VllmEngineGenerator` (in-process) |
| `generate/json_items.py` | Parse model JSON → `ProblemItem` |
| `correctness/` | Verifier + `DifficultyEstimator` |
| `utils/` | `schemas`, `difficulty_scale`, config, jsonl anchors, `anchor_select` |
| `tests/` | Unit tests |

---

## Example: OpenAI-compatible vLLM server

```python
from correctness import ChainedVerifier, NonEmptyFieldsVerifier, OpenAICompatQaMatchVerifier, QaMatchJudgeSettings
from generate.vllm_generators import VllmOpenAICompatGenerator, VllmOpenAICompatSettings
from run.pipeline import run_pipeline
from utils.anchor_select import select_gap_end_anchors
from utils.anchors import load_anchors_from_jsonl
from utils.config import default_run_config

# You still implement DifficultyEstimator for acceptance gating (0–10 scale).
pool = load_anchors_from_jsonl("pool.jsonl", normalize_difficulty_0_10=True)
anchors = select_gap_end_anchors(pool, gap_low=5.5, gap_high=7.5)  # default: 2 per side
gen = VllmOpenAICompatGenerator(
    VllmOpenAICompatSettings(base_url="http://127.0.0.1:8010/v1", model="generator-model-name")
)
correctness = ChainedVerifier(
    [
        NonEmptyFieldsVerifier(),
        OpenAICompatQaMatchVerifier(
            # Use a different model (and/or URL) than `gen` so the judge is independent.
            QaMatchJudgeSettings(base_url="http://127.0.0.1:8011/v1", model="judge-model-name")
        ),
    ]
)
# state = run_pipeline(anchors, gen, correctness, your_difficulty_estimator, default_run_config())
```

## Example: In-process vLLM

```python
from generate.vllm_generators import VllmEngineGenerator, VllmEngineSettings

gen = VllmEngineGenerator(VllmEngineSettings(model="/path/to/weights", tensor_parallel_size=1))
```

## How to run

- **BENCHFILL run:** from the repo root, `bash run_experiment.sh`. That calls `split_gpu_pipeline.py`.
- **Full pipeline:** `from run.pipeline import run_pipeline` in your script; wire real `ItemGenerator` / `CorrectnessVerifier` / `DifficultyEstimator`.
- **Preview prompt text:** load anchors (e.g. `load_anchors_from_jsonl(..., normalize_difficulty_0_10=True)`), then `from generate.generator import build_prompt` and `print(build_prompt(anchors, target_difficulty=6.5, feedback_context="", k=4))`.
- **Unit tests:**

```bash
cd propose   # from the repo root
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

Deps: `openai` and `json-repair` are in `requirements_diff_qwen.txt`; the proposal server uses `requirements_diff_minimax.txt`.
