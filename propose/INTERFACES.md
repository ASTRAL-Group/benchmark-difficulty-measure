# Propose: interface contracts

This document defines the **stable integration surface** for `generate/`, `correctness/`, and `run/`.

---

## 1. Data objects (`utils/schemas.py`)

### `ProblemItem` (anchor, generator output, and stored candidates)

Same four fields everywhere:

| Field | Rule |
|-------|------|
| `id` | Stable string (uuid or model-provided). |
| `question` | Full task text. |
| `reference_answer` | Canonical / claimed answer string. |
| `difficulty` | Optional float on the project’s **0–10** scale (anchors / post-estimation); generator JSON omits it. |

Anchors are `list[ProblemItem]`. The generator returns `list[ProblemItem]` with the same shape.

**IRT → 0–10:** `utils/difficulty_scale.to_difficulty_0_10(b, b_min, b_max)` and `normalize_items_to_0_10(pool, b_min, b_max)`. Optional: `utils/anchors.load_anchors_from_jsonl(..., normalize_difficulty_0_10=True)` min–maxes over the file. `merge_q_indexed_irt_csv_with_eval_jsonl` still emits **raw *b***; normalize before prompts/pipeline.

**Default selection:** `utils/anchor_select.select_gap_end_anchors(pool, gap_low, gap_high, per_side=2)` builds the reference list: `per_side` items at or below `gap_low` with highest difficulty (closest from below), then `per_side` items at or above `gap_high` with lowest difficulty (closest from above). Items without `difficulty` are ignored. `gap_low` / `gap_high` must use the **same scale** as `pool` (typically **0–10**).

**Max gap (optional):** `utils/max_gap.largest_normalized_gap_bracket_b(difficulties)` applies min–max normalization only to identify which adjacent pair has the largest gap. It returns endpoints in the **same units as `difficulties`** (often raw IRT *b* from a CSV). Convert those endpoints to 0–10 with `to_difficulty_0_10` before `select_gap_end_anchors` / `RunConfig.target_difficulty`.

### `CorrectnessResult`

| Field | Rule |
|-------|------|
| `ok` | If `False`, the pipeline **must not** run difficulty estimation on this item. |
| `failure_code` | One of the literals in `FailureCode` in code. |
| `evidence` | Short reason. |
| `judge_raw` | Optional raw judge output. |

### `DifficultyResult` / `RunConfig`

Scalar `estimate` vs `target` on the **0–10** scale; `difficulty_tolerance` is interpreted on that scale. `RunConfig` defaults: `target_difficulty=5.0`, `difficulty_tolerance=0.8`. Optional `irt_b_min` / `irt_b_max` record the IRT span used to build 0–10 (metadata only). Optional **`run_id_for_artifacts`** / **`current_round_index`** are set by `run_pipeline` (and generation-only demos) so generators can persist raw model I/O per round.

---

## 2. `ItemGenerator` (`generate/generator.py`)

**Method**

```text
generate(anchors, *, k, target_difficulty, feedback_context, config) -> list[ProblemItem]
```

**Rules**

1. Return at least `k` items (implementations trim to `k`); on failure, raise.
2. Do not mutate `anchors`.
3. `feedback_context` is opaque text from prior rounds.
4. `target_difficulty` matches the **0–10** scale used in `ProblemItem.difficulty` and `DifficultyResult`.

**Prompt + JSON:** `build_prompt(..., k)` asks the model for a **JSON array of exactly `k` objects** with keys `id`, `question`, `answer`, and **`modification_policy`** (one of `small_edit`, `medium_edit`, `large_edit` — the model chooses per item). No `difficulty` in model output; legacy `reference_answer` is still accepted by the parser. A 0–10 score is produced later by `DifficultyEstimator` when the pipeline runs. Parsing: `generate/json_items.py`. JSONL export uses `answer` via `problem_item_to_json_dict`.

### vLLM implementations (`generate/vllm_generators.py`)

Two supported backends:

1. **`VllmOpenAICompatGenerator`** — HTTP **OpenAI-compatible** Chat Completions API (e.g. `vllm serve` / `--api` server). Configure `VllmOpenAICompatSettings` (`base_url`, `api_key`, `model`, sampling). Requires **`openai`** Python package.

2. **`VllmEngineGenerator`** — **In-process** `vllm.LLM` (load weights in the same process, no server). Configure `VllmEngineSettings` (`model`, `tensor_parallel_size`, etc.). Requires **`vllm`** installed.

Both call `build_prompt`, then `parse_problem_items_from_model_text` on the generated text.

When `RunConfig.run_id_for_artifacts` is set (`run_pipeline` sets it to the run id; `split_gpu_pipeline.py generate` uses `propose_gen_<benchmark>_round<r>`), both implementations also write **`model_round_<r>_response.txt`**, **`model_round_<r>_prompt.txt`**, and **`model_round_<r>_meta.json`** under `artifact_dir / run_id_for_artifacts` next to the per-round candidate JSONL. Set **`PROPOSE_NO_FULL_RESPONSE_SAVE=1`** to skip these files.

Each run directory also includes **`anchor_dataset_ids.txt`**: anchor `ProblemItem.id` values in prompt order (same ids as the source eval jsonl / matrix columns when applicable), written by `run.pipeline.write_anchor_dataset_ids_txt`.

**Helper:** `PromptLoggingGenerator` wraps another `ItemGenerator` and sets `last_prompt`.

---

## 3. `CorrectnessVerifier` (`correctness/verifier.py`)

```text
verify(item: ProblemItem) -> CorrectnessResult
```

`NonEmptyFieldsVerifier` only checks non-empty `question` and `reference_answer` (no LLM).

**QA match (LLM judge):** `correctness/qa_match.py` — `OpenAICompatQaMatchVerifier` uses the same OpenAI-compatible Chat Completions API as `VllmOpenAICompatGenerator`. Configure `QaMatchJudgeSettings` (`base_url`, `model`, etc.). **Use a different served model (and optionally a different server) than the generator** so evaluation is not the same model that wrote the item. The judge must reply with JSON `{"match": bool, "reason": str}`; on `match: false`, `failure_code` is `QA_MISMATCH`; unparseable or API errors use `JUDGE_REJECT`.

**Chaining:** `ChainedVerifier([NonEmptyFieldsVerifier(), OpenAICompatQaMatchVerifier(...)])` runs cheap checks before the LLM call.

---

## 4. `DifficultyEstimator` (`correctness/difficulty.py`)

```text
estimate(item, *, target, anchors) -> DifficultyResult
```

`anchors` are `Sequence[ProblemItem]`.

**Demo stub:** `ConstantDifficultyEstimator(fixed_estimate)` always returns that value on the 0–10 scale (for smoke runs without a real scorer).

**Verifier IRT:** `adapters/verifier_difficulty.py` defines `VerifierDifficultyAdapter`, wrapping `verifier.estimators.FullRolloutEstimator` so it implements this ABC (`ProblemItem` → `question` / `reference_answer`). Build via `build_verifier_difficulty_estimator(workspace, benchmark, endpoints_json)`; JSON format in `adapters/examples/verifier_endpoints.example.json`. Requires repo-root `verifier` deps (e.g. numpy, pandas). The search driver uses `split_gpu_pipeline.py` and `verifier/run_rollout.sh` instead of this adapter.

---

## 5. Pipeline (`run/pipeline.py`)

```text
run_pipeline(anchors, generator, correctness, difficulty, config, *, run_id=None) -> RunState
```

Per-round JSONL rows include `id`, `question`, `answer`, `modification_policy`, `item_difficulty` (from the item), plus `estimated_difficulty` from your estimator when run.

---

## 6. Extending

1. New backends: subclass ABCs; keep method signatures.
2. New fields: optional dataclass fields or `schema_version` on `RunState`.
3. New failure codes: extend `FailureCode` in `utils/schemas.py`.
