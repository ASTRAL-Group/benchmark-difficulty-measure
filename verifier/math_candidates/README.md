# Math Candidates

Place candidate math problems here for difficulty calibration via the verifier pipeline.

## Directory structure

```
math_candidates/
└── <benchmark_name>/
    └── test.jsonl          # one JSON object per line
```

`<benchmark_name>` must match a key in `config.py:BENCHMARK_META` (e.g. `aime24`,
`math-500`, `gsm8k`).  The rollout script passes this name as `--data_names` to
`math_eval_general.py`, which in turn calls `data_loader.load_data(data_name, "test", data_dir)`.

## JSONL format

Each line must be a JSON object with **at least** these fields:

| Field      | Type   | Description                                    |
|------------|--------|------------------------------------------------|
| `question` | string | The problem statement (LaTeX OK)               |
| `gt`       | string | Ground-truth answer (numeric string)           |
| `gt_cot`   | string | Chain-of-thought solution (can be empty `""`)  |

Optional but recommended:

| Field    | Type       | Description                          |
|----------|------------|--------------------------------------|
| `idx`    | int/string | Stable item id (auto-assigned if missing) |
| `answer` | string     | Same as `gt` (for compatibility with existing parsers) |
| `type`   | string     | Problem category / source            |

### Why `gt_cot` + `gt`?

`parse_ground_truth` in `verifier/diff_eval/parser.py` has a fast path (line 576):
if both `gt_cot` and `gt` are present, it returns them directly without
dispatching on `data_name`.  This makes the candidate format work with **any**
benchmark name, even ones not explicitly listed in the parser.

### Example

```jsonl
{"idx": 0, "question": "Find the sum of all positive integers $n$ such that $n^2 + 12n - 2007$ is a perfect square.", "gt_cot": "", "gt": "88", "answer": "88", "type": "number_theory"}
{"idx": 1, "question": "What is $2^{10} \\mod 7$?", "gt_cot": "", "gt": "2", "answer": "2", "type": "modular_arithmetic"}
```

## Running

```bash
# from the repo root
bash verifier/run_rollout.sh --benchmark aime24 --n 32 --framework math
```
