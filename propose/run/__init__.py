"""Multi-round run: see run/pipeline.py (single entrypoint + disk helpers)."""

from run.pipeline import (
    append_round_jsonl,
    new_run_id,
    run_pipeline,
    save_run_state,
)

__all__ = [
    "append_round_jsonl",
    "new_run_id",
    "run_pipeline",
    "save_run_state",
]
