"""Rollout runner: run inference on candidate items and judge responses.

Supports two backends:
  1. OpenAI-compatible vLLM server (requires a running `vllm serve` endpoint)
  2. Direct lmms_eval invocation (heavier, but matches existing pipeline exactly)

For the verifier we default to the OpenAI-compatible approach since it's
simpler to call per-item and matches the proposal repo's convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BENCHMARK_META, VerifierConfig
from .judge import judge_answer


@dataclass
class ModelEndpoint:
    """Connection info for one model served via OpenAI-compatible API."""

    name: str
    base_url: str  # e.g. "http://localhost:8000/v1"
    model: str  # model name as registered in the server
    api_key: str = "EMPTY"
    # Generation parameters — should match scripts/*.sh
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 10240


@dataclass
class RolloutResult:
    """Result of running one item on one or more models."""

    # model_name -> n_correct out of n_trials
    scores: dict[str, int] = field(default_factory=dict)
    # model_name -> list of (response_text, correct_bool) per trial
    details: dict[str, list[tuple[str, bool]]] = field(default_factory=dict)

    def to_y_vector(self, model_order: list[str]) -> NDArray:
        """Convert to y_m array aligned with model_order."""
        return np.array([self.scores.get(m, 0) for m in model_order])


class RolloutRunner:
    """Run inference for candidate items against a panel of models."""

    def __init__(
        self,
        endpoints: list[ModelEndpoint],
        cfg: VerifierConfig,
    ):
        self.endpoints = {ep.name: ep for ep in endpoints}
        self.cfg = cfg
        self.meta = cfg.meta

    def _call_model(
        self, endpoint: ModelEndpoint, prompt: str, n: int = 1
    ) -> list[str]:
        """Call one model endpoint and return n response texts."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  (needed for rollout)")

        client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
        responses: list[str] = []
        for _ in range(n):
            completion = client.chat.completions.create(
                model=endpoint.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=endpoint.temperature,
                top_p=endpoint.top_p,
                max_tokens=endpoint.max_tokens,
            )
            text = completion.choices[0].message.content or ""
            responses.append(text)
        return responses

    def run_single_item(
        self,
        question: str,
        reference_answer: str,
        model_subset: list[str] | None = None,
    ) -> RolloutResult:
        """Run one candidate item on (a subset of) models.

        Parameters
        ----------
        question : the full prompt / question text
        reference_answer : gold answer for judging
        model_subset : if given, only run these model names

        Returns
        -------
        RolloutResult with scores and per-trial details
        """
        targets = model_subset or list(self.endpoints.keys())
        result = RolloutResult()

        for model_name in targets:
            ep = self.endpoints[model_name]
            responses = self._call_model(ep, question, n=self.meta.n_trials)

            trials: list[tuple[str, bool]] = []
            n_correct = 0
            for resp_text in responses:
                correct = judge_answer(resp_text, reference_answer, self.meta.judge)
                trials.append((resp_text, correct))
                n_correct += int(correct)

            result.scores[model_name] = n_correct
            result.details[model_name] = trials

        return result

    def run_batch(
        self,
        items: list[dict[str, str]],
        model_subset: list[str] | None = None,
    ) -> list[RolloutResult]:
        """Run multiple items. Each item is {"question": ..., "reference_answer": ...}."""
        return [
            self.run_single_item(
                it["question"], it["reference_answer"], model_subset
            )
            for it in items
        ]


# ---------------------------------------------------------------------------
# Offline rollout: load from existing logs (for validation experiments)
# ---------------------------------------------------------------------------


def load_rollout_from_matrix(
    matrix_path: str | Path,
    item_ids: list[str] | None = None,
) -> tuple[NDArray, list[str], list[str]]:
    """Load response matrix CSV as (array, model_names, item_ids).

    Returns
    -------
    arr : (M, N) array of correct counts
    models : list of model names
    items : list of item ids
    """
    import pandas as pd

    df = pd.read_csv(matrix_path, index_col=0)
    models = df.index.tolist()
    items = df.columns.tolist()
    arr = df.to_numpy()

    if item_ids is not None:
        mask = [items.index(i) for i in item_ids]
        arr = arr[:, mask]
        items = item_ids

    return arr, models, items
