"""Shared dataclasses. Main loop: run/pipeline.py. Contracts: INTERFACES.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal


# Generator may set per item; model must use one of these strings when emitting ``modification_policy``.
MODIFICATION_POLICY_CANONICAL_VALUES: Final[frozenset[str]] = frozenset(
    {"small_edit", "medium_edit", "large_edit"}
)

FailureCode = Literal[
    "OK",
    "PARSE_ERROR",
    "AMBIGUOUS",
    "POLICY",
]


@dataclass
class ProblemItem:
    """One benchmark item: anchor, model output, or final dataset row.

    In-memory field is ``reference_answer``. Serialized JSON (``problem_item_to_json_dict``,
    eval-style jsonl) uses the key ``answer`` to match ``evaluation/data/*/test.jsonl``.
    Model prompts ask for ``answer`` and ``modification_policy`` in the returned JSON array;
    ``difficulty`` is left ``None`` until a later estimator (or set for anchors from IRT on a **0–10**
    scale; see ``utils.difficulty_scale``).
    ``parent_id`` tracks which parent item generated this child (None for anchor/initial items).
    """

    id: str
    question: str
    reference_answer: str
    difficulty: float | None = None
    modification_policy: str | None = None
    parent_id: str | None = None
    # MCQ benchmarks (mmlu_pro family) carry the option strings separately so the lmms-eval
    # task yaml can render them via its own template (Question:/Options:/A./B./...). For
    # bbh-style MCQs the options are baked into ``question`` and this stays None.
    options: list[str] | None = None
    # ifeval-only: the program-checked instruction-id list and per-instruction kwargs
    # used by lmms-eval's process_results (strict instruction-following score).
    # Strategy (c): generators inherit these from the parent / reference item unchanged,
    # only rewriting the natural-language prompt.
    instruction_id_list: list[str] | None = None
    kwargs: list[dict] | None = None


def problem_item_to_json_dict(item: ProblemItem) -> dict[str, Any]:
    """JSON-serializable row: ``id``, ``question``, ``answer``, ``difficulty``, ``modification_policy``, ``parent_id``."""
    out: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answer": item.reference_answer,
        "difficulty": item.difficulty,
        "modification_policy": item.modification_policy,
        "parent_id": item.parent_id,
    }
    if item.options is not None:
        out["options"] = list(item.options)
    if item.instruction_id_list is not None:
        out["instruction_id_list"] = list(item.instruction_id_list)
    if item.kwargs is not None:
        out["kwargs"] = list(item.kwargs)
    return out


@dataclass
class CorrectnessResult:
    ok: bool
    failure_code: FailureCode = "OK"
    evidence: str = ""
    judge_raw: str = ""


@dataclass
class DifficultyResult:
    """Scalar difficulty on the same scale as ``RunConfig.target_difficulty`` (default 0–10)."""

    estimate: float
    target: float
    confidence: Literal["low", "medium", "high"] = "low"
    notes: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class VerifierFeedback:
    validity: bool
    difficulty: DifficultyResult | None
    failure_codes: list[FailureCode] = field(default_factory=list)
    edit_hints: list[str] = field(default_factory=list)

    def to_prompt_bullets(self, max_hints: int = 6) -> str:
        lines: list[str] = []
        if self.failure_codes:
            lines.append("Failures: " + ", ".join(self.failure_codes))
        for h in self.edit_hints[:max_hints]:
            lines.append(f"- {h}")
        if self.difficulty:
            d = self.difficulty
            lines.append(
                f"Difficulty: estimate={d.estimate:.3f} target={d.target:.3f} ({d.confidence})"
            )
            if d.notes:
                lines.append(f"Notes: {d.notes}")
        return "\n".join(lines) if lines else "(no feedback)"


@dataclass
class RunConfig:
    """``target_difficulty`` / ``difficulty_tolerance`` use the **0–10** normalized scale.

    Parent-child tree algorithm parameters:
      - ``k_per_round``: K parents selected each round (after round 0; round 0 uses the anchors).
      - ``n_children``: N child items generated per parent per round.
      - ``memory_size``: M items in the memory list shown to the LLM (M/2 below target, M/2 above).
      - ``parent_selection``: policy for selecting parents; currently only ``"greedy"`` (K nearest to target).
    """

    target_difficulty: float = 5.0
    difficulty_tolerance: float = 0.15
    k_per_round: int = 4          # K: parents selected per round after round 0
    n_children: int = 2            # N: children generated per parent
    memory_size: int = 8           # M: memory list size (M/2 per side of target)
    parent_selection: str = "greedy"
    max_rounds: int = 3
    single_round_mode: bool = False
    artifact_dir: str = "./results"
    seed: int | None = 42
    # Optional: IRT *b* range used to build the 0–10 mapping (for logs / reproducibility).
    irt_b_min: float | None = None
    irt_b_max: float | None = None
    # When set (e.g. by ``run_pipeline`` / ``--generation-only``), generators may write
    # per-task artifacts under ``artifact_dir / run_id_for_artifacts``.
    run_id_for_artifacts: str | None = None
    current_round_index: int = 0
    # When set, the generator emits an MCQ-format prompt. Dict shape:
    #   {"style": "bbh"|"mmlu_pro", "num_options": int}
    # bbh-style bakes Options into the question and answers as "(X)";
    # mmlu_pro-style uses a separate options list and answers as a single letter.
    # None = free-form (math).
    mcq_spec: dict | None = None
    # When True, the generator builds an ifeval-specific prompt asking only for a
    # rewritten ``prompt`` text; the pipeline injects parent's instruction_id_list/kwargs
    # into the produced item (strategy c — fixed instruction structure).
    ifeval_mode: bool = False


@dataclass
class ScoredCandidate:
    candidate: ProblemItem
    correctness: CorrectnessResult
    difficulty: DifficultyResult | None
    accepted: bool


@dataclass
class RoundRecord:
    round_index: int
    feedback_context: str
    scored: list[ScoredCandidate] = field(default_factory=list)


@dataclass
class RunState:
    run_id: str
    config: RunConfig
    anchors: list[ProblemItem]
    rounds: list[RoundRecord] = field(default_factory=list)
    accepted: list[ScoredCandidate] = field(default_factory=list)
