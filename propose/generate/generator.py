"""ItemGenerator protocol and helpers. See INTERFACES.md §2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from utils.schemas import MODIFICATION_POLICY_CANONICAL_VALUES, ProblemItem, RunConfig

_POLICY_ENUM_TEXT = ", ".join(f'"{v}"' for v in sorted(MODIFICATION_POLICY_CANONICAL_VALUES))


def _modification_policy_block(k: int) -> str:
    return f"""
**Modification policy (you choose per item):** For each of the **{k}** new items, decide how heavily you
modify relative to the parent/anchor ideas (do not copy them). Use exactly one of these values for
``modification_policy`` on that object:
- **small_edit** — local refinement: tweak parameters, conditions, or wording slightly.
- **medium_edit** — change one main constraint, or add/remove one conceptual step in the solution path.
- **large_edit** — structural variation: change problem type, combine subtasks, or reframe the core task.

Guidance: items targeting **higher** difficulty than a **lower-scoring** parent often suit **small_edit**
or **medium_edit** moving upward; items targeting **easier** problems than a **harder** parent may use
**medium_edit** or **large_edit** to simplify. Each array element **must** include ``modification_policy``
as one of: {_POLICY_ENUM_TEXT}.
""".strip()


def _diff_line(difficulty: float, target: float) -> str:
    delta = target - difficulty
    direction = "harder" if delta > 0 else "easier" if delta < 0 else "same difficulty"
    return f"Difficulty (0–10): {difficulty:.3f} | delta {delta:+.3f} | generate {direction}"


def _mcq_bbh_block(num_options: int) -> str:
    """Schema for BBH-style MCQs: options baked into the ``question`` string,
    answer wrapped as ``(X)``. Matches what BBH yamls + ``flexible-extract`` filter expect.
    """
    letters = ", ".join(f"({chr(65 + i)})" for i in range(num_options))
    return (
        "**This is a multiple-choice problem.** The new item MUST follow this exact format:\n"
        f"- The ``question`` string must end with ``\\nOptions:\\n(A) ...\\n(B) ...`` for all "
        f"{num_options} options ({letters}). Exactly {num_options} options, no more, no fewer.\n"
        f"- The ``answer`` string must be exactly one letter wrapped in parentheses, one of: {letters}.\n"
        "- Do NOT include the answer letter inside the question or options text.\n"
        "- The correct option must be among the listed choices.\n"
        "- The returned JSON object has these keys: \"id\", \"question\", \"answer\".\n"
    )


def _mcq_mmlu_pro_block(num_options: int) -> str:
    """Schema for MMLU-Pro-style MCQs: ``question`` is the question text only (no options),
    ``options`` is a JSON list of exactly ``num_options`` strings (no letter prefixes), and
    ``answer`` is a single capital letter (e.g. "B") indexing into ``options``.

    This mirrors the original MMLU-Pro source schema so the lmms-eval template's
    ``format_cot_example`` can render the prompt the same way the calibration matrix used.
    """
    letters_with_paren = ", ".join(f"({chr(65 + i)})" for i in range(num_options))
    letters_plain = ", ".join(chr(65 + i) for i in range(num_options))
    return (
        "**This is a multiple-choice problem.** The new item MUST follow this exact format:\n"
        "- The ``question`` field is the question text ONLY — it must NOT contain an "
        "``Options:`` block, letter labels (A./B./..), or the answer.\n"
        f"- The ``options`` field is a JSON list of EXACTLY {num_options} strings, in order "
        f"corresponding to choices {letters_with_paren}. Do NOT prefix letter labels (no "
        '"A." or "(A)") inside the strings — just the option text itself.\n'
        f"- The ``answer`` field is a single capital letter, one of: {letters_plain}.\n"
        "- The correct option must be among the listed choices.\n"
        "- The returned JSON object has these keys: \"id\", \"question\", \"options\", \"answer\".\n"
    )


def _mcq_schema_block(spec: dict | None) -> str | None:
    if not spec:
        return None
    style = spec.get("style", "bbh")
    n = int(spec["num_options"])
    if style == "bbh":
        return _mcq_bbh_block(n)
    if style == "mmlu_pro":
        return _mcq_mmlu_pro_block(n)
    raise ValueError(f"unknown mcq style: {style!r}")


def _ifeval_constraint_description(iid: str, kw: dict) -> str:
    """Resolve an ifeval instruction id + kwargs to its **natural-language** description
    (what ifeval's grader would expect the prompt to ask for). Falls back to a structured
    repr if the registry can't render it.
    """
    clean_kw = {k: v for k, v in (kw or {}).items() if v is not None}
    try:
        from lmms_eval.tasks.ifeval import instructions_registry  # type: ignore[import-untyped]

        cls = instructions_registry.INSTRUCTION_DICT.get(iid)
        if cls is None:
            return f"id={iid!r} params={clean_kw}"
        inst = cls(iid)
        desc = inst.build_description(**clean_kw)
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    except Exception:
        pass
    return f"id={iid!r} params={clean_kw}"


def _ifeval_constraint_block(item: ProblemItem) -> str:
    """Render the program-checked instruction structure of an ifeval item using
    ifeval's own ``build_description`` so the LLM sees the exact wording the
    grader will check against (e.g. *"At the end of your response, please
    explicitly add a postscript starting with P.S."*) rather than opaque ids.
    Strategy (c): the new item inherits ``instruction_id_list`` + ``kwargs`` from
    its donor unchanged; only the natural-language ``prompt`` is rewritten.
    """
    if not item.instruction_id_list:
        return ""
    lines = []
    kwargs_list = item.kwargs or []
    for i, iid in enumerate(item.instruction_id_list):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        desc = _ifeval_constraint_description(iid, kw)
        clean_kw = {k: v for k, v in (kw or {}).items() if v is not None}
        lines.append(f"  {i + 1}. {desc}\n     (grader id: {iid}; params: {clean_kw})")
    return (
        "**Program-checked constraints — your new prompt MUST require ALL of these, "
        "phrased clearly enough that any responder reading the prompt would be told to "
        "do exactly what each line below describes:**\n"
        + "\n".join(lines)
        + "\n"
    )


def build_ifeval_child_prompt(
    parent: ProblemItem,
    *,
    memory: Sequence[ProblemItem],
    target_difficulty: float,
    feedback_context: str,
) -> str:
    """ifeval-specific prompt: rewrite the *natural-language* instruction text only;
    the new item must enforce the **same** program-checked constraints as the parent
    (instruction_id_list + kwargs are inherited unchanged by the pipeline).
    """
    parts: list[str] = [
        "Generate **exactly 1** NEW instruction-following item by rewriting the parent's "
        f"prompt. Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
        "**Hard rule 1:** the program-grader checks the model's output against a fixed list of "
        "constraints. The new prompt MUST require the responder to satisfy ALL of these constraints "
        "verbatim — same parameters, same checks. Do not add, remove, or relax any constraint.\n",
        "**Hard rule 2 (critical):** do NOT introduce ANY constraint in the new prompt beyond the "
        "listed ones. The natural-language instruction must require ONLY the listed constraint(s) "
        "and nothing else — no extra length / format / placeholder / keyword / section / paragraph / "
        "bullet / postscript / language requirements. If the listed constraints don't include it, "
        "your prompt must NOT mention it. Memory items often combine multiple constraints — DO NOT "
        "imitate their full constraint sets; only borrow the topic/style.\n",
    ]

    parent_diff = (
        _diff_line(parent.difficulty, target_difficulty)
        if parent.difficulty is not None
        else "Difficulty: unknown"
    )
    parts.append(
        "**Parent item — gold prompt** (this prompt already correctly enforces every constraint "
        "below and passes the grader). Treat it as the *template*: keep every constraint phrase "
        "(numbers, markers, format names, languages, etc.) and only change the surface topic / "
        "context / examples so your new prompt is not a verbatim copy.\n"
        f"PARENT PROMPT: {parent.question}\n{parent_diff}\n"
    )
    parts.append(_ifeval_constraint_block(parent))

    if memory:
        parts.append("**Memory items** (style/level reference only, do not copy):\n")
        for j, m in enumerate(memory):
            mem_diff = (
                _diff_line(m.difficulty, target_difficulty)
                if m.difficulty is not None
                else "Difficulty: unknown"
            )
            parts.append(f"--- Memory {j + 1} ---\n{m.question}\n{mem_diff}\n")

    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
        "Do not put anything after the closing fence.\n"
        "Do not stop until the JSON is **syntactically complete**: every string value closed with \", "
        "every object with }, the array ends with ], then the closing ``` line.\n"
        "The array must have **exactly 1** element — an object with exactly these keys:\n"
        '"id" (string), "prompt" (string).\n'
        "The ``prompt`` field is the new natural-language instruction; it must implicitly require "
        "every constraint in the list above.\n"
    )
    return "\n".join(parts)


def build_ifeval_scratch_prompt(
    *,
    memory: Sequence[ProblemItem],
    target_difficulty: float,
    feedback_context: str,
    constraint_donor: ProblemItem | None = None,
) -> str:
    """Stage-1 ifeval scratch prompt. ``constraint_donor`` supplies the
    instruction_id_list / kwargs the new prompt must enforce (the pipeline picks
    one and the generated item inherits its constraint structure)."""
    parts: list[str] = [
        "Generate **exactly 1** NEW instruction-following item from scratch. "
        f"Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
        "**Hard rule 1:** the program-grader checks the model's output against a fixed list of "
        "constraints. The new prompt MUST require the responder to satisfy ALL of these constraints "
        "verbatim — same parameters, same checks.\n",
        "**Hard rule 2 (critical):** do NOT introduce ANY constraint in the new prompt beyond the "
        "listed donor constraints. The instruction must require ONLY the donor constraint(s) — no "
        "extra length / format / placeholder / keyword / section / paragraph / bullet / postscript / "
        "language requirements. Reference items often combine multiple constraints — DO NOT imitate "
        "their full constraint sets; only borrow topic/style.\n",
    ]
    if feedback_context.strip():
        parts.append("Constraints from prior attempts:\n" + feedback_context.strip() + "\n")
    if constraint_donor is not None:
        parts.append("**Donor constraints (your new prompt MUST enforce these):**\n")
        parts.append(_ifeval_constraint_block(constraint_donor))
        parts.append(
            "**Donor's gold prompt** — this prompt already correctly expresses the constraints "
            "above and passes the grader. Treat it as the *template*: keep every constraint phrase "
            "(numbers, markers, format names, languages, etc.) and only change the surface "
            "topic / context / examples so the new prompt is not a verbatim copy.\n"
            f"DONOR PROMPT: {constraint_donor.question}\n"
        )
    if memory:
        parts.append("**Reference items** (style/difficulty gauge, do NOT copy):\n")
        for j, m in enumerate(memory):
            d_line = (
                f"Difficulty (0–10): {m.difficulty:.3f}\n" if m.difficulty is not None else ""
            )
            parts.append(f"--- Reference {j + 1} ---\n{m.question}\n{d_line}")
    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
        "Do not put anything after the closing fence.\n"
        "Do not stop until the JSON is **syntactically complete**.\n"
        "The array must have **exactly 1** element — an object with exactly these keys:\n"
        '"id" (string), "prompt" (string).\n'
        "The ``prompt`` field is the new natural-language instruction; it must implicitly require "
        "every constraint in the donor list above.\n"
    )
    return "\n".join(parts)


def _render_item_block(label: str, item: ProblemItem, target_difficulty: float) -> str:
    """Render one parent/memory item for the LLM prompt. If the item has structured
    ``options`` (mmlu_pro source), render them as a labeled Options block so the LLM
    sees the full problem context."""
    diff_line = (
        _diff_line(item.difficulty, target_difficulty)
        if item.difficulty is not None
        else "Difficulty: unknown"
    )
    body = item.question
    if item.options:
        opts_block = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(item.options))
        body = f"{item.question}\nOptions:\n{opts_block}"
    return f"--- {label} ---\n{body}\nAnswer: {item.reference_answer}\n{diff_line}\n"


def build_child_prompt(
    parent: ProblemItem,
    *,
    memory: Sequence[ProblemItem],
    target_difficulty: float,
    feedback_context: str,
    mcq_spec: dict | None = None,
) -> str:
    parts: list[str] = [
        "Generate **exactly 1** NEW benchmark item inspired by the parent item below. "
        f"Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
    ]

    parts.append(
        "**Parent item** (generate a new problem inspired by this one — do not copy its specific "
        "wording, numbers, or answer):\n"
        + _render_item_block("Parent", parent, target_difficulty)
    )

    if memory:
        parts.append("**Memory items** (use as style/level reference only, do not copy):\n")
        for j, m in enumerate(memory):
            parts.append(_render_item_block(f"Memory {j + 1}", m, target_difficulty))

    schema = _mcq_schema_block(mcq_spec)
    if schema is not None:
        parts.append(schema)
        keys_line = (
            '"id" (string), "question" (string), "options" (array of strings), "answer" (string).\n'
            if mcq_spec and mcq_spec.get("style") == "mmlu_pro"
            else '"id" (string), "question" (string), "answer" (string).\n'
        )
    else:
        keys_line = '"id" (string), "question" (string), "answer" (string).\n'

    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
        "Do not put anything after the closing fence.\n"
        "If you reason step-by-step first, you **must still** output the complete ```json block at the end; "
        "reserve enough output for the JSON — incomplete JSON counts as failure.\n"
        "Do not stop until the JSON is **syntactically complete**: every string value closed with \", "
        "every object with }, the array ends with ], then the closing ``` line.\n"
        "The array must have **exactly 1** element — an object with exactly these keys:\n"
        + keys_line
    )
    return "\n".join(parts)


def _render_parent_block_no_diff(label: str, item: ProblemItem) -> str:
    """Parent block for budget-match baseline: question/answer only (no difficulty delta)."""
    body = item.question
    if item.options:
        opts_block = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(item.options))
        body = f"{item.question}\nOptions:\n{opts_block}"
    return f"--- {label} ---\n{body}\nAnswer: {item.reference_answer}\n"


def build_baseline_child_prompt(
    parent: ProblemItem,
    *,
    target_difficulty: float,
    mcq_spec: dict | None = None,
    ifeval_mode: bool = False,
) -> str:
    """Budget-match baseline rewrite prompt (paper).

    Conditioned on the parent item and target difficulty only — no memory / reference
    list, no parent difficulty delta line, no iterative feedback.
    """
    if ifeval_mode:
        parts: list[str] = [
            "Generate **exactly 1** NEW instruction-following item by rewriting the parent's "
            f"prompt. Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
            "**Hard rule 1:** the program-grader checks the model's output against a fixed list of "
            "constraints. The new prompt MUST require the responder to satisfy ALL of these constraints "
            "verbatim — same parameters, same checks. Do not add, remove, or relax any constraint.\n",
            "**Hard rule 2 (critical):** do NOT introduce ANY constraint in the new prompt beyond the "
            "listed ones. The natural-language instruction must require ONLY the listed constraint(s) "
            "and nothing else.\n",
            "**Parent item — gold prompt** (rewrite topic/context only; keep every constraint phrase):\n"
            f"PARENT PROMPT: {parent.question}\n",
            _ifeval_constraint_block(parent),
            "Return **only** one markdown code block: start the first line with ```json then a newline, "
            "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
            "Do not put anything after the closing fence.\n"
            "The array must have **exactly 1** element — an object with exactly these keys:\n"
            '"id" (string), "prompt" (string).\n',
        ]
        return "\n".join(parts)

    parts = [
        "Generate **exactly 1** NEW benchmark item inspired by the parent item below. "
        f"Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
        "**Parent item** (generate a new problem inspired by this one — do not copy its specific "
        "wording, numbers, or answer):\n"
        + _render_parent_block_no_diff("Parent", parent),
    ]

    # mmlu_pro baseline: rewrite question stem only; options/answer inherited from parent
    # by the verifier converter (same as historical random_baseline tooling).
    if mcq_spec and mcq_spec.get("style") == "mmlu_pro":
        ans = str(parent.reference_answer or "").strip()
        parts.append(
            "**Hard rule:** rewrite the **question text only**. Keep the same option list and the "
            f"same correct answer letter (**{ans}**). Do not invent new options.\n"
            "The ``question`` field must NOT contain an Options block or letter labels.\n"
        )
        keys_line = '"id" (string), "question" (string), "answer" (string).\n'
    else:
        schema = _mcq_schema_block(mcq_spec)
        if schema is not None:
            parts.append(schema)
            keys_line = '"id" (string), "question" (string), "answer" (string).\n'
        else:
            keys_line = '"id" (string), "question" (string), "answer" (string).\n'

    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
        "Do not put anything after the closing fence.\n"
        "If you reason step-by-step first, you **must still** output the complete ```json block at the end; "
        "reserve enough output for the JSON — incomplete JSON counts as failure.\n"
        "Do not stop until the JSON is **syntactically complete**: every string value closed with \", "
        "every object with }, the array ends with ], then the closing ``` line.\n"
        "The array must have **exactly 1** element — an object with exactly these keys:\n"
        + keys_line
    )
    return "\n".join(parts)


def build_scratch_prompt(
    *,
    memory: Sequence[ProblemItem],
    target_difficulty: float,
    feedback_context: str,
    mcq_spec: dict | None = None,
) -> str:
    """Prompt for generating ONE item **from scratch** (no parent to modify).

    The ``memory`` items are reference problems from the original benchmark that are
    closest to the target difficulty — use as style/difficulty gauge only. The model
    must create a genuinely new problem, not paraphrase or modify the references.
    """
    parts: list[str] = [
        "Generate **exactly 1** NEW benchmark item **from scratch**. "
        "Do **NOT** copy, paraphrase, rewrite, or translate any reference problem below — "
        "invent a genuinely new problem with a different setup, context, and numbers. "
        f"Target difficulty (0–10 scale, higher = harder): **{target_difficulty:.3f}**.\n",
    ]
    if feedback_context.strip():
        parts.append("Constraints from prior attempts:\n" + feedback_context.strip() + "\n")

    if memory:
        parts.append(
            "**Reference problems** from the original benchmark, sorted by closeness to the target "
            "difficulty (use them only to gauge the expected style and difficulty level; "
            "do NOT copy wording, numbers, structure, or answers):\n"
        )
        for j, m in enumerate(memory):
            parts.append(_render_item_block(f"Reference {j + 1}", m, target_difficulty))

    schema = _mcq_schema_block(mcq_spec)
    if schema is not None:
        parts.append(schema)
        keys_line = (
            '"id" (string), "question" (string), "options" (array of strings), "answer" (string).\n'
            if mcq_spec and mcq_spec.get("style") == "mmlu_pro"
            else '"id" (string), "question" (string), "answer" (string).\n'
        )
    else:
        keys_line = '"id" (string), "question" (string), "answer" (string).\n'

    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array with **exactly 1** element, then close with ``` on its own line. "
        "Do not put anything after the closing fence.\n"
        "If you reason step-by-step first, you **must still** output the complete ```json block at the end; "
        "reserve enough output for the JSON — incomplete JSON counts as failure.\n"
        "Do not stop until the JSON is **syntactically complete**: every string value closed with \", "
        "every object with }, the array ends with ], then the closing ``` line.\n"
        "The array must have **exactly 1** element — an object with exactly these keys:\n"
        + keys_line
    )
    return "\n".join(parts)

def build_prompt(
    anchors: Sequence[ProblemItem],
    *,
    target_difficulty: float,
    feedback_context: str,
    k: int,
) -> str:
    """Legacy batch prompt: model returns a JSON array of ``k`` objects.

    Kept for inspection / debugging. The active pipeline uses ``build_child_prompt``
    (one parent → one child per API call).
    """
    if k < 1:
        raise ValueError("build_prompt requires k >= 1")
    parts: list[str] = [
        f"Generate **exactly {k}** NEW benchmark items (not {k - 1}, not {k + 1} — the JSON array must have "
        f"**{k}** elements). Each item must include: id, question, answer, and modification_policy.\n"
        f"Target difficulty for the new items (0–10 scale, higher = harder): {target_difficulty:.3f}. "
        "Use this as a style/level guide only; do not put a difficulty field in the JSON.\n",
    ]
    if feedback_context.strip():
        parts.append("Constraints from prior rounds:\n" + feedback_context.strip() + "\n")
    if anchors:
        parts.append(
            "Below are anchor problems you can refer to for style, format, and approximate difficulty level "
            "(each includes a difficulty hint on the 0–10 scale when known). Do not copy them or reuse their specific wording or ideas.\n"
        )
    for j, a in enumerate(anchors):
        diff_line = (
            f"Difficulty (0–10): {a.difficulty:.3f}\n" if a.difficulty is not None else "Difficulty: unknown\n"
        )
        parts.append(
            f"--- Anchor {j + 1} ---\n{a.question}\nAnswer: {a.reference_answer}\n{diff_line}"
        )
    parts.append(_modification_policy_block(k))
    parts.append(
        "Return **only** one markdown code block: start the first line with ```json then a newline, "
        "then a single JSON array, then close with ``` on its own line. Do not put anything after the "
        "closing fence.\n"
        "Do not stop until the JSON is **syntactically complete**: every string value closed with \", "
        "every object with }, the array ends with ], then the closing ``` line.\n"
        f"The array length must be **exactly {k}**.\n"
        "Each array element must be an object with exactly these keys:\n"
        '"id" (string), "question" (string), "answer" (string), "modification_policy" (string).\n'
    )
    return "\n".join(parts)


class ItemGenerator(ABC):
    """Generates one new ProblemItem from a parent item and memory context."""

    @abstractmethod
    def generate_one(
        self,
        parent: ProblemItem,
        *,
        memory: Sequence[ProblemItem],
        target_difficulty: float,
        feedback_context: str,
        config: RunConfig,
        artifact_tag: str = "",
        constraint_donor: ProblemItem | None = None,
    ) -> ProblemItem:
        """Return one new item derived from ``parent``.

        ``target_difficulty`` is on the project **0–10** scale.
        ``memory`` contains reference items (up to M/2 below and M/2 above target difficulty).
        ``artifact_tag`` is an optional string appended to artifact filenames to distinguish
        parallel calls (e.g. ``"parent_q3_child_0"``).
        ``constraint_donor`` (ifeval only): see VllmOpenAICompatGenerator.generate_one.
        """


class PromptLoggingGenerator(ItemGenerator):
    """Delegates to inner; records last_prompt for debugging."""

    def __init__(self, inner: ItemGenerator) -> None:
        self.inner = inner
        self.last_prompt: str = ""

    def generate_one(
        self,
        parent: ProblemItem,
        *,
        memory: Sequence[ProblemItem],
        target_difficulty: float,
        feedback_context: str,
        config: RunConfig,
        artifact_tag: str = "",
        constraint_donor: ProblemItem | None = None,
    ) -> ProblemItem:
        self.last_prompt = build_child_prompt(
            parent,
            memory=memory,
            target_difficulty=target_difficulty,
            feedback_context=feedback_context,
        )
        return self.inner.generate_one(
            parent,
            memory=memory,
            target_difficulty=target_difficulty,
            feedback_context=feedback_context,
            config=config,
            artifact_tag=artifact_tag,
            constraint_donor=constraint_donor,
        )
