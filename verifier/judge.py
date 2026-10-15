"""Answer extraction and judging.

For math benchmarks (AIME, MATH, etc.):
  Uses parser.extract_answer + grader.math_equal (vendored under verifier/math_eval/).

For MC benchmarks (GPQA, MMLU-Pro, MedQA, BBH, ScienceQA):
  Uses regex patterns from logs/extract_answers.py.

For free-response (IFEval, ZebraLogic):
  Exact string match.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

JudgeType = Literal["numeric", "letter", "medqa", "string"]

# ---------------------------------------------------------------------------
# Import math eval utilities (parser + grader) from the copied code
# ---------------------------------------------------------------------------

from .math_eval.parser import extract_answer, strip_string  # noqa: E402
from .math_eval.grader import math_equal  # noqa: E402


# ---------------------------------------------------------------------------
# Math answer extraction and grading (AIME, MATH, etc.)
# ---------------------------------------------------------------------------

def extract_and_judge_math(
    response_text: str,
    reference_answer: str,
    data_name: str = "aime24",
) -> bool:
    """Extract answer from response and compare with reference using math_equal.

    This is the same pipeline used in verifier/diff_eval/evaluate.py.
    """
    pred = extract_answer(response_text, data_name)
    gt = strip_string(reference_answer)
    if not pred:
        return False
    try:
        return math_equal(pred, gt, timeout=True)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Letter answer extraction (MC: A-J for MMLU-Pro 10-choice)
# ---------------------------------------------------------------------------

def extract_letter_answer(resp: str) -> str | None:
    """Extract letter answer (A-J) from a model response."""
    if isinstance(resp, list):
        resp = resp[0] if resp else ""

    # \\boxed{\\text{X}} or \\boxed{X}
    m = re.search(r"\\boxed\{\\text\{([A-Ja-j])\}\}", resp)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-Ja-j])\}", resp)
    if m:
        return m.group(1).upper()

    # "answer is (X)"
    m = re.search(r"[Aa]nswer is \(([A-Ja-j])\)", resp)
    if m:
        return m.group(1).upper()

    # "answer: X" or "answer X" at end
    m = re.search(r"[Aa]nswer[:\s]+\(?([A-Ja-j])\)?[\s.]*$", resp.strip())
    if m:
        return m.group(1).upper()

    # "option (X)" — last occurrence
    matches = re.findall(r"option \(([A-Ja-j])\)", resp)
    if matches:
        return matches[-1].upper()

    # "the answer is X"
    m = re.search(r"[Tt]he answer is ([A-Ja-j])[\.\s]", resp)
    if m:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# MedQA answer extraction
# ---------------------------------------------------------------------------

def extract_medqa_answer(resp: str) -> str | None:
    """Extract letter answer from a MedQA model response."""
    if isinstance(resp, list):
        resp = resp[0] if resp else ""

    m = re.search(r"Answer:\s*([A-Ea-e])\b", resp)
    if m:
        return m.group(1).upper()

    m = re.search(r"答案[：:]\s*([A-Ea-e])", resp)
    if m:
        return m.group(1).upper()

    m = re.search(r"\\boxed\{\\text\{([A-Ea-e])\}\}", resp)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-Ea-e])\}", resp)
    if m:
        return m.group(1).upper()

    m = re.search(r"(?:best|my|the) answer (?:is|should be)\s+([A-Ea-e])[\.\s\n]", resp)
    if m:
        return m.group(1).upper()

    m = re.search(r"\n([A-Ea-e])[\.\s]*$", resp.strip())
    if m:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Unified judge
# ---------------------------------------------------------------------------

def judge_answer(
    response_text: str,
    reference_answer: str,
    judge_type: JudgeType,
    data_name: str = "aime24",
) -> bool:
    """Score a single response against the reference answer.

    Parameters
    ----------
    response_text : raw model output
    reference_answer : gold answer string
    judge_type : "numeric" | "letter" | "medqa" | "string"
    data_name : dataset name, used by math eval parser (e.g. "aime24", "math")

    Returns True if correct.
    """
    if judge_type == "numeric":
        return extract_and_judge_math(response_text, reference_answer, data_name)

    if judge_type == "letter":
        extracted = extract_letter_answer(response_text)
        if extracted is None:
            return False
        ref = reference_answer.strip().upper()
        m = re.search(r"\(?([A-Ja-j])\)?", ref)
        if m:
            ref = m.group(1).upper()
        return extracted == ref

    if judge_type == "medqa":
        extracted = extract_medqa_answer(response_text)
        if extracted is None:
            return False
        return extracted == reference_answer.strip().upper()

    if judge_type == "string":
        return response_text.strip() == reference_answer.strip()

    raise ValueError(f"Unknown judge_type: {judge_type}")
