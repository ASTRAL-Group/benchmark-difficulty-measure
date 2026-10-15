"""Parse model output into ProblemItem list."""

from __future__ import annotations

import json
import re
from typing import Any

from utils.schemas import ProblemItem


def _double_invalid_json_string_escapes(s: str) -> str:
    """Inside JSON double-quoted strings, turn invalid ``\\c`` into ``\\\\c``.

    LLMs often emit LaTeX like ``\\le`` using a **single** backslash. In JSON only
    ``\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX`` are valid; ``\\l`` raises
    ``Invalid \\escape``. Doubling the stray backslash yields valid JSON that still
    decodes to the intended ``\\le`` in the problem string.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    in_string = False

    while i < n:
        c = s[i]
        if c == '"':
            bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                in_string = not in_string
            out.append(c)
            i += 1
            continue

        if in_string and c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in '"\\/bfnrt':
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < n:
                hx = s[i + 2 : i + 6]
                if len(hx) == 4 and all(x in "0123456789abcdefABCDEF" for x in hx):
                    out.append(s[i : i + 6])
                    i += 6
                    continue
            out.append("\\\\")
            i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


def _json_loads_lenient(s: str) -> Any:
    """``json.loads``, then fix invalid LaTeX backslashes in strings, then ``json_repair``."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    fixed = _double_invalid_json_string_escapes(s)
    if fixed != s:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    try:
        import json_repair  # type: ignore[import-untyped]
    except ImportError as e:
        raise ValueError(
            "model JSON is invalid for stdlib json.loads (after LaTeX backslash fix); "
            "`pip install json-repair` (listed in requirements_diff_qwen.txt) for heavier repairs."
        ) from e
    jr_err: Exception | None = None
    try:
        return json_repair.loads(s)
    except Exception as e:
        jr_err = e
    if fixed != s:
        try:
            return json_repair.loads(fixed)
        except Exception as e2:
            raise ValueError(f"json_repair could not parse model JSON: {e2}") from e2
    raise ValueError(f"json_repair could not parse model JSON: {jr_err}") from jr_err


def _problem_dict_list(data: list[Any]) -> list[dict[str, Any]] | None:
    """Keep only dict elements; None if there are no dicts (e.g. parsed ``[1,2,3]`` by mistake)."""
    out = [x for x in data if isinstance(x, dict)]
    return out if out else None


# parse the model output into a list of ProblemItem objects
def _strip_thinking_wrappers(text: str) -> str:
    """Remove Qwen3 / common reasoning blocks so JSON can be found."""
    t = text
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<redacted_reasoning>[\s\S]*?</redacted_reasoning>", "", t, flags=re.IGNORECASE)
    return t.strip()


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
        t = t.strip()
    return t


def _balanced_array_from(s: str, start: int) -> str | None:
    """Balanced ``[...]`` substring starting at ``start`` (respects quotes), or None."""
    if start < 0 or start >= len(s) or s[start] != "[":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _object_array_bracket_starts(s: str) -> list[int]:
    """Indices of ``[`` that begin ``[ { ...`` (JSON array of objects), in document order."""
    out: list[int] = []
    i = 0
    while True:
        j = s.find("[", i)
        if j < 0:
            break
        k = j + 1
        while k < len(s) and s[k] in " \t\n\r":
            k += 1
        if k < len(s) and s[k] == "{":
            out.append(j)
        i = j + 1
    return out


def _load_json_array_loose(s: str) -> list[Any] | None:
    """Return a parsed JSON list of objects (dict rows), or None."""

    def _try_chunk(chunk: str) -> list[Any] | None:
        chunk = chunk.strip()
        if not chunk:
            return None
        try:
            data = _json_loads_lenient(chunk)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        if not isinstance(data, list):
            return None
        rows = _problem_dict_list(data)
        return rows if rows is not None else None

    s = (s or "").strip()
    if not s:
        return None
    s = _strip_markdown_fence(s)
    if not s:
        return None
    got = _try_chunk(s)
    if got is not None:
        return got
    # Prefer ``[{`` arrays (avoids grabbing an earlier numeric ``[a,b]`` slice).
    for st in reversed(_object_array_bracket_starts(s)):
        chunk = _balanced_array_from(s, st)
        if not chunk:
            continue
        got = _try_chunk(chunk)
        if got is not None:
            return got
    st = s.find("[")
    if st >= 0:
        chunk = _balanced_array_from(s, st)
        if chunk:
            got = _try_chunk(chunk)
            if got is not None:
                return got
    return None


_JSON_FENCE_RE = re.compile(r"```\s*json\s*\n?([\s\S]*?)```", re.IGNORECASE)


def _greedy_json_fence_inner(raw: str) -> str | None:
    """Content between first `` ```json `` and the **last** `` ``` `` in ``raw``.

    Non-greedy fence regex can truncate early if a JSON string accidentally contains a
    `` ``` `` substring; taking text up to the final fence often recovers the full array.
    """
    m = re.search(r"```\s*json\s*\n?", raw, flags=re.IGNORECASE)
    if not m:
        return None
    after = raw[m.end() :]
    if "```" not in after:
        return None
    inner = after.rsplit("```", 1)[0].strip()
    return inner or None


def _parse_json_array_from_text(raw: str) -> Any:
    """Parse a JSON array from model text.

    Prefer `` ```json ... ``` `` fences; fall back to the last balanced ``[{...}]``
    in the text (after stripping reasoning wrappers) for models that omit fences.
    """
    if not (raw or "").strip():
        raise ValueError(
            "model returned empty assistant text; if using vLLM + Qwen3 reasoning, "
            "ensure the client merges `message.content` and `message.reasoning_content`."
        )
    stripped = _strip_thinking_wrappers(raw)

    matches = list(_JSON_FENCE_RE.finditer(stripped))
    if matches:
        seen: set[str] = set()
        for m in reversed(matches):
            inner = m.group(1).strip()
            if not inner or inner in seen:
                continue
            seen.add(inner)
            got = _load_json_array_loose(inner)
            if got is not None:
                return got
        greedy = _greedy_json_fence_inner(stripped)
        if greedy and greedy not in seen:
            got = _load_json_array_loose(greedy)
            if got is not None:
                return got
        raise ValueError(
            "found ```json fence(s) but could not parse a JSON array of dict objects inside. "
            "Common causes: (1) the model stopped early — JSON is truncated (unclosed string or missing ]); "
            "(2) invalid JSON escapes in LaTeX (built-in fix handles common ``\\\\le``-style cases; else `json-repair`); "
            "(3) a ``` substring inside a question confused the fence — retry or shorten wording. "
            f"(assistant len={len(raw)}, first 500 chars): {raw[:500]!r}"
        )

    got = _load_json_array_loose(stripped)
    if got is not None:
        return got
    raise ValueError(
        "no parseable JSON array of objects in model output (no ```json fence and no "
        f"balanced `[{{...}}]` array). (first 500 chars): {raw[:500]!r}"
    )


def parse_single_item_from_model_text(text: str) -> ProblemItem:
    """Parse exactly one ProblemItem from model text (expects a 1-element JSON array).

    Used by ``generate_one`` in the parent-child pipeline. Returns the first parsed item;
    raises ``ValueError`` if none are found.
    """
    items = parse_problem_items_from_model_text(text)
    if not items:
        raise ValueError(
            "model returned an empty array; expected exactly 1 item. "
            f"(first 500 chars): {text[:500]!r}"
        )
    return items[0]


# parse the model output into a list of ProblemItem objects
def parse_problem_items_from_model_text(text: str) -> list[ProblemItem]:
    """Expect a JSON array of objects with ``id``, ``question``, ``answer`` (or legacy ``reference_answer``).

    ``modification_policy`` and ``difficulty`` are optional; both default to ``None`` if absent.
    """
    data: Any = _parse_json_array_from_text(text)
    if not isinstance(data, list):
        raise ValueError("model output must be a JSON array")
    if _problem_dict_list(data) is None:
        raise ValueError("model JSON array contains no objects (dict rows)")
    out: list[ProblemItem] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        pid = str(obj.get("id") if obj.get("id") is not None else f"gen_{i}")
        # ifeval generators emit "prompt" instead of "question" since the lmms-eval
        # process_results function reads doc["prompt"]. Accept either spelling.
        q = str(obj.get("question") or obj.get("problem") or obj.get("prompt") or "")
        ra = str(obj.get("answer") or obj.get("reference_answer") or "")
        raw_d = obj.get("difficulty")
        diff = float(raw_d) if isinstance(raw_d, (int, float)) else None
        raw_mp = obj.get("modification_policy")
        mp = str(raw_mp).strip() if raw_mp is not None and str(raw_mp).strip() else None
        raw_opts = obj.get("options")
        opts: list[str] | None = None
        if isinstance(raw_opts, list) and raw_opts:
            opts = [str(o) for o in raw_opts]
        raw_il = obj.get("instruction_id_list")
        il: list[str] | None = [str(x) for x in raw_il] if isinstance(raw_il, list) and raw_il else None
        raw_kw = obj.get("kwargs")
        kw: list[dict] | None = [dict(x) if isinstance(x, dict) else {} for x in raw_kw] if isinstance(raw_kw, list) and raw_kw else None
        out.append(
            ProblemItem(
                id=pid,
                question=q,
                reference_answer=ra,
                difficulty=diff,
                modification_policy=mp,
                options=opts,
                instruction_id_list=il,
                kwargs=kw,
            )
        )
    return out
