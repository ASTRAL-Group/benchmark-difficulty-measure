import dataclasses
from typing import Dict, Optional, Union

from lmms_eval.tasks.ifeval import instructions_registry
from lmms_eval.utils import eval_logger


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _flatten_responses(results):
    """``results`` may be a list of strings (preferred, --repeats N + take_first_k filter)
    or a list-of-list (one element wrapping the per-sample list). Flatten one level so
    we always iterate over per-sample response strings."""
    flat: list[str] = []
    for r in results:
        if isinstance(r, list):
            flat.extend(s for s in r if isinstance(s, str))
        elif isinstance(r, str):
            flat.append(r)
    return flat


def process_results(doc, results):
    """Override of upstream ifeval process_results to score every sample (not just
    ``results[0]``). With ``--repeats N`` and a filter that keeps all N strings
    (e.g. ``take_first_k k=N``), this returns *list-valued* metrics so the
    downstream IRT loader can compute ``y_m`` from N trials per (model, item)
    instead of collapsing to N=1."""
    eval_logger.warning(
        "This task is meant for chat-finetuned models, and may not give meaningful results for models other than `openai` or `anthropic` if `doc_to_text` in its YAML is not wrapped in the appropriate chat template string. This warning will be removed when chat templating support is added natively to local models"
    )

    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    responses = _flatten_responses(results)
    if not responses:
        responses = [""]

    prompt_strict: list[bool] = []
    prompt_loose: list[bool] = []
    inst_strict: list[list[bool]] = []
    inst_loose: list[list[bool]] = []
    for response in responses:
        out_strict = test_instruction_following_strict(inp, response)
        out_loose = test_instruction_following_loose(inp, response)
        prompt_strict.append(bool(out_strict.follow_all_instructions))
        prompt_loose.append(bool(out_loose.follow_all_instructions))
        inst_strict.append([bool(x) for x in out_strict.follow_instruction_list])
        inst_loose.append([bool(x) for x in out_loose.follow_instruction_list])

    if len(responses) == 1:
        return {
            "prompt_level_strict_acc": prompt_strict[0],
            "inst_level_strict_acc": inst_strict[0],
            "prompt_level_loose_acc": prompt_loose[0],
            "inst_level_loose_acc": inst_loose[0],
        }
    return {
        "prompt_level_strict_acc": prompt_strict,
        "inst_level_strict_acc": inst_strict,
        "prompt_level_loose_acc": prompt_loose,
        "inst_level_loose_acc": inst_loose,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


def _flatten_to_bools(items):
    """Recursively flatten nested lists/bools into a flat list of bool-like values."""
    out = []
    for it in items:
        if isinstance(it, list):
            out.extend(_flatten_to_bools(it))
        else:
            out.append(it)
    return out


def agg_prompt_level_acc_multi(items):
    """Aggregation that tolerates per-doc metric values that are either a single bool
    (legacy / N=1) or a list of bools (our multi-sample override). Returns the mean
    over all individual sample outcomes."""
    flat = _flatten_to_bools(items)
    if not flat:
        return 0.0
    return sum(bool(x) for x in flat) / len(flat)


def agg_inst_level_acc_multi(items):
    """Like ``agg_inst_level_acc`` but tolerates per-doc values that are a list of
    list-of-bools (one inner list per sample) instead of a single list-of-bools."""
    flat = _flatten_to_bools(items)
    if not flat:
        return 0.0
    return sum(bool(x) for x in flat) / len(flat)
