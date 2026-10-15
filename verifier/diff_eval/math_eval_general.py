"""
General math evaluation script for any chat/completion model.
Supports multi-GPU runs via CUDA_VISIBLE_DEVICES and --tensor_parallel_size / --pipeline_parallel_size (e.g. 8 GPUs with TP=4 PP=2).
Use script args (prompt_type, n_sampling, etc.) to get different behaviors; no per-model Python needed.
"""
import random
import os
import sys
import argparse
import time
import json

# enviroment variables to disable torch.compile to avoid conflict. no influence on accuracy
# os.environ["TORCH_COMPILE_DEBUG"] = "0"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
import torch
try:
    torch._dynamo.config.suppress_errors = True
    # torch._dynamo.config.disable = True
except Exception:
    pass
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt, get_n_sampling_for_data, get_data_list
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser(description="General math eval: any model; set CUDA_VISIBLE_DEVICES and TP/PP to match GPU count.")
    parser.add_argument("--data_names", default="gsm8k,math", type=str, help="Comma-separated dataset names to run (e.g. aime24,math500). Ignored if --all is set.")
    parser.add_argument("--all", action="store_true", help="Run all datasets (utils.ALL_DATA_NAMES).")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str, help="Prompt template (e.g. qwen25-math-cot, qwen-math-cot-standard, olmo-math-cot).")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--max_tokens_per_call", default=10240, type=int)
    parser.add_argument("--max_model_len", default=85000, type=int, help="vLLM max model length (context).")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Override TP size; default = len(CUDA_VISIBLE_DEVICES)//pipeline_parallel_size.")
    parser.add_argument("--adapt_few_shot", action="store_true")
    parser.add_argument("--n_sampling", type=int, default=None, help="Override n_sampling for all datasets (default: use utils.get_n_sampling_for_data per dataset). If set, output filename includes _n{N} so 1 vs 8 do not overwrite.")
    parser.add_argument("--quantization", type=str, default=None, help="vLLM quantization (e.g. fp8, modelopt_fp4 for nvidia/Qwen3.5-397B-A17B-NVFP4).")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        help="vLLM attention backend (e.g. TRITON_ATTN for Qwen3.5 on GPUs without nvcc/FLASHINFER JIT).",
    )
    parser.add_argument(
        "--moe_backend",
        type=str,
        default=None,
        help="vLLM --moe-backend (e.g. marlin for NVFP4 MoE on Blackwell).",
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action="store_true",
        help="vLLM --enable-expert-parallel (spread MoE experts across GPUs; saves VRAM on large MoE).",
    )
    parser.add_argument(
        "--no_enforce_eager",
        action="store_true",
        help="Set enforce_eager=False for vLLM (e.g. Marlin MoE + CUDA graphs).",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Optional vLLM max_num_seqs (scheduler concurrency). Omit to use vLLM default.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vLLM prefix caching (default: ON). Dedupes n_sampling identical prompts and multi-epoch growing prefix re-prefill. Disable via --no-enable_prefix_caching if hybrid-attention models hit cache corruption.",
    )
    parser.add_argument(
        "--inference_chunk_size",
        type=int,
        default=0,
        help="If >0 and --use_vllm, run llm.generate in chunks of this many prompts (clearer tqdm; avoids one huge 960-seq batch). 0 = single batch (legacy).",
    )
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    out_file_prefix = f"{args.prompt_type}_seed{args.seed}_t{args.temperature}_{args.max_tokens_per_call}_n{args.n_sampling}"
    output_dir = os.path.abspath(args.output_dir)
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))
    processed_samples = {s["idx"]: s for s in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [e for e in examples if e["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def _vllm_generate_chunked(llm, prompts: list, sampling_params, chunk_size: int, desc: str) -> list:
    """Run vLLM generate; optionally chunk so tqdm advances between chunks (logs stay readable)."""
    if not prompts:
        return []
    if chunk_size <= 0 or chunk_size >= len(prompts):
        outputs = llm.generate(prompts, sampling_params)
        return [x.outputs[0].text for x in sorted(outputs, key=lambda x: int(x.request_id))]
    texts = []
    n = len(prompts)
    starts = list(range(0, n, chunk_size))
    bar = tqdm(starts, desc=desc, file=sys.stderr, mininterval=1.0, dynamic_ncols=True)
    for start in bar:
        chunk = prompts[start : start + chunk_size]
        sub = llm.generate(chunk, sampling_params)
        sub = sorted(sub, key=lambda x: int(x.request_id))
        texts.extend([x.outputs[0].text for x in sub])
        bar.set_postfix_str(f"{min(start + len(chunk), n)}/{n} seqs", refresh=False)
    return texts


def _stop_token_ids_for_model(model_path: str):
    """Return stop token ids for Qwen/QwQ/DeepSeek-R1-Distill-Qwen family; else None.
    IDs 151645=<|im_end|>, 151643=<|endoftext|> are shared by Qwen2/Qwen2.5/Qwen3 tokenizers (see HF tokenizer_config.json).
    Non-Qwen models (Llama, OLMo, GLM, etc.) get None and rely on stop_words only."""
    p = (model_path or "").lower()
    if "qwen2" in p or "qwen3" in p or "qwq" in p or "deepseek-r1-distill-qwen" in p:
        return [151645, 151643]
    return None


def setup(args):
    available_gpus = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    tp_size = args.tensor_parallel_size
    if tp_size is None:
        tp_size = max(1, len(available_gpus) // args.pipeline_parallel_size)

    if args.use_vllm:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.reset()
        # Default True: fewer CUDA-graph edge cases; use --no_enforce_eager when you need eager=False.
        enforce_eager = True
        if args.no_enforce_eager:
            enforce_eager = False
        llm_kwargs = dict(
            model=args.model_name_or_path,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            enforce_eager=enforce_eager,
        )
        if args.quantization is not None:
            llm_kwargs["quantization"] = args.quantization
        if args.attention_backend is not None:
            llm_kwargs["attention_config"] = {"backend": args.attention_backend}
        if args.moe_backend is not None:
            llm_kwargs["moe_backend"] = args.moe_backend
        if args.enable_expert_parallel:
            llm_kwargs["enable_expert_parallel"] = True
        if args.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(args.max_num_seqs)
        if args.enable_prefix_caching:
            llm_kwargs["enable_prefix_caching"] = True
        llm = LLM(**llm_kwargs)
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    data_list = get_data_list(args.all, args.data_names)
    results = []
    n_sampling_override = getattr(args, "n_sampling", None)
    for data_name in data_list:
        args.n_sampling = n_sampling_override if n_sampling_override is not None else get_n_sampling_for_data(data_name)
        results.append(main(llm, tokenizer, data_name, args))

    data_list.append("avg")
    results.append({"acc": sum(r["acc"] for r in results) / len(results)})
    pad = max(len(d) for d in data_list)
    print("\t".join(d.ljust(pad) for d in data_list))
    print("\t".join([f"{r['acc']:.1f}".ljust(pad) for r in results]))


def is_multi_choice(answer):
    if answer is None:
        return False
    return all(c in ["A", "B", "C", "D", "E"] for c in answer)


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    run_tag = os.environ.get("MATH_EVAL_RUN_TAG", "").strip()
    print("=" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if run_tag:
        print(f"MATH_EVAL_RUN_TAG={run_tag}", flush=True)
    if examples:
        print(examples[0])

    executor = PythonExecutor(get_answer_expr="solution()") if "pal" in args.prompt_type else PythonExecutor(get_answer_from_stdout=True)
    samples = []
    prep_desc = f"prep[{run_tag}]" if run_tag else "prep"
    for example in tqdm(examples, total=len(examples), desc=prep_desc, file=sys.stderr, mininterval=0.5, dynamic_ncols=True):
        idx = example["idx"]
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        if idx == args.start:
            print(full_prompt)
        sample = {"idx": idx, "question": example["question"], "gt_cot": gt_cot, "gt": gt_ans, "prompt": full_prompt}
        for key in ["level", "type", "unit", "solution_type", "choices", "solution", "ques_type", "ans_type", "answer_type", "dataset", "subfield", "filed", "theorem", "answer"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    input_prompts = [s["prompt"] for s in samples for _ in range(args.n_sampling)]
    print(
        f"[math_eval] model={args.model_name_or_path} | items={len(samples)} | n_sampling={args.n_sampling} "
        f"=> {len(input_prompts)} generation requests | max_tokens={args.max_tokens_per_call} | "
        f"inference_chunk_size={getattr(args, 'inference_chunk_size', 0) or 'ALL'}",
        flush=True,
    )
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p.strip()}], tokenize=False, add_generation_prompt=True)
            for p in input_prompts
        ]
    remain_prompts = [(i, p) for i, p in enumerate(input_prompts)]
    end_prompts = []
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    stop_token_ids = _stop_token_ids_for_model(args.model_name_or_path)
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if not current_prompts:
            break
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            sp = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
            )
            gen_desc = f"vLLM[{run_tag}]e{epoch}" if run_tag else f"vLLM e{epoch}"
            outputs = _vllm_generate_chunked(
                llm, prompts, sp, getattr(args, "inference_chunk_size", 0) or 0, desc=gen_desc
            )
        else:
            outputs = generate_completions(llm, tokenizer, prompts, max_new_tokens=args.max_tokens_per_call, batch_size=16, stop_id_sequences=stop_words)
        assert len(outputs) == len(current_prompts)

        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                remain_codes.append(extract_program(query) if "```python" in output else output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                remain_prompts.append((i, query))
                remain_codes.append(extract_program(query))
            else:
                end_prompts.append((i, query))

        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    end_prompts.sort(key=lambda x: x[0])
    codes = []
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for w in stop_words:
            if w in code:
                code = code.split(w)[0].strip()
        codes.append(code)

    score_desc = f"score[{run_tag}]" if run_tag else "score"
    results = [
        run_execute(executor, code, args.prompt_type, data_name)
        for code in tqdm(codes, desc=score_desc, file=sys.stderr, mininterval=1.0, dynamic_ncols=True)
    ]
    time_use = time.time() - start_time

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                preds[j] = choice_answer_clean(code[j]) if code[j] else ""
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(c for c in (preds[j] or "") if c in ["A", "B", "C", "D", "E"])
        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True)

    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    with open(out_file.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
