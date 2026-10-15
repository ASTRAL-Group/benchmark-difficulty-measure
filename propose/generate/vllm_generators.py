"""vLLM-backed ItemGenerator: OpenAI-compatible HTTP server, or in-process LLM."""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from generate.generator import (
    ItemGenerator,
    build_child_prompt,
    build_ifeval_child_prompt,
    build_ifeval_scratch_prompt,
    build_scratch_prompt,
)
from generate.json_items import parse_single_item_from_model_text
from utils.schemas import ProblemItem, RunConfig

# Chat Completions `max_tokens` is **completion-only**; vLLM still enforces
# prompt_tokens + max_tokens <= max_model_len. Defaults like 32768 break when
# max_model_len is also 32768 (zero room for the user prompt).
DEFAULT_MAX_COMPLETION_TOKENS = 30000  # thinking chains can be long; leaves ~2768 tokens headroom for prompts

# MiniMax-M2.7 card: https://huggingface.co/MiniMaxAI/MiniMax-M2.7
MINIMAX_M2_7_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
)
MINIMAX_M2_7_TEMPERATURE = 1.0
MINIMAX_M2_7_TOP_P = 0.95
MINIMAX_M2_7_TOP_K = 40


def _save_debug_model_io(
    config: RunConfig,
    *,
    prompt: str,
    response: str,
    source: str,
    artifact_tag: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write last prompt / raw response / meta under ``artifact_dir`` for offline inspection."""
    base = Path(config.artifact_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = f"debug_last_model_{artifact_tag}" if artifact_tag else "debug_last_model"
    prompt_path = base / f"{stem}_prompt.txt"
    resp_path = base / f"{stem}_response.txt"
    meta_path = base / f"{stem}_meta.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    resp_path.write_text(response, encoding="utf-8")
    meta: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "artifact_tag": artifact_tag,
        "response_chars": len(response),
        "prompt_chars": len(prompt),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return prompt_path, resp_path, meta_path


def _full_response_save_disabled() -> bool:
    return os.environ.get("PROPOSE_NO_FULL_RESPONSE_SAVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _task_artifact_paths(
    config: RunConfig, artifact_tag: str
) -> tuple[Path, Path, Path] | None:
    """Return (prompt_path, response_path, meta_path) or None if artifacts are disabled."""
    if _full_response_save_disabled():
        return None
    rid = getattr(config, "run_id_for_artifacts", None)
    if not rid or not str(rid).strip():
        return None
    r = int(getattr(config, "current_round_index", 0))
    run_dir = Path(config.artifact_dir).resolve() / str(rid).strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model_round_{r}_{artifact_tag}" if artifact_tag else f"model_round_{r}"
    return (
        run_dir / f"{stem}_prompt.txt",
        run_dir / f"{stem}_response.txt",
        run_dir / f"{stem}_meta.json",
    )


def _save_prompt(config: RunConfig, *, prompt: str, artifact_tag: str) -> None:
    """Write the prompt immediately before the API call so it's visible while the task runs."""
    paths = _task_artifact_paths(config, artifact_tag)
    if paths is None:
        return
    prompt_path, _, meta_path = paths
    prompt_path.write_text(prompt, encoding="utf-8")
    # Write a stub meta so the file exists and shows the task is in-flight.
    meta_path.write_text(
        json.dumps({"status": "in_flight", "artifact_tag": artifact_tag}, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_response(config: RunConfig, *, text: str, artifact_tag: str) -> None:
    """Write the raw model response immediately after receipt, before parsing."""
    paths = _task_artifact_paths(config, artifact_tag)
    if paths is None:
        return
    _, resp_path, _ = paths
    resp_path.write_text(text, encoding="utf-8")


def _save_meta(
    config: RunConfig,
    *,
    prompt: str,
    text: str,
    content: str,
    reasoning: str,
    artifact_tag: str,
    parsed_ok: bool,
    source: str,
    extra_meta: dict[str, Any],
) -> None:
    """Finalize the meta.json for a completed (or failed) task."""
    paths = _task_artifact_paths(config, artifact_tag)
    if paths is None:
        return
    _, _, meta_path = paths
    meta: dict[str, Any] = {
        "status": "ok" if parsed_ok else "parse_failed",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "round_index": int(getattr(config, "current_round_index", 0)),
        "artifact_tag": artifact_tag,
        "parsed_ok": parsed_ok,
        "response_chars": len(text),
        "prompt_chars": len(prompt),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
    }
    meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_run_round_task_io(
    config: RunConfig,
    *,
    prompt: str,
    text: str,
    content: str,
    reasoning: str,
    artifact_tag: str,
    parsed_ok: bool,
    source: str,
    extra_meta: dict[str, Any],
) -> Path | None:
    """Back-compat wrapper: write all three files at once (used by VllmEngineGenerator)."""
    paths = _task_artifact_paths(config, artifact_tag)
    if paths is None:
        return None
    prompt_path, resp_path, _ = paths
    prompt_path.write_text(prompt, encoding="utf-8")
    resp_path.write_text(text, encoding="utf-8")
    _save_meta(
        config,
        prompt=prompt, text=text, content=content, reasoning=reasoning,
        artifact_tag=artifact_tag, parsed_ok=parsed_ok, source=source,
        extra_meta=extra_meta,
    )
    return resp_path


@dataclass
class VllmOpenAICompatSettings:
    """Talk to a local (or remote) server that exposes the OpenAI Chat Completions API (vLLM --api)."""

    base_url: str = "http://127.0.0.1:8010/v1"
    api_key: str = "EMPTY"
    model: str = ""
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    system_prompt: str | None = None
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    extra_client_kwargs: dict[str, Any] = field(default_factory=dict)


class VllmOpenAICompatGenerator(ItemGenerator):
    def __init__(self, settings: VllmOpenAICompatSettings) -> None:
        if not settings.model:
            raise ValueError("VllmOpenAICompatSettings.model is required")
        self.settings = settings

    def generate_one(
        self,
        parent: ProblemItem | None,
        *,
        memory: Sequence[ProblemItem],
        target_difficulty: float,
        feedback_context: str,
        config: RunConfig,
        artifact_tag: str = "",
        constraint_donor: ProblemItem | None = None,
    ) -> ProblemItem:
        """Generate one item via an OpenAI-compatible API.

        If ``parent`` is provided, the item is a modification of that parent (build_child_prompt).
        If ``parent`` is ``None``, the item is generated from scratch using ``memory`` as
        style/difficulty references (build_scratch_prompt) — Stage 1 of the two-stage pipeline.

        ``constraint_donor`` (ifeval only): the item whose instruction_id_list / kwargs the
        new prompt must enforce. Caller should pass the SAME donor that will later be used
        to populate ``item.instruction_id_list`` so the LLM's prompt and the grader's
        constraints stay in sync. If ``None``, falls back to legacy parent / memory[0].
        """
        mcq_spec = getattr(config, "mcq_spec", None)
        is_ifeval = bool(getattr(config, "ifeval_mode", False))
        if is_ifeval:
            # Strategy (c): instruction structure inherited from caller-supplied donor;
            # legacy fallback = parent / memory[0] (kept for back-compat).
            if constraint_donor is None:
                constraint_donor = parent if parent is not None else (memory[0] if memory else None)
            if parent is None:
                prompt = build_ifeval_scratch_prompt(
                    memory=memory,
                    target_difficulty=target_difficulty,
                    feedback_context=feedback_context,
                    constraint_donor=constraint_donor,
                )
            else:
                prompt = build_ifeval_child_prompt(
                    parent,
                    memory=memory,
                    target_difficulty=target_difficulty,
                    feedback_context=feedback_context,
                )
        elif parent is None:
            prompt = build_scratch_prompt(
                memory=memory,
                target_difficulty=target_difficulty,
                feedback_context=feedback_context,
                mcq_spec=mcq_spec,
            )
        else:
            prompt = build_child_prompt(
                parent,
                memory=memory,
                target_difficulty=target_difficulty,
                feedback_context=feedback_context,
                mcq_spec=mcq_spec,
            )
        # Set PROPOSE_PRINT_PROMPTS=1 to inspect the raw prompt before it is sent.
        # The chat template is applied server-side by vLLM; what you see here is the
        # user-turn content only. Check the server logs or use VllmEngineGenerator with
        # PROPOSE_PRINT_CHAT_TEMPLATE=1 to see the fully-templated string.
        if os.environ.get("PROPOSE_PRINT_PROMPTS", "").strip() == "1":
            sep = "=" * 60
            parent_tag = "scratch" if parent is None else repr(parent.id)
            print(f"\n{sep}\n[PROMPT] tag={artifact_tag!r} parent={parent_tag}\n{sep}\n{prompt}\n{sep}\n", flush=True)

        # Write prompt immediately so it's visible while the task is in-flight.
        _save_prompt(config, prompt=prompt, artifact_tag=artifact_tag)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install openai") from e

        client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=None,
            **self.settings.extra_client_kwargs,
        )
        messages: list[dict[str, str]] = []
        if self.settings.system_prompt:
            messages.append({"role": "system", "content": self.settings.system_prompt})
        messages.append({"role": "user", "content": prompt})
        create_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.top_p is not None:
            create_kwargs["top_p"] = self.settings.top_p
        # OpenAI SDK has no top_k param; vLLM accepts it via extra_body.
        if self.settings.top_k is not None:
            extra = dict(create_kwargs.get("extra_body") or {})
            extra["top_k"] = self.settings.top_k
            create_kwargs["extra_body"] = extra
        if config.seed is not None:
            create_kwargs["seed"] = config.seed

        parse_retries = int(os.environ.get("PROPOSE_VLLM_PARSE_RETRIES", "5"))
        text = ""
        c = ""
        rc = ""
        item: ProblemItem | None = None
        finish_reason: str | None = None
        for attempt in range(max(1, parse_retries)):
            print(
                f"[VllmOpenAICompatGenerator] POST {self.settings.base_url} model={self.settings.model!r} "
                f"tag={artifact_tag!r} attempt={attempt + 1}/{max(1, parse_retries)} — waiting for completion…",
                flush=True,
            )
            resp = client.chat.completions.create(**create_kwargs)
            msg = resp.choices[0].message
            c = (getattr(msg, "content", None) or "").strip()
            rc = str(getattr(msg, "reasoning_content", None) or "").strip()
            text = "\n\n".join(x for x in (rc, c) if x) if (rc or c) else ""
            finish_reason = str(getattr(resp.choices[0], "finish_reason", None) or "")
            # Write response immediately — visible before parsing even starts.
            _save_response(config, text=text, artifact_tag=artifact_tag)
            print(
                f"[VllmOpenAICompatGenerator] Got text len={len(text)} "
                f"(content_len={len(c)}, reasoning_len={len(rc)}, finish_reason={finish_reason!r}); parsing JSON…\n"
                f"[VllmOpenAICompatGenerator] Response tag={artifact_tag!r}:\n{text}\n",
                flush=True,
            )
            try:
                item = parse_single_item_from_model_text(text)
            except ValueError as e:
                if attempt + 1 >= max(1, parse_retries):
                    _save_meta(
                        config,
                        prompt=prompt, text=text, content=c, reasoning=rc,
                        artifact_tag=artifact_tag, parsed_ok=False,
                        source="VllmOpenAICompatGenerator",
                        extra_meta={
                            "model": self.settings.model,
                            "base_url": self.settings.base_url,
                            "error": str(e),
                            "finish_reason": finish_reason,
                        },
                    )
                    raise
                truncated = finish_reason == "length"
                print(
                    f"[VllmOpenAICompatGenerator] Parse failed"
                    f"{' (output truncated at max_tokens)' if truncated else ''}: {e} — retrying…",
                    flush=True,
                )
                if create_kwargs.get("seed") is not None:
                    create_kwargs["seed"] = int(create_kwargs["seed"]) + 1
                else:
                    create_kwargs["seed"] = random.randint(0, 2**31 - 1)
                if truncated:
                    # Replace user message with a short recovery prompt so the model
                    # spends tokens on JSON instead of another long reasoning chain.
                    recovery = (
                        "Your previous response was cut off before the JSON was delivered. "
                        "Output **only** one ```json code block containing a single-element array "
                        "with keys id, question, answer. No analysis.\n\n"
                        f"Original task:\n{prompt}"
                    )
                    create_kwargs["messages"] = list(messages)
                    create_kwargs["messages"][-1] = {"role": "user", "content": recovery}
                continue
            break

        assert item is not None
        _save_meta(
            config,
            prompt=prompt, text=text, content=c, reasoning=rc,
            artifact_tag=artifact_tag, parsed_ok=True,
            source="VllmOpenAICompatGenerator",
            extra_meta={"model": self.settings.model, "base_url": self.settings.base_url},
        )
        item.parent_id = parent.id if parent is not None else None
        return item


@dataclass
class VllmEngineSettings:
    """Load weights in-process with vllm.LLM (no HTTP server)."""

    model: str = ""
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    temperature: float = 0.7
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    llm_init_kwargs: dict[str, Any] = field(default_factory=dict)


class VllmEngineGenerator(ItemGenerator):
    def __init__(self, settings: VllmEngineSettings) -> None:
        if not settings.model:
            raise ValueError("VllmEngineSettings.model is required")
        self.settings = settings
        self._llm: Any = None

    def _engine(self) -> Any:
        if self._llm is None:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError("Install vllm for in-process inference") from e
            self._llm = LLM(
                model=self.settings.model,
                tensor_parallel_size=self.settings.tensor_parallel_size,
                trust_remote_code=self.settings.trust_remote_code,
                **self.settings.llm_init_kwargs,
            )
        return self._llm

    def shutdown(self) -> None:
        """Destroy the in-process engine and free GPU VRAM."""
        if self._llm is None:
            return
        import gc

        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del self._llm
        self._llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[VllmEngineGenerator] Engine shut down; GPU VRAM released.", flush=True)

    def generate_one(
        self,
        parent: ProblemItem,
        *,
        memory: Sequence[ProblemItem],
        target_difficulty: float,
        feedback_context: str,
        config: RunConfig,
        artifact_tag: str = "",
    ) -> ProblemItem:
        """Generate one child item from one parent via in-process vLLM.

        Note: the in-process engine is not thread-safe for concurrent calls.
        Use VllmOpenAICompatGenerator for parallel child generation.
        """
        prompt = build_child_prompt(
            parent,
            memory=memory,
            target_difficulty=target_difficulty,
            feedback_context=feedback_context,
            mcq_spec=getattr(config, "mcq_spec", None),
        )
        # Set PROPOSE_PRINT_PROMPTS=1 to see the raw user-turn content.
        if os.environ.get("PROPOSE_PRINT_PROMPTS", "").strip() == "1":
            sep = "=" * 60
            print(f"\n{sep}\n[PROMPT] tag={artifact_tag!r} parent={parent.id!r}\n{sep}\n{prompt}\n{sep}\n", flush=True)
        # Set PROPOSE_PRINT_CHAT_TEMPLATE=1 to see the fully chat-template-applied string
        # (i.e. exactly what the tokenizer will encode). Only available for the engine backend
        # because we have direct access to the tokenizer here.
        if os.environ.get("PROPOSE_PRINT_CHAT_TEMPLATE", "").strip() == "1":
            try:
                tok = self._engine().get_tokenizer()
                messages = [{"role": "user", "content": prompt}]
                templated = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                sep = "=" * 60
                print(f"\n{sep}\n[CHAT TEMPLATE] tag={artifact_tag!r}\n{sep}\n{templated}\n{sep}\n", flush=True)
            except Exception as e:
                print(f"[VllmEngineGenerator] chat-template preview failed: {e}", flush=True)
        from vllm import SamplingParams

        sp_kwargs: dict[str, Any] = {
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if config.seed is not None:
            sp_kwargs["seed"] = config.seed

        parse_retries = int(os.environ.get("PROPOSE_VLLM_PARSE_RETRIES", "3"))
        text = ""
        item: ProblemItem | None = None
        for attempt in range(max(1, parse_retries)):
            sp = SamplingParams(**sp_kwargs)
            print(
                f"[VllmEngineGenerator] model={self.settings.model!r} "
                f"tag={artifact_tag!r} attempt={attempt + 1}/{max(1, parse_retries)} — generating…",
                flush=True,
            )
            outputs = self._engine().chat(
                messages=[{"role": "user", "content": prompt}],
                sampling_params=sp,
            )
            text = outputs[0].outputs[0].text.strip()
            print(
                f"[VllmEngineGenerator] Got text len={len(text)}; parsing JSON…",
                flush=True,
            )
            try:
                item = parse_single_item_from_model_text(text)
            except ValueError as e:
                if attempt + 1 >= max(1, parse_retries):
                    _save_run_round_task_io(
                        config,
                        prompt=prompt,
                        text=text,
                        content=text,
                        reasoning="",
                        artifact_tag=artifact_tag,
                        parsed_ok=False,
                        source="VllmEngineGenerator",
                        extra_meta={"model": self.settings.model, "error": str(e)},
                    )
                    raise
                print(
                    f"[VllmEngineGenerator] Parse failed: {e} — retrying with a new seed…",
                    flush=True,
                )
                sp_kwargs["seed"] = (
                    int(sp_kwargs["seed"]) + 1
                    if sp_kwargs.get("seed") is not None
                    else random.randint(0, 2**31 - 1)
                )
                continue
            break

        assert item is not None
        _save_run_round_task_io(
            config,
            prompt=prompt,
            text=text,
            content=text,
            reasoning="",
            artifact_tag=artifact_tag,
            parsed_ok=True,
            source="VllmEngineGenerator",
            extra_meta={"model": self.settings.model},
        )
        item.parent_id = parent.id
        return item
