#!/usr/bin/env python3
"""Minimal MiniMax-M2.7 inference environment test.

Verifies (without running the full gap-filling pipeline):
  1. vLLM OpenAI API is reachable at VLLM_BASE_URL
  2. Model id matches MODEL (default minimax-m2.7)
  3. One chat completion succeeds with official card parameters

Usage (vLLM must already be running):
  conda activate diff_minimax   # or any env with `openai` installed
  bash propose/sh/serve_minimax_m2_7.sh   # in another terminal

  python scripts/test_minimax_inference.py

Optional env:
  VLLM_BASE_URL   default http://127.0.0.1:8010/v1
  MODEL           default minimax-m2.7
  SKIP_COMPLETION set 1 to only probe /v1/models (no generation)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "minimax-m2.7")
SKIP_COMPLETION = os.environ.get("SKIP_COMPLETION", "").strip().lower() in ("1", "true", "yes")

SYSTEM_PROMPT = os.environ.get(
    "MINIMAX_SYSTEM_PROMPT",
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.",
)
TEMPERATURE = float(os.environ.get("MINIMAX_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("MINIMAX_TOP_P", "0.95"))
TOP_K = int(os.environ.get("MINIMAX_TOP_K", "40"))


def _probe_models() -> list[str]:
    url = f"{BASE_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"FAIL: cannot reach {url}\n  {e}\n"
            "Start vLLM first, e.g.:\n"
            "  conda activate diff_minimax\n"
            "  CUDA_VISIBLE_DEVICES=0,1,2,3 bash propose/sh/serve_minimax_m2_7.sh"
        ) from e
    ids = [m["id"] for m in payload.get("data", [])]
    print(f"OK: {url} → models: {ids}")
    if MODEL not in ids:
        raise SystemExit(f"FAIL: expected model {MODEL!r} in {ids!r}")
    return ids


def _run_completion() -> None:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("FAIL: pip install openai") from e

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=120.0)
    print(
        f"POST chat.completions model={MODEL!r} "
        f"temperature={TEMPERATURE} top_p={TOP_P} top_k={TOP_K}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Reply with exactly one JSON object, no markdown: "
                    '{"question":"What is 2+2?","answer":"4"}'
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=256,
        extra_body={"top_k": TOP_K},
    )
    msg = resp.choices[0].message
    content = (getattr(msg, "content", None) or "").strip()
    reasoning = str(getattr(msg, "reasoning_content", None) or "").strip()
    text = "\n\n".join(x for x in (reasoning, content) if x)
    finish = getattr(resp.choices[0], "finish_reason", None)
    print(f"OK: completion len={len(text)} finish_reason={finish!r}")
    if not text:
        raise SystemExit("FAIL: empty completion")
    preview = text[:400] + ("…" if len(text) > 400 else "")
    print(f"--- response preview ---\n{preview}\n------------------------")


def main() -> None:
    print(f"[test_minimax_inference] base_url={BASE_URL} model={MODEL}")
    _probe_models()
    if SKIP_COMPLETION:
        print("SKIP_COMPLETION=1 — models probe only.")
        return
    _run_completion()
    print("PASS: MiniMax inference environment looks healthy.")


if __name__ == "__main__":
    main()
