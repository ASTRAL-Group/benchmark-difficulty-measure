#!/usr/bin/env bash
# vLLM OpenAI API for MiniMaxAI/MiniMax-M2.7 (230B MoE, ~10B active/token).
# Model card: https://huggingface.co/MiniMaxAI/MiniMax-M2.7
# Deploy guide: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/vllm_deploy_guide.md
#
# Parallel layouts (single vLLM process — MoE sharded over TP×DP GPUs):
#   4-GPU (default):  TP=4, DP=1, EP=1  →  CUDA_VISIBLE_DEVICES=0,1,2,3
#   8-GPU proposal:   TP=4, DP=2, EP=1  →  serve_minimax_m2_7_8gpu.sh
#
# Requires conda env **diff_minimax** (see requirements_diff_minimax.txt).
#
# Override:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash propose/sh/serve_minimax_m2_7.sh
#   TP=4 ENABLE_EP=1 MAX_MODEL_LEN=32768 bash ...
#
# Client: MODEL=minimax-m2.7 must match --served-model-name.
# Inference (per request, from run_experiment.sh → OpenAI API):
#   temperature=1.0, top_p=0.95, top_k=40
#   system: "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
# Quick env check: python scripts/test_minimax_inference.py

set -euo pipefail

# Avoid NCCL net-plugin segfault on some driver/GPU stacks (same as verifier/run_rollout.sh).
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# FlashInfer JIT needs a matched nvcc + CUDA headers toolchain. On mixed pip cuda
# packages (cu13 nvcc 13.3 vs runtime 13.0) ninja build fails at profile_run.
# PyTorch-native sampler is fine for proposal generation throughput.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# FlashInfer JIT needs nvcc; pip-installed toolkit lives under nvidia/cu13 in diff_minimax.
if [[ -z "${CUDA_HOME:-}" ]]; then
  _nvcc=""
  if command -v nvcc >/dev/null 2>&1; then
    _nvcc="$(command -v nvcc)"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    for _cand in \
      "${CONDA_PREFIX}/bin/nvcc" \
      "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc \
      "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/cuda_nvcc/bin/nvcc; do
      if [[ -x "$_cand" ]]; then _nvcc="$_cand"; break; fi
    done
  fi
  if [[ -n "$_nvcc" ]]; then
    export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
    export PATH="$(dirname "$_nvcc"):${PATH}"
    echo "[serve_minimax_m2_7] CUDA_HOME=${CUDA_HOME}" >&2
  fi
fi

_visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local _s="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "${_s}" ]]; then
      echo 0
      return
    fi
    IFS=',' read -ra _ids <<< "${_s}"
    echo "${#_ids[@]}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]'
    return
  fi
  echo 1
}

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES

_num="$(_visible_gpu_count)"
_num="${_num// /}"
if ! [[ "${_num}" =~ ^[0-9]+$ ]] || (( _num < 1 )); then
  _num=1
fi

: "${TP:=4}"
: "${DP:=1}"
: "${PP:=1}"
: "${ENABLE_EP:=1}"

_expected=$((TP * PP * DP))
if (( _expected != _num )); then
  echo "serve_minimax_m2_7.sh: TP=${TP} × PP=${PP} × DP=${DP} = ${_expected} but ${_num} GPU(s) visible." >&2
  echo "  4-GPU: TP=4 DP=1.  8-GPU proposal: bash propose/sh/serve_minimax_m2_7_8gpu.sh (TP=4 DP=2)." >&2
  exit 1
fi

: "${PORT:=8010}"
: "${MAX_MODEL_LEN:=65536}"
: "${SERVED_NAME:=minimax-m2.7}"
: "${GPU_MEM_UTIL:=0.9}"
: "${VLLM_MODEL:=MiniMaxAI/MiniMax-M2.7}"

# Bash ${var:-...} truncates at unescaped '}' — assign JSON in a plain if-block instead.
if [[ -z "${MINIMAX_COMPILATION_CONFIG:-}" ]]; then
  MINIMAX_COMPILATION_CONFIG='{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true},"cudagraph_mode":"PIECEWISE"}'
fi

_extra=()
if (( DP > 1 )); then
  _extra+=(--data-parallel-size "${DP}")
fi
if [[ "${ENABLE_EP}" == "1" ]]; then
  _extra+=(--enable-expert-parallel)
fi
if [[ -n "${MINIMAX_COMPILATION_CONFIG:-}" ]]; then
  _extra+=(--compilation-config "${MINIMAX_COMPILATION_CONFIG}")
fi

echo "[serve_minimax_m2_7] model=${VLLM_MODEL} TP=${TP} DP=${DP} PP=${PP} EP=${ENABLE_EP} visible_gpus=${_num} max_model_len=${MAX_MODEL_LEN} port=${PORT} served_model_name=${SERVED_NAME}" >&2

export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

exec vllm serve "${VLLM_MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  "${_extra[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice
