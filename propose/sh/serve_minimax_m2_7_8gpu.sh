#!/usr/bin/env bash
# MiniMax-M2.7 proposal generator: **single** vLLM with DP=2 × TP=4 × EP on 8 GPUs.
#
# Uses all 8× RTX PRO 6000 (physical 0–7) in one process.
#
#   conda activate diff_minimax
#   bash propose/sh/serve_minimax_m2_7_8gpu.sh
#
# Override:
#   MAX_MODEL_LEN=32768 GPU_MEM_UTIL=0.85 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TP="${TP:-4}"
export DP="${DP:-2}"
export PP="${PP:-1}"
export ENABLE_EP="${ENABLE_EP:-1}"

exec bash "${SCRIPT_DIR}/serve_minimax_m2_7.sh"
