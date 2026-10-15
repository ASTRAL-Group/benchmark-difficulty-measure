#!/usr/bin/env bash
# Create isolated conda env for MiniMax-M2.7 proposal generation (vLLM serve).
set -euo pipefail

ENV_NAME="${ENV_NAME:-diff_minimax}"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/requirements_diff_minimax.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup_diff_minimax] env ${ENV_NAME} already exists — upgrading packages"
else
  echo "[setup_diff_minimax] creating ${ENV_NAME} (python 3.12)"
  conda create -n "$ENV_NAME" python=3.12 -y
fi

echo "[setup_diff_minimax] pip install -r ${REQ}"
conda run --no-capture-output -n "$ENV_NAME" pip install -U pip
conda run --no-capture-output -n "$ENV_NAME" pip install -r "$REQ"

echo ""
echo "[setup_diff_minimax] done. Verify:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import vllm; print('vllm', vllm.__version__)\""
echo "  bash propose/sh/serve_minimax_m2_7.sh   # after model weights cached"
