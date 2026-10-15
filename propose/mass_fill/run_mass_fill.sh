#!/usr/bin/env bash
# Mass-fill entrypoint (MiniMax generate + 12-model verify).
#
#   bash propose/mass_fill/run_mass_fill.sh init aime24
#   bash propose/mass_fill/run_mass_fill.sh run aime24
#   bash propose/mass_fill/run_mass_fill.sh status aime24
#   bash propose/mass_fill/run_mass_fill.sh export aime24   # OCR sheet (photo → file)
#   bash propose/mass_fill/run_mass_fill.sh import-release  # rebuild state from mass_result/release
#
# Hardware: 8× NVIDIA RTX PRO 6000 Blackwell (~96GB each), CUDA devices 0–7;
#   the verifier runs Qwen3.5-397B-A17B / GLM-4.6 at TP=8.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.local"
fi

export REPO_ROOT="${REPO_ROOT:-$ROOT}"
export GAP_ROOT="${GAP_ROOT:-$ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CONDA_SERVE_ENV="${CONDA_SERVE_ENV:-diff_minimax}"
export MODEL="${MODEL:-minimax-m2.7}"
export PYTHONPATH="${ROOT}/propose${PYTHONPATH:+:$PYTHONPATH}"

CMD="${1:-help}"
shift || true

_py() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n diff --no-capture-output python -m mass_fill.driver "$@"
  else
    python3 -m mass_fill.driver "$@"
  fi
}

_py_mod() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n diff --no-capture-output python -m "$@"
  else
    python3 -m "$@"
  fi
}

case "$CMD" in
  init)
    BM="${1:?init <benchmark>}"
    _py init --benchmark "$BM" "${@:2}"
    ;;
  run|all)
    BM="${1:?run <benchmark>}"
    echo "[mass_fill] benchmark=$BM"
    _py run --benchmark "$BM" "${@:2}"
    ;;
  status)
    BM="${1:?status <benchmark>}"
    _py status --benchmark "$BM"
    ;;
  export)
    if [[ -n "${1:-}" ]]; then
      _py export --benchmark "$1" "${@:2}"
    else
      _py export "${@:1}"
    fi
    ;;
  import-release)
    _py_mod mass_fill.import_release "$@"
    ;;
  help|-h|--help)
    echo "Usage: $0 init|run|status|export <aime24|aime25|aime26|amc23>"
    echo "       $0 export                 # all benches that have state"
    echo "       $0 import-release         # rebuild state.json from mass_result/release"
    ;;
  *)
    echo "Unknown: $CMD"
    exit 1
    ;;
esac
