#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE_PATH="${SCENE_PATH:-data/test/derain/4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/4}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${OUTPUT_ROOT}}"

PRIOR_CKPT="${PRIOR_CKPT:-ckpt/adair5d.ckpt}"

usage() {
  cat <<'EOF'
Usage:
  ./infer/infer_scene4.sh

Examples:
  ./infer/infer_scene4.sh

Optional environment variables:
  PYTHON_BIN       Python executable to use
  SCENE_PATH       Input scene path
  OUTPUT_ROOT      Root output directory
  LOG_ROOT         Log directory
  FINAL_OUTPUT     Final output root directory
  PRIOR_CKPT       Checkpoint for inference
EOF
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path not found: ${path}" >&2
    exit 1
  fi
}

run_infer() {
  local log_file="${LOG_ROOT}/prior.log"

  require_path "test2.py"
  require_path "${SCENE_PATH}"
  require_path "${PRIOR_CKPT}"
  mkdir -p "${FINAL_OUTPUT}"

  echo "Running scene 4 inference..."
  if ! "${PYTHON_BIN}" -u test2.py \
    --mode 1 \
    --derain_path "${SCENE_PATH}" \
    --ckpt_name "${PRIOR_CKPT}" \
    --output_path "${FINAL_OUTPUT}" \
    --use_video_prior \
    --prior_window_size 5 \
    --prior_align_mode none \
    --prior_warp_mode affine \
    --prior_temporal_semantics transient \
    --prior_stable_band_ratio 0.10 \
    --prior_strength 0.25 \
    >"${log_file}" 2>&1; then
    echo "Inference failed. Check log: ${log_file}" >&2
    exit 1
  fi

  echo "Inference finished. Log: ${log_file}"
  echo "Final result saved to: ${FINAL_OUTPUT}/derain/4"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"

run_infer
