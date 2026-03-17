#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE_PATH="${SCENE_PATH:-data/test/deblur/6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/6}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${OUTPUT_ROOT}}"

NOPRIOR_CKPT="${NOPRIOR_CKPT:-AdaIR/snow_falling_noprior_stage1_e30_bs4/epoch=99-step=23200.ckpt}"
PRIOR_CKPT="${PRIOR_CKPT:-AdaIR/vp_adair_snow_from_adair5d_single_e30/epoch=99-step=80700.ckpt}"
STAGE1_OUTPUT=""

usage() {
  cat <<'EOF'
Usage:
  ./infer/infer_scene6.sh

Examples:
  ./infer/infer_scene6.sh

Optional environment variables:
  PYTHON_BIN       Python executable to use
  SCENE_PATH       Input scene path
  OUTPUT_ROOT      Root output directory
  LOG_ROOT         Log directory
  FINAL_OUTPUT     Final output root directory
  NOPRIOR_CKPT     Checkpoint for no-prior inference
  PRIOR_CKPT       Checkpoint for video-prior inference
EOF
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path not found: ${path}" >&2
    exit 1
  fi
}

cleanup_intermediate() {
  if [[ -n "${STAGE1_OUTPUT}" && -d "${STAGE1_OUTPUT}" ]]; then
    rm -rf "${STAGE1_OUTPUT}"
  fi
}

run_noprior_stage() {
  local log_file="${LOG_ROOT}/noprior.log"

  require_path "test2.py"
  require_path "${SCENE_PATH}"
  require_path "${NOPRIOR_CKPT}"
  STAGE1_OUTPUT="$(mktemp -d "${OUTPUT_ROOT}/.stage1_tmp.XXXXXX")"

  echo "Running stage 1: scene 6 without video prior..."
  if ! "${PYTHON_BIN}" -u test2.py \
    --mode 3 \
    --gopro_path "${SCENE_PATH}" \
    --ckpt_name "${NOPRIOR_CKPT}" \
    --output_path "${STAGE1_OUTPUT}" \
    --prior_strength 0.0 \
    >"${log_file}" 2>&1; then
    echo "Stage 1 failed. Check log: ${log_file}" >&2
    exit 1
  fi

  echo "Stage 1 finished. Log: ${log_file}"
}

run_prior_stage() {
  local log_file="${LOG_ROOT}/prior.log"
  local stage1_input="${STAGE1_OUTPUT}/deblur/6"

  require_path "test2.py"
  require_path "${stage1_input}"
  require_path "${PRIOR_CKPT}"
  mkdir -p "${FINAL_OUTPUT}"

  echo "Running stage 2: use stage 1 output as input with video prior..."
  if ! "${PYTHON_BIN}" -u test2.py \
    --mode 3 \
    --gopro_path "${stage1_input}" \
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
    echo "Stage 2 failed. Check log: ${log_file}" >&2
    exit 1
  fi

  echo "Stage 2 finished. Log: ${log_file}"
  echo "Final result saved to: ${FINAL_OUTPUT}/deblur/6"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"
trap cleanup_intermediate EXIT

run_noprior_stage
run_prior_stage
