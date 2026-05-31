#!/usr/bin/bash

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
INPUT_DIR="${INPUT_DIR:-data/external/Dolci-Think-SFT-32B}"
OUTPUT_DIR="${OUTPUT_DIR:-data/external/Dolci-Think-SFT-32B_sampled}"
TARGET_SIZE="${TARGET_SIZE:-20000}"
NUM_OUTPUT_SHARDS="${NUM_OUTPUT_SHARDS:-1}"
SEED="${SEED:-42}"
EXCLUDE_DIR="${EXCLUDE_DIR:-}"

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python3
  later/src/train/data/preprocess/sample_Dolci-Think-SFT-32B.py
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --target_size "${TARGET_SIZE}"
  --num_output_shards "${NUM_OUTPUT_SHARDS}"
  --seed "${SEED}"
)

if [ -n "${EXCLUDE_DIR}" ]; then
  CMD+=(--exclude_dir "${EXCLUDE_DIR}")
fi

printf 'Running sample_data:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
