#!/usr/bin/bash

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
BASE_DIR="${BASE_DIR:-data/external/Dolci-Think-SFT-32B_sampled}"
ADDITIONAL_DIRS="${ADDITIONAL_DIRS:-data/external/Dolci-Think-SFT-32B_sampled_additional_180k_20260321_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-data/external/Dolci-Think-SFT-32B_sampled_200k}"
ASSEMBLE_MODE="${ASSEMBLE_MODE:-symlink}"
SKIP_VALIDATE_IDS="${SKIP_VALIDATE_IDS:-0}"

if [ -z "${ADDITIONAL_DIRS}" ]; then
  echo "ADDITIONAL_DIRS is required and should be a colon-separated list of sampled dataset roots" >&2
  exit 1
fi

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python
  later/src/train/data/preprocess/assemble_sampled_dataset.py
  --base_dir "${BASE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --mode "${ASSEMBLE_MODE}"
)

IFS=':' read -r -a ADDITIONAL_DIR_ARRAY <<< "${ADDITIONAL_DIRS}"
for dir_path in "${ADDITIONAL_DIR_ARRAY[@]}"; do
  if [ -n "${dir_path}" ]; then
    CMD+=(--additional_dir "${dir_path}")
  fi
done

if [ "${SKIP_VALIDATE_IDS}" = "1" ]; then
  CMD+=(--skip_validate_ids)
fi

printf 'Running assemble_sampled_data:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
