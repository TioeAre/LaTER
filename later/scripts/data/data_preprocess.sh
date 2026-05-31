#!/usr/bin/bash

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"

MODE="${MODE:-sft}"  # sft rl both

INPUT="${INPUT:-data/latent_reasoning_distill_ds/distilled_latent_reasoning.jsonl,data/latent_reasoning_distill/distilled_latent_reasoning.jsonl}"
GOOSE_DIR="${GOOSE_DIR:-data/external/Nemotron-Research-GooseReason-0.7M}"

OUTPUT_DIR="${OUTPUT_DIR:-data/latent-switch-69k-processed}"
RESET_PROGRESS="${RESET_PROGRESS:-1}"
IGNORE_CONFIG_CHANGE="${IGNORE_CONFIG_CHANGE:-0}"

COMMIT_EVERY="${COMMIT_EVERY:-200}"

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python
  later/src/train/data/preprocess/preprocess.py
  --input "${INPUT}"
  --goose_dir "${GOOSE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --mode "${MODE}"
  --tokenizer "${MODEL_PATH}"
  --commit_every "${COMMIT_EVERY}"
)

if [ "${RESET_PROGRESS}" = "1" ]; then
  CMD+=(--reset_progress)
fi
if [ "${IGNORE_CONFIG_CHANGE}" = "1" ]; then
  CMD+=(--ignore_config_change)
fi

printf 'Running data_preprocess:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
