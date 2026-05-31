#!/usr/bin/bash

set -euo pipefail

export MODEL="${MODEL:-Qwen3-32B}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to an OpenAI-compatible /v1 endpoint}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export STAGE1_OPENAI_BASE_URL="${STAGE1_OPENAI_BASE_URL:-${OPENAI_BASE_URL}}"
# export JUDGE_OPENAI_BASE_URL="${JUDGE_OPENAI_BASE_URL:-}"
export JUDGE_OPENAI_BASE_URL="${JUDGE_OPENAI_BASE_URL:-${OPENAI_BASE_URL}}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-32B}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"

INPUT_DIR="${INPUT_DIR:-data/external/Dolci-Think-SFT-32B_sampled_200k_20260321_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-data/latent_reasoning_distill}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-14B}"

MAX_SAMPLES="${MAX_SAMPLES:--1}"
WORKERS="${WORKERS:-120}"

BATCH_SIZE="${BATCH_SIZE:--1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
STAGE2_CORRECT_THRESHOLD="${STAGE2_CORRECT_THRESHOLD:-0.7}"
STAGE2_REROLLOUTS="${STAGE2_REROLLOUTS:-5}"
STAGE1_MAX_TOKENS="${STAGE1_MAX_TOKENS:-16384}"
STAGE2_MAX_TOKENS="${STAGE2_MAX_TOKENS:-40960}"
MAX_CHARS_PER_OUTPUT="${MAX_CHARS_PER_OUTPUT:-40960}"

TIMEOUT="${TIMEOUT:-3600}"
MAX_RETRIES="${MAX_RETRIES:-3}"

LOG_EVERY="${LOG_EVERY:-100}"
LIMIT="${LIMIT:-1000000}"

RESUME="${RESUME:-1}"

DISABLE_SEED="${DISABLE_SEED:-1}"

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python
  later/src/train/data/preprocess/data_construct.py
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model "${MODEL}"
  --base_url "${OPENAI_BASE_URL}"
  --api_key_env OPENAI_API_KEY
  --tokenizer_path "${TOKENIZER_PATH}"
  --max_samples "${MAX_SAMPLES}"
  --workers "${WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --stage2_correct_threshold "${STAGE2_CORRECT_THRESHOLD}"
  --stage2_rerollouts "${STAGE2_REROLLOUTS}"
  --stage1_max_tokens "${STAGE1_MAX_TOKENS}"
  --stage2_max_tokens "${STAGE2_MAX_TOKENS}"
  --max_chars_per_output "${MAX_CHARS_PER_OUTPUT}"
  --timeout "${TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --log_every "${LOG_EVERY}"
  --limit "${LIMIT}"
)

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi
if [ "${DISABLE_SEED}" = "1" ]; then
  CMD+=(--disable_seed)
fi

printf 'Running data_construct:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
