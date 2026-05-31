#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export CONDA_ENV="${CONDA_ENV:-later}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export SPLIT="${SPLIT:-test}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export JUDGE_MODEL="qwen3_5_27b"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

export TASK="math500"

export METHOD="latent_qwen3"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-later-14b}"
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"
export LATENT_STEPS=128
export MAX_NEW_TOKENS=28192
export TEMPERATURE=0.6
export TOP_P=0.95
export GENERATE_BS=10
export TOP_K="${TOP_K:-20}"
export DO_SAMPLE="${DO_SAMPLE:-1}"
export DRAW_ENTROPY=false
export DRAW_ATTENTION=false
export SAVE_STATES=false
export PENDDING=2

# export LATENT_QWEN3_EARLY_EXIT_ON_SPECIAL=1

CMD=(
    ${CONDA_BIN:-conda} run -n ${CONDA_ENV:-later} --live-stream python
    later/src/eval/eval.py
    --method "$METHOD"
    --model_name "$BASE_MODEL_NAME"
    --task "$TASK"
    --generate_bs "$GENERATE_BS"
    --max_samples "$MAX_SAMPLES"
    --split "$SPLIT"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --latent_steps "$LATENT_STEPS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
)

if [ "$DO_SAMPLE" = "1" ]; then
    CMD+=(--do_sample)
fi

"${CMD[@]}"
