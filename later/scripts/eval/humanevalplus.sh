#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export TASK="humanevalplus"
export EXPERIMENT_NAME="other_benchmark"

export METHOD="latent_switch"
export DRAW_ENTROPY=false
export DRAW_ATTENTION=false
export SAVE_STATES=false

${CONDA_BIN:-conda} run -n later --live-stream python later/src/eval/eval.py --method "$METHOD" \
    --model_name "$BASE_MODEL_NAME" \
    --task "$TASK" \
    --generate_bs "$GENERATE_BS" \
    --max_samples "$MAX_SAMPLES" \
    --split "$SPLIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --latent_steps "$LATENT_STEPS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P"


export METHOD="baseline"
export DRAW_ATTENTION=false
export SAVE_STATES=false

${CONDA_BIN:-conda} run -n later --live-stream python later/src/eval/eval.py --method "$METHOD" \
    --model_name "$BASE_MODEL_NAME" \
    --task "$TASK" \
    --generate_bs "$GENERATE_BS" \
    --max_samples "$MAX_SAMPLES" \
    --split "$SPLIT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --latent_steps "$LATENT_STEPS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P"