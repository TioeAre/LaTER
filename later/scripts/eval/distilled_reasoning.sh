#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export IF_EVALUATE_WITH_INSIGHT=true
export TASK="distilled_reasoning"

export EXPERIMENT_NAME="training"

export GENERATE_BS=1
export MAX_SAMPLES=-1

export COT_SWITCH_ENTROPY_THRESHOLD=1

export METHOD="latent_switch"
export DRAW_ENTROPY=true
export DRAW_ATTENTION=true
export SAVE_STATES=true
export IF_EXPLICIT_MODEL=true
# export SKIP_SPECIAL_TOKENS=false

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

export IF_EVALUATE_WITH_INSIGHT=true

export METHOD="baseline"
export DRAW_ENTROPY=true
export DRAW_ATTENTION=true
export SAVE_STATES=true

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