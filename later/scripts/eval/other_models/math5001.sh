#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export JUDGE_MODEL="qwen3_5_27b"
export CUDA_VISIBLE_DEVICES="4,5"

export TASK="math500"
export EXPERIMENT_NAME="other_models"

export METHOD="latent_switch"
export DRAW_ATTENTION=false
export SAVE_STATES=false

export LATENT_ENTROPY_THRESHOLD=5

export METHOD="baseline"
export DRAW_ATTENTION=false
export SAVE_STATES=false
export GENERATE_BS=1

export BASE_MODEL_NAME="models/DeepSeek-R1-Distill-Llama-8B"
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

export BASE_MODEL_NAME="models/Olmo-3-32B-Think"
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
