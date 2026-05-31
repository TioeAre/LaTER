#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export TASK="aime2025"
export EXPERIMENT_NAME="latent_qwen3_38192_analysis"

export METHOD="latent_switch"
export DRAW_ATTENTION=true
export SAVE_STATES=true

export ENTROPY_THRESHOLD=10    # red
export LATENT_ENTROPY_THRESHOLD=70    # red
export COT_SWITCH_ENTROPY_THRESHOLD=30    # red
export LATENT_TOKENS_LIMIT=96     # for latent_switch, 0-80 red

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
export SAVE_STATES=true

# ${CONDA_BIN:-conda} run -n later --live-stream python later/src/eval/eval.py --method "$METHOD" \
#     --model_name "$BASE_MODEL_NAME" \
#     --task "$TASK" \
#     --generate_bs "$GENERATE_BS" \
#     --max_samples "$MAX_SAMPLES" \
#     --split "$SPLIT" \
#     --max_new_tokens "$MAX_NEW_TOKENS" \
#     --latent_steps "$LATENT_STEPS" \
#     --temperature "$TEMPERATURE" \
#     --top_p "$TOP_P"

# export METHOD="latent_qwen3"
# # export EXPERIMENT_NAME="latent_qwen3"
# # export BASE_MODEL_NAME="checkpoints/deepspeed/sft_latent_qwen3_8b_2/step_0010000"
# # export EXPERIMENT_NAME="latent_qwen3_38192"
# # export BASE_MODEL_NAME="checkpoints/deepspeed/sft_latent_qwen3_8b_38192_2/step_0009500"
# export EXPERIMENT_NAME="latent_qwen3_14b_38192_3"
# export BASE_MODEL_NAME="checkpoints/deepspeed/sft_latent_qwen3_14b_38192_3/step_0003250"
# export LATENT_STEPS=128
# export MAX_NEW_TOKENS=38192
# export TOP_K="${TOP_K:-20}"
# export DO_SAMPLE="${DO_SAMPLE:-1}"
# export DRAW_ENTROPY=false
# export DRAW_ATTENTION=false
# export SAVE_STATES=false

# CMD=(
#     ${CONDA_BIN:-conda} run -n later --live-stream python
#     later/src/eval/eval.py
#     --method "$METHOD"
#     --model_name "$BASE_MODEL_NAME"
#     --task "$TASK"
#     --generate_bs "$GENERATE_BS"
#     --max_samples "$MAX_SAMPLES"
#     --split "$SPLIT"
#     --max_new_tokens "$MAX_NEW_TOKENS"
#     --latent_steps "$LATENT_STEPS"
#     --temperature "$TEMPERATURE"
#     --top_p "$TOP_P"
#     --top_k "$TOP_K"
# )

# if [ "$DO_SAMPLE" = "1" ]; then
#     CMD+=(--do_sample)
# fi

# "${CMD[@]}"
