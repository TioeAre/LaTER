#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export TASK="math500"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export JUDGE_MODEL="qwen3_5_27b"
export CUDA_VISIBLE_DEVICES="4,5"

export WITH_ENTROPY=false

export METHOD="baseline"
export EXPERIMENT_NAME="latent_qwen3_14b_38192_no_latent"
export BASE_MODEL_NAME="checkpoints/fsdp_cp/sft_no_latent_qwen3_14b_mix_ds/step_0004000"
export LATENT_STEPS=128
export MAX_NEW_TOKENS=38192
export TOP_K="${TOP_K:-20}"
export DO_SAMPLE="${DO_SAMPLE:-1}"
export DRAW_ENTROPY=false
export DRAW_ATTENTION=false
export SAVE_STATES=false

CMD=(
    ${CONDA_BIN:-conda} run -n later --live-stream python
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
