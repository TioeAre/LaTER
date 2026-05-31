#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"


# Qwen3-VL-32B-Instruct
VLLM_CACHE_ROOT=".cache/vllm" ${CONDA_BIN:-conda} run -n qwenvl --live-stream vllm serve Qwen/Qwen3-32B --served-model-name Qwen3-32B --tensor-parallel-size 2 --port 12503 --max-model-len 20480

VLLM_CACHE_ROOT=".cache/vllm" ${CONDA_BIN:-conda} run -n qwenvl --live-stream vllm serve models/Qwen3-VL-32B-Instruct --served-model-name Qwen3-VL-32B-Instruct --tensor-parallel-size 2 --gpu_memory_utilization 0.85 --mm-processor-cache-gb 0 --no-enable-prefix-caching --limit-mm-per-prompt.video 0 --host 0.0.0.0 --port 12503

uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

VLLM_CACHE_ROOT=".cache/vllm" vllm serve models/Qwen3.5-27B --served-model-name qwen3_5_27b --port 8000 --tensor-parallel-size 2 --max-model-len 262144 --reasoning-parser qwen3


# SGLANG_DG_CACHE_DIR=".cache/sglang" ${CONDA_BIN:-conda} run -n qwenvl --live-stream -m sglang.launch_server \
#    --model-path models/Qwen3-VL-32B-Instruct \
#    --host 0.0.0.0 \
#    --port 12503 \
#    --tp 4

# python3 -m
# vllm.entrypoints.openai.api_server
# --model ${MODEL_PATH}
# --served-model-name ${MODEL_NAME}
# --tensor-parallel-size ${MLP_GPU_NUM}
# --enable-prefix-caching
# --enable-chunked-prefill
# --max-model-len 32768
# --port 8000
# --disable-log-requests

