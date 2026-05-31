#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
CONFIG="${CONFIG:-later/src/config/sft_lora_config_14b.yaml}"
PYTHON_BIN=("${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

detect_ngpus() {
    "${PYTHON_BIN[@]}" - <<'PY'
import torch
count = torch.cuda.device_count()
print(count if count > 0 else 1)
PY
}

NGPUS="${NGPUS:-$(detect_ngpus)}"

CMD=(
    "${PYTHON_BIN[@]}"
    -m torch.distributed.run
    --standalone
    --nnodes=1
    --nproc_per_node="${NGPUS}"
    -m later.src.train.train_lora
    --config "${CONFIG}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
