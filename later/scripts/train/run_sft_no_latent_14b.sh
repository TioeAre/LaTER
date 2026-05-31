#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
CONFIG="${CONFIG:-later/src/config/sft_no_latent_config_14b.yaml}"
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

detect_context_parallel_size() {
    "${PYTHON_BIN[@]}" - "${CONFIG}" <<'PY'
import sys
import yaml
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
print(int(config.get("context_parallel_size", 1) or 1))
PY
}

CP_SIZE="$(detect_context_parallel_size)"
AVAILABLE_GPUS="$(detect_ngpus)"
NGPUS="${NGPUS:-${CP_SIZE}}"
if [ "${CP_SIZE}" -gt 1 ] && [ "${NGPUS}" -ne "${CP_SIZE}" ]; then
    echo "For no-latent context parallel training, NGPUS (${NGPUS}) must equal context_parallel_size (${CP_SIZE})." >&2
    exit 1
fi
if [ "${AVAILABLE_GPUS}" -lt "${NGPUS}" ]; then
    echo "Requested ${NGPUS} processes but only ${AVAILABLE_GPUS} CUDA device(s) are visible." >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN[@]}"
    -m torch.distributed.run
    --standalone
    --nnodes=1
    --nproc_per_node="${NGPUS}"
    -m later.src.train.train_no_latent
    --config "${CONFIG}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
