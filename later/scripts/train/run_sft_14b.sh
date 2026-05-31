#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
CONFIG="${CONFIG:-later/src/config/sft_config_14b.yaml}"
PYTHON_BIN=("${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

TRAIN_DATA=$(CONFIG_PATH="${CONFIG}" "${PYTHON_BIN[@]}" - <<'PYCFG'
import os
import yaml
with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(cfg.get("train_data", ""))
PYCFG
)
if [ -n "${TRAIN_DATA}" ] && [ ! -f "${TRAIN_DATA}" ]; then
    echo "Training data not found: ${TRAIN_DATA}" >&2
    echo "Run: bash later/scripts/data/prepare_latent_switch_69k.sh" >&2
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
    -m later.src.train.train
    --config "${CONFIG}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
