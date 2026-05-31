#!/bin/bash
#
# Offline teacher precompute launcher for latent-reasoning SFT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
CONFIG="${CONFIG:-later/src/config/sft_config_14b.yaml}"
RUNNER=("${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"

if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

detect_ngpus() {
    "${RUNNER[@]}" python - <<'PY'
import torch
count = torch.cuda.device_count()
print(count if count > 0 else 1)
PY
}

NGPUS="${NGPUS:-$(detect_ngpus)}"

if ! [[ "${NGPUS}" =~ ^[0-9]+$ ]] || [ "${NGPUS}" -lt 1 ]; then
    echo "Invalid NGPUS=${NGPUS}. Set NGPUS to a positive integer." >&2
    exit 1
fi

readarray -t CONFIG_INFO < <("${RUNNER[@]}" python - <<'PY' "${CONFIG}"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

print(str(cfg.get("model_path", "Qwen/Qwen3-14B")))
print(str(cfg.get("train_data", "data/processed/sft_train.parquet")))
print(str(cfg.get("precomputed_teacher_dir", cfg.get("teacher_cache_dir", ""))))
print(str(cfg.get("precomputed_teacher_topk", cfg.get("kl_topk_dim", 32))))
PY
)

MODEL_PATH="${MODEL_PATH:-${CONFIG_INFO[0]}}"
TRAIN_DATA="${TRAIN_DATA:-${CONFIG_INFO[1]}}"
PRECOMPUTED_TEACHER_TOPK="${PRECOMPUTED_TEACHER_TOPK:-${CONFIG_INFO[3]}}"
PRECOMPUTED_TEACHER_DIR="${PRECOMPUTED_TEACHER_DIR:-${CONFIG_INFO[2]}}"
if [ -z "${PRECOMPUTED_TEACHER_DIR}" ]; then
    PRECOMPUTED_TEACHER_DIR="data/processed/teacher_cache/sft_teacher_topk${PRECOMPUTED_TEACHER_TOPK}_${RUN_TIMESTAMP}"
fi

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "=== Preprocessing SFT data ==="
    "${RUNNER[@]}" python -m later.src.train.data.preprocess.preprocess \
        --output_dir data/processed \
        --mode sft \
        --tokenizer "${MODEL_PATH}"
fi

RUNTIME_CONFIG="$(mktemp /tmp/latent_teacher_precompute.XXXXXX.yaml)"
cleanup() {
    rm -f "${RUNTIME_CONFIG}"
}
trap cleanup EXIT

"${RUNNER[@]}" python - <<'PY' "${CONFIG}" "${RUNTIME_CONFIG}"
from pathlib import Path
import os
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

string_overrides = {
    "model_path": os.environ.get("MODEL_PATH"),
    "teacher_model_path": os.environ.get("TEACHER_MODEL_PATH"),
    "train_data": os.environ.get("TRAIN_DATA"),
    "precomputed_teacher_dir": os.environ.get("PRECOMPUTED_TEACHER_DIR"),
    "precomputed_teacher_prob_dtype": os.environ.get("PRECOMPUTED_TEACHER_PROB_DTYPE"),
}
for key, value in string_overrides.items():
    if value:
        cfg[key] = value

int_overrides = {
    "max_length": os.environ.get("MAX_LENGTH"),
    "teacher_max_length": os.environ.get("TEACHER_MAX_LENGTH"),
    "num_workers": os.environ.get("NUM_WORKERS"),
    "teacher_precompute_batch_size": os.environ.get("TEACHER_PRECOMPUTE_BATCH_SIZE"),
    "teacher_precompute_num_workers": os.environ.get("TEACHER_PRECOMPUTE_NUM_WORKERS"),
    "teacher_precompute_rows_per_shard": os.environ.get("TEACHER_PRECOMPUTE_ROWS_PER_SHARD"),
    "precomputed_teacher_topk": os.environ.get("PRECOMPUTED_TEACHER_TOPK"),
}
for key, value in int_overrides.items():
    if value is not None and value != "":
        cfg[key] = int(value)

bool_overrides = {
    "precomputed_teacher_validate_metadata": os.environ.get("PRECOMPUTED_TEACHER_VALIDATE_METADATA"),
}
for key, value in bool_overrides.items():
    if value is None or value == "":
        continue
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        cfg[key] = True
    elif lowered in {"0", "false", "no", "off"}:
        cfg[key] = False
    else:
        raise ValueError(f"Invalid boolean override for {key}: {value}")

with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, allow_unicode=False, sort_keys=False)
PY

CMD=(
    "${RUNNER[@]}"
    python -m torch.distributed.run
    --standalone
    --nnodes=1
    --nproc_per_node="${NGPUS}"
    -m later.src.train.data.preprocess.precompute_teacher
    --config "${RUNTIME_CONFIG}"
)

echo "=== Starting offline teacher precompute (${NGPUS} GPUs) ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Config: ${CONFIG}"
echo "Runtime config: ${RUNTIME_CONFIG}"
echo "Model path: ${MODEL_PATH}"
echo "Train data: ${TRAIN_DATA}"
echo "Precomputed teacher dir: ${PRECOMPUTED_TEACHER_DIR}"
echo "Precomputed teacher top-k: ${PRECOMPUTED_TEACHER_TOPK}"

if [ "${DRY_RUN}" = "1" ]; then
    printf 'DRY_RUN command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"

echo "=== offline teacher precompute complete ==="
