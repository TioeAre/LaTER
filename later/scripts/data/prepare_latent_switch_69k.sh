#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-later}"
DATASET_NAME="${DATASET_NAME:-}"
SPLIT="${SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-data/latent-switch-69k/sft_train.parquet}"
MAX_ROWS="${MAX_ROWS:-}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --live-stream python
  -c '
import os
from pathlib import Path

from datasets import load_dataset

name = os.environ["DATASET_NAME"]
split = os.environ["SPLIT"]
output_path = Path(os.environ["OUTPUT_PATH"])
max_rows = os.environ.get("MAX_ROWS", "").strip()

ds = load_dataset(name, split=split)
if max_rows:
    ds = ds.select(range(min(int(max_rows), len(ds))))

output_path.parent.mkdir(parents=True, exist_ok=True)
ds.to_parquet(str(output_path))
print(f"Wrote {len(ds)} rows to {output_path}")
'
)

printf 'Preparing LaTER training data:'
printf ' %q' "${CMD[@]}"
printf '\n'
DATASET_NAME="${DATASET_NAME}" SPLIT="${SPLIT}" OUTPUT_PATH="${OUTPUT_PATH}" MAX_ROWS="${MAX_ROWS}" "${CMD[@]}"
