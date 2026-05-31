#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "${PROJECT_DIR}"

export BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export SPLIT="${SPLIT:-test}"

TASK_SCRIPTS=(
  later/scripts/eval/sft/gsm8k.sh
  later/scripts/eval/sft/aime25.sh
  later/scripts/eval/sft/math500.sh
  later/scripts/eval/sft/gpqa.sh
  later/scripts/eval/sft/arc_challenge.sh
  later/scripts/eval/sft/mbppplus.sh
  later/scripts/eval/sft/humanevalplus.sh
)

for script in "${TASK_SCRIPTS[@]}"; do
  echo "===== Running ${script} with BASE_MODEL_NAME=${BASE_MODEL_NAME} ====="
  bash "${script}"
done
