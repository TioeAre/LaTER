#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

export EXPERIMENT_NAME="visualize_3D_trace"
export PRINT_FILE=false

${CONDA_BIN:-conda} run -n later --live-stream python later/src/analysis/visualize_reasoning_trace.py
