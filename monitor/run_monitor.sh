#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_FILE=${1:?Usage: run_monitor.sh TRAIN.log [GPU_SAMPLES.csv] [REPORT.json]}
GPU_SAMPLES=${2:-}
OUTPUT=${3:-}
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

args=(--log "${LOG_FILE}")
if [[ -n "${GPU_SAMPLES}" ]]; then
    args+=(--gpu-samples "${GPU_SAMPLES}")
fi
if [[ -n "${OUTPUT}" ]]; then
    args+=(--output "${OUTPUT}")
fi
exec python3 "${SCRIPT_DIR}/monitor_cli.py" "${args[@]}"
