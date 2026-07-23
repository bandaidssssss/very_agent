#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PLATFORM=${PLATFORM:-V5000}
export MAX_TRIALS=${MAX_TRIALS:-1}
RUN_TIME=$(date +%m%d_%H%M_%Y)
export OUTPUT_PATH=${OUTPUT_PATH:-${SCRIPT_DIR}/output}/${RUN_TIME}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_PATH}"
RUN_LOG="${OUTPUT_PATH}/run_circle.log"

if [[ -z "${VERL_ENV_SCRIPT:-}" ]]; then
    PLATFORM_UPPER=$(printf '%s' "${PLATFORM}" | tr '[:lower:]' '[:upper:]')
    case "${PLATFORM_UPPER}" in
        V5000) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_V5000.sh" ;;
        C550|METAX) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_C550.sh" ;;
        A100|NVIDIA|CUDA) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_NVIDIA.sh" ;;
        *) echo "Unsupported PLATFORM=${PLATFORM}. Set VERL_ENV_SCRIPT and GPU_SMI explicitly." >&2; exit 2 ;;
    esac
fi

python3 "${SCRIPT_DIR}/run_agent.py" \
    --base-config "${BASE_CONFIG_FILE:-${SCRIPT_DIR}/config/base_parameters.json}" \
    --agent-config "${AGENT_CONFIG_FILE:-${SCRIPT_DIR}/config/agent_config.json}" \
    --max-trials "${MAX_TRIALS}" \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
