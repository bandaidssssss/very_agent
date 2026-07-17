#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROLE=${1:?Usage: run_analyze.sh proposal|feasibility|diagnosis CONTEXT.json [OUTPUT.json]}
CONTEXT=${2:?Usage: run_analyze.sh proposal|feasibility|diagnosis CONTEXT.json [OUTPUT.json]}
OUTPUT=${3:-}
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

args=(--role "${ROLE}" --context "${CONTEXT}")
if [[ -n "${OUTPUT}" ]]; then
    args+=(--output "${OUTPUT}")
fi
exec python3 "${SCRIPT_DIR}/role_cli.py" "${args[@]}"
