#!/usr/bin/env bash
# Use the locked standalone workbench environment and the server-local TTS API.
set -euo pipefail

repository_root=${SGLANG_OMNI_REPOSITORY_ROOT:-/root/workspace/sglang-omni}
operations_root=${SGLANG_OMNI_OPERATIONS_ROOT:-/root/data/sglang-omni/operations}
cd "$repository_root/projects/tts-workbench"
exec env -u TTS_WORKBENCH_PASSWORD .venv/bin/tts-workbench \
    --api-base http://127.0.0.1:8000 --data-dir "$repository_root/tts_test_data" \
    --results-dir ./results/server --log-file "$operations_root/server.log" \
    --host 0.0.0.0 --port 8081 --allow-unauthenticated-public
