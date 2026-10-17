#!/usr/bin/env bash
# Install the T4 stack inside the isolated userspace, leaving base Torch intact.
set -euo pipefail

runtime_root=${SGLANG_OMNI_RUNTIME_ROOT:-/root/.local/share/sglang-omni-runtime/ubuntu24}
repository_root=${SGLANG_OMNI_REPOSITORY_ROOT:-/root/workspace/sglang-omni}
chroot "$runtime_root" /usr/bin/env REPOSITORY_ROOT="$repository_root" /bin/bash -eu -c '
cd "$REPOSITORY_ROOT"
if ! test -f .venv-cu126/pyvenv.cfg; then
  uv venv --python /usr/bin/python3 .venv-cu126
fi
uv pip install --python .venv-cu126/bin/python \
  --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
  --overrides deploy/t4/overrides-cu126.txt --excludes deploy/t4/excludes-cu126.txt \
  --requirements deploy/t4/requirements-cu126.txt
uv pip install --python .venv-cu126/bin/python --no-deps sglang==0.5.21 qwen-tts==0.1.1
uv pip install --python .venv-cu126/bin/python --no-deps --editable .
uv pip freeze --python .venv-cu126/bin/python > /root/data/sglang-omni/operations/environment-freeze.txt
'
