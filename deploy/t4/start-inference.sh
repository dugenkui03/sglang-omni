#!/usr/bin/env bash
# Start the formal inference CLI with T4-compatible kernels and precision.
set -euo pipefail

runtime_root=${SGLANG_OMNI_RUNTIME_ROOT:-/root/.local/share/sglang-omni-runtime/ubuntu24}
repository_root=${SGLANG_OMNI_REPOSITORY_ROOT:-/root/workspace/sglang-omni}
operations_root=${SGLANG_OMNI_OPERATIONS_ROOT:-/root/data/sglang-omni/operations}
for bind_path in /proc /sys /dev "$repository_root" /root/data/sglang-omni /usr/local/cuda-12.4; do
    mkdir -p "$runtime_root$bind_path"
    if ! mountpoint -q "$runtime_root$bind_path"; then
        mount --rbind "$bind_path" "$runtime_root$bind_path"
        mount --make-rslave "$runtime_root$bind_path"
    fi
done
exec chroot "$runtime_root" /usr/bin/env \
    PATH="$repository_root/.venv-cu126/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat:/opt/nvidia/lib \
    CUDA_HOME=/usr/local/cuda-12.4 \
    TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas \
    CUDA_VISIBLE_DEVICES=0 SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0 \
    SGLANG_FORCE_FUSED_OP_BACKEND=torch SGLANG_OMNI_STRICT_PORT=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    "$repository_root/.venv-cu126/bin/sgl-omni" serve \
    --config "$repository_root/deploy/t4/qwen3-tts.yaml" \
    --host 127.0.0.1 --port 8000 --model-name qwen3-tts-1.7b-base \
    --allowed-local-media-path "$operations_root" "$@"
