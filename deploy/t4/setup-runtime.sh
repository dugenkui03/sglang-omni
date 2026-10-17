#!/usr/bin/env bash
# Prepare an isolated userspace without changing the host OS or GPU driver.
set -euo pipefail

runtime_root=${SGLANG_OMNI_RUNTIME_ROOT:-/root/.local/share/sglang-omni-runtime/ubuntu24}
operations_root=${SGLANG_OMNI_OPERATIONS_ROOT:-/root/data/sglang-omni/operations}
repository_root=${SGLANG_OMNI_REPOSITORY_ROOT:-/root/workspace/sglang-omni}
ubuntu_release=24.04.5
ubuntu_archive=ubuntu-base-${ubuntu_release}-base-amd64.tar.gz
ubuntu_source=https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release

test "$(id -u)" = 0
mkdir -p "$operations_root" "$runtime_root"
if ! test -f "$runtime_root/etc/os-release"; then
    curl --fail --location --retry 3 "$ubuntu_source/$ubuntu_archive" -o "$operations_root/$ubuntu_archive"
    curl --fail --location --retry 3 "$ubuntu_source/SHA256SUMS" -o "$operations_root/ubuntu-SHA256SUMS"
    (cd "$operations_root" && grep "[ *]$ubuntu_archive\$" ubuntu-SHA256SUMS | sha256sum --check -)
    tar --extract --gzip --file "$operations_root/$ubuntu_archive" --directory "$runtime_root"
fi

for bind_path in /proc /sys /dev "$repository_root" /root/data/sglang-omni /usr/local/cuda-12.4; do
    mkdir -p "$runtime_root$bind_path"
    if ! mountpoint -q "$runtime_root$bind_path"; then
        mount --rbind "$bind_path" "$runtime_root$bind_path"
        mount --make-rslave "$runtime_root$bind_path"
    fi
done
cp --dereference /etc/resolv.conf "$runtime_root/etc/resolv.conf"
install -D /root/.local/bin/uv "$runtime_root/usr/local/bin/uv"
chroot "$runtime_root" /usr/bin/env DEBIAN_FRONTEND=noninteractive /bin/bash -eu -c '
apt-get update
apt-get install -y --no-install-recommends ca-certificates python3 python3-venv python3-dev build-essential git curl sox ffmpeg libsndfile1 libnuma1 libibverbs1
'
