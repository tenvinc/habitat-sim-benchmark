#!/usr/bin/env bash
# Run CPU, GPU, GPU-without-measurements, then threaded, vector, and
# GPU-resident vector sweeps from 1 env up to the GPU cap in steps of 2.
# At that cap, also run vector with the batch renderer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
exec python3 "${ROOT}/benchmark/compare_cpu_gpu.py" "$@"
