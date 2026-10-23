#!/usr/bin/env bash
# ThreadedVectorEnv, then VectorEnv, at --num-envs (default 2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
exec python3 "${ROOT}/benchmark/compare_parallel.py" "$@"
