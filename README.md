# Habitat PointNav step benchmarks

Measure `env.step()` time on the MP3D sample scene `17DRP5sb8fy` (64×64 RGB + depth).

One Docker image (`habitat-pointnav-bench:latest`) is used for every run.

## Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU, threaded, and vector runs)
- Host `nvidia-smi` (used to sample SM% while a run is in progress)

Build is limited to two Habitat-Sim compile jobs (`HABITAT_BUILD_PARALLEL=2`).

## Build the image

```bash
docker compose -f docker-compose.gpu.yml build
```

CPU runs reuse this image. Do not build `docker-compose.cpu.yml`.

## Full sweep (recommended)

From the repo root, after the image exists:

```bash
./run_benchmark.sh
```

This runs:

1. Single-env GPU (task measurements on)
2. Single-env CPU (no GPU passthrough)
3. Single-env GPU with measurements off
4. For env counts `1, 3, 5, 7, 9` (up to `--max-envs`, step `--env-step`):
   - GPU `ThreadedVectorEnv`
   - GPU `VectorEnv` (one process per env)
   - GPU `VectorEnv` with frames kept in CUDA (`--gpu-gpu`)
5. At the largest env count only: `VectorEnv` with the batch renderer

It does **not** rebuild the image. Rebuild with:

```bash
./run_benchmark.sh --build
```

Useful flags:

```bash
./run_benchmark.sh --max-envs 5 --env-step 2
./run_benchmark.sh --warmup-steps 20 --measure-steps 200 --episodes 1
./run_benchmark.sh --gpu-id 0
./run_benchmark.sh --no-gpu-monitor
```

`./run_cpu_gpu_compare.sh` is an alias for `./run_benchmark.sh`.

Results print to the terminal. Full JSON is written to `benchmark/cpu_gpu_comparison.json` (gitignored).

## Threaded vs vector only

Fixed env count, GPU only (default 2 envs). Uses the existing image.

```bash
./run_parallel_benchmark.sh
./run_parallel_benchmark.sh --num-envs 4
```

Writes `benchmark/parallel_comparison.json`.

## Single compose runs

```bash
# GPU, one env
docker compose -f docker-compose.gpu.yml run --rm pointnav-gpu

# CPU, one env
docker compose -f docker-compose.cpu.yml run --rm pointnav-cpu
```

Override the command for other modes:

```bash
# ThreadedVectorEnv
docker compose -f docker-compose.gpu.yml run --rm --no-deps pointnav-gpu \
  bash -lc 'source activate habitat && python benchmark/pointnav_step_benchmark.py --device gpu --parallel threaded --num-envs 2'

# VectorEnv
docker compose -f docker-compose.gpu.yml run --rm --no-deps pointnav-gpu \
  bash -lc 'source activate habitat && python benchmark/pointnav_step_benchmark.py --device gpu --parallel vector --num-envs 2'

# VectorEnv, RGB/depth stay in CUDA (no host readback)
docker compose -f docker-compose.gpu.yml run --rm --no-deps pointnav-gpu \
  bash -lc 'source activate habitat && python benchmark/pointnav_step_benchmark.py --device gpu --parallel vector --num-envs 2 --gpu-gpu'

# VectorEnv + batch renderer (largest-count test in the full sweep)
docker compose -f docker-compose.gpu.yml run --rm --no-deps pointnav-gpu \
  bash -lc 'source activate habitat && python benchmark/pointnav_step_benchmark.py --device gpu --parallel vector --num-envs 9 --batch-renderer'

# GPU without task measurements (distance, success, SPL, reward)
docker compose -f docker-compose.gpu.yml run --rm --no-deps pointnav-gpu \
  bash -lc 'source activate habitat && python benchmark/pointnav_step_benchmark.py --device gpu --no-measures'
```

`--batch-renderer` requires `--parallel vector` and `--device gpu`. It cannot be combined with `--gpu-gpu`. `--no-measures` is for single-env only.
