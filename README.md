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

## Expected results

The following results were measured on a machine with these specifications., 

| | |
|---|---|
| CPU | AMD Ryzen 7 3700X (8 cores / 16 threads) |
| RAM | 32 GiB |
| GPU | NVIDIA GeForce RTX 2070 SUPER (8192 MiB) |
| Driver | 615.71.09 |
| Image | Habitat-Sim v0.3.3 (CUDA, headless, bullet), Habitat-Lab stable |

They were run using the scene `17DRP5sb8fy`, 64×64 RGB + depth, 1 episode, seed 7.  Each experiment was ran for 20 warmup steps which were not timed, and then for 200 steps which were timed.

### Single env

Below are the results for a single environment.

| | CPU | GPU | GPU no measures |
|---|---:|---:|---:|
| Step mean (ms) | 11.66 | 0.75 | 0.67 |
| Step median (ms) | 8.71 | 0.65 | 0.50 |
| Step p95 (ms) | 23.97 | 1.73 | 1.82 |
| Steps/sec | 85.8 | 1325.1 | 1499.9 |
| Reset mean (ms) | 141.99 | 17.99 | 15.03 |

### Parallel GPU sweep

Steps/sec is computed by taking the cumulative number of steps for all environments and dividing over the total time taken. VRAM represents the amount of additional GPU VRAM allocated during the experiments.

| envs | threaded (ms per multi-step) | threaded (steps/s) | vector (ms per multi-step) | vector (steps/s) | vector VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.89 | 1123.8 | 1.34 | 743.7 | 701 |
| 3 | 2.97 | 1011.4 | 2.12 | 1412.4 | 2103 |
| 5 | 5.10 | 981.2 | 2.73 | 1834.4 | 3489 |
| 7 | 8.40 | 833.3 | 3.23 | 2169.5 | 4906 |
| 9 | 9.61 | 936.7 | 4.11 | 2192.3 | 6311 |

At 9 envs, batch renderer (RGB only) was 9.46 ms / 951.8 steps/s and 975 MiB VRAM. Most of that step time was `post_step()` draw GPU→CPU (8.33 ms).

### Vector, frames kept in CUDA

These results are obtained using the gpu2gpu transfer function in habitat, keeping the frames entirely in GPU, but converting it to CUDA pytorch tensors. In practice, this allows you to work with the output of the simulation as pytorch tensors, while not incurring the GPU to CPU read cost.


| envs | step ms | steps/s | VRAM (MiB) |
|---:|---:|---:|---:|
| 1 | 0.64 | 1555.7 | 805 |
| 3 | 0.89 | 3358.9 | 2378 |
| 5 | 0.95 | 5259.6 | 3990 |
| 7 | 1.57 | 4467.9 | 5634 |

Note that 9-env configuration were not tested, because the program would run out of GPU memory for the test machine.