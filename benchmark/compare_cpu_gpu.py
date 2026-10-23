#!/usr/bin/env python3
"""Build once, run single-env and parallel PointNav step benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

COMPOSE_GPU = "docker-compose.gpu.yml"
COMPOSE_CPU = "docker-compose.cpu.yml"
SERVICE_GPU = "pointnav-gpu"
SERVICE_CPU = "pointnav-cpu"
IMAGE = "habitat-pointnav-bench:latest"
# RTX 2070 Super, 8192 MiB. One VectorEnv of this scene is about 699 MiB.
# Nine envs used 7053 MiB and left 742 MiB free, above a 384 MiB desktop reserve.
DEFAULT_MAX_ENVS = 9
DEFAULT_ENV_STEP = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PointNav CPU, GPU, GPU without measurements, "
            "then threaded and vector envs. At the largest count, also "
            "vector with the batch renderer."
        )
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--max-envs",
        type=int,
        default=DEFAULT_MAX_ENVS,
        help="Largest parallel env count. Sweep is 1, 1+step, ... up to this.",
    )
    parser.add_argument(
        "--env-step",
        type=int,
        default=DEFAULT_ENV_STEP,
        help="Gap between parallel env counts. Default 2 gives 1, 3, 5, ...",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the Docker image before running. Default is to use the existing image.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-gpu-monitor", action="store_true")
    return parser.parse_args()


def parse_nvidia_dmon(text: str) -> dict[str, float]:
    sm_vals: list[float] = []
    mem_vals: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            sm_vals.append(float(parts[1]))
            mem_vals.append(float(parts[2]))
        except ValueError:
            continue
    if not sm_vals:
        return {"samples": 0.0, "sm_avg_pct": 0.0, "sm_max_pct": 0.0, "mem_avg_pct": 0.0}
    return {
        "samples": float(len(sm_vals)),
        "sm_avg_pct": sum(sm_vals) / len(sm_vals),
        "sm_max_pct": max(sm_vals),
        "mem_avg_pct": sum(mem_vals) / len(mem_vals),
    }


def start_gpu_monitor() -> subprocess.Popen[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    return subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def stop_gpu_monitor(proc: subprocess.Popen[str] | None) -> dict[str, float]:
    if proc is None:
        return {"samples": 0.0, "sm_avg_pct": 0.0, "sm_max_pct": 0.0, "mem_avg_pct": 0.0}
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return parse_nvidia_dmon(out or "")


def bench_shell(
    device: str,
    gpu_id: int,
    warmup: int,
    measure: int,
    episodes: int,
    seed: int,
    no_measures: bool = False,
    parallel: str = "single",
    num_envs: int = 1,
    batch_renderer: bool = False,
    gpu_resident: bool = False,
) -> str:
    args = [
        "python",
        "benchmark/pointnav_step_benchmark.py",
        "--device",
        device,
        "--json",
        "--warmup-steps",
        str(warmup),
        "--measure-steps",
        str(measure),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
    ]
    if device == "gpu":
        args.extend(["--gpu-id", str(gpu_id)])
    if no_measures:
        args.append("--no-measures")
    if batch_renderer:
        args.extend(
            [
                "--parallel",
                "vector",
                "--num-envs",
                str(num_envs),
                "--batch-renderer",
            ]
        )
    elif gpu_resident:
        args.extend(
            [
                "--parallel",
                "vector",
                "--num-envs",
                str(num_envs),
                "--gpu-gpu",
            ]
        )
    elif parallel != "single":
        args.extend(["--parallel", parallel, "--num-envs", str(num_envs)])
    return "source activate habitat && " + " ".join(args)


def compose_run(
    *,
    root: Path,
    compose_file: str,
    service: str,
    shell_cmd: str,
    monitor_gpu: bool,
) -> dict[str, Any]:
    cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "run",
        "--rm",
        "--no-deps",
        service,
        "bash",
        "-lc",
        shell_cmd,
    ]
    dmon = start_gpu_monitor() if monitor_gpu else None
    time.sleep(0.5)
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    gpu_stats = stop_gpu_monitor(dmon)

    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        tail = (proc.stderr or proc.stdout)[-3000:]
        raise RuntimeError(f"{service} failed (exit {proc.returncode}):\n{tail}")

    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"No JSON from {service}:\n{proc.stderr[-1500:]}")
    result: dict[str, Any] = json.loads(proc.stdout[start:])
    result["host_nvidia_smi"] = gpu_stats
    return result


def ensure_data(root: Path) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            COMPOSE_GPU,
            "run",
            "--rm",
            "--no-deps",
            SERVICE_GPU,
            "bash",
            "-lc",
            f"/workspace/scripts/setup_mp3d_sample_data.sh /workspace/data "
            f"&& chown -R {os.getuid()}:{os.getgid()} /workspace/data 2>/dev/null || true",
        ],
        cwd=root,
        check=True,
    )


def gpu_ok(device: str, result: dict[str, Any]) -> tuple[bool, str]:
    sm_max = float(result.get("host_nvidia_smi", {}).get("sm_max_pct", 0.0))
    if device == "gpu":
        if sm_max >= 5.0:
            return True, f"GPU active (SM peak {sm_max:.0f}%)"
        return False, f"Low GPU use (SM peak {sm_max:.0f}%) — check NVIDIA runtime"
    if sm_max <= 5.0:
        return True, f"No GPU load (SM peak {sm_max:.0f}%)"
    return False, f"GPU still busy (SM peak {sm_max:.0f}%) — CPU compose may be wrong"


def _measure_label(result: dict[str, Any]) -> str:
    names = result.get("measurements") or []
    if result.get("no_measures") or not names:
        return "(none)"
    return ",".join(names)


def env_counts(max_envs: int, step: int) -> list[int]:
    if step < 1:
        raise SystemExit("--env-step must be >= 1")
    if max_envs < 1:
        raise SystemExit("--max-envs must be >= 1")
    counts = list(range(1, max_envs + 1, step))
    if counts[-1] != max_envs:
        counts.append(max_envs)
    return counts


def steps_per_sec(result: dict[str, Any]) -> float:
    step = result["step"]
    if "steps_per_sec" in step:
        return float(step["steps_per_sec"])
    num_envs = int(result.get("num_envs", 1))
    return num_envs * 1000.0 / float(step["mean_ms"])


def print_digest(
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    gpu_bare: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    columns = (
        ("CPU", cpu),
        ("GPU", gpu),
        ("GPU no meas.", gpu_bare),
    )
    means = [(name, result["step"]["mean_ms"]) for name, result in columns]
    fastest_name, fastest_ms = min(means, key=lambda item: item[1])
    slowest_name, slowest_ms = max(means, key=lambda item: item[1])
    if fastest_ms <= 0 or fastest_ms == slowest_ms:
        verdict = f"{fastest_name} is fastest (mean step time)"
    else:
        verdict = (
            f"{fastest_name} is {slowest_ms / fastest_ms:.2f}x faster than "
            f"{slowest_name} (mean step)"
        )

    gpu_mean = gpu["step"]["mean_ms"]
    bare_mean = gpu_bare["step"]["mean_ms"]
    if bare_mean > 0 and gpu_mean > bare_mean:
        measure_note = (
            f"Dropping measurements speeds GPU by {gpu_mean / bare_mean:.2f}x "
            f"({gpu_mean:.2f} ms -> {bare_mean:.2f} ms)"
        )
    elif gpu_mean > 0 and bare_mean > gpu_mean:
        measure_note = (
            f"GPU with measurements is faster ({gpu_mean:.2f} ms vs "
            f"{bare_mean:.2f} ms without)"
        )
    else:
        measure_note = "Measurements did not change GPU mean step time"

    gpu_check, gpu_msg = gpu_ok("gpu", gpu)
    bare_check, bare_msg = gpu_ok("gpu", gpu_bare)
    cpu_check, cpu_msg = gpu_ok("cpu", cpu)

    w = 78
    print()
    print("=" * w)
    print("  POINTNAV STEP BENCHMARK — MP3D sample (17DRP5sb8fy)")
    print("=" * w)
    rgb = gpu.get("rgb_resolution", ["?", "?"])
    depth = gpu.get("depth_resolution", ["?", "?"])
    print(f"  Config: warmup={args.warmup_steps}  measured={args.measure_steps}  "
          f"episodes={args.episodes}  seed={args.seed}")
    print(f"  Sensors: rgb={rgb[0]}x{rgb[1]}  depth={depth[0]}x{depth[1]}")
    print(f"  Image:  {IMAGE}")
    print(f"  Measures: CPU={_measure_label(cpu)}  GPU={_measure_label(gpu)}  "
          f"GPU no meas.={_measure_label(gpu_bare)}")
    print("-" * w)
    print(f"  {'':18} {'CPU':>14} {'GPU':>14} {'GPU no meas.':>14}")

    def row(label: str, values: list[float], spec: str) -> None:
        cells = " ".join(f"{value:14{spec}}" for value in values)
        print(f"  {label:18} {cells}")

    row(
        "Step mean (ms)",
        [result["step"]["mean_ms"] for _, result in columns],
        ".2f",
    )
    row(
        "Step median (ms)",
        [result["step"]["median_ms"] for _, result in columns],
        ".2f",
    )
    row(
        "Step p95 (ms)",
        [result["step"]["p95_ms"] for _, result in columns],
        ".2f",
    )
    row(
        "Steps/sec (mean)",
        [steps_per_sec(result) for _, result in columns],
        ".1f",
    )
    row(
        "Reset mean (ms)",
        [result["reset"]["mean_ms"] for _, result in columns],
        ".2f",
    )
    print("-" * w)
    print(f"  VERDICT: {verdict}")
    print(f"  MEASURES: {measure_note}")
    print("-" * w)
    print("  Runtime checks (host nvidia-smi during each run):")
    mark = "OK" if gpu_check else "!!"
    print(f"    [{mark}] GPU compose: {gpu_msg}")
    mark = "OK" if bare_check else "!!"
    print(f"    [{mark}] GPU no measures: {bare_msg}")
    mark = "OK" if cpu_check else "!!"
    print(f"    [{mark}] CPU compose: {cpu_msg}")
    if not gpu.get("cuda_enabled") or not gpu_bare.get("cuda_enabled"):
        print("    [!!] GPU run: habitat-sim cuda_enabled=False")
    if cpu.get("gpu_device_id") != -1:
        print(
            f"    [!!] CPU run: expected gpu_device_id=-1, got {cpu.get('gpu_device_id')}"
        )
    if gpu_bare.get("measurements"):
        print(
            "    [!!] GPU no-measures run still has measurements: "
            + ",".join(gpu_bare["measurements"])
        )
    print("=" * w)
    print()


def print_sweep(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    w = 92
    print("=" * w)
    print(
        f"  PARALLEL SWEEP — envs {env_counts(args.max_envs, args.env_step)} "
        f"(GPU, measurements on)"
    )
    print(
        f"  GPU cap is {args.max_envs} envs "
        f"({DEFAULT_MAX_ENVS} fit on this 8192 MiB card with a 384 MiB reserve)"
    )
    print("  Steps/sec counts every environment in the parallel step.")
    print("-" * w)
    print(
        f"  {'envs':>4}  {'thread ms':>10}  {'thread /s':>10}  "
        f"{'vector ms':>10}  {'vector /s':>10}  {'vram MiB':>8}"
    )
    for row in rows:
        threaded = row["threaded"]["step"]
        vector = row["vector"]["step"]
        vram = row["vector"].get("vram_delta_mib")
        vram_cell = f"{vram:8.0f}" if isinstance(vram, (int, float)) else f"{'—':>8}"
        print(
            f"  {row['num_envs']:4d}  {threaded['mean_ms']:10.2f}  "
            f"{steps_per_sec(row['threaded']):10.1f}  "
            f"{vector['mean_ms']:10.2f}  {steps_per_sec(row['vector']):10.1f}  "
            f"{vram_cell}"
        )
    batch_row = next((row for row in rows if "vector_batch" in row), None)
    if batch_row is not None:
        print("-" * w)
        print(
            f"  {batch_row['num_envs']}-ENV — threaded, vector, "
            "vector with batch renderer"
        )
        print(
            "  Batch renderer draws RGB only. Threaded and vector keep RGB and depth."
        )
        print(
            f"  {'mode':<22}  {'step ms':>10}  {'steps/s':>10}  {'vram MiB':>8}"
        )

        def batch_line(label: str, result: dict[str, Any]) -> None:
            vram = result.get("vram_delta_mib")
            vram_cell = (
                f"{vram:8.0f}" if isinstance(vram, (int, float)) else f"{'—':>8}"
            )
            print(
                f"  {label:<22}  {result['step']['mean_ms']:10.2f}  "
                f"{steps_per_sec(result):10.1f}  {vram_cell}"
            )

        batch_line("threaded", batch_row["threaded"])
        batch_line("vector", batch_row["vector"])
        batch_line("vector batch", batch_row["vector_batch"])
    gpu_rows = [row for row in rows if "vector_gpu" in row]
    if gpu_rows:
        print("-" * w)
        print("  VECTOR, RGB and depth kept in CUDA memory")
        print(
            f"  {'envs':>4}  {'step ms':>10}  {'steps/s':>10}  {'vram MiB':>8}"
        )
        for row in gpu_rows:
            result = row["vector_gpu"]
            if "step" not in result:
                print(f"  {row['num_envs']:4d}  {'OOM':>10}  {'—':>10}  {'—':>8}")
                continue
            vram = result.get("vram_delta_mib")
            vram_cell = (
                f"{vram:8.0f}" if isinstance(vram, (int, float)) else f"{'—':>8}"
            )
            print(
                f"  {row['num_envs']:4d}  {result['step']['mean_ms']:10.2f}  "
                f"{steps_per_sec(result):10.1f}  {vram_cell}"
            )
        phases = batch_row["vector_batch"].get("phases") or {}
        if phases:
            print("-" * w)
            print("  Batch phases at that count (mean ms of the measured steps)")
            phase_labels = (
                ("vector_env_step", "VectorEnv.step()"),
                ("batch_post_step", "post_step()"),
                ("keyframe_apply", "  keyframe apply"),
                ("draw_gpu_to_cpu", "  draw GPU to CPU"),
                ("observation_conversion", "  observation conversion"),
            )
            for key, label in phase_labels:
                mean = (phases.get(key) or {}).get("mean_ms")
                if mean is None:
                    continue
                print(f"  {label:<28} {mean:8.3f}")
    print("=" * w)
    print()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    monitor = not args.no_gpu_monitor

    print("PointNav benchmark runner")
    print("-" * 40)

    counts = env_counts(args.max_envs, args.env_step)
    # image, data, three single-env runs, threaded, vector, and GPU-resident
    # vector at each count, plus one batch-renderer run at the largest count
    total_stages = 2 + 3 + 3 * len(counts) + 1
    stage = 0

    def next_stage(message: str) -> None:
        nonlocal stage
        stage += 1
        print(f"[{stage}/{total_stages}] {message}")

    if args.build:
        next_stage("Building Docker image...")
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_GPU, "build"],
            cwd=root,
            check=True,
        )
    else:
        next_stage(f"Using existing image {IMAGE} (pass --build to rebuild)")

    next_stage("Ensuring MP3D sample data...")
    ensure_data(root)

    common = (
        args.gpu_id,
        args.warmup_steps,
        args.measure_steps,
        args.episodes,
        args.seed,
    )

    next_stage("Running GPU benchmark (measurements on)...")
    gpu = compose_run(
        root=root,
        compose_file=COMPOSE_GPU,
        service=SERVICE_GPU,
        shell_cmd=bench_shell("gpu", *common),
        monitor_gpu=monitor,
    )

    next_stage("Running CPU benchmark (measurements on, no GPU passthrough)...")
    cpu = compose_run(
        root=root,
        compose_file=COMPOSE_CPU,
        service=SERVICE_CPU,
        shell_cmd=bench_shell("cpu", *common),
        monitor_gpu=monitor,
    )

    next_stage("Running GPU benchmark with measurements disabled...")
    gpu_bare = compose_run(
        root=root,
        compose_file=COMPOSE_GPU,
        service=SERVICE_GPU,
        shell_cmd=bench_shell("gpu", *common, no_measures=True),
        monitor_gpu=monitor,
    )

    sweep: list[dict[str, Any]] = []
    for count in counts:
        next_stage(f"Running threaded benchmark ({count} envs)...")
        threaded = compose_run(
            root=root,
            compose_file=COMPOSE_GPU,
            service=SERVICE_GPU,
            shell_cmd=bench_shell(
                "gpu", *common, parallel="threaded", num_envs=count
            ),
            monitor_gpu=monitor,
        )
        next_stage(f"Running vector benchmark ({count} envs)...")
        vector = compose_run(
            root=root,
            compose_file=COMPOSE_GPU,
            service=SERVICE_GPU,
            shell_cmd=bench_shell(
                "gpu", *common, parallel="vector", num_envs=count
            ),
            monitor_gpu=monitor,
        )
        next_stage(
            f"Running vector benchmark, frames kept on GPU ({count} envs)..."
        )
        try:
            vector_gpu = compose_run(
                root=root,
                compose_file=COMPOSE_GPU,
                service=SERVICE_GPU,
                shell_cmd=bench_shell(
                    "gpu",
                    *common,
                    parallel="vector",
                    num_envs=count,
                    gpu_resident=True,
                ),
                monitor_gpu=monitor,
            )
        except RuntimeError as exc:
            print(f"  GPU-resident {count} envs did not finish: {exc}")
            vector_gpu = {"error": str(exc)}
        row: dict[str, Any] = {
            "num_envs": count,
            "threaded": threaded,
            "vector": vector,
            "vector_gpu": vector_gpu,
        }
        if count == args.max_envs:
            next_stage(
                f"Running vector benchmark with batch renderer ({count} envs)..."
            )
            row["vector_batch"] = compose_run(
                root=root,
                compose_file=COMPOSE_GPU,
                service=SERVICE_GPU,
                shell_cmd=bench_shell(
                    "gpu",
                    *common,
                    parallel="vector",
                    num_envs=count,
                    batch_renderer=True,
                ),
                monitor_gpu=monitor,
            )
        sweep.append(row)

    print_digest(cpu, gpu, gpu_bare, args)
    print_sweep(sweep, args)

    out_path = root / "benchmark" / "cpu_gpu_comparison.json"
    out_path.write_text(
        json.dumps(
            {
                "max_envs": args.max_envs,
                "env_step": args.env_step,
                "env_counts": counts,
                "cpu": cpu,
                "gpu": gpu,
                "gpu_no_measures": gpu_bare,
                "parallel_sweep": sweep,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Full JSON: {out_path}")


if __name__ == "__main__":
    main()
