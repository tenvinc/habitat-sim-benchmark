#!/usr/bin/env python3
"""PointNav step timing for threaded env, then vector env, at a fixed env count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compare_cpu_gpu import (
    COMPOSE_GPU,
    IMAGE,
    SERVICE_GPU,
    compose_run,
    gpu_ok,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ThreadedVectorEnv and VectorEnv at N parallel PointNav envs."
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    return parser.parse_args()


def bench_shell(kind: str, args: argparse.Namespace) -> str:
    cmd = [
        "python",
        "benchmark/pointnav_step_benchmark.py",
        "--device",
        "gpu",
        "--json",
        "--parallel",
        kind,
        "--num-envs",
        str(args.num_envs),
        "--gpu-id",
        str(args.gpu_id),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
    ]
    return "source activate habitat && " + " ".join(cmd)


def _cell(result: dict[str, Any], key: str, spec: str) -> str:
    step = result["step"]
    if key == "construct_ms":
        value = result.get("construct_ms")
    elif key == "reset_mean_ms":
        value = result["reset"]["mean_ms"]
    else:
        value = step[key]
    if value is None:
        return f"{'—':>16}"
    return f"{value:16{spec}}"


def print_digest(
    threaded: dict[str, Any],
    vector: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    columns = (("threaded", threaded), ("vector", vector))
    rgb = threaded.get("rgb_resolution", ["?", "?"])
    depth = threaded.get("depth_resolution", ["?", "?"])
    w = 62
    print()
    print("=" * w)
    print(f"  PARALLEL POINTNAV — {args.num_envs} envs (MP3D 17DRP5sb8fy)")
    print("=" * w)
    print(
        f"  Config: warmup={args.warmup_steps}  measured={args.measure_steps}  "
        f"episodes={args.episodes}  seed={args.seed}"
    )
    print(f"  Sensors: rgb={rgb[0]}x{rgb[1]}  depth={depth[0]}x{depth[1]}")
    print(f"  Image:  {IMAGE}")
    print("-" * w)
    print(f"  {'':22} {'threaded':>16} {'vector':>16}")
    rows = (
        ("Construct (ms)", "construct_ms", ".0f"),
        ("Step wall mean (ms)", "mean_ms", ".2f"),
        ("Step wall median (ms)", "median_ms", ".2f"),
        ("Per-env mean (ms)", "per_env_mean_ms", ".2f"),
        ("Env-steps/sec", "env_steps_per_sec", ".1f"),
        ("Reset mean (ms)", "reset_mean_ms", ".1f"),
    )
    for label, key, spec in rows:
        cells = " ".join(_cell(result, key, spec) for _, result in columns)
        print(f"  {label:22} {cells}")
    print("-" * w)
    print("  Runtime checks (host nvidia-smi during each run):")
    for name, result in columns:
        ok, msg = gpu_ok("gpu", result)
        mark = "OK" if ok else "!!"
        print(f"    [{mark}] {name}: {msg}")
    print("=" * w)
    print()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        sys.exit("--num-envs must be >= 1")
    root = Path(__file__).resolve().parents[1]
    monitor = not args.no_gpu_monitor

    print(f"Parallel PointNav benchmark ({args.num_envs} envs)")
    print("-" * 40)
    print("[1/2] ThreadedVectorEnv...")
    threaded = compose_run(
        root=root,
        compose_file=COMPOSE_GPU,
        service=SERVICE_GPU,
        shell_cmd=bench_shell("threaded", args),
        monitor_gpu=monitor,
    )
    print("[2/2] VectorEnv (processes)...")
    vector = compose_run(
        root=root,
        compose_file=COMPOSE_GPU,
        service=SERVICE_GPU,
        shell_cmd=bench_shell("vector", args),
        monitor_gpu=monitor,
    )
    print_digest(threaded, vector, args)

    out_path = root / "benchmark" / "parallel_comparison.json"
    out_path.write_text(
        json.dumps(
            {"num_envs": args.num_envs, "threaded": threaded, "vector": vector},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Full JSON: {out_path}")


if __name__ == "__main__":
    main()
