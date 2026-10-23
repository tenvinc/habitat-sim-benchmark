#!/usr/bin/env python3
"""Largest VectorEnv count that fits on the current GPU for the PointNav sample."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pointnav_step_benchmark as bench

# Leave this much GPU memory free so the desktop is not pushed into OOM.
RESERVE_MIB = 384


def namespace(num_envs: int) -> argparse.Namespace:
    return argparse.Namespace(
        device="gpu",
        config=bench.DEFAULT_CONFIG,
        no_measures=False,
        parallel="vector",
        num_envs=num_envs,
        warmup_steps=0,
        measure_steps=1,
        episodes=1,
        seed=7,
        gpu_id=0,
        gpu_gpu=False,
    )


def release(env: object | None) -> None:
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    gc.collect()
    time.sleep(1.0)


def try_count(config: object, num_envs: int) -> dict:
    args = namespace(num_envs)
    before = bench.query_gpu_memory()
    env = None
    try:
        env, construct_s = bench.open_parallel_env(args, config)
        bench.run_parallel_episode(env, __import__("numpy").random.default_rng(7), num_envs, 0, 1)
        alive = bench.query_gpu_memory()
        return {
            "num_envs": num_envs,
            "ok": True,
            "construct_ms": construct_s * 1000.0,
            "vram_before_mib": before.get("used_mib"),
            "vram_used_mib": alive.get("used_mib"),
            "vram_free_mib": alive.get("free_mib"),
            "vram_total_mib": alive.get("total_mib"),
            "vram_delta_mib": alive.get("used_mib", 0.0) - before.get("used_mib", 0.0),
        }
    except Exception as exc:
        return {
            "num_envs": num_envs,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "vram_before_mib": before.get("used_mib"),
        }
    finally:
        release(env)


def main() -> None:
    args = namespace(1)
    bench.apply_device_env("gpu", 0)
    config = bench.build_config(args, Path(__file__).resolve().parents[1])
    baseline = bench.query_gpu_memory()
    print(f"baseline used={baseline.get('used_mib')} free={baseline.get('free_mib')}", file=sys.stderr, flush=True)

    first = try_count(config, 1)
    print(json.dumps({"probe": first}), file=sys.stderr, flush=True)
    if not first.get("ok"):
        print(json.dumps({"baseline": baseline, "max_envs": 0, "probes": [first]}))
        sys.exit(1)

    per_env = float(first["vram_delta_mib"])
    if per_env < 1.0:
        per_env = 1.0
    free_for_envs = float(baseline.get("free_mib", 0.0)) - RESERVE_MIB
    estimate = max(1, int(free_for_envs // per_env))
    print(
        f"per_env={per_env:.0f} MiB reserve={RESERVE_MIB} estimate={estimate}",
        file=sys.stderr,
        flush=True,
    )

    probes = [first]
    best = 1
    if estimate > 1:
        trial = try_count(config, estimate)
        probes.append(trial)
        print(json.dumps({"probe": trial}), file=sys.stderr, flush=True)
        if trial.get("ok") and float(trial.get("vram_free_mib", 0.0)) >= RESERVE_MIB:
            best = estimate
            # One more if the leftover free memory can hold another env.
            if float(trial["vram_free_mib"]) >= per_env + RESERVE_MIB:
                extra = try_count(config, estimate + 1)
                probes.append(extra)
                print(json.dumps({"probe": extra}), file=sys.stderr, flush=True)
                if extra.get("ok") and float(extra.get("vram_free_mib", 0.0)) >= RESERVE_MIB:
                    best = estimate + 1
        else:
            # Estimate did not fit. Step down until one does.
            for count in range(estimate - 1, 1, -1):
                trial = try_count(config, count)
                probes.append(trial)
                print(json.dumps({"probe": trial}), file=sys.stderr, flush=True)
                if trial.get("ok") and float(trial.get("vram_free_mib", 0.0)) >= RESERVE_MIB:
                    best = count
                    break

    result = {
        "gpu_total_mib": baseline.get("total_mib"),
        "baseline_used_mib": baseline.get("used_mib"),
        "reserve_mib": RESERVE_MIB,
        "per_env_mib": per_env,
        "max_envs": best,
        "probes": probes,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
