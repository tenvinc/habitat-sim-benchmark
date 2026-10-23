#!/usr/bin/env python3
"""Section timings for threaded, CPU-read vector, and GPU-memory vector."""

from __future__ import annotations

import json
import subprocess
import sys


MODES = (
    ("threaded", ["--parallel", "threaded"]),
    ("vector", ["--parallel", "vector"]),
    ("vector_gpu", ["--parallel", "vector", "--gpu-gpu"]),
)


def main() -> None:
    results = []
    for n in (1, 3, 5, 7):
        for name, extra in MODES:
            print(f"RUN {name} n={n}", flush=True)
            cmd = [
                sys.executable,
                "benchmark/pointnav_step_benchmark.py",
                "--device",
                "gpu",
                "--num-envs",
                str(n),
                "--sections",
                "--warmup-steps",
                "20",
                "--measure-steps",
                "200",
                "--episodes",
                "1",
                "--json",
                *extra,
            ]
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                print(f"FAIL {name} n={n} code={proc.returncode}", flush=True)
                print(proc.stderr[-4000:], flush=True)
                results.append({"name": name, "num_envs": n, "error": proc.stderr[-2000:]})
                continue
            payload = json.loads(proc.stdout)
            step = payload.get("step") or {}
            sections = payload.get("sections") or {}
            util = payload.get("gpu_util") or {}
            per = sections.get("per_env_mean_ms") or {}
            row = {
                "name": name,
                "num_envs": n,
                "wall_ms": step.get("mean_ms"),
                "steps_per_sec": step.get("steps_per_sec"),
                "vram_delta_mib": payload.get("vram_delta_mib"),
                "per_env_mean_ms": per,
                "sum_across_envs_ms": sections.get("sum_across_envs_ms"),
                "slowest_env_ms": sections.get("slowest_env_ms"),
                "gpu_util": util,
            }
            results.append(row)
            gpu = util.get("gpu_util_mean_pct", util.get("error"))
            print(
                "DONE {name} n={n} wall={wall:.3f} ms steps/s={sps:.0f} "
                "act={act:.3f} draw={draw:.3f} read={read:.3f} "
                "phys2={phys2:.3f} meas={meas:.3f} other={other:.3f} gpu={gpu}".format(
                    name=name,
                    n=n,
                    wall=row["wall_ms"],
                    sps=row["steps_per_sec"],
                    act=per.get("act_physics", 0.0),
                    draw=per.get("draw", 0.0),
                    read=per.get("read", 0.0),
                    phys2=per.get("second_physics", 0.0),
                    meas=per.get("measures", 0.0),
                    other=per.get("other", 0.0),
                    gpu=gpu,
                ),
                flush=True,
            )
    print("JSON_START")
    print(json.dumps(results))
    print("JSON_END")


if __name__ == "__main__":
    main()
