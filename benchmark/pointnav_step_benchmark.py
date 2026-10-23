#!/usr/bin/env python3
"""Benchmark wall-clock time per PointNav env.step() on MP3D sample scene."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import habitat
import habitat_sim
import numpy as np
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PointNav simulation step latency (MP3D sample)."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="gpu: habitat_sim_v0.gpu_device_id; cpu: gpu_device_id=-1 (no GPU).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pointnav_mp3d_sample.yaml",
        help="Hydra config path relative to habitat-lab or workspace.",
    )
    parser.add_argument(
        "--no-measures",
        action="store_true",
        help=(
            "Drop task measurements (distance_to_goal, success, SPL, reward). "
            "Selects configs/pointnav_mp3d_no_measures.yaml when --config is "
            "left at the default."
        ),
    )
    parser.add_argument(
        "--parallel",
        choices=("single", "threaded", "vector"),
        default="single",
        help=(
            "single: one habitat.Env. "
            "threaded: habitat.ThreadedVectorEnv. "
            "vector: habitat.VectorEnv (one process per env)."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel environments. Defaults to 2 for threaded/vector, 1 for single.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Steps excluded from timing statistics.",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=200,
        help="Steps included in timing statistics.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (each triggers env.reset()).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for action sampling.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device index when --device=gpu (2070 Super is usually 0).",
    )
    parser.add_argument(
        "--gpu-gpu",
        action="store_true",
        help=(
            "Keep RGB and depth in CUDA memory. The framebuffer is copied "
            "with cudaMemcpyDeviceToDevice and is not read back to the CPU. "
            "Separate from the batch-renderer test."
        ),
    )
    parser.add_argument(
        "--batch-renderer",
        action="store_true",
        help=(
            "VectorEnv with Habitat's batch renderer. Workers record gfx-replay "
            "keyframes and do not draw. The main process renders every env in "
            "one pass inside env.post_step(), which is included in the step time. "
            "The batch renderer accepts one visual sensor, so depth is removed."
        ),
    )
    parser.add_argument(
        "--sections",
        action="store_true",
        help=(
            "Split each step into act+physics, draw, pixel read, the second "
            "physics step, and task measurements. Also sample GPU utilization "
            "over a 2 second continuation of the same loop."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object with results (for automation).",
    )
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 1 if args.parallel == "single" else 2
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.parallel == "single" and args.num_envs != 1:
        parser.error("--num-envs > 1 requires --parallel threaded or vector")
    if args.parallel != "single" and args.no_measures:
        parser.error(
            "--parallel threaded/vector steps an RL env that reads reward "
            "and success measurements; omit --no-measures"
        )
    if args.batch_renderer:
        if args.parallel != "vector":
            parser.error("--batch-renderer requires --parallel vector")
        if args.device != "gpu":
            parser.error("--batch-renderer requires --device gpu")
        if args.gpu_gpu:
            parser.error("the batch renderer does not support --gpu-gpu")
        if args.sections:
            parser.error("--sections is for the classic renderer, not --batch-renderer")
    if args.sections and args.parallel == "single":
        parser.error("--sections requires --parallel threaded or vector")
    return args


def query_gpu_memory() -> dict[str, float]:
    """Global GPU memory from nvidia-smi, in MiB. Includes desktop and this process."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    line = out.strip().splitlines()[0]
    total_s, used_s, free_s = [part.strip() for part in line.split(",")]
    return {
        "total_mib": float(total_s),
        "used_mib": float(used_s),
        "free_mib": float(free_s),
    }


def apply_device_env(device: str, gpu_id: int) -> None:
    if device == "gpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
        os.environ.setdefault("HABITAT_SIM_DEFAULT_GPU_ID", str(gpu_id))


def sample_action(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 4))


def run_episode(
    env: habitat.Env,
    rng: np.random.Generator,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[list[float], list[float]]:
    step_times: list[float] = []
    reset_time = 0.0

    t0 = time.perf_counter()
    env.reset()
    reset_time = time.perf_counter() - t0

    total_steps = warmup_steps + measure_steps
    for step_idx in range(total_steps):
        action = sample_action(rng)
        if action == HabitatSimActions.stop:
            action = HabitatSimActions.move_forward

        t_step = time.perf_counter()
        env.step(action)
        elapsed = time.perf_counter() - t_step

        if step_idx >= warmup_steps:
            step_times.append(elapsed)

    return step_times, [reset_time]


def summarize(times_s: list[float]) -> dict[str, float]:
    if not times_s:
        return {}
    ms = [t * 1000.0 for t in times_s]
    return {
        "count": float(len(ms)),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "stdev_ms": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "p95_ms": float(np.percentile(ms, 95)),
    }


DEFAULT_CONFIG = "configs/pointnav_mp3d_sample.yaml"
NO_MEASURES_CONFIG = "configs/pointnav_mp3d_no_measures.yaml"
# Keys composed by benchmark/nav/pointnav/pointnav.yaml. Deleted only when
# --no-measures is combined with a config that still includes them.
_POINTNAV_MEASURES = (
    "distance_to_goal",
    "success",
    "spl",
    "distance_to_goal_reward",
)


def resolve_config_path(args: argparse.Namespace) -> str:
    if args.no_measures and args.config == DEFAULT_CONFIG:
        return NO_MEASURES_CONFIG
    return args.config


def build_config(args: argparse.Namespace, workspace: Path) -> Any:
    # -1 selects the non-GPU renderer path (EGL does not require a CUDA device).
    gpu_id = args.gpu_id if args.device == "gpu" else -1
    gpu_gpu = args.gpu_gpu and args.device == "gpu"

    overrides = [
        "habitat.dataset.split=train",
        "habitat.dataset.content_scenes=[17DRP5sb8fy]",
        "habitat.dataset.scenes_dir=data/scene_datasets",
        "habitat.simulator.scene_dataset=data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_id}",
        f"habitat.simulator.habitat_sim_v0.gpu_gpu={str(gpu_gpu).lower()}",
    ]
    if args.parallel != "single":
        # Random actions should not end the episode mid-timing via success.
        overrides.append("habitat.task.end_on_success=false")
    if args.sections:
        # The utilization sample keeps stepping after the timed window.
        # The PointNav default of 500 would end the episode and hang the worker.
        overrides.append("habitat.environment.max_episode_steps=100000")
    if args.batch_renderer:
        # EnvBatchRenderer draws every env in one ReplayRenderer. Workers must
        # emit keyframes and must not open their own GL contexts. It asserts
        # exactly one visual sensor, so depth is dropped and RGB is kept.
        overrides.extend(
            [
                "habitat.simulator.renderer.enable_batch_renderer=true",
                "habitat.simulator.habitat_sim_v0.enable_gfx_replay_save=true",
                "habitat.simulator.create_renderer=false",
                "habitat.simulator.debug_render=false",
                "~habitat.simulator.agents.main_agent.sim_sensors.depth_sensor",
            ]
        )

    config_rel = resolve_config_path(args)
    # The no-measures yaml already clears the group. Deleting again fails.
    if args.no_measures and Path(config_rel).name != Path(NO_MEASURES_CONFIG).name:
        overrides.extend(
            f"~habitat.task.measurements.{name}" for name in _POINTNAV_MEASURES
        )
        overrides.extend(
            [
                "habitat.task.reward_measure=null",
                "habitat.task.success_measure=null",
            ]
        )

    config_path = config_rel
    if not Path(config_path).is_absolute():
        config_path = str(workspace / config_path)

    config = habitat.get_config(config_path, overrides=overrides)

    with read_write(config):
        config.habitat.seed = args.seed
    return config


def parallel_env_fn_args(config: Any, num_envs: int) -> tuple[tuple[Any], ...]:
    """One independent config per env, copied on the calling thread.

    habitat.ThreadedVectorEnv and habitat.VectorEnv both call
    habitat.gym.make_gym_from_config(config) inside the worker. Copying a
    shared OmegaConf node from those workers is not safe; the habitat-baselines
    factory and test_threaded_vectorized_env copy first, then pass
    env_fn_args=(config,) per env.
    """
    configs = []
    base_seed = int(config.habitat.seed)
    for rank in range(num_envs):
        proc_config = config.copy()
        with read_write(proc_config):
            proc_config.habitat.seed = base_seed + rank
        configs.append(proc_config)
    return tuple((proc_config,) for proc_config in configs)


def sample_actions(rng: np.random.Generator, num_envs: int) -> list[int]:
    actions: list[int] = []
    for _ in range(num_envs):
        action = sample_action(rng)
        if action == HabitatSimActions.stop:
            action = HabitatSimActions.move_forward
        actions.append(action)
    return actions


_BATCH_PHASES = (
    "vector_env_step",
    "batch_post_step",
    "keyframe_apply",
    "draw_gpu_to_cpu",
    "observation_conversion",
)


def install_batch_phase_timers(env: Any) -> None:
    """Split one batch step into simulator, draw, and conversion time.

    VectorEnv.step() is the worker step and the keyframe transfer. post_step()
    applies those keyframes, draws, and writes images back into the
    observations. The draw time is _draw_observations_gpu_to_cpu(), which
    blocks in ReplayRenderer.render() while the GPU finishes and the pixels
    are written into the NumPy buffer. The conversion time is the following
    per-environment slice assignment and sensor-suite pass.
    """
    from habitat.core.batch_rendering.env_batch_renderer_constants import (
        KEYFRAME_OBSERVATION_KEY,
        KEYFRAME_SENSOR_PREFIX,
    )

    renderer = env._batch_renderer
    phases: dict[str, list[float]] = {name: [] for name in _BATCH_PHASES}
    env._batch_phases = phases
    env._batch_phases_record = False
    draw_seconds = {"value": 0.0}

    original_draw = renderer._draw_observations_gpu_to_cpu

    def timed_draw(sensor_spec: Any) -> Any:
        t0 = time.perf_counter()
        result = original_draw(sensor_spec)
        draw_seconds["value"] += time.perf_counter() - t0
        return result

    renderer._draw_observations_gpu_to_cpu = timed_draw

    def timed_post_step(observations: list[Any]) -> list[Any]:
        draw_seconds["value"] = 0.0
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        for env_index in range(renderer._num_envs):
            env_obs = observations[env_index]
            keyframe = env_obs.pop(KEYFRAME_OBSERVATION_KEY)
            renderer._replay_renderer.set_environment_keyframe(env_index, keyframe)
            renderer._replay_renderer.set_sensor_transforms_from_keyframe(
                env_index, KEYFRAME_SENSOR_PREFIX
            )
        keyframe_s = time.perf_counter() - t0

        batch_observations: dict[str, Any] = {}
        for sensor_spec in renderer._sensor_specifications:
            batch_observations[sensor_spec.uuid] = renderer._draw_observations(
                sensor_spec
            )

        t0 = time.perf_counter()
        output: list[Any] = []
        for env_index in range(renderer._num_envs):
            env_observations = observations[env_index]["observation"]
            for sensor_spec in renderer._sensor_specifications:
                env_observations[sensor_spec.uuid] = batch_observations[
                    sensor_spec.uuid
                ][env_index]
            processed_obs = renderer._sensor_suite.get_observations(env_observations)
            for key, value in processed_obs.items():
                env_observations[key] = value
            output.append(env_observations)
        convert_s = time.perf_counter() - t0
        post_s = time.perf_counter() - t_all

        if env._batch_phases_record:
            phases["keyframe_apply"].append(keyframe_s)
            phases["draw_gpu_to_cpu"].append(draw_seconds["value"])
            phases["observation_conversion"].append(convert_s)
            phases["batch_post_step"].append(post_s)
        return output

    renderer.post_step = timed_post_step


def _section_rows(outputs: Any) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not isinstance(outputs, list):
        return rows
    for output in outputs:
        if not isinstance(output, tuple) or len(output) < 4:
            continue
        info = output[3]
        if isinstance(info, dict) and isinstance(info.get("step_sections"), dict):
            rows.append(info["step_sections"])
    return rows


def _aggregate_sections(
    section_rows: list[list[dict[str, float]]],
) -> dict[str, Any]:
    from step_sections import SECTION_KEYS

    def values_for(key: str, rows: list[dict[str, float]]) -> list[float]:
        vals: list[float] = []
        for row in rows:
            if key == "other":
                named = sum(float(row.get(name, 0.0)) for name in SECTION_KEYS)
                vals.append(float(row.get("env_step", 0.0)) - named)
            else:
                vals.append(float(row.get(key, 0.0)))
        return vals

    keys = list(SECTION_KEYS) + ["other", "env_step"]
    per_env: dict[str, float] = {}
    summed: dict[str, float] = {}
    slowest: dict[str, float] = {}
    for key in keys:
        per_env_vals: list[float] = []
        sums: list[float] = []
        maxes: list[float] = []
        for rows in section_rows:
            vals = values_for(key, rows)
            if not vals:
                continue
            per_env_vals.extend(vals)
            sums.append(sum(vals))
            maxes.append(max(vals))
        if not per_env_vals:
            continue
        per_env[key] = statistics.mean(per_env_vals) * 1000.0
        summed[key] = statistics.mean(sums) * 1000.0
        slowest[key] = statistics.mean(maxes) * 1000.0
    return {
        "per_env_mean_ms": per_env,
        "sum_across_envs_ms": summed,
        "slowest_env_ms": slowest,
    }


def _sample_gpu_util(env: Any, rng: np.random.Generator, num_envs: int) -> dict[str, Any]:
    """Step for 2 seconds and read NVML's rolling GPU utilization."""
    from step_sections import GpuUtilSampler

    sampler = GpuUtilSampler()
    if sampler.error:
        return {"error": sampler.error}
    # The NVML counter is a rolling window. Fill it, then keep the later samples.
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        env.step(sample_actions(rng, num_envs))
    sampler.samples.clear()
    sampler.start()
    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        env.step(sample_actions(rng, num_envs))
    sampler.stop()
    summary = sampler.summary()
    if summary is None:
        return {"error": "no NVML samples"}
    return summary


def run_parallel_episode(
    env: Any,
    rng: np.random.Generator,
    num_envs: int,
    warmup_steps: int,
    measure_steps: int,
    batch_renderer: bool = False,
    sections: bool = False,
) -> tuple[list[float], list[float]]:
    t0 = time.perf_counter()
    observations = env.reset()
    if batch_renderer:
        # First draw allocates the GPU-to-CPU buffer. Leave it out of the means.
        env._batch_phases_record = False
        env.post_step(observations)
    reset_time = time.perf_counter() - t0

    step_times: list[float] = []
    section_rows: list[list[dict[str, float]]] = []
    total_steps = warmup_steps + measure_steps
    for step_idx in range(total_steps):
        actions = sample_actions(rng, num_envs)
        if batch_renderer:
            env._batch_phases_record = step_idx >= warmup_steps
        t_step = time.perf_counter()
        outputs = env.step(actions)
        step_s = time.perf_counter() - t_step
        if batch_renderer:
            # Workers return (obs, reward, done, info). The draw is post_step.
            observations = [output[0] for output in outputs]
            env.post_step(observations)
            if env._batch_phases_record:
                env._batch_phases["vector_env_step"].append(step_s)
        elapsed = time.perf_counter() - t_step
        if step_idx >= warmup_steps:
            step_times.append(elapsed)
            if sections:
                section_rows.append(_section_rows(outputs))
    if sections:
        env._section_report = _aggregate_sections(section_rows)
        env._gpu_util = _sample_gpu_util(env, rng, num_envs)
    return step_times, [reset_time]


def open_parallel_env(args: argparse.Namespace, config: Any) -> tuple[Any, float]:
    from habitat.gym import make_gym_from_config

    cls = (
        habitat.ThreadedVectorEnv
        if args.parallel == "threaded"
        else habitat.VectorEnv
    )
    env_fn_args = parallel_env_fn_args(config, args.num_envs)
    if args.batch_renderer:
        # Imported by the forkserver workers. A parent-process patch would not
        # be visible there, because forkserver is a new interpreter.
        from batch_env import make_batch_gym_from_config

        make_env_fn = make_batch_gym_from_config
    elif args.sections and args.gpu_gpu:
        from step_sections import make_section_gpu_gym_from_config

        make_env_fn = make_section_gpu_gym_from_config
    elif args.gpu_gpu:
        from gpu_resident import make_gpu_resident_gym_from_config

        make_env_fn = make_gpu_resident_gym_from_config
    elif args.sections:
        from step_sections import make_section_gym_from_config

        make_env_fn = make_section_gym_from_config
    else:
        make_env_fn = make_gym_from_config
    kwargs: dict[str, Any] = {
        "make_env_fn": make_env_fn,
        "env_fn_args": env_fn_args,
        "auto_reset_done": False,
    }
    # ThreadedVectorEnv does not spawn processes. VectorEnv needs forkserver
    # so worker processes stay compatible with an already-initialized CUDA context.
    if args.parallel == "vector":
        kwargs["multiprocessing_start_method"] = "forkserver"
    t0 = time.perf_counter()
    env = cls(**kwargs)
    if args.batch_renderer:
        env.initialize_batch_renderer(config)
        install_batch_phase_timers(env)
    return env, time.perf_counter() - t0


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    apply_device_env(args.device, args.gpu_id)
    workspace = Path(__file__).resolve().parents[1]
    config = build_config(args, workspace)

    if args.gpu_gpu and args.device == "gpu" and not habitat_sim.cuda_enabled:
        raise RuntimeError(
            "habitat-sim was built without CUDA; rebuild the Docker image "
            "with --with-cuda or omit --gpu-gpu."
        )

    rng = np.random.default_rng(args.seed)
    all_step_times: list[float] = []
    all_reset_times: list[float] = []
    construct_s: float | None = None
    vram_before = query_gpu_memory()
    vram_after_construct: dict[str, float] = {}
    vram_while_alive: dict[str, float] = {}

    if args.parallel == "single":
        if args.gpu_gpu:
            from gpu_resident import prepare_gpu_resident

            prepare_gpu_resident()
        with habitat.Env(config=config) as env:
            vram_after_construct = query_gpu_memory()
            for _ in range(args.episodes):
                step_times, reset_times = run_episode(
                    env, rng, args.warmup_steps, args.measure_steps
                )
                all_step_times.extend(step_times)
                all_reset_times.extend(reset_times)
            vram_while_alive = query_gpu_memory()
    else:
        env, construct_s = open_parallel_env(args, config)
        vram_after_construct = query_gpu_memory()
        try:
            for _ in range(args.episodes):
                step_times, reset_times = run_parallel_episode(
                    env,
                    rng,
                    args.num_envs,
                    args.warmup_steps,
                    args.measure_steps,
                    batch_renderer=args.batch_renderer,
                    sections=args.sections,
                )
                all_step_times.extend(step_times)
                all_reset_times.extend(reset_times)
            vram_while_alive = query_gpu_memory()
        finally:
            env.close()
    vram_after_close = query_gpu_memory()

    step_stats = summarize(all_step_times)
    reset_stats = summarize(all_reset_times)
    num_envs = 1 if args.parallel == "single" else args.num_envs
    mean_ms = step_stats.get("mean_ms", 0.0)
    if mean_ms > 0:
        # Environment steps per second. A parallel step counts every env.
        steps_per_sec = num_envs * 1000.0 / mean_ms
        step_stats = {
            **step_stats,
            "per_env_mean_ms": mean_ms / num_envs,
            "steps_per_sec": steps_per_sec,
            "env_steps_per_sec": steps_per_sec,
        }
    measurements = list(config.habitat.task.measurements.keys())
    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    rgb = sensors.rgb_sensor
    depth_resolution = (
        [int(sensors.depth_sensor.width), int(sensors.depth_sensor.height)]
        if "depth_sensor" in sensors
        else None
    )

    return {
        "device": args.device,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", os.environ.get("HABITAT_CONDA_ENV", "")),
        "gpu_device_id": int(config.habitat.simulator.habitat_sim_v0.gpu_device_id),
        "gpu_gpu": args.gpu_gpu and args.device == "gpu",
        "cuda_enabled": bool(habitat_sim.cuda_enabled),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "(unset)"),
        "config": resolve_config_path(args),
        "no_measures": len(measurements) == 0,
        "measurements": measurements,
        "rgb_resolution": [int(rgb.width), int(rgb.height)],
        "depth_resolution": depth_resolution,
        "batch_renderer": bool(args.batch_renderer),
        "visual_sensors": list(sensors.keys()),
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "episodes": args.episodes,
        "seed": args.seed,
        "parallel": args.parallel,
        "num_envs": num_envs,
        "construct_ms": None if construct_s is None else construct_s * 1000.0,
        "vram_before_mib": vram_before.get("used_mib"),
        "vram_after_construct_mib": vram_after_construct.get("used_mib"),
        "vram_while_alive_mib": vram_while_alive.get("used_mib"),
        "vram_after_close_mib": vram_after_close.get("used_mib"),
        "vram_total_mib": vram_before.get("total_mib"),
        "vram_delta_mib": (
            None
            if "used_mib" not in vram_before or "used_mib" not in vram_while_alive
            else vram_while_alive["used_mib"] - vram_before["used_mib"]
        ),
        "step": step_stats,
        "reset": reset_stats,
        "phases": (
            {
                name: summarize(getattr(env, "_batch_phases", {}).get(name, []))
                for name in _BATCH_PHASES
            }
            if args.batch_renderer and args.parallel != "single"
            else None
        ),
        "sections": getattr(env, "_section_report", None) if args.parallel != "single" else None,
        "gpu_util": getattr(env, "_gpu_util", None) if args.sections else None,
    }


def print_human_report(result: dict[str, Any]) -> None:
    print("PointNav step benchmark (MP3D sample scene 17DRP5sb8fy)")
    measures = result["measurements"]
    measure_label = "(none)" if result["no_measures"] else ",".join(measures)
    print(
        f"device={result['device']} warmup={result['warmup_steps']} "
        f"measured={result['measure_steps']} episodes={result['episodes']} "
        f"measurements={measure_label} "
        f"parallel={result.get('parallel', 'single')} num_envs={result.get('num_envs', 1)} "
        f"batch_renderer={result.get('batch_renderer', False)}"
    )
    rgb = result["rgb_resolution"]
    depth = result["depth_resolution"]
    if depth is None:
        print(f"sensors: rgb={rgb[0]}x{rgb[1]} depth=off")
    else:
        print(f"sensors: rgb={rgb[0]}x{rgb[1]} depth={depth[0]}x{depth[1]}")
    if result["device"] == "gpu":
        print(
            f"GPU: device_id={result['gpu_device_id']} cuda_enabled={result['cuda_enabled']} "
            f"gpu_gpu={result['gpu_gpu']}"
        )
    else:
        print(
            f"CPU mode: habitat_sim_v0.gpu_device_id={result['gpu_device_id']} "
            f"gpu_gpu={result['gpu_gpu']}"
        )

    step = result["step"]
    reset = result["reset"]
    print("\nAggregate step timing:")
    for key in ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "stdev_ms"):
        if key in step:
            print(f"  {key}: {step[key]:.3f}")
    if "steps_per_sec" in step:
        print(f"  steps_per_sec: {step['steps_per_sec']:.1f}")
    if result.get("num_envs", 1) > 1 and "per_env_mean_ms" in step:
        print(f"  per_env_mean_ms: {step['per_env_mean_ms']:.3f}")
    if result.get("construct_ms") is not None:
        print(f"  construct_ms: {result['construct_ms']:.1f}")
    if reset:
        print(f"  reset_mean_ms: {reset['mean_ms']:.3f}")
    print_batch_phases(result.get("phases"))
    print_sections(result.get("sections"), result.get("gpu_util"))


def print_batch_phases(phases: dict[str, Any] | None) -> None:
    if not phases:
        return
    labels = (
        ("vector_env_step", "VectorEnv.step()", "simulators and keyframe transfer"),
        ("batch_post_step", "EnvBatchRenderer.post_step()", "keyframes, draw, conversion"),
        ("keyframe_apply", "  keyframe apply", "replay state from the workers"),
        ("draw_gpu_to_cpu", "  _draw_observations_gpu_to_cpu()", "blocked GPU readback"),
        ("observation_conversion", "  observation conversion", "slice, copy, sensor suite"),
    )
    print("\nBatch phases (measured steps, mean):")
    print(f"  {'phase':<42} {'ms':>8}  ")
    for key, label, _note in labels:
        stats = phases.get(key) or {}
        mean = stats.get("mean_ms")
        if mean is None:
            continue
        print(f"  {label:<42} {mean:8.3f}")


def print_sections(sections: dict[str, Any] | None, gpu_util: dict[str, Any] | None) -> None:
    if not sections:
        return
    labels = (
        ("act_physics", "act + first physics"),
        ("draw", "draw"),
        ("read", "read"),
        ("second_physics", "second physics"),
        ("measures", "measurements"),
        ("other", "other"),
        ("env_step", "env step"),
    )
    print("\nStep sections (measured steps, mean ms):")
    print(f"  {'section':<22} {'one env':>8} {'sum':>8} {'slowest':>8}")
    per_env = sections.get("per_env_mean_ms") or {}
    summed = sections.get("sum_across_envs_ms") or {}
    slowest = sections.get("slowest_env_ms") or {}
    for key, label in labels:
        if key not in per_env:
            continue
        print(
            f"  {label:<22} {per_env[key]:8.3f} {summed.get(key, 0.0):8.3f} "
            f"{slowest.get(key, 0.0):8.3f}"
        )
    if not gpu_util:
        return
    if gpu_util.get("error"):
        print(f"\nGPU utilization: {gpu_util['error']}")
        return
    print(
        "\nGPU utilization over the following 1s of steps: "
        f"mean {gpu_util['gpu_util_mean_pct']:.0f}%  "
        f"max {gpu_util['gpu_util_max_pct']:.0f}%  "
        f"memory {gpu_util['mem_util_mean_pct']:.0f}%"
    )


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)

    if args.json:
        # Keep stdout machine-readable for compare_cpu_gpu.py (logs go to stderr).
        print(json.dumps(result, indent=2), file=sys.stdout)
    else:
        print_human_report(result)


if __name__ == "__main__":
    main()
