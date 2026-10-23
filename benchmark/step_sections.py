"""Split one PointNav step into CPU work and the GPU draw/read.

Imported from the environment worker. VectorEnv uses forkserver, so a patch
applied only in the parent process would not run.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any


SECTION_KEYS = (
    "act_physics",
    "draw",
    "read",
    "second_physics",
    "measures",
)

_tls = threading.local()
_installed = False


def _bucket() -> dict[str, float]:
    bucket = getattr(_tls, "bucket", None)
    if bucket is None:
        bucket = {}
        _tls.bucket = bucket
    return bucket


def reset_bucket() -> None:
    _tls.bucket = {}


def _add(name: str, seconds: float) -> None:
    bucket = _bucket()
    bucket[name] = bucket.get(name, 0.0) + seconds


def install_step_sections() -> None:
    """Time act, draw, read, the extra physics step, and task measurements."""
    global _installed
    if _installed:
        return

    import habitat_sim.simulator as sim_module
    from habitat.core.embodied_task import Measurements
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

    sensor_cls = sim_module.Sensor
    simulator_cls = sim_module.Simulator

    original_draw = sensor_cls.draw_observation
    original_get = sensor_cls.get_observation
    original_sim_step = simulator_cls.step
    original_physics = HabitatSim.step_physics
    original_measures = Measurements.update_measures

    def draw_observation(self: Any) -> None:
        t0 = time.perf_counter()
        try:
            original_draw(self)
        finally:
            _add("draw", time.perf_counter() - t0)

    def get_observation(self: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original_get(self)
        finally:
            _add("read", time.perf_counter() - t0)

    def sim_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        draw_before = _bucket().get("draw", 0.0)
        read_before = _bucket().get("read", 0.0)
        try:
            return original_sim_step(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            draw = _bucket().get("draw", 0.0) - draw_before
            read = _bucket().get("read", 0.0) - read_before
            _add("act_physics", elapsed - draw - read)

    def step_physics(self: Any, dt: float) -> None:
        t0 = time.perf_counter()
        try:
            original_physics(self, dt)
        finally:
            _add("second_physics", time.perf_counter() - t0)

    def update_measures(self: Any, *args: Any, **kwargs: Any) -> None:
        t0 = time.perf_counter()
        try:
            original_measures(self, *args, **kwargs)
        finally:
            _add("measures", time.perf_counter() - t0)

    sensor_cls.draw_observation = draw_observation
    sensor_cls.get_observation = get_observation
    simulator_cls.step = sim_step
    HabitatSim.step_physics = step_physics
    Measurements.update_measures = update_measures
    _installed = True


class SectionEnv:
    """Copies this step's section times onto the gym info dict."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.reset(*args, **kwargs)

    def step(self, action: Any) -> Any:
        reset_bucket()
        t0 = time.perf_counter()
        result = self._env.step(action)
        env_step = time.perf_counter() - t0
        sections = dict(_bucket())
        sections["env_step"] = env_step
        if isinstance(result, tuple) and len(result) >= 4 and isinstance(result[3], dict):
            result[3]["step_sections"] = sections
        return result

    def close(self) -> None:
        self._env.close()


def _wrap(config: Any, dataset: Any) -> Any:
    install_step_sections()
    from habitat.gym import make_gym_from_config

    if dataset is None:
        env = make_gym_from_config(config)
    else:
        env = make_gym_from_config(config, dataset)
    return SectionEnv(env)


def make_section_gym_from_config(config: Any, dataset: Any = None) -> Any:
    return _wrap(config, dataset)


def make_section_gpu_gym_from_config(config: Any, dataset: Any = None) -> Any:
    from gpu_resident import prepare_gpu_resident

    # Install the CUDA read path first so the section timer wraps it.
    prepare_gpu_resident()
    return _wrap(config, dataset)


class _NvmlUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class GpuUtilSampler:
    """NVML GPU-util samples. The driver reports a short rolling average."""

    def __init__(self) -> None:
        self.samples: list[tuple[int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle = ctypes.c_void_p()
        self.error: str | None = None
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            nvml.nvmlInit_v2.restype = ctypes.c_int
            rc = nvml.nvmlInit_v2()
            if rc != 0:
                self.error = f"nvmlInit_v2 returned {rc}"
                return
            nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            rc = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self._handle))
            if rc != 0:
                self.error = f"nvmlDeviceGetHandleByIndex_v2 returned {rc}"
                return
            nvml.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NvmlUtil),
            ]
            nvml.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._nvml = nvml
        except OSError as exc:
            self.error = str(exc)

    def _loop(self) -> None:
        util = _NvmlUtil()
        while not self._stop.is_set():
            if self._nvml.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)) == 0:
                self.samples.append((int(util.gpu), int(util.memory)))
            self._stop.wait(0.05)

    def start(self) -> None:
        if self._nvml is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict[str, float] | None:
        if not self.samples:
            return None
        gpu = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {
            "samples": float(len(gpu)),
            "gpu_util_mean_pct": sum(gpu) / len(gpu),
            "gpu_util_max_pct": float(max(gpu)),
            "mem_util_mean_pct": sum(memory) / len(memory),
        }
