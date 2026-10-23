"""Keep visual observations in CUDA memory for the step benchmark.

Habitat's gpu2gpu path copies the framebuffer into a CUDA buffer with
cudaMemcpyDeviceToDevice. That path normally allocates the buffer with
torch, which this image does not have. cudaMalloc provides the same
destination pointer. Pixels are not copied to the CPU.
"""

from __future__ import annotations

import ctypes
from typing import Any


def prepare_gpu_resident() -> None:
    import habitat.sims.habitat_simulator.habitat_simulator as lab_sim
    import habitat_sim.simulator as sim_module
    from habitat_sim.sensor import SensorType
    from habitat_sim.sensors.noise_models import make_sensor_noise_model

    sensor_cls = sim_module.Sensor
    if getattr(sensor_cls._initialize_sensor, "_gpu_resident", False):
        return

    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int

    original_init = sensor_cls._initialize_sensor
    original_get = sensor_cls.get_observation

    def _initialize_sensor(self: Any) -> None:
        if not self._spec.gpu2gpu_transfer:
            return original_init(self)
        if self._spec.sensor_type == SensorType.AUDIO:
            return
        if self._sim.renderer is not None:
            self._sim.renderer.bind_render_target(self._sensor_object)

        device_id = int(self._sim.gpu_device)
        err = cudart.cudaSetDevice(device_id)
        if err != 0:
            raise RuntimeError(
                f"cudaSetDevice({device_id}) failed ({err}). "
                "Error 2 is out of memory while creating the CUDA context."
            )

        size = self._sensor_object.framebuffer_size
        # Color is RGBA8. Depth and semantic are one 32-bit channel.
        nbytes = int(size.x) * int(size.y) * 4
        ptr = ctypes.c_void_p()
        err = cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
        if err != 0 or not ptr.value:
            raise RuntimeError(f"cudaMalloc failed ({err})")
        self._buffer = int(ptr.value)

        noise_model_kwargs = self._spec.noise_model_kwargs
        self._noise_model = make_sensor_noise_model(
            self._spec.noise_model,
            {"gpu_device_id": self._sim.gpu_device, **noise_model_kwargs},
        )

    def _get_observation(self: Any) -> Any:
        if not self._spec.gpu2gpu_transfer:
            return original_get(self)
        if self._spec.sensor_type == SensorType.AUDIO:
            return self._get_audio_observation()
        target = self._sensor_object.render_target
        if self._spec.sensor_type == SensorType.DEPTH:
            target.read_frame_depth_gpu(self._buffer)
        elif self._spec.sensor_type == SensorType.SEMANTIC:
            target.read_frame_object_id_gpu(self._buffer)
        else:
            target.read_frame_rgba_gpu(self._buffer)
        return self._buffer

    _initialize_sensor._gpu_resident = True  # type: ignore[attr-defined]
    sensor_cls._initialize_sensor = _initialize_sensor
    sensor_cls.get_observation = _get_observation

    def _keep_device_pointer(original: Any) -> Any:
        def get_observation(self: Any, sim_obs: dict[str, Any]) -> Any:
            obs = sim_obs.get(self.uuid, None)
            lab_sim.check_sim_obs(obs, self)
            if isinstance(obs, int):
                return obs
            return original(self, sim_obs)

        return get_observation

    lab_sim.HabitatSimRGBSensor.get_observation = _keep_device_pointer(
        lab_sim.HabitatSimRGBSensor.get_observation
    )
    lab_sim.HabitatSimDepthSensor.get_observation = _keep_device_pointer(
        lab_sim.HabitatSimDepthSensor.get_observation
    )


def make_gpu_resident_gym_from_config(config: Any, dataset: Any = None) -> Any:
    prepare_gpu_resident()
    from habitat.gym import make_gym_from_config

    if dataset is None:
        return make_gym_from_config(config)
    return make_gym_from_config(config, dataset)
