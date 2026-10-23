"""VectorEnv worker entry point for Habitat's batch renderer.

forkserver starts a new interpreter, so patches made in the parent process
are not visible here. This module is imported by each worker.
"""

from __future__ import annotations

from typing import Any


def prepare_batch_worker() -> None:
    """Record gfx-replay keyframes and do not open a GL context.

    habitat.simulator.create_renderer is a lab flag. Habitat-sim's own
    SimulatorConfiguration still defaults create_renderer to True, and the
    lab flag is not copied onto it.

    add_keyframe_to_observations looks up sim.sensors. This habitat-sim build
    exposes the agent sensors as sim._sensors, and each entry is a Python
    wrapper whose scene node is _sensor_object.node.
    """
    import magnum as mn

    import habitat.sims.habitat_simulator.habitat_simulator as hs
    from habitat.core.batch_rendering.env_batch_renderer_constants import (
        KEYFRAME_OBSERVATION_KEY,
        KEYFRAME_SENSOR_PREFIX,
    )

    create_sim_config = hs.HabitatSim.create_sim_config
    if not getattr(create_sim_config, "_batch_no_worker_renderer", False):

        def create_sim_config_without_renderer(self, sensor_suite):
            output = create_sim_config(self, sensor_suite)
            output.sim_cfg.create_renderer = False
            return output

        create_sim_config_without_renderer._batch_no_worker_renderer = True  # type: ignore[attr-defined]
        hs.HabitatSim.create_sim_config = create_sim_config_without_renderer

    add_keyframe = hs.HabitatSim.add_keyframe_to_observations
    if getattr(add_keyframe, "_batch_sensor_fix", False):
        return

    def add_keyframe_to_observations(self, observations):
        assert self.config.enable_batch_renderer
        assert KEYFRAME_OBSERVATION_KEY not in observations
        for sensor_uuid, sensor in self._sensors.items():
            node = sensor._sensor_object.node
            transform = node.absolute_transformation()
            rotation = mn.Quaternion.from_matrix(transform.rotation())
            self.gfx_replay_manager.add_user_transform_to_keyframe(
                KEYFRAME_SENSOR_PREFIX + sensor_uuid,
                transform.translation,
                rotation,
            )
        observations[KEYFRAME_OBSERVATION_KEY] = (
            self.gfx_replay_manager.extract_keyframe()
        )

    add_keyframe_to_observations._batch_sensor_fix = True  # type: ignore[attr-defined]
    hs.HabitatSim.add_keyframe_to_observations = add_keyframe_to_observations


def make_batch_gym_from_config(config: Any, dataset: Any = None) -> Any:
    prepare_batch_worker()
    from habitat.gym import make_gym_from_config

    if dataset is None:
        return make_gym_from_config(config)
    return make_gym_from_config(config, dataset)
