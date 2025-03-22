import time
from functools import cached_property
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from whisker_simulation.config import (
    Config,
    ExperimentConfig,
    MujocoBodyConfig,
    MujocoGeomConfig,
    MujocoMeshConfig,
    RendererConfig,
)
from whisker_simulation.controller import Controller
from whisker_simulation.demo_assets import generate_demo_assets, has_demo_assets
from whisker_simulation.models import SensorData
from whisker_simulation.preprocessor import DataPreprocessor
from whisker_simulation.utils import get_logger, normalize, prettify


class Renderer:
    def __init__(self, model: mujoco.MjModel, config: RendererConfig):
        self.config = config
        self.model: mujoco.MjModel = model
        self.mj_renderer: mujoco.Renderer = mujoco.Renderer(
            self.model,
            width=self.config.width,
            height=self.config.height,
        )
        self.frames = []

    def reset(self, model: mujoco.MjModel):
        self.model = model
        self.mj_renderer = mujoco.Renderer(
            self.model,
            width=self.config.width,
            height=self.config.height,
        )
        self.frames = []

    def render(self, data: mujoco.MjData):
        if len(self.frames) < data.time * self.config.fps:
            self.mj_renderer.update_scene(data, camera=self.config.camera)
            self.frames.append(self.mj_renderer.render())

    def export_video(self, path: Path):
        import mediapy as media

        media.write_video(path, self.frames, fps=self.config.fps)


class Simulation:
    def __init__(self, *, config: Config | None = None):
        if config is None:
            config = Config()
        self.config = config
        self.logger = get_logger(__file__, log_level=config.log_level)

        if not has_demo_assets(self.config):
            if self.config.generate_demo_assets:
                self.logger.info("Generating the missing demo assets...")
                generate_demo_assets(self.config)
                self.logger.info("All demo assets have been generated")
            else:
                self.logger.warning("Some demo assets are missing, the simulation might crash")

        self._sensor_data_last_updated: float = 0
        self._sensor_data_cache: SensorData | None = None

        self.model_path = config.model_path
        # noinspection PyArgumentList
        self.spec: mujoco.MjSpec = mujoco.MjSpec.from_file(str(self.model_path))
        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.control_rps = config.control_rps
        self.monitor = self.get_monitor()
        self.preprocessor = DataPreprocessor(config)
        self.controller = Controller(
            initial_data=self.sensor_data,
            config=self.config,
            monitor=self.monitor,
        )

        self.is_paused: bool = False
        self.is_controlled: bool = True

    @cached_property
    def sensor_data(self) -> SensorData:
        # needs to be cleared every iteration
        # the filter is set to a certain frequency, so we can't update it more than once per control step
        return SensorData.from_mujoco_data(self.data, self.preprocessor, self.config)

    def get_monitor(self):
        from whisker_simulation.monitor import Monitor

        if self.config.use_monitor:
            return Monitor(config=self.config)

        class Dummy:
            def __getattr__(self, name):
                # Return a no-op function for any attribute access.
                return lambda *args, **kwargs: None

        return Dummy()

    def control(self, _: mujoco.MjModel, __: mujoco.MjData):
        if not self.is_controlled:
            return
        try:
            control = self.controller.control(self.sensor_data)
        except Exception as e:
            self.logger.exception(f"Error in control: {e}", exc_info=e)
            self.is_controlled = False
            self.is_paused = True
            return
        if control is not None:
            self.data.ctrl[0:3] = [
                control.body_vx_w,
                control.body_vy_w,
                control.body_omega_w,
            ]

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.is_paused = not self.is_paused

    def _add_mesh(self, mesh: MujocoMeshConfig):
        for mj_mesh in self.spec.meshes:
            if mj_mesh.name == mesh.name:
                return
        self.spec.add_mesh(**mesh.to_kwargs())

    def _add_geom(self, body: mujoco.MjsBody, geom: MujocoGeomConfig):
        for mj_geom in self.spec.geoms:
            if mj_geom.name == geom.name:
                return
        if geom.type == "mesh":
            self._add_mesh(geom.mesh)
        body.add_geom(**geom.to_kwargs())

    def add_body(self, body_config: MujocoBodyConfig) -> mujoco.MjsBody:
        # Check if the body already exists
        for body in self.spec.bodies:
            if body.name == body_config.name:
                return body
        # Add a new body to the worldbody with a unique name
        body = self.spec.worldbody.add_body()
        body.name = body_config.name
        # Add geoms to the body
        for geom in body_config.geoms:
            self._add_geom(body, geom)
        # Recompile model & data preserving the state
        self.model, self.data = self.spec.recompile(self.model, self.data)
        return body

    def remove_body(self, body: mujoco.MjsBody):
        # Locate the body by its unique name and delete it if present
        self.spec.detach_body(body)
        # Recompile model & data preserving the state
        self.model, self.data = self.spec.recompile(self.model, self.data)

    def run(self):
        if self.config.debug:
            self.logger.info(prettify(self.config))

        self.logger.info(f"Running the simulation with model: {self.model_path}")

        for experiment in self.config.experiments:
            self.run_experiment(experiment)

    def run_experiment(self, experiment: ExperimentConfig):
        self.logger.info(f"Initializing the experiment: {experiment.name}")

        # add the test body to the model
        test_body = self.add_body(self.config.bodies[experiment.test_body])

        # initialize the renderer
        renderer = Renderer(self.model, self.config.renderer)

        # set the initial control values
        total_v = self.config.body.total_v
        initial_control = experiment.initial_control
        initial_body_v_w = total_v * normalize(np.array([initial_control.body_vx_w, initial_control.body_vy_w]))
        self.data.ctrl[0:3] = *initial_body_v_w, initial_control.body_omega_w

        # set the control function
        mujoco.set_mjcb_control(self.control)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
        ) as viewer:
            self.logger.info(f"Running the experiment: {experiment.name}")
            start_time = self.data.time
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = self.data.body(self.config.body.mj_body_name).id
            while viewer.is_running():
                if self.data.time - start_time > experiment.timeout:
                    self.logger.info(f"Timeout reached for experiment: {experiment.name}")
                    break

                if self.is_paused:
                    time.sleep(self.model.opt.timestep)
                    continue

                # render the scene for the video
                renderer.render(self.data)

                # reset the sensor data cache
                # noinspection PyPropertyAccess
                del self.sensor_data

                # step the physics
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)

                # call the monitor to draw the trajectories
                self.monitor.on_simulation_step(viewer, self.sensor_data)

                # update the display
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0 and self.config.track_time:
                    time.sleep(time_until_next_step)

        # reset the control
        mujoco.set_mjcb_control(None)

        # export the video
        renderer.export_video(self.config.project_path / "outputs" / "simulation.mp4")

        # remove the test body from the model and reset the data
        self.remove_body(test_body)
        mujoco.mj_resetData(self.model, self.data)

        self.logger.info(f"Finished experiment: {experiment.name}")
