import time
from functools import cached_property
from pathlib import Path

import jinja2
import mujoco
import mujoco.viewer
import numpy as np

from whisker_simulation.config import (
    Config,
    ExperimentConfig,
    Flag,
    MujocoBodyConfig,
    MujocoGeomConfig,
    MujocoMeshConfig,
    RendererConfig,
)
from whisker_simulation.contours import extract_contours, plot_contours
from whisker_simulation.controller import Controller
from whisker_simulation.demo_assets import generate_demo_assets, has_demo_assets
from whisker_simulation.models import SensorData
from whisker_simulation.monitor import Monitor
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
            self.mj_renderer.update_scene(data, camera=self.config.platform_camera)
            self.frames.append(self.mj_renderer.render())

    def export_video(self, path: Path):
        import mediapy as media

        media.write_video(path, self.frames, fps=self.config.fps)


class Experiment:
    def __init__(self, *, exp_config: ExperimentConfig, config: Config, spec: mujoco.MjSpec, monitor: Monitor):
        self.config = config
        self.exp_config = exp_config
        self.monitor = monitor
        self.logger = get_logger("Experiment", log_level=config.log_level)
        self.spec: mujoco.MjSpec = spec
        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.preprocessor = DataPreprocessor(config)
        self.controller = Controller(
            initial_data=self.sensor_data,
            config=self.config,
            monitor=monitor,
        )
        self.is_healthy: bool = True
        self.set_initial_control()

    @cached_property
    def sensor_data(self) -> SensorData:
        # needs to be cleared every iteration
        # the filter is set to a certain frequency, so we can't update it more than once per control step
        return SensorData.from_mujoco_data(self.data, self.preprocessor, self.config)

    def control(self, _: mujoco.MjModel, __: mujoco.MjData):
        if not self.is_healthy:
            return
        try:
            control = self.controller.control(self.sensor_data)
        except Exception as e:
            self.logger.exception(f"Error in control: {e}", exc_info=e)
            self.is_healthy = False
            return
        if control is not None:
            self.data.ctrl[0:3] = [
                control.body_vx_w,
                control.body_vy_w,
                control.body_omega_w,
            ]

    def step(self):
        # reset the sensor data cache
        # noinspection PyPropertyAccess
        del self.sensor_data

        # step the physics
        mujoco.mj_step(self.model, self.data)

    def set_initial_control(self):
        # set the initial control values
        total_v = self.config.body.total_v
        initial_control = self.exp_config.initial_control
        initial_body_v_w = total_v * normalize(np.array([initial_control.body_vx_w, initial_control.body_vy_w]))
        self.data.ctrl[0:3] = *initial_body_v_w, initial_control.body_omega_w

    def update_spec(self, spec: mujoco.MjSpec):
        self.spec = spec
        self.model, self.data = self.spec.recompile(self.model, self.data)

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
        self.update_spec(self.spec)
        return body

    def remove_body(self, body: mujoco.MjsBody):
        # Locate the body by its unique name and delete it if present
        self.spec.detach_body(body)
        # Recompile model & data preserving the state
        self.update_spec(self.spec)


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

        self.model_path = config.model_path
        self.environment = jinja2.Environment()
        self.is_paused: bool = False

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.is_paused = not self.is_paused

    def _get_mujoco_spec(
        self,
        path: Path,
        flags: set[Flag],
    ) -> mujoco.MjSpec:
        with open(path) as f:
            template = self.environment.from_string(f.read())
        # noinspection PyArgumentList
        return mujoco.MjSpec.from_string(
            xml=template.render(**{flag.value: True for flag in flags}),
        )

    def run(self):
        if self.config.debug:
            self.logger.info(prettify(self.config))

        self.logger.info(f"Running the simulation with model: {self.model_path}")

        for experiment in self.config.experiments:
            self.run_experiment(experiment)

    def run_experiment(self, exp_config: ExperimentConfig):
        self.logger.info(f"Initializing the experiment: {exp_config.name}")

        # create the experiment
        spec = self._get_mujoco_spec(self.model_path, exp_config.flags)
        monitor = Monitor(config=self.config) if self.config.use_monitor else None
        experiment = Experiment(config=self.config, spec=spec, monitor=monitor, exp_config=exp_config)

        # extract the true test body contours
        contours = self.extract_true_contours(
            self._get_mujoco_spec(self.model_path, exp_config.flags - {Flag.USE_PLATFORM})
        )

        # initialize the renderer
        mujoco.mj_forward(experiment.model, experiment.data)
        renderer = Renderer(experiment.model, self.config.renderer)

        # set the control function
        mujoco.set_mjcb_control(experiment.control)

        with mujoco.viewer.launch_passive(
            experiment.model,
            experiment.data,
            key_callback=self.key_callback,
            show_left_ui=False,
        ) as viewer:
            self.logger.info(f"Running the experiment: {exp_config.name}")
            start_time = experiment.data.time
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = experiment.data.body(self.config.body.mj_body_name).id
            while viewer.is_running():
                if experiment.data.time - start_time > exp_config.timeout:
                    self.logger.info(f"Timeout reached for experiment: {exp_config.name}")
                    break

                if self.is_paused:
                    time.sleep(experiment.model.opt.timestep)
                    continue

                step_start = time.time()
                # render the scene for the video
                renderer.render(experiment.data)
                # step the experiment
                experiment.step()
                # call the monitor to draw the trajectories
                monitor.on_simulation_step(viewer, experiment.sensor_data)
                # update the display
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = experiment.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0 and self.config.track_time:
                    time.sleep(time_until_next_step)

        # reset the control
        mujoco.set_mjcb_control(None)

        # export the video
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        renderer.export_video(self.config.project_path / "outputs" / f"{exp_config.name}-{timestamp}.mp4")

        self.logger.info(f"Finished experiment: {exp_config.name}")

    def extract_true_contours(self, spec: mujoco.MjSpec):
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        width, height = self.config.renderer.test_camera_width, self.config.renderer.test_camera_height
        renderer = mujoco.Renderer(model, width=width, height=height)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=self.config.renderer.test_camera)
        frame = renderer.render()
        contours = extract_contours(frame, center=np.array([0, 0]), width=3, height=3)
        plot_contours(contours)
        return contours
