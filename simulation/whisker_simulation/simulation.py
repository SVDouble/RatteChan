import datetime
import math
import time
from functools import cached_property

import mediapy as media
import mujoco
import mujoco.viewer
from tqdm import tqdm

from whisker_simulation.config import Config, MujocoBodyConfig
from whisker_simulation.controller import Controller
from whisker_simulation.demo_assets import generate_demo_assets, has_demo_assets
from whisker_simulation.models import SensorData
from whisker_simulation.preprocessor import DataPreprocessor
from whisker_simulation.utils import get_logger, prettify


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
        if self.config.test_body is not None:
            self.add_dynamic_body(config.test_body)
        self.control_rps = config.control_rps
        self.monitor = self.get_monitor()
        self.preprocessor = DataPreprocessor(config)
        self.controller = Controller(
            initial_data=self.sensor_data,
            config=self.config,
            monitor=self.monitor,
        )

        self.duration = config.recording_duration
        self.camera_fps = config.recording_camera_fps
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

    def record(self):
        renderer = mujoco.Renderer(self.model, width=720, height=512)

        # calculate the steps and their sizes
        dt = self.model.opt.timestep
        frames = []
        total_steps = math.ceil(self.duration / dt)
        camera_step = round(1 / self.camera_fps / dt)
        control_step = round(1 / self.control_rps / dt)
        if camera_step < 1 or control_step < 1:
            raise ValueError("Camera and control FPS are too high.")

        # run the simulation
        pbar = tqdm(total=total_steps, desc="Simulating {:.1f}s".format(self.duration))
        pbar.update(0)
        for step in range(0, total_steps, camera_step):
            renderer.update_scene(self.data, camera="whisker_cam")
            frames.append(renderer.render())
            mujoco.mj_step(self.model, self.data, camera_step)
            pbar.update(step)
        pbar.close()

        # write the video
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Writing video to {timestamp}.mp4")
        media.write_video(f"outputs/{timestamp}.mp4", frames, fps=self.camera_fps)
        print("Done!")

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.is_paused = not self.is_paused

    def add_dynamic_body(self, body_config: MujocoBodyConfig):
        # Add a new body to the worldbody with a unique name
        body = self.spec.worldbody.add_body()
        body.name = body_config.name
        # Add geoms to the body
        for geom in body_config.geoms:
            if geom.type == "mesh":
                self.spec.add_mesh(**geom.mesh.to_kwargs())
            body.add_geom(**geom.to_kwargs())
        # Recompile model & data preserving the state
        self.model, self.data = self.spec.recompile(self.model, self.data)

    def remove_dynamic_body(self, body_name: str):
        # Locate the body by its unique name and delete it if present
        self.spec.detach_body(body_name)
        # Recompile model & data preserving the state
        self.model, self.data = self.spec.recompile(self.model, self.data)

    def run(self):
        if self.config.debug:
            self.logger.info(prettify(self.config))

        self.logger.info(f"Running the simulation with model: {self.model_path}")

        # set the control function
        mujoco.set_mjcb_control(self.control)

        # set the initial control values
        total_v = self.config.body.total_v
        self.data.ctrl[0:3] = total_v * 0, total_v * 1, 0

        # launch the viewer
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
        ) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = self.data.body(self.config.body.mj_body_name).id
            while viewer.is_running():
                if self.is_paused:
                    time.sleep(self.model.opt.timestep)
                    continue

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
