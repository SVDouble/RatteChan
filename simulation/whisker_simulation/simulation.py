import datetime
import math

import mediapy as media
import mujoco
from mujoco import viewer
from tqdm import tqdm

from whisker_simulation.config import Config
from whisker_simulation.controller import Controller
from whisker_simulation.models import SensorData


class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.model_path = str(config.model_path)
        # noinspection PyArgumentList
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.control_rps = config.control_rps
        initial_data = self.get_sensor_data_from_mujoco()
        self.controller = Controller(
            initial_data=initial_data,
            config=self.config,
        )

        self.duration = config.recording_duration
        self.camera_fps = config.recording_camera_fps

    def control(self, _: mujoco.MjModel, __: mujoco.MjData):
        sensor_data = self.get_sensor_data_from_mujoco()
        control = self.controller.control(sensor_data)
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

    def run(self):
        # set the control function
        mujoco.set_mjcb_control(self.control)

        # set the initial control values
        self.data.ctrl[1] = self.controller.total_v

        # launch the viewer
        mujoco.viewer.launch(self.model, self.data)

    def get_sensor_data_from_mujoco(self) -> SensorData:
        # noinspection PyTypeChecker
        fields: dict = SensorData.model_fields
        data = {
            sensor: self.data.sensor(sensor).data.item()
            for sensor in fields.keys() - {"time"}
        }
        return SensorData(**data, time=self.data.time)
