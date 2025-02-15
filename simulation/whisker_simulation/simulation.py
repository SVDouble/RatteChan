import datetime
import math

import mediapy as media
import mujoco
from mujoco import viewer
from tqdm import tqdm

from whisker_simulation.config import Config
from whisker_simulation.controller import WhiskerController
from whisker_simulation.models import WorldState


class WhiskerSimulation:
    def __init__(self, config: Config):
        self.model_path = str(config.model_path)
        # noinspection PyArgumentList
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.control_rps = config.control_rps
        initial_world_state = self.get_world_state_from_mujoco_data()
        self.controller = WhiskerController(
            initial_state=initial_world_state,
            dt=self.model.opt.timestep,
            control_rps=self.control_rps,
        )

        self.duration = config.recording_duration
        self.camera_fps = config.recording_camera_fps

    def control(self, _: mujoco.MjModel, __: mujoco.MjData):
        world_state = self.get_world_state_from_mujoco_data()
        control = self.controller.control(world_state)
        if control is not None:
            self.data.ctrl[0:3] = [control.body_vx, control.body_vy, control.body_omega]

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
        self.data.ctrl[1] = self.controller.total_velocity

        # launch the viewer
        mujoco.viewer.launch(self.model, self.data)

    def get_world_state_from_mujoco_data(self) -> WorldState:
        # noinspection PyTypeChecker
        fields: dict = WorldState.model_fields
        data = {
            sensor: self.data.sensor(sensor).data.item()
            for sensor in fields.keys() - {"time"}
        }
        return WorldState(**data, time=self.data.time)
