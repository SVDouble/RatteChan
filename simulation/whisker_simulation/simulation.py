import datetime
import math

import mediapy as media
import mujoco
from mujoco import viewer
from tqdm import tqdm

from whisker_simulation.controller import WhiskerController


class WhiskerSimulation:
    def __init__(
        self,
        model_path: str,
        duration: float,
        camera_fps: int,
        control_rps: int,
    ):
        self.model_path = model_path
        self.duration = duration
        self.camera_fps = camera_fps
        self.control_rps = control_rps

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = WhiskerController(self.model.opt.timestep, self.control_rps)

    def control(self, _: mujoco.MjModel, __: mujoco.MjData):
        time = self.data.time
        deflection = self.data.sensor("wr0_deflection").data.item()
        x, y = (
            self.data.sensor("body_x").data.item(),
            self.data.sensor("body_y").data.item(),
        )
        theta = self.data.sensor("body_yaw").data.item()
        control_values = self.controller.control(time, deflection, x, y, theta)
        if control_values is not None:
            body_vx, body_vy, angle = control_values
            self.data.ctrl[0:3] = [body_vx, body_vy, angle]

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
