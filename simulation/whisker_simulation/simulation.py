import datetime
import math
from collections import deque

import mediapy as media
import mujoco
import numpy as np
from tqdm import tqdm

from whisker_simulation.controller import WhiskerController
from whisker_simulation.deflection_model import DeflectionModel
from whisker_simulation.kalman_filter import WhiskerKalmanFilter


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
        self.controller = WhiskerController()
        self.renderer = mujoco.Renderer(self.model, width=720, height=512)

        self.filter = WhiskerKalmanFilter()
        self.tip_pos_s_deq = deque(maxlen=20)
        self.tip_pos_w_filtered_que = []

        self.df_theta_next_measured = []
        self.df_theta_next_desired = []
        self.df_deflection_moment = []
        self.loop_index = 0
        self.touch_index = 0

    def send_control(self):
        deflection = self.data.sensor("base2whisker_z").data.item()
        # if the deflection is too small, keep the control values
        if abs(deflection) < 1e-4:
            return

        x, y = (
            self.data.sensor("whisker_joint_x").data.item(),
            self.data.sensor("whisker_joint_y").data.item(),
        )
        theta = self.data.sensor("whisker_joint_z").data.item()

        self.touch_index += 1
        tx, ty = (
            DeflectionModel.fx(deflection),
            DeflectionModel.fy(deflection),
        )
        self.tip_pos_s_deq.append([tx, ty])
        if len(self.tip_pos_s_deq) == self.tip_pos_s_deq.maxlen:
            self.filter.update_noise_matrices(self.tip_pos_s_deq)
            tip_s_filt = self.filter.predict_and_update(self.tip_pos_s_deq[-1])
        else:
            tip_s_filt = np.array([tx, ty])
        transform = np.array(
            [
                [np.cos(theta), -np.sin(theta), x],
                [np.sin(theta), np.cos(theta), y],
                [0, 0, 1],
            ]
        )
        tip_w = transform @ np.array([[tip_s_filt[0]], [tip_s_filt[1]], [1]])
        self.tip_pos_w_filtered_que.append([tip_w[0, 0], tip_w[1, 0]])
        next_desired, next_measured = self.controller.compute_rotation(
            self.tip_pos_w_filtered_que, self.touch_index
        )
        if next_measured is not None:
            self.df_theta_next_measured.append(next_measured)
        self.df_theta_next_desired.append(next_desired)
        self.df_deflection_moment.append(deflection)
        xvel_s, yvel_s = self.controller.compute_translation(deflection)
        tw = next_desired - 0.5 * np.pi
        xvel_w = xvel_s * np.cos(tw) - yvel_s * np.sin(tw)
        yvel_w = xvel_s * np.sin(tw) + yvel_s * np.cos(tw)
        if self.touch_index >= self.controller.STABLE_DISTANCE:
            self.data.ctrl[0] = xvel_w
            self.data.ctrl[1] = yvel_w
            self.data.ctrl[2] = tw

    def run(self):
        # set the initial control values
        self.data.ctrl[1] = self.controller.TOTAL_VEL

        # calculate the steps and their sizes
        dt = self.model.opt.timestep
        frames = []
        total_steps = math.ceil(self.duration / dt)
        camera_step = round(1 / self.camera_fps / dt)
        control_step = round(1 / self.control_rps / dt)
        if camera_step < 1 or control_step < 1:
            raise ValueError("Camera and control FPS are too high.")

        # run the simulation
        step = 0
        last_control_step, last_camera_step = -math.inf, -math.inf
        pbar = tqdm(total=total_steps, desc="Simulating {:.1f}s".format(self.duration))
        pbar.update(0)
        while step < total_steps:
            # send control and update camera
            if step - last_control_step >= control_step:
                last_control_step = step
                self.send_control()
            if step - last_camera_step >= camera_step:
                last_camera_step = step
                self.renderer.update_scene(self.data, camera="whisker_cam")
                frames.append(self.renderer.render())

            # step_size is the minimum of the remaining steps to the next control or camera step
            step_size = min(
                last_camera_step + camera_step - step,
                last_control_step + control_step - step,
            )

            # step the simulation
            mujoco.mj_step(self.model, self.data, step_size)
            step += step_size
            pbar.update(step_size)
        pbar.close()

        # write the video
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Writing video to {timestamp}.mp4")
        media.write_video(f"outputs/{timestamp}.mp4", frames, fps=self.camera_fps)
        print("Done!")
