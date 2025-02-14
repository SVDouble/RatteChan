from collections import deque

import numpy as np
import scipy.interpolate as interpolate

from whisker_simulation.deflection_model import DeflectionModel
from whisker_simulation.kalman_filter import WhiskerKalmanFilter


__all__ = ["WhiskerController"]


class WhiskerController:
    def __init__(self, control_rps: int):
        self.filter = WhiskerKalmanFilter()
        self.control_period = 1 / control_rps
        self.tip_pos_s_deq = deque(maxlen=20)
        self.tip_pos_w_filtered_que = []
        self.loop_index = 0
        self.touch_index = 0
        self.last_control_time = 0

        self.wrap_count = 0
        self.last_measured = 0.5 * np.pi
        self.last_desired = 0.5 * np.pi
        self.SERVOING_RATIO = 5
        self.STABLE_DISTANCE = 100
        self.keypoint_interval = 6
        self.keypoint_length = 10
        self.n_interior_knots = 5
        self.spline_degree = 3
        self.u_next = 1 + 1 / (self.keypoint_length - 1)
        self.keypoints_deq = deque(maxlen=self.keypoint_length)
        # PI
        self.X_VEL = 0.01
        self.TOTAL_VEL = 0.2
        self.PI_scale_bound = 15
        self.DEF_TARGET = -3.2e-4
        self.GAIN_P = 300000
        self.GAIN_I = 1000
        self.def2Target_integral = 0
        self.dt = 1.0

    def control(self, time, deflection, x, y, theta):
        # if not enough time has passed, keep the control values
        if time - self.last_control_time < self.control_period:
            return
        self.last_control_time = time

        # if the deflection is too small, keep the control values
        if abs(deflection) < 1e-4:
            return

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
        next_desired, next_measured = self.compute_rotation(
            self.tip_pos_w_filtered_que, self.touch_index
        )
        # if next_measured is not None:
        #     self.df_theta_next_measured.append(next_measured)
        # self.df_theta_next_desired.append(next_desired)
        # self.df_deflection_moment.append(deflection)
        xvel_s, yvel_s = self.compute_translation(deflection)
        tw = next_desired - 0.5 * np.pi
        xvel_w = xvel_s * np.cos(tw) - yvel_s * np.sin(tw)
        yvel_w = xvel_s * np.sin(tw) + yvel_s * np.cos(tw)
        if self.touch_index >= self.STABLE_DISTANCE:
            return xvel_w, yvel_w, tw

    def refine_orientation(self, raw_angle, touch_index):
        if raw_angle - self.last_measured > 2 * np.pi:
            self.wrap_count += 1
        elif raw_angle - self.last_measured < -2 * np.pi:
            self.wrap_count -= 1
        refined = raw_angle - self.wrap_count * 2 * np.pi
        if touch_index >= self.STABLE_DISTANCE:
            limit = 0.00023 * self.SERVOING_RATIO * self.keypoint_interval
            err = refined - self.last_desired
            if (
                (limit < err < 0.5 * np.pi)
                or (-1.5 * np.pi > err > -2 * np.pi + limit)
                or (-0.5 * np.pi >= err >= -np.pi)
                or (np.pi <= err <= 1.5 * np.pi)
            ):
                refined = self.last_desired + limit
            elif (
                (-limit > err > -0.5 * np.pi)
                or (1.5 * np.pi < err < 2 * np.pi - limit)
                or (0.5 * np.pi <= err <= np.pi)
                or (-np.pi >= err >= -1.5 * np.pi)
            ):
                refined = self.last_desired - limit
        return refined

    def compute_rotation(self, tip_pos_list, touch_index):
        if touch_index % self.keypoint_interval == 0:
            self.keypoints_deq.append(tip_pos_list[-1])
            if len(self.keypoints_deq) == self.keypoint_length:
                xk, yk = np.array(self.keypoints_deq).T
                qs = np.linspace(0, 1, self.n_interior_knots + 2)[1:-1]
                knots = np.quantile(yk, qs)
                tck, u = interpolate.splprep([xk, yk], t=knots, k=self.spline_degree)
                pred = interpolate.splev(self.u_next, tck)
                raw_angle = np.arctan2(pred[1] - yk[-1], pred[0] - xk[-1])
                refined = self.refine_orientation(raw_angle, touch_index)
                self.last_measured = raw_angle
                self.last_desired = refined
        return self.last_desired, self.last_measured

    def compute_translation(self, deflection):
        err = deflection - self.DEF_TARGET
        self.def2Target_integral += err * self.dt
        pi_scale = self.GAIN_P * err + self.GAIN_I * self.def2Target_integral
        pi_scale = max(min(pi_scale, self.PI_scale_bound), -self.PI_scale_bound)
        xvel_s = pi_scale * self.X_VEL
        yvel_s = np.sqrt(self.TOTAL_VEL**2 - xvel_s**2)
        return xvel_s, yvel_s
