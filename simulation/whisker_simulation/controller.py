from collections import deque

import numpy as np
import scipy.interpolate as interpolate

__all__ = ['WhiskerController']


class WhiskerController:
    def __init__(self):
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
        yvel_s = np.sqrt(self.TOTAL_VEL ** 2 - xvel_s ** 2)
        return xvel_s, yvel_s
