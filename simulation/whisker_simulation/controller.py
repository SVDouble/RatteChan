from collections import deque

import numpy as np
import scipy.interpolate as interpolate
from filterpy.kalman import KalmanFilter

from whisker_simulation.deflection_model import DeflectionModel

__all__ = ["WhiskerController"]


class WhiskerTipXYFilter(KalmanFilter):
    def __init__(self, dim=2, init_state=(-0.015, 0.098), init_var=10, q_scale=0.01):
        super().__init__(dim_x=dim, dim_z=dim)
        self.x = np.array(init_state)
        self.H = np.eye(dim)
        self.P *= init_var
        self.Q = np.eye(dim) * q_scale


class WhiskerController:
    def __init__(self, control_rps: int):
        self.control_period = 1 / control_rps

        # tip position estimation using deflection model and kalman filter
        self.deflection_model = DeflectionModel()
        self.tip_xy_filter = WhiskerTipXYFilter()
        self.tip_raw_xy_deque = deque(maxlen=20)

        # tip position prediction using spline
        self.spline_degree = 3
        self.n_keypoints = 10
        self.n_knots = 7
        self.min_keypoint_distance = 0.01  # TODO: use velocity to determine this
        self.next_tip_u = 1 + 1 / (self.n_keypoints - 1)
        self.keypoints = deque(maxlen=self.n_keypoints)

        self.touch_index = 0
        self.last_control_time = 0

        self.wrap_count = 0
        self.last_measured = 0.5 * np.pi
        self.last_desired = 0.5 * np.pi
        self.SERVOING_RATIO = 5
        self.STABLE_DISTANCE = 100
        self.keypoint_interval = 6

        # PI
        self.X_VEL = 0.01
        self.TOTAL_VEL = 0.2
        self.PI_scale_bound = 15
        self.DEF_TARGET = -3.2e-4
        self.GAIN_P = 300000
        self.GAIN_I = 1000
        self.def2Target_integral = 0
        self.dt = 1.0

    def control(self, time, deflection, x, y, a):
        # if not enough time has passed, keep the control values
        if time - self.last_control_time < self.control_period:
            return
        self.last_control_time = time

        # if the deflection is too small, keep the control values
        if abs(deflection) < 1e-4:
            return
        else:
            self.touch_index += 1

        # calculate the whisker tip position
        tip = self.get_tip_position(deflection, x, y, a)

        # predict next tip position
        next_tip = self.predict_next_tip_position(tip)
        if next_tip is None:
            return

        # calculate the target rotation
        rotation = self.get_target_rotation(tip, next_tip, self.touch_index)

        # calculate the control values
        xvel_s, yvel_s = self.compute_translation(deflection)  # ???
        tw = rotation - 0.5 * np.pi
        xvel_w = xvel_s * np.cos(tw) - yvel_s * np.sin(tw)
        yvel_w = xvel_s * np.sin(tw) + yvel_s * np.cos(tw)
        if self.touch_index >= self.STABLE_DISTANCE:
            return xvel_w, yvel_w

    def get_tip_position(self, deflection, x, y, a):
        """Get the tip position in world coordinates"""

        # get raw, local tip position from deflection
        tip = self.deflection_model.get_position(deflection)

        # filter the tip position using the kalman filter
        self.tip_raw_xy_deque.append(tip)
        if len(self.tip_raw_xy_deque) > 1:
            self.tip_xy_filter.R = np.cov(np.array(self.tip_raw_xy_deque), rowvar=False)
        else:
            self.tip_xy_filter.x = tip
        self.tip_xy_filter.predict()
        self.tip_xy_filter.update(tip)
        tip = self.tip_xy_filter.x.copy()

        # transform the tip position to world coordinates
        c, s = np.cos(a), np.sin(a)
        tip = np.array([c * tip[0] - s * tip[1] + x, s * tip[0] + c * tip[1] + y])

        # return the filtered tip position in world coordinates
        return tip

    def predict_next_tip_position(self, tip):
        # If no key points exist, add the first one
        if not self.keypoints:
            self.keypoints.append(tip)
        else:
            last_tip = np.array(self.keypoints[-1])
            if np.linalg.norm(np.array(tip) - last_tip) >= self.min_keypoint_distance:
                self.keypoints.append(tip)
        if len(self.keypoints) <= self.spline_degree:
            return None

        keypoints = np.array(self.keypoints)
        # Compute arc-length parameterization for the keypoints
        deltas = np.diff(np.array(keypoints), axis=0)
        distances = np.sqrt((deltas**2).sum(axis=1))
        arc_length = np.concatenate(([0], np.cumsum(distances)))
        u = arc_length / arc_length[-1]  # normalized parameter [0,1]
        # Compute interior quantiles from the normalized parameter as knot locations
        quantile_levels = np.linspace(0, 1, self.n_knots)[1:-1]
        knots = np.quantile(u, quantile_levels)
        # Fit a parametric spline through the keypoints with the computed knots
        # noinspection PyTupleAssignmentBalance
        tck, _ = interpolate.splprep(keypoints.T, t=knots, k=self.spline_degree)
        # Evaluate the spline at the desired parameter value (self.u_next)
        predicted = interpolate.splev(self.next_tip_u, tck)
        return np.array([predicted[0], predicted[1]])

    def get_target_rotation(self, tip, next_tip, touch_index):
        # calculate and unwrap the angle between the current and predicted tip
        angle_wrapped = np.arctan2(next_tip[1] - tip[1], next_tip[0] - tip[0])
        angle = np.unwrap(np.array([self.last_measured, angle_wrapped]))[-1]

        # After a stable period, limit rapid changes in orientation
        if touch_index >= self.STABLE_DISTANCE:
            limit = 0.00023 * self.SERVOING_RATIO * self.keypoint_interval
            error = angle - self.last_desired
            # Clamp the error within [-limit, limit]
            error_clamped = np.clip(error, -limit, limit)
            angle = self.last_desired + error_clamped
        self.last_measured = angle
        return angle

    def compute_translation(self, deflection):
        err = deflection - self.DEF_TARGET
        self.def2Target_integral += err * self.dt
        pi_scale = self.GAIN_P * err + self.GAIN_I * self.def2Target_integral
        pi_scale = max(min(pi_scale, self.PI_scale_bound), -self.PI_scale_bound)
        xvel_s = pi_scale * self.X_VEL
        yvel_s = np.sqrt(self.TOTAL_VEL**2 - xvel_s**2)
        return xvel_s, yvel_s
