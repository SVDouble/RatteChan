from collections import deque

import numpy as np
import scipy.interpolate as interpolate
from filterpy.kalman import KalmanFilter

from whisker_simulation.deflection_model import DeflectionModel
from whisker_simulation.utils import PID

__all__ = ["WhiskerController"]


class WhiskerController:
    def __init__(self, dt: float, control_rps: int):
        self.control_period = 1 / control_rps

        # tip position estimation using deflection model and kalman filter
        self.deflection_model = DeflectionModel()
        self.tip_xy_filter = KalmanFilter(dim_x=2, dim_z=2)
        self.tip_xy_filter.H = np.eye(2)
        self.tip_xy_filter.P *= 10
        self.tip_xy_filter.Q = np.eye(2) * 0.01
        self.tip_raw_xy_deque = deque(maxlen=20)

        # tip position prediction using spline
        self.spline_degree = 3
        self.n_keypoints = 7
        self.n_knots = 5
        self.min_keypoint_distance = 1e-2  # TODO: use velocity to determine this
        self.keypoints = deque(maxlen=self.n_keypoints)
        self.spline = None
        self.spline_last_body = None

        # velocity and angle control
        self.last_control_time = 0
        self.total_velocity = 0.25
        self.target_deflection_range = np.array([-2e-4, -3.5e-4])
        self.yaw_angle_range = np.array([0, np.pi / 4])
        self.deflection_detection_threshold = 1e-6
        self.pid_deflection = PID(kp=3000, ki=10, kd=0, dt=dt, out_limits=(-self.total_velocity, self.total_velocity))

        self.angular_velocity_limit = 4 * np.pi
        self.pid_body_yaw = PID(
            kp=10, ki=0.01, kd=0.1, dt=dt, out_limits=(-self.angular_velocity_limit, self.angular_velocity_limit)
        )
        self.target_body_yaw = 1e-6

    def control(self, time, deflection, x, y, yaw):
        # if not enough time has passed, keep the control values
        if time - self.last_control_time < self.control_period:
            return
        self.last_control_time = time

        # if the deflection is too small, keep the control values
        if abs(deflection) < self.deflection_detection_threshold:
            return

        # calculate the whisker tip position
        body = np.array([x, y])
        tip_now = self.get_tip_position(deflection, body, yaw)

        # update the spline and predict the next tip position
        self.update_tip_spline(tip_now, body)
        if not self.spline:
            return

        # set the target deflection based on the spline curvature
        target_body_yaw = self.get_target_body_yaw()
        target_deflection = np.interp(target_body_yaw, self.yaw_angle_range, self.target_deflection_range)

        # body velocity is relative to the body frame, so we need to rotate it to the world frame
        body_vx_s = self.pid_deflection(-(float(target_deflection) - deflection))
        body_vy_s = np.sqrt(self.total_velocity ** 2 - body_vx_s ** 2)
        body_v_w = self.rotate_ccw(np.array([body_vx_s, body_vy_s]), target_body_yaw)
        body_omega = self.pid_body_yaw(target_body_yaw - yaw)
        return body_v_w[0], body_v_w[1], body_omega

    def get_tip_position(self, deflection, body, yaw):
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
        try:
            self.tip_xy_filter.update(tip)
        except np.linalg.LinAlgError:
            pass
        tip = self.tip_xy_filter.x.copy()

        # transform the tip position to world coordinates
        return self.rotate_ccw(tip, yaw) + body

    def update_tip_spline(self, new_tip, body):
        # add the new tip point
        has_new_point = False
        if not self.keypoints:
            self.keypoints.append(new_tip)
            self.spline_last_body = body
            has_new_point = True
        else:
            last_tip = np.array(self.keypoints[-1])
            tip_d = np.linalg.norm(np.array(new_tip) - last_tip)
            body_d = np.linalg.norm(body - self.spline_last_body)
            if min(tip_d, body_d) >= self.min_keypoint_distance:
                self.keypoints.append(new_tip)
                self.spline_last_body = body
                has_new_point = True
        if len(self.keypoints) <= self.spline_degree:
            return has_new_point
        if not has_new_point:
            return False

        # update the spline
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
        tck, _ = interpolate.splprep(keypoints.T, t=knots, k=self.spline_degree, s=0.1)
        self.spline = tck
        return True

    def spline_estimate_tip(self, u) -> np.ndarray:
        return interpolate.splev(u, self.spline)

    def spline_future_point(self, k):
        return 1 + k / (len(self.keypoints) - 1)

    def get_target_body_yaw(self):
        # get predicted tip positions
        tip_from = self.spline_estimate_tip(self.spline_future_point(-1))
        tip_to = self.spline_estimate_tip(self.spline_future_point(1))

        # calculate and unwrap the angle between the first and predicted tip
        angle_wrapped = (
            np.arctan2(tip_to[1] - tip_from[1], tip_to[0] - tip_from[0]) - np.pi / 2
        )
        angle = np.unwrap(np.array([self.target_body_yaw, angle_wrapped]))[-1]
        limit = 0.0015
        self.target_body_yaw += np.clip(angle - self.target_body_yaw, -limit, limit)
        return self.target_body_yaw

    @staticmethod
    def rotate_ccw(v, theta):
        # noinspection PyPep8Naming
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R @ v

    @staticmethod
    def normalize_and_scale(v, scale):
        norm = np.linalg.norm(v)
        return v * (scale / norm) if norm > 0 else np.array([0.0, 0.0])

    def draw_spline(self, u: np.ndarray, **kwargs: np.ndarray):
        if self.spline is None:
            return

        import matplotlib.pyplot as plt

        u_fine = np.linspace(0, 1, 100)
        spline_points = interpolate.splev(u_fine, self.spline)
        predicted = interpolate.splev(u, self.spline)
        plt.figure()
        plt.plot(spline_points[0], spline_points[1], "r-", label="Spline")
        keypoints = np.array(self.keypoints)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="b", label="Keypoints")
        plt.scatter(
            predicted[0], predicted[1], c="g", marker="*", s=100, label="Predicted"
        )
        for i, (key, p) in enumerate(kwargs.items()):
            plt.scatter(p[0], p[1], c="cmykw"[i], marker="x", s=100, label=key.title())
        plt.legend()
        plt.title("Spline Fit to Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()
