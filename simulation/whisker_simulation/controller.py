from collections import deque
from typing import Tuple, Optional

import numpy as np
import scipy.interpolate as interpolate
from filterpy.kalman import KalmanFilter

from whisker_simulation.deflection_model import DeflectionModel

__all__ = ["WhiskerController"]


class WhiskerTipXYFilter(KalmanFilter):
    def __init__(self, dim=2, init_var=10, q_scale=0.01):
        super().__init__(dim_x=dim, dim_z=dim)
        self.H = np.eye(dim)
        self.P *= init_var
        self.Q = np.eye(dim) * q_scale


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        out_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.dt: float = dt
        self.integral: float = 0.0
        self.last_error: float = 0.0
        self.out_limits: Tuple[Optional[float], Optional[float]] = out_limits

    def __call__(self, error: float) -> float:
        self.integral += error * self.dt
        derivative: float = (error - self.last_error) / self.dt
        output: float = self.kp * error + self.ki * self.integral + self.kd * derivative
        min_out, max_out = self.out_limits
        if min_out is not None:
            output = max(min_out, output)
        if max_out is not None:
            output = min(max_out, output)
        self.last_error = error
        return output


class WhiskerController:
    def __init__(self, dt: float, control_rps: int):
        self.control_period = 1 / control_rps

        # tip position estimation using deflection model and kalman filter
        self.deflection_model = DeflectionModel()
        self.tip_xy_filter = WhiskerTipXYFilter()
        self.tip_raw_xy_deque = deque(maxlen=20)

        # tip position prediction using spline
        self.spline_degree = 3
        self.n_keypoints = 7
        self.n_knots = 5
        self.min_keypoint_distance = 5e-3  # TODO: use velocity to determine this
        self.keypoints = deque(maxlen=self.n_keypoints)
        self.spline = None
        self.spline_last_body = None

        # velocity and angle control
        self.last_control_time = 0
        self.total_velocity = 0.25
        self.target_deflection = -3e-4
        self.deflection_detection_threshold = 1e-5
        self.deflected_whisker_dims = self.deflection_model.get_position(
            self.target_deflection
        )
        self.target_body_yaw = 0.5 * np.pi
        vlims = (-self.total_velocity, self.total_velocity)
        self.pid_x = PID(kp=1, ki=0, kd=0, dt=dt, out_limits=vlims)
        self.pid_y = PID(kp=1, ki=0, kd=0, dt=dt, out_limits=vlims)
        self.pid_yaw = PID(kp=0.1, ki=0, kd=0, dt=dt, out_limits=(-0.5, 0.5))
        self.last_body = None

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
        if not self.update_tip_spline(tip_now, body) or not self.spline:
            return

        tip_anchor = self.spline_estimate_tip(0.5)
        tip_next1 = self.spline_estimate_tip(self.spline_future_point(1))
        tip_next2 = self.spline_estimate_tip(self.spline_future_point(2))

        # calculate the target body position
        # when the body reaches the target deflection,
        # it should be at the angle parallel to (tip_next2 - tip_now)
        # and at the position tip_next1 with the whisker deflection offset

        # calculate and unwrap the angle between the first and predicted tip
        angle_wrapped = (
            np.arctan2(tip_next2[1] - tip_anchor[1], tip_next2[0] - tip_anchor[0]) - np.pi / 2
        )
        angle = np.unwrap(np.array([self.target_body_yaw, angle_wrapped]))[-1]
        # self.target_body_yaw += np.clip(angle - prev_angle, -limit, limit)
        self.target_body_yaw = angle

        rotated_whisker_dims = self.rotate_ccw(
            self.deflected_whisker_dims, self.target_body_yaw
        )
        body_ideal_target = tip_next1 - rotated_whisker_dims
        body_target = body * 0.95 + body_ideal_target * 0.05

        # calculate the control values
        control_omega = self.pid_yaw(self.target_body_yaw - yaw)
        raw_vx = self.pid_x(body_target[0] - x)
        raw_vy = self.pid_y(body_target[1] - y)

        # Scale the translational velocity to have magnitude self.total_velocity
        raw_v = np.array([raw_vx, raw_vy])
        # raw_v = target_body - body
        norm = np.linalg.norm(raw_v)
        control_v = (
            raw_v * (self.total_velocity / norm) if norm > 0 else np.array([0.0, 0.0])
        )
        print(f"body: {body}, target_body: {body_target}")
        print(
            f"offset: {self.deflected_whisker_dims}, rotated_offset: {rotated_whisker_dims}"
        )
        print(f"body yaw: {yaw}, target yaw: {self.target_body_yaw}")
        print(f"control_v: {control_v}, control_omega: {control_omega}")
        print("\n")
        # self.draw_spline(
        #     np.array([self.spline_future_point(1), self.spline_future_point(2)]),
        #     contact=tip_now,
        #     body=body,
        #     target=body_target,
        # )
        #input()
        return control_v[0], control_v[1], 0

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
        self.tip_xy_filter.update(tip)
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


    @staticmethod
    def rotate_ccw(v, theta):
        # noinspection PyPep8Naming
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R @ v

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
