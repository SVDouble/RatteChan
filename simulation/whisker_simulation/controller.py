from collections import deque

import numpy as np
import scipy.interpolate as interpolate
from filterpy.kalman import KalmanFilter

from whisker_simulation.deflection_model import DeflectionModel
from whisker_simulation.models import Control, WorldState, Mode
from whisker_simulation.pid import PID
from whisker_simulation.utils import get_monitor, get_logger

__all__ = ["Controller"]

logger = get_logger(__file__)
monitor = get_monitor()


class Controller:
    def __init__(self, *, initial_state: WorldState, dt: float, control_rps: int):
        self.time_step = dt
        self.control_period = 1 / control_rps

        # runtime
        self.mode = Mode.IDLE
        self.state = initial_state
        self.last_state = initial_state
        self.last_control_time = 0

        # tip position estimation using deflection model and kalman filter
        self.deflection_model = DeflectionModel()
        self.tip_xy_filter = KalmanFilter(dim_x=2, dim_z=2)
        self.tip_xy_filter.H = np.eye(2)
        self.tip_xy_filter.P *= 10
        self.tip_xy_filter.Q = np.eye(2) * 0.01
        self.tip_raw_xy_deque = deque(maxlen=20)

        # tip position prediction using spline
        self.spline_degree = 3
        self.spline_n_keypoints = 7
        self.spline_n_knots = 5
        self.spline_smoothness = 0.1
        self.keypoint_distance = 1e-3  # TODO: use velocity to determine this
        self.keypoints = deque(maxlen=self.spline_n_keypoints)
        self.spline = None
        self.spline_last_body = None

        # velocity and angle control
        self.total_velocity = 0.5
        self.target_deflection = -3.2e-4
        self.deflection_detection_threshold = 5e-5
        self.deflected_whisker_dims = self.deflection_model.get_position(
            self.target_deflection
        )
        self.pid_deflection = PID(
            kp=3000,
            ki=10,
            kd=0,
            dt=dt,
            out_limits=(-self.total_velocity, self.total_velocity),
        )

        self.pid_body_yaw = PID(
            kp=10,
            ki=0.001,
            kd=0,
            dt=dt,
            out_limits=(-2 * np.pi, 2 * np.pi),
        )
        self.target_body_yaw = 1e-6
        self.body_yaw_step_limit = 0.003

    def control(self, state: WorldState) -> Control | None:
        self.state = state

        # rate limit the control
        if self.state.time - self.last_control_time < self.control_period:
            return None
        self.last_control_time = self.state.time

        # exit idle mode if the whisker is deflected
        is_deflected = (
            abs(self.state.wr0_deflection) < self.deflection_detection_threshold
        )
        if self.mode == Mode.IDLE and is_deflected:
            self.mode = Mode.ENGAGED
        else:
            return None

        # detect anomalies
        tip_now = self.get_tip_position()

        anomaly = self.detect_anomaly(tip_now)
        if anomaly is not None and anomaly != self.mode:
            logger.warning(f"Anomaly detected: {anomaly}")
            self.mode = anomaly

            # the anomaly has just been detected, introduce the countermeasures
            # TODO

        # if in failure mode, send idle control
        if self.mode == Mode.FAILURE:
            return Control(body_vx=0, body_vy=0, body_omega=0)

        if self.mode == Mode.ENGAGED:
            # if the deflection is too small, keep the control values
            if is_deflected:
                return None
            # swipe the whisker along the surface
            return self.follow_spline(tip_now)

        # the whisker is disengaged
        # TODO

    def detect_anomaly(self, tip_now: np.ndarray) -> Mode | None:
        # assume that when the system has exited idle, steady body velocity has been reached
        body_dr = self.state.body_r - self.last_state.body_r
        body_ds = np.linalg.norm(body_dr)
        expected_body_ds = self.total_velocity * self.time_step
        if (diff_p := abs(body_ds / expected_body_ds - 1)) > 0.1:
            logger.error(f"Expected body velocity differs from actual by {diff_p:.2%}")
            return Mode.FAILURE

        # now we are sure that the body has covered some distance
        # check whether the whisker is slipping (sliding backwards)
        # control might not be respected, so rely on the previous world state
        tip_dr = tip_now - self.last_state.tip_position
        tip_ds = np.linalg.norm(tip_dr)
        # the tip might be stuck, so ignore small movements
        if tip_ds / expected_body_ds >= 0.2:
            if np.dot(body_dr, tip_dr) < 0:
                logger.warning("Whisker is slipping backwards")
                return Mode.SLIPPING

        # the whisker might be slipping forward, but that's alright
        # still, we want to detect this
        # the indicator is that the tip is moving faster than the body
        if tip_ds / body_ds - 1 > 0.5:
            logger.warning("Whisker is slipping forwards")

        # the whisker might become disengaged
        # the indicator is that the tip is moving faster than the body
        # and the deflection is vanishing
        # TODO
        return None


    def follow_spline(self, tip_now: np.ndarray) -> Control | None:
        # update the spline and predict the next tip position
        self.update_tip_spline(tip_now)
        if not self.spline:
            return None

        # estimate the control values
        target_body_yaw = self.get_target_body_yaw()
        body_v = self.get_target_body_velocity(
            self.state.wr0_deflection, target_body_yaw
        )
        body_omega = self.pid_body_yaw(target_body_yaw - self.state.body_yaw)
        return Control(
            body_vx=float(body_v[0]), body_vy=float(body_v[1]), body_omega=body_omega
        )

    def get_tip_position(self):
        """Get the tip position in world coordinates"""

        # get raw, local tip position from deflection
        deflection = self.state.wr0_deflection
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
        return self.rotate_ccw(tip, self.state.body_yaw) + self.state.body_r

    def update_tip_spline(self, new_tip):
        has_new_point = False
        # add the new tip point
        if self.keypoints:
            last_tip = np.array(self.keypoints[-1])
            tip_d = np.linalg.norm(np.array(new_tip) - last_tip)
            body_d = np.linalg.norm(self.state.body_r - self.spline_last_body)
            if min(tip_d, body_d) >= self.keypoint_distance:
                has_new_point = True
        if not self.keypoints:
            has_new_point = True
        if has_new_point:
            self.keypoints.append(new_tip)
            self.spline_last_body = self.state.body_r
            monitor.add_keypoint(self.state.time, new_tip)
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
        quantile_levels = np.linspace(0, 1, self.spline_n_knots)[1:-1]
        knots = np.quantile(u, quantile_levels)
        # Fit a parametric spline through the keypoints with the computed knots
        # noinspection PyTupleAssignmentBalance
        tck, _ = interpolate.splprep(
            keypoints.T, t=knots, k=self.spline_degree, s=self.spline_smoothness
        )
        self.spline = tck
        return True

    def spline_interpolate(self, u: float) -> np.ndarray:
        return interpolate.splev(u, self.spline)

    def spline_end_kth_point_u(self, k: float) -> float:
        return 1 + k / (len(self.keypoints) - 1)

    def get_target_body_yaw(self) -> float:
        # get predicted tip positions
        tip_from = self.spline_interpolate(self.spline_end_kth_point_u(-1))
        tip_to = self.spline_interpolate(self.spline_end_kth_point_u(1))

        # calculate and unwrap the angle between the first and predicted tip
        angle_wrapped = (
            np.arctan2(tip_to[1] - tip_from[1], tip_to[0] - tip_from[0]) - np.pi / 2
        )
        angle = np.unwrap(np.array([self.target_body_yaw, angle_wrapped]))[-1]
        limit = self.body_yaw_step_limit
        self.target_body_yaw += np.clip(angle - self.target_body_yaw, -limit, limit)
        return self.target_body_yaw

    def get_target_body_velocity(self, deflection, target_body_yaw):
        # why -deflection? it just works
        body_vx_s = self.pid_deflection(-(self.target_deflection - deflection))
        body_vy_s = np.sqrt(self.total_velocity**2 - body_vx_s**2)
        return self.rotate_ccw(np.array([body_vx_s, body_vy_s]), target_body_yaw)

    @staticmethod
    def rotate_ccw(v: np.ndarray, theta: float) -> np.ndarray:
        # noinspection PyPep8Naming
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R @ v
