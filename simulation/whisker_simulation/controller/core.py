import numpy as np

from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import BodyMotionController
from whisker_simulation.controller.spline import Spline
from whisker_simulation.controller.tip_estimator import TipEstimator
from whisker_simulation.models import Control, WorldState, Mode
from whisker_simulation.utils import get_monitor, get_logger

__all__ = ["Controller"]

logger = get_logger(__file__)
monitor = get_monitor()


class Controller:
    def __init__(self, *, initial_state: WorldState, control_rps: float):
        # runtime
        self.control_period = 1 / control_rps
        self.mode = Mode.IDLE
        self.state = initial_state
        self.prev_state = initial_state

        def __get_state():
            return self.state

        # tip position estimation using deflection model and kalman filter
        self.tip_estimator = TipEstimator(__get_state)
        self.defl_detect_threshold = 5e-5

        # tip position prediction using spline
        self.spline = Spline(__get_state)

        # body yaw control
        self.tgt_body_yaw_w = 1e-6  # basically 0+, so that unwrapping works properly
        self.body_yaw_step_limit = 0.003

        # velocity and angle control
        self.total_v = 0.05
        self.body_motion_controller = BodyMotionController(
            total_v=self.total_v, control_rps=control_rps, get_state=__get_state
        )

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(total_v=self.total_v)

    def control(self, state: WorldState) -> Control | None:
        # rate limit the control
        if state.time - self.prev_state.time < self.control_period:
            return None

        # ATTENTION: time step might be variable due to rate limiting

        # before the state got updated, previous state is accessible
        tip_w_prev = self.tip_estimator.get_w()

        # update the controller state given the current world state
        self.prev_state = self.state
        self.state = state
        self.tip_estimator.update_wr0_yaw_s()

        # detect anomalies (ignore them if mode is IDLE)
        tip_w = self.tip_estimator.get_w()
        anomaly = self.anomaly_detector.run(
            state=self.state,
            prev_state=self.prev_state,
            tip_w=tip_w,
            tip_w_prev=tip_w_prev,
        )
        if anomaly is not None and anomaly[0] != self.mode:
            anomaly_mode, anomaly_msg = anomaly
            logger.warning(f"Anomaly detected: {anomaly_mode}: {anomaly_msg}")

        # exit idle mode if the whisker is deflected
        is_deflected = abs(self.state.wr0_yaw_s) > self.defl_detect_threshold
        if self.mode == Mode.IDLE:
            if is_deflected:
                logger.info("Whisker has come into contact with the surface")
                self.mode = Mode.ENGAGED
            else:
                return None

        # handle anomalies
        if anomaly is not None and (anomaly_mode := anomaly[0]) != self.mode:
            self.mode = anomaly_mode

            # the anomaly has just been detected, introduce the countermeasures
            # TODO

        # if in failure mode, send idle control
        if self.mode == Mode.FAILURE:
            return Control(body_vx_w=0, body_vy_w=0, body_omega_w=0)

        if self.mode == Mode.ENGAGED:
            # if the deflection is too small, keep the control values
            if not is_deflected:
                return None
            # swipe the whisker along the surface
            logger.info("Following the surface")
            return self.policy_swipe_surface(tip_w)

        # the whisker is disengaged
        raise NotImplementedError("Disengaged mode is not implemented")

    def policy_swipe_surface(self, tip_w: np.ndarray) -> Control | None:
        # 1. Update the spline and predict the next tip position
        self.spline.add_keypoint(tip_w)
        if not self.spline:
            return None

        # 2. Calculate the target body yaw
        # get predicted tip positions
        tip_from = self.spline.interpolate(self.spline.end_kth_point_u(-1))
        tip_to = self.spline.interpolate(self.spline.end_kth_point_u(1))

        # calculate and unwrap the angle between the first and predicted tip
        angle_wrapped = (
            np.arctan2(tip_to[1] - tip_from[1], tip_to[0] - tip_from[0]) - np.pi / 2
        )
        limit = self.body_yaw_step_limit
        angle = np.unwrap(np.array([self.tgt_body_yaw_w, angle_wrapped]))[-1]
        self.tgt_body_yaw_w += np.clip(angle - self.tgt_body_yaw_w, -limit, limit)

        # 3. Calculate the required control values to reach the target body yaw
        return self.body_motion_controller.control(self.tgt_body_yaw_w)
