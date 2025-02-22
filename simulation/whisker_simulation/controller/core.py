import numpy as np

from whisker_simulation.config import Config
from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import BodyMotionController
from whisker_simulation.controller.deflection_model import DeflectionModel
from whisker_simulation.controller.spline import Spline
from whisker_simulation.controller.tip_estimator import TipEstimator
from whisker_simulation.models import ControllerState, ControlMessage, SensorData
from whisker_simulation.utils import get_logger, get_monitor, normalize, rotate_ccw

__all__ = ["Controller"]

logger = get_logger(__file__)
monitor = get_monitor()


class Controller:
    def __init__(self, *, initial_data: SensorData, config: Config):
        self.config = config

        # runtime
        self.control_dt = 1 / config.control_rps
        self.__state = self.__prev_state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        self.data = self.prev_data = initial_data

        # tip position estimation using deflection model and kalman filter
        self.defl_model = DeflectionModel()
        self.tip_estimator = TipEstimator(self.defl_model, initial_data)
        self.defl_detect_threshold = 5e-5
        self.is_deflected = False

        # tip position prediction using spline
        self.keypoint_distance = 2e-3
        self.n_keypoints = 7
        self.spline = Spline(
            keypoint_distance=self.keypoint_distance,
            n_keypoints=self.n_keypoints,
        )

        # body yaw control
        self.tgt_defl = -3e-4
        self.tilt = 0.2

        # velocity and angle control
        self.total_v = 0.05
        self.motion_ctrl = BodyMotionController(total_v=self.total_v, tilt=self.tilt)

        # disengagement policy
        self.disengaged_duration_threshold = 0.1
        self.exploration_omega = np.pi / 6
        self.prev_spline: Spline | None = None
        self.orientation: int | None = None

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(controller=self)

    @property
    def state(self) -> ControllerState:
        return self.__state

    @state.setter
    def state(self, value: ControllerState) -> None:
        self.__prev_state = self.__state
        logger.info(f"State transition: {self.__state} -> {value}")
        self.__state = value

    @property
    def prev_state(self) -> ControllerState:
        return self.__prev_state

    def control(self, new_data: SensorData) -> ControlMessage | None:
        # rate limit the control
        if new_data.time - self.data.time < self.control_dt:
            return None

        # ATTENTION: time step might be variable due to rate limiting

        # update the controller given the new sensor data
        self.prev_data, self.data = self.data, new_data

        # update the tip position
        tip_w_prev = self.tip_estimator.get_w(self.prev_data)
        self.tip_estimator.update_wr0_yaw_s(self.data)
        tip_w = self.tip_estimator.get_w(self.data)
        self.is_deflected = abs(self.data.wr0_yaw_s) > self.defl_detect_threshold

        # detect anomalies (ignore them if state is EXPLORING)
        anomaly = self.anomaly_detector.run(tip_w=tip_w, tip_w_prev=tip_w_prev, is_deflected=self.is_deflected)
        if anomaly is not None:
            anomaly_state, anomaly_msg = anomaly
            logger.debug(f"Anomaly detected: {anomaly_state}: {anomaly_msg}")

        # exit idle state if the whisker is deflected
        if self.state == ControllerState.EXPLORING:
            if self.is_deflected:
                logger.debug("Whisker has come into contact with the surface")
                assert self.desired_next_state is not None
                self.state = self.desired_next_state
                self.desired_next_state = None
            else:
                return None

        # handle anomalies
        if anomaly is not None and (anomaly_state := anomaly[0]) != self.state:
            self.state = anomaly_state

        # if in failure state, send idle control
        if self.state == ControllerState.FAILURE:
            return ControlMessage(body_vx_w=0, body_vy_w=0, body_omega_w=0)

        if self.state == ControllerState.SWIPING:
            # follow the surface
            return self.policy_swiping(tip_w)

        if self.state == ControllerState.WHISKING:
            # swipe the whisker up to the edge
            return self.policy_whisking(tip_w)

        if self.state == ControllerState.DISENGAGED:
            if self.prev_state == ControllerState.SWIPING:
                # the whisker has disengaged, needs to swipe back
                return self.policy_initiate_whisking()
            if self.prev_state == ControllerState.WHISKING:
                # we swiped the other side of the edge, now we rotate and engage
                return self.policy_reattaching()

        raise NotImplementedError(f"State {self.state} is not implemented")

    def policy_swiping(self, tip_w: np.ndarray) -> ControlMessage | None:
        # 0. If the deflection is too small, keep the control values
        if not self.is_deflected:
            return None

        # 1. Update the spline and predict the next tip position
        is_new_keypoint = self.spline.add_keypoint(keypoint=tip_w, data=self.data)
        if not self.spline:
            return None

        # 2. Calculate spline curvature
        spl_k0_w = self.spline(self.spline.end_kth_point_u(0))
        spl_k1_w = self.spline(self.spline.end_kth_point_u(1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])

        # 3. Calculate the delta offset between the target and current deflection
        zero_defl_offset_l = self.defl_model(0)
        cur_defl_offset_l = self.defl_model(self.data.wr0_yaw_s)
        tgt_defl_offset_l = self.defl_model(self.tgt_defl)
        defl_doffset_w = rotate_ccw(tgt_defl_offset_l - cur_defl_offset_l, self.data.body_yaw_w)
        defl_doffset_w_n = normalize(-defl_doffset_w)
        defl_offset_weight = np.linalg.norm(defl_doffset_w) / np.linalg.norm(tgt_defl_offset_l - zero_defl_offset_l)

        # 4. Choose the target direction as weighted average of the spline and deflection offset
        k = np.clip(defl_offset_weight * 1.5, 0, 1)
        tgt_body_dr_n_w = normalize(defl_doffset_w_n * k + spl_dk_w_n * (1 - k))

        # 5. Calculate the control values
        # for v to follow tgt_body_dr_n_w and for omega to follow the spline slightly tilted
        control = self.motion_ctrl(
            data=self.data,
            prev_data=self.prev_data,
            tgt_body_dr_n_w=tgt_body_dr_n_w,
            tgt_body_yaw_w=spline_angle - self.tilt,
        )

        if self.config.debug and is_new_keypoint:
            ds = self.spline.keypoint_distance * self.spline.n_keypoints
            poi = {
                "body": self.data.body_r_w,
                "d_defl": self.data.body_r_w + defl_doffset_w_n * ds,
                "d_spl": self.data.body_r_w + spl_dk_w_n * ds,
                "d_tgt": self.data.body_r_w + tgt_body_dr_n_w * ds,
            }
            if k < 0.05:
                del poi["d_defl"]
            if k > 0.5:
                monitor.draw_spline(self.spline, **poi)

        return control

    def policy_initiate_whisking(self) -> ControlMessage | None:
        # assume the whisker has disengaged just now
        assert self.prev_state == ControllerState.SWIPING

        # 0. If the spline is not defined, keep the control values
        # There is no better strategy anyway
        if not self.spline:
            logger.warning("Spline is not defined, cannot reengage")
            return None

        # 1. Calculate new control:
        # The goal is to swipe the whisker along the other side of the edge
        # The linear velocity follows the spline backwards,
        # while the rotation brings the whisker tip to the new plane
        spl_start_w, spl_end_w = self.spline(0), self.spline(1)
        spl_k1_w = self.spline(self.spline.end_kth_point_u(1))
        spl_dk_w_n = normalize(spl_k1_w - spl_end_w)

        spl_span_w = spl_end_w - spl_start_w
        nose_yaw_w = self.data.body_yaw_w + np.pi / 2
        body_span_w = np.array([np.cos(nose_yaw_w), np.sin(nose_yaw_w)])
        # orientation > 0 means that the body is on the left side of the spline
        orientation = np.sign(np.cross(spl_span_w, body_span_w))
        self.orientation = orientation

        tgt_body_dr_n_w = rotate_ccw(-spl_dk_w_n, -orientation * np.pi / 4)
        tgt_body_yaw_w = self.exploration_omega * orientation

        control = self.motion_ctrl(
            data=self.data,
            prev_data=self.prev_data,
            tgt_body_dr_n_w=tgt_body_dr_n_w,
            tgt_body_yaw_w=tgt_body_yaw_w,
        )

        # 2. Reset the spline and the tip estimator
        self.tip_estimator.reset()
        self.prev_spline = self.spline
        self.spline = Spline(keypoint_distance=self.keypoint_distance / 10, n_keypoints=self.n_keypoints)

        # 3. Set the desired next state
        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.WHISKING

        return control

    def policy_whisking(self, tip_w: np.ndarray) -> ControlMessage | None:
        # If the whisker has reached over the edge,
        # lock the body position and keep swiping until the whisker is disengaged
        if self.state == ControllerState.WHISKING:
            # if the deflection is too small, keep the control values
            if not self.is_deflected:
                return None
            self.spline.add_keypoint(keypoint=tip_w, data=self.data)
            # TODO: improve whisking
            return ControlMessage(body_vx_w=0, body_vy_w=0, body_omega_w=self.orientation * self.exploration_omega)

    def policy_reattaching(self) -> ControlMessage | None:
        monitor.draw_spline(self.spline)
        raise RuntimeError(f"bool(self.spline) = {bool(self.spline)}")
        # TODO: using the new spline, determine the approach and engage the whisker
