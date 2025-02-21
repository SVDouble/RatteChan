import numpy as np

from whisker_simulation.config import Config
from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import BodyMotionController
from whisker_simulation.controller.deflection_model import DeflectionModel
from whisker_simulation.controller.spline import Spline
from whisker_simulation.controller.tip_estimator import TipEstimator
from whisker_simulation.models import Control, WorldState, Mode
from whisker_simulation.utils import get_monitor, get_logger, normalize, rotate_ccw

__all__ = ["Controller"]

logger = get_logger(__file__)
monitor = get_monitor()


class Controller:
    def __init__(self, *, initial_state: WorldState, config: Config):
        self.config = config

        # runtime
        self.control_period = 1 / config.control_rps
        self.mode = Mode.IDLE
        self.state = initial_state
        self.prev_state = initial_state
        self.initial_state = initial_state

        # tip position estimation using deflection model and kalman filter
        self.defl_model = DeflectionModel()
        self.tip_estimator = TipEstimator(self.defl_model, self.initial_state)
        self.defl_detect_threshold = 5e-5

        # tip position prediction using spline
        self.spline = Spline()

        # body yaw control
        self.tgt_defl = -3e-4
        self.tilt = 0.2

        # velocity and angle control
        self.total_v = 0.05
        self.body_motion_controller = BodyMotionController(
            total_v=self.total_v, tilt=self.tilt, control_rps=config.control_rps
        )

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(total_v=self.total_v)

    def control(self, state: WorldState) -> Control | None:
        # rate limit the control
        if state.time - self.prev_state.time < self.control_period:
            return None

        # ATTENTION: time step might be variable due to rate limiting

        # before the state got updated, previous state is accessible
        tip_w_prev = self.tip_estimator.get_w(self.prev_state)

        # update the controller state given the current world state
        self.prev_state = self.state
        self.state = state
        self.tip_estimator.update_wr0_yaw_s(self.state)

        # detect anomalies (ignore them if mode is IDLE)
        tip_w = self.tip_estimator.get_w(self.state)
        anomaly = None
        if self.config.detect_anomalies:
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
            return self.policy_swipe_surface(tip_w)

        # the whisker is disengaged
        raise NotImplementedError("Disengaged mode is not implemented")

    def policy_swipe_surface(self, tip_w: np.ndarray) -> Control | None:
        # 1. Update the spline and predict the next tip position
        is_new_keypoint = self.spline.add_keypoint(keypoint=tip_w, state=self.state)
        if not self.spline:
            return None

        # 2. Calculate spline curvature
        spl_k0_w = self.spline(self.spline.end_kth_point_u(0))
        spl_k1_w = self.spline(self.spline.end_kth_point_u(1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])

        # 3. Calculate the delta offset between the target and current deflection
        zero_defl_offset_l = self.defl_model.get_position(0)
        cur_defl_offset_l = self.defl_model.get_position(self.state.wr0_yaw_s)
        tgt_defl_offset_l = self.defl_model.get_position(self.tgt_defl)
        defl_doffset_w = rotate_ccw(
            tgt_defl_offset_l - cur_defl_offset_l, self.state.body_yaw_w
        )
        defl_doffset_w_n = normalize(-defl_doffset_w)
        defl_offset_weight = np.linalg.norm(defl_doffset_w) / np.linalg.norm(
            tgt_defl_offset_l - zero_defl_offset_l
        )

        # 4. Choose the target direction as weighted average of the spline and deflection offset
        k = np.clip(defl_offset_weight * 1.5, 0, 1)
        tgt_body_dr_n_w = normalize(defl_doffset_w_n * k + spl_dk_w_n * (1 - k))

        # 5. Calculate the control values
        # for v to follow tgt_body_dr_n_w and for omega to follow the spline slightly tilted
        control = self.body_motion_controller.control(
            state=self.state,
            prev_state=self.prev_state,
            tgt_body_dr_n_w=tgt_body_dr_n_w,
            spline_angle=spline_angle,
        )

        if self.config.debug and is_new_keypoint:
            ds = self.spline.keypoint_distance * self.spline.n_keypoints
            poi = {
                "body": self.state.body_r_w,
                "d_defl": self.state.body_r_w + defl_doffset_w_n * ds,
                "d_spl": self.state.body_r_w + spl_dk_w_n * ds,
                "d_tgt": self.state.body_r_w + tgt_body_dr_n_w * ds,
            }
            if k < 0.05:
                del poi["d_defl"]
            if k > 0.5:
                monitor.draw_spline(self.spline, **poi)

        return control
