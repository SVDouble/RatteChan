import numpy as np

from whisker_simulation.config import Config
from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import BodyMotionController
from whisker_simulation.controller.deflection_model import DeflectionModel
from whisker_simulation.controller.spline import Spline
from whisker_simulation.models import ControllerState, ControlMessage, MotionAnalyzer, SensorData
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
        self.motion: MotionAnalyzer = MotionAnalyzer(data=self.data, prev_data=self.prev_data)

        # tip position estimation
        self.defl_model = DeflectionModel()
        self.defl_threshold = 5e-5
        self.wr0_defl_sign: int = 0
        # is set once for every engagement
        self.tgt_wr0_defl_sign: int = self.wr0_defl_sign

        # tip position prediction using spline
        self.keypoint_distance = 2e-3
        self.n_keypoints = 7
        self.spline = Spline(
            keypoint_distance=self.keypoint_distance,
            n_keypoints=self.n_keypoints,
        )

        # body yaw control
        self.tgt_defl_abs = 3e-4
        self.tilt = 0.2

        # velocity and angle control
        self.total_v = 0.05
        self.motion_ctrl = BodyMotionController(total_v=self.total_v, tilt=self.tilt)

        # disengagement policy
        self.disengaged_duration_threshold = 0.1
        self.exploration_omega = np.pi / 10

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(controller=self)
        self.anomaly_blacklist = [(ControllerState.EXPLORING, ControllerState.DISENGAGED)]

        # exploration policy
        self.exploration_instructions = None

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
        self.motion = MotionAnalyzer(data=self.data, prev_data=self.prev_data)

        # update the spline
        self.wr0_defl_sign = np.sign(self.data.wr0_defl) if abs(self.data.wr0_defl) > self.defl_threshold else 0
        if self.wr0_defl_sign:
            has_new_point = self.spline.add_keypoint(keypoint=self.data.tip_r_w, data=self.data)
            if has_new_point and len(self.spline.keypoints) == 1:
                logger.debug("Whisker has come into contact with the surface")

        # detect anomalies (ignore them if state is EXPLORING)
        if anomaly := self.anomaly_detector():
            anomaly_state, anomaly_msg = anomaly
            if (self.state, anomaly_state) not in self.anomaly_blacklist:
                logger.warning(f"Anomaly detected: {anomaly_msg}")
                self.state = anomaly_state

        if self.state == ControllerState.EXPLORING:
            # explore the map according to the exploration instructions
            return self.exploration_policy()

        if self.state == ControllerState.SWIPING:
            # follow the surface
            return self.policy_swiping()

        if self.state == ControllerState.WHISKING:
            # swipe the whisker up to the edge
            return self.policy_whisking()

        if self.state == ControllerState.DISENGAGED:
            if self.prev_state == ControllerState.SWIPING:
                # the whisker has disengaged, needs to swipe back
                return self.policy_initiate_whisking()
            if self.prev_state == ControllerState.WHISKING:
                # we swiped the other side of the edge, now we rotate and engage
                return self.policy_reattaching()

        # if in failure state, send idle control
        if self.state == ControllerState.FAILURE:
            return ControlMessage(body_vx_w=0, body_vy_w=0, body_omega_w=0)

        raise NotImplementedError(f"State {self.state} is not implemented")

    def exploration_policy(self) -> ControlMessage | None:
        def apply_instructions() -> ControlMessage | None:
            if self.exploration_instructions is None:
                return None
            return self.exploration_instructions()

        if not self.wr0_defl_sign or not self.spline:
            self.tgt_wr0_defl_sign = 0
            return apply_instructions()

        # At this point we have a spline to work with
        # This means that exploration has ended and the state should be updated
        logger.debug("The spline has been defined")
        monitor.draw_spline(self.spline, title="Exploration End", body=self.data.body_r_w)
        assert self.desired_next_state is not None
        self.state = self.desired_next_state
        self.desired_next_state = None
        assert self.tgt_wr0_defl_sign == 0
        self.tgt_wr0_defl_sign = self.wr0_defl_sign

        # Use the exploration instructions one last time to let it prepare for the engaged state
        control = apply_instructions()
        self.exploration_instructions = None
        return control

    def policy_swiping(self) -> ControlMessage | None:
        # 1. If the deflection is too small, keep the control values
        if not self.wr0_defl_sign:
            return None

        # 2. Calculate spline curvature
        # be quite conservative with the angle as the most recent points might be unstable
        spl_k0_w = self.spline(self.spline.end_kth_point_u(-2))
        spl_k1_w = self.spline(self.spline.end_kth_point_u(-1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])

        # 3. Calculate the delta offset between the target and current deflection
        zero_defl_offset_l = self.defl_model(0)
        cur_defl_offset_l = self.defl_model(self.data.wr0_defl)
        tgt_defl_offset_l = self.defl_model(self.tgt_defl_abs * self.wr0_defl_sign)
        defl_doffset_w = rotate_ccw(tgt_defl_offset_l - cur_defl_offset_l, self.data.wr0_yaw_w)
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

        if self.config.debug and np.array_equiv(self.spline.keypoints[-1], self.data.tip_r_w):
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
                monitor.draw_spline(self.spline, title="Swiping", **poi)

        return control

    def policy_initiate_whisking(self) -> ControlMessage | None:
        # assume the whisker has disengaged just now
        assert self.prev_state == ControllerState.SWIPING

        # 0. If the spline is not defined, keep the control values
        # There is no better strategy anyway
        if not self.spline:
            logger.warning("Spline is not defined, cannot initiate whisking")
            self.state = ControllerState.FAILURE
            return None

        # 1. Calculate new control:
        # The goal is to swipe the whisker along the other side of the edge
        # The linear velocity follows the spline backwards,
        # while the rotation brings the whisker tip to the new plane
        spl_k0_w, spl_k1_w = self.spline(self.spline.end_kth_point_u(0)), self.spline(self.spline.end_kth_point_u(1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)

        tgt_body_dr_n_w = rotate_ccw(-spl_dk_w_n, -self.tgt_wr0_defl_sign * np.pi / 2)
        tgt_body_yaw_w = self.exploration_omega * self.tgt_wr0_defl_sign

        control_reach_over_the_edge = self.motion_ctrl(
            data=self.data,
            prev_data=self.prev_data,
            tgt_body_dr_n_w=tgt_body_dr_n_w,
            tgt_body_yaw_w=tgt_body_yaw_w,
        )

        # 2. Reset the spline
        prev_spline = self.spline
        monitor.draw_spline(self.spline, title="Swiping End", body=self.data.body_r_w)
        self.spline = Spline(keypoint_distance=self.keypoint_distance, n_keypoints=self.n_keypoints)

        # 3. Set the desired next state
        has_reached_surface = False

        def initial_whisking_instructions() -> ControlMessage | None:
            nonlocal has_reached_surface
            if self.wr0_defl_sign:
                if has_reached_surface:
                    return None
                has_reached_surface = True

                # we've got the first estimate on the other side,
                # use it to construct a reasonable spline and give control to the swiping policy
                tip_r_w = self.data.tip_r_w
                side_tangent_n_w = normalize(tip_r_w - spl_k0_w)
                side_tangent_yaw_w = np.arctan2(side_tangent_n_w[1], side_tangent_n_w[0])
                tgt_wr0_yaw_w = side_tangent_yaw_w + self.data.body_wr0_angle_s
                tgt_defl_offset_w = rotate_ccw(self.defl_model(self.tgt_defl_abs * self.wr0_defl_sign), tgt_wr0_yaw_w)
                tgt_body_dr_w = spl_k0_w - tgt_defl_offset_w

                monitor.draw_spline(
                    prev_spline,
                    title="Whisking Start",
                    body=self.data.body_r_w,
                    tip=tip_r_w,
                    edge=spl_k0_w,
                    body_tgt=tgt_body_dr_w,
                )

                return self.motion_ctrl(
                    data=self.data,
                    prev_data=self.prev_data,
                    tgt_body_dr_n_w=normalize(tgt_body_dr_w),
                    tgt_body_yaw_w=side_tangent_yaw_w,
                )

            return control_reach_over_the_edge

        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.WHISKING
        self.exploration_instructions = initial_whisking_instructions

        return control_reach_over_the_edge

    def policy_whisking(self):
        control = self.policy_swiping()
        monitor.draw_spline(self.spline, title="Whisking", body=self.data.body_r_w)
        return control

    def policy_reattaching(self) -> ControlMessage | None:
        if not self.spline:
            logger.warning("Spline is not defined, cannot reattach")
            self.state = ControllerState.FAILURE
            return None

        # 1. Calculate the target body yaw
        spline_start_w = self.spline(0)
        spline_end_w = self.spline(1)
        spl_dk_w_n = normalize(spline_start_w - spline_end_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])
        tgt_body_yaw_w = spline_angle - self.tilt

        # 2. Calculate the target body movement direction
        tgt_defl_offset_l = self.defl_model(self.tgt_defl_abs * self.wr0_defl_sign)
        tgt_defl_offset_w = rotate_ccw(tgt_defl_offset_l, tgt_body_yaw_w + self.data.body_wr0_angle_s)
        tgt_body_r_w = spline_end_w - tgt_defl_offset_w

        monitor.draw_spline(
            self.spline,
            title="The other side of the edge",
            tgt_body_r_w=tgt_body_r_w,
            body=self.data.body_r_w,
        )

        # 3. Reset the spline and the tip estimator
        self.spline = Spline(keypoint_distance=self.keypoint_distance, n_keypoints=self.n_keypoints)

        # 4. Set the exploration policy to reattach the whisker

        def reattach_instructions() -> ControlMessage | None:
            tgt_body_dr_n_w = normalize(tgt_body_r_w - self.data.body_r_w)
            return self.motion_ctrl(
                data=self.data,
                prev_data=self.prev_data,
                tgt_body_dr_n_w=tgt_body_dr_n_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
            )

        self.exploration_instructions = reattach_instructions
        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        return self.exploration_instructions()
