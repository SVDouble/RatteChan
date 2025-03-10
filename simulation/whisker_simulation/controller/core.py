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
        self.config: Config = config

        # runtime
        self.control_dt = 1 / config.control_rps
        self.__state = self.__prev_state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        self.data = self.prev_data = initial_data
        self.motion: MotionAnalyzer = MotionAnalyzer(data=self.data, prev_data=self.prev_data)

        # tip position estimation
        self.defl_model = DeflectionModel()
        self.defl_threshold = 2e-5
        self.orient: int = 0
        # target orientation is reset at exploration start and set at exploration end
        self.tgt_orient: int = self.orient

        # tip position prediction using spline
        self.keypoint_distance = 1e-3
        self.n_keypoints = 7
        self.spline = Spline(keypoint_distance=self.keypoint_distance, n_keypoints=self.n_keypoints)

        # body yaw control
        self.tgt_defl_abs = 3e-4
        self.tilt = 0.2

        # velocity and angle control
        self.total_v = 0.05
        self.motion_ctrl = BodyMotionController(total_v=self.total_v)

        # disengagement policy
        self.disengaged_duration_threshold = 0.1

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

        # if in failure state, send idle control
        if self.state == ControllerState.FAILURE:
            return self.motion_ctrl.idle()

        # update the controller given the new sensor data
        self.prev_data, self.data = self.data, new_data
        self.motion = MotionAnalyzer(data=self.data, prev_data=self.prev_data)

        # update the spline
        self.orient = np.sign(self.data.wr0_defl) if abs(self.data.wr0_defl) > self.defl_threshold else 0
        if self.orient:
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
            # reduce the deflection, as close alignment is not needed
            # tilt the whisker to increase the chance of proper detachment
            # TODO: stop swiping when the previous spline end has been reached
            return self.policy_swiping(tgt_defl_abs=self.tgt_defl_abs * 0.7, tilt=-self.tilt)

        if self.state == ControllerState.DISENGAGED:
            if self.prev_state == ControllerState.SWIPING:
                # the whisker has disengaged, needs to swipe back
                return self.policy_initiate_whisking()
            if self.prev_state == ControllerState.WHISKING:
                # we swiped the other side of the edge, now we rotate and engage
                return self.policy_reattaching()

        raise NotImplementedError(f"State {self.state} is not implemented")

    def exploration_policy(self) -> ControlMessage | None:
        def apply_instructions() -> ControlMessage | None:
            if self.exploration_instructions is None:
                return None
            return self.exploration_instructions()

        if not self.spline:
            self.tgt_orient = 0
            return apply_instructions()

        # At this point we have a spline to work with
        # This means that exploration has ended and the state should be updated
        logger.debug("The spline has been defined")
        if self.config.debug:
            monitor.draw_spline(self.spline, title="Exploration End", body=self.data.body_r_w)
        assert self.desired_next_state is not None
        self.state = self.desired_next_state
        self.desired_next_state = None
        if self.tgt_orient == 0:
            self.tgt_orient = self.orient
        assert self.tgt_orient != 0

        # Use the exploration instructions one last time to let it prepare for the engaged state
        control = apply_instructions()
        self.exploration_instructions = None
        return control

    def policy_swiping(self, tgt_defl_abs: float | None = None, tilt: float | None = None) -> ControlMessage | None:
        tgt_defl_abs = tgt_defl_abs if tgt_defl_abs is not None else self.tgt_defl_abs
        tilt = tilt if tilt is not None else self.tilt

        # 1. If the deflection is too small, keep the control values
        if not self.orient:
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
        tgt_defl_offset_l = self.defl_model(tgt_defl_abs * self.orient)
        defl_doffset_w = rotate_ccw(tgt_defl_offset_l - cur_defl_offset_l, self.data.wr0_yaw_w)
        defl_doffset_w_n = normalize(-defl_doffset_w)
        defl_offset_weight = np.linalg.norm(defl_doffset_w) / np.linalg.norm(tgt_defl_offset_l - zero_defl_offset_l)

        # 4. Choose the target direction as weighted average of the spline and deflection offset
        k = np.clip(defl_offset_weight * 1.5, 0, 1)
        tgt_body_dr_n_w = normalize(defl_doffset_w_n * k + spl_dk_w_n * (1 - k))

        # 5. Calculate the control values
        # for v to follow tgt_body_dr_w and for omega to follow the spline slightly tilted
        control = self.motion_ctrl(
            data=self.data,
            prev_data=self.prev_data,
            tgt_body_dr_w=tgt_body_dr_n_w,
            tgt_body_yaw_w=spline_angle + tilt * self.orient,
            orient=self.orient,
        )

        if self.config.debug and np.array_equiv(self.spline.keypoints[-1], self.data.tip_r_w):
            ds = self.spline.keypoint_distance * self.spline.n_keypoints
            poi = {
                "body": self.data.body_r_w,
                "d_defl": self.data.body_r_w + defl_doffset_w_n * ds,
                "d_spl": self.data.body_r_w + spl_dk_w_n * ds,
                "d_tgt": self.data.body_r_w + tgt_body_dr_n_w * ds,
            }
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
        # Rotate the whisker tip around the edge to guarantee sufficient swipe distance after reattachment

        # stabilize the spline
        self.spline.stabilize()

        # keep the old swiping orientation
        tgt_orient = self.tgt_orient
        prev_spline = self.spline
        # the actual edge is continuous, we use the spline middle point for stability
        spl_start = self.spline(0)
        edge = self.spline(1)
        spl_tangent = normalize(edge - spl_start)
        spl_normal = rotate_ccw(spl_tangent, -tgt_orient * np.pi / 2)
        # the whisker is detached, so the deflection is zero
        body_tip_offset_s = self.defl_model(0)
        body_tip_offset_l = np.linalg.norm(body_tip_offset_s)
        # the tip might be oscillating, so use its neutral position
        radius = body_tip_offset_l / 4
        tgt_tip_r_w = edge + rotate_ccw(radius * spl_tangent, tgt_orient * np.pi / 8)
        # total velocity is fixed for the body, so account for its wider radius
        omega = self.total_v / radius * tgt_orient

        def control_reach_over_the_edge():
            nonlocal tgt_tip_r_w
            if abs(np.linalg.norm(tgt_tip_r_w - self.data.tip_r_w)) < radius / 2:
                # rotate the tip around the center by expected angle at dt
                tgt_tip_r_w = edge + rotate_ccw(tgt_tip_r_w - edge, omega * self.motion.dt)
            # get the normal and tangent to the circle at the tip
            normal = normalize(tgt_tip_r_w - edge)
            tangent = rotate_ccw(normal, tgt_orient * np.pi / 2)
            # get the new body rotation and position
            tgt_body_yaw_w = np.arctan2(normal[1], normal[0]) - tgt_orient * self.tilt
            tgt_wr0_yaw_w = tgt_body_yaw_w - tgt_orient * self.data._wr0_angle_s
            tgt_body_r_w = tgt_tip_r_w - rotate_ccw(body_tip_offset_s, tgt_wr0_yaw_w)

            if self.config.debug:
                space = np.linspace(0, 2 * np.pi, 100)
                monitor.draw_spline(
                    prev_spline,
                    title="Reaching Over The Edge",
                    body=self.data.body_r_w,
                    tgt_body_r_w=tgt_body_r_w,
                    tip=self.data.tip_r_w,
                    tgt_tip_r_w=tgt_tip_r_w,
                    circle=edge + radius * np.array([np.cos(space), np.sin(space)]).T,
                    normal=tgt_tip_r_w + normal * radius / 4,
                    tangent=tgt_tip_r_w + tangent * radius / 4,
                )
            return self.motion_ctrl(
                data=self.data,
                prev_data=self.prev_data,
                tgt_body_dr_w=tgt_body_r_w - self.data.body_r_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
                orient=tgt_orient,
            )

        # 2. Reset the spline
        self.spline = Spline(keypoint_distance=self.keypoint_distance, n_keypoints=self.n_keypoints, track=False)

        # 3. Set the desired next state and the exploration instructions
        has_reached_surface = False

        def initial_whisking_instructions() -> ControlMessage | None:
            nonlocal has_reached_surface
            if not has_reached_surface and not self.orient:
                # while we haven't reached the surface, keep steering the whisker around the edge
                return control_reach_over_the_edge()

            # we've got the first estimate on the other side,
            # use it to construct a reasonable spline and give control to the swiping policy
            has_reached_surface = True
            tgt_tip_outside_w = edge + (radius / 4) * spl_normal
            tgt_tip_dr_w = tgt_tip_outside_w - self.data.tip_r_w
            if abs(np.linalg.norm(tgt_tip_dr_w)) < self.keypoint_distance * 2 and self.orient == 0:
                # the length was not enough to reconstruct the full spline
                # and we have already reached the edge and even went beyond
                # fake the spline and start swiping
                if not self.spline.keypoints:
                    self.state = ControllerState.FAILURE
                    return None
                fake_keypoint = self.spline.keypoints[0].copy()
                while not self.spline:
                    fake_keypoint -= spl_normal * self.keypoint_distance * 5
                    self.spline.prepend_fake_keypoint(keypoint=fake_keypoint)
                # we have faked the swipe back
                self.tgt_orient = -tgt_orient
                return None

            if self.config.debug:
                monitor.draw_spline(
                    prev_spline,
                    title="Reaching Over The Edge",
                    body=self.data.body_r_w,
                    tip=self.data.tip_r_w,
                    tgt_tip_r_w=tgt_tip_outside_w,
                    normal=tgt_tip_outside_w + spl_normal * radius / 4,
                    tangent=tgt_tip_outside_w + spl_tangent * radius / 4,
                )

            return self.motion_ctrl(
                data=self.data,
                prev_data=self.prev_data,
                tgt_body_dr_w=tgt_tip_dr_w,
                tgt_body_yaw_w=None,
                orient=-tgt_orient,
            )

        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.WHISKING
        self.exploration_instructions = initial_whisking_instructions

        return self.exploration_instructions()

    def policy_reattaching(self) -> ControlMessage | None:
        if not self.spline:
            logger.warning("Spline is not defined, cannot reattach")
            self.state = ControllerState.FAILURE
            return None

        # 0. Calculate the spline orientation (flip the direction, as we were moving backwards)
        orient = -self.tgt_orient
        self.spline.stabilize()
        spl_start_w, spl_end_w = self.spline(1), self.spline(0)
        spl_tangent_n = normalize(spl_end_w - spl_start_w)
        spl_normal_n = rotate_ccw(spl_tangent_n, -orient * np.pi / 2)
        spl_tangent_angle = np.arctan2(spl_tangent_n[1], spl_tangent_n[0])

        # 1. Calculate the target body yaw (tilt a bit more for better edge engagement)
        tgt_body_yaw_w = spl_tangent_angle + (self.tilt * 2) * orient
        tgt_wr_yaw_w = tgt_body_yaw_w - orient * self.data._wr0_angle_s

        # 2. Calculate the target body position
        # Keep in mind that the current deflection is zero
        optimal_tip_overshoot_d = np.linalg.norm(self.defl_model(0)) / 24
        tgt_tip_r_w = spl_start_w - spl_normal_n * optimal_tip_overshoot_d
        tgt_body_r_w = tgt_tip_r_w - rotate_ccw(self.defl_model(0), tgt_wr_yaw_w)

        monitor.draw_spline(
            self.spline,
            title="The other side of the edge",
            body=self.data.body_r_w,
            tip=self.data.tip_r_w,
            tgt_body_r_w=tgt_body_r_w,
            tgt_tip_r_w=tgt_tip_r_w,
            spl_tangent=spl_end_w + spl_tangent_n * np.linalg.norm(spl_end_w - spl_start_w) * 2,
            spl_normal=spl_end_w + spl_normal_n * np.linalg.norm(spl_end_w - spl_start_w) * 2,
        )

        # 3. Reset the spline and the tip estimator
        self.spline = Spline(keypoint_distance=self.keypoint_distance, n_keypoints=self.n_keypoints)

        # 4. Set the exploration policy to reattach the whisker

        def reattach_instructions() -> ControlMessage | None:
            # orientation has changed by now
            return self.motion_ctrl(
                data=self.data,
                prev_data=self.prev_data,
                tgt_body_dr_w=tgt_body_r_w - self.data.body_r_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
                orient=orient,
            )

        self.exploration_instructions = reattach_instructions
        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        return self.exploration_instructions()
