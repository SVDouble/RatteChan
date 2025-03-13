import numpy as np

from whisker_simulation.config import Config, WhiskerId, WhiskerOrientation
from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import MotionController
from whisker_simulation.controller.spline import Spline
from whisker_simulation.models import ControllerState, ControlMessage, Motion, SensorData, WhiskerData
from whisker_simulation.monitor import Monitor
from whisker_simulation.utils import get_logger, normalize, rotate

__all__ = ["Controller"]


class Controller:
    def __init__(self, *, initial_data: SensorData, config: Config, monitor: Monitor):
        self.config: Config = config
        self.monitor = monitor
        self.logger = get_logger(__file__, log_level=self.config.log_level)

        # runtime
        self.control_dt = 1 / config.control_rps
        self.__state = self.__prev_state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        self.data = self.prev_data = initial_data
        self.motion: Motion = Motion(data=self.data, prev_data=self.prev_data)
        self.splines: dict[WhiskerId, Spline] = {
            wsk_id: Spline(name=wsk_id, config=self.config.spline, monitor=self.monitor)
            for wsk_id in self.data.whiskers
        }
        self.midpoint_spline = Spline(name="midpoint", config=self.config.spline, monitor=self.monitor, track=False)

        # single-whisker control
        self.active_wsk_id: WhiskerId | None = None
        self.is_wsk_locked: bool = False
        self.tgt_orient: WhiskerOrientation = 0
        self.motion_ctrl = MotionController(total_v=self.config.body.total_v)

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(controller=self)
        self.anomaly_blacklist = [(ControllerState.EXPLORING, ControllerState.DISENGAGED)]

        # additional instruction for different policies
        self.exploration_instructions = None
        self.whisking_instructions = None

    @property
    def state(self) -> ControllerState:
        return self.__state

    @state.setter
    def state(self, value: ControllerState) -> None:
        self.__prev_state = self.__state
        self.logger.info(f"State transition: {self.__state} -> {value}")
        self.__state = value
        match value:
            case ControllerState.EXPLORING:
                self.logger.info("Switched to 0-whisker control")
            case ControllerState.SWIPING, ControllerState.WHISKING:
                self.logger.info(f"Switched to 1-whisker control, active whisker: {self.wsk}")
            case ControllerState.TUNNELING:
                self.logger.info("Switched to 2-whisker control")

    @property
    def prev_state(self) -> ControllerState:
        return self.__prev_state

    @property
    def wsk(self) -> WhiskerData:
        if self.active_wsk_id is None:
            raise RuntimeError("Active whisker id not set")
        return self.data.whiskers[self.active_wsk_id]

    @property
    def spline(self) -> Spline | None:
        if self.active_wsk_id is None:
            raise RuntimeError("Active whisker id not set")
        return self.splines[self.active_wsk_id]

    @spline.setter
    def spline(self, value: Spline) -> None:
        if self.active_wsk_id is None:
            raise RuntimeError("Active whisker id not set")
        self.splines[self.active_wsk_id] = value

    def control(self, new_data: SensorData) -> ControlMessage | None:  # noqa: C901
        # rate limit the control
        if new_data.time - self.data.time < self.control_dt:
            return None

        # ATTENTION: time step might be variable due to rate limiting

        # if in failure state, send idle control
        if self.state == ControllerState.FAILURE:
            return self.motion_ctrl.idle()

        # update the controller given the new sensor data
        self.prev_data, self.data = self.data, new_data
        self.motion = Motion(data=self.data, prev_data=self.prev_data)

        whiskers = self.data.whiskers
        left_wsk, right_wsk = whiskers["l0"], whiskers["r0"]
        both_wsk_deflected = left_wsk.is_deflected and right_wsk.is_deflected
        any_wsk_deflected = left_wsk.is_deflected or right_wsk.is_deflected
        if both_wsk_deflected and self.state != ControllerState.TUNNELING:
            self.active_wsk_id = None
            self.is_wsk_locked = False
            self.tgt_orient = 0
            self.state = ControllerState.TUNNELING

        if any_wsk_deflected and not both_wsk_deflected:
            if self.state == ControllerState.TUNNELING:
                self.state = ControllerState.SWIPING
            new_wsk_id = next(wsk_id for wsk_id, wsk in whiskers.items() if wsk.is_deflected)
            if not self.is_wsk_locked and self.active_wsk_id != new_wsk_id:
                # reset the splines of the other whiskers
                for wsk_id in whiskers:
                    if wsk_id != new_wsk_id:
                        self.splines[wsk_id].reset()
                self.active_wsk_id = new_wsk_id

        # update the splines
        self.midpoint_spline.add_keypoint(keypoint=(left_wsk.tip_r_w + right_wsk.tip_r_w) / 2, data=self.data)
        for wsk_id, wsk in whiskers.items():
            if wsk.is_deflected:
                spline = self.splines[wsk_id]
                has_new_point = spline.add_keypoint(keypoint=wsk.tip_r_w, data=self.data)
                if has_new_point and len(spline.keypoints) == 1:
                    self.logger.debug(f"Whisker '{wsk}' has come into contact with the surface")

        # detect anomalies (ignore them if state is EXPLORING)
        if anomaly := self.anomaly_detector():
            anomaly_state, anomaly_msg = anomaly
            if (self.state, anomaly_state) not in self.anomaly_blacklist:
                self.logger.info(f"Anomaly detected: {anomaly_msg}")
                self.state = anomaly_state

        # run the policy matching the state
        match self.state:
            case ControllerState.EXPLORING:
                # explore the map according to the exploration instructions
                return self.exploration_policy()
            case ControllerState.SWIPING:
                # follow the surface
                self.is_wsk_locked = False
                return self.policy_swiping()
            case ControllerState.WHISKING:
                # pull the whisker back to the edge (simple linear movement)
                return self.whisking_instructions()
            case ControllerState.DISENGAGED:
                if self.prev_state == ControllerState.SWIPING:
                    # the whisker has disengaged, needs to swipe back
                    self.is_wsk_locked = True
                    return self.policy_initiate_whisking()
                if self.prev_state == ControllerState.WHISKING:
                    # we swiped the other side of the edge, now we rotate and engage
                    return self.policy_reattaching()
            case ControllerState.TUNNELING:
                return self.policy_tunneling()
            case _:
                raise NotImplementedError(f"State {self.state} is not implemented")

    def exploration_policy(self) -> ControlMessage | None:
        def apply_instructions() -> ControlMessage | None:
            if self.exploration_instructions is None:
                return None
            return self.exploration_instructions()

        if self.active_wsk_id is None or not self.spline:
            self.tgt_orient = 0
            return apply_instructions()

        # At this point we have a spline to work with
        # This means that exploration has ended and the state should be updated
        self.logger.debug(f"Spline '{self.spline}' has been defined")
        if self.config.debug:
            self.monitor.draw_spline(self.spline, title="Exploration End", wsk=self.wsk.r_w)
        assert self.desired_next_state is not None
        self.state = self.desired_next_state
        self.desired_next_state = None
        if self.tgt_orient == 0:
            self.tgt_orient = self.wsk.orientation
        assert self.tgt_orient != 0

        # Use the exploration instructions one last time to let it prepare for the engaged state
        control = apply_instructions()
        self.exploration_instructions = None
        return control

    def policy_swiping(self) -> ControlMessage | None:
        # 1. If the deflection is too small, keep the control values
        if not self.wsk.is_deflected:
            return None

        # 2. Calculate spline curvature
        # be quite conservative with the angle as the most recent points might be unstable
        spl_k0_w = self.spline(self.spline.end_kth_point_u(-2))
        spl_k1_w = self.spline(self.spline.end_kth_point_u(-1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])

        # 3. Calculate the delta offset between the target and current deflection
        tgt_defl_offset_s = self.wsk.defl_model(self.wsk.config.tgt_defl_abs * self.wsk.orientation)
        defl_doffset_w = rotate(tgt_defl_offset_s - self.wsk.defl_offset_s, self.wsk.yaw_w)
        defl_doffset_w_n = normalize(-defl_doffset_w)
        defl_offset_weight = np.linalg.norm(defl_doffset_w) / np.linalg.norm(
            tgt_defl_offset_s - self.wsk.neutral_defl_offset
        )

        # 4. Choose the target direction as weighted average of the spline and deflection offset
        k = np.clip(defl_offset_weight * 1.5, 0, 1)
        tgt_wsk_dr_n_w = normalize(defl_doffset_w_n * k + spl_dk_w_n * (1 - k))

        # 5. Calculate the control values
        # for v to follow tgt_wsk_dr_w and for omega to follow the spline slightly tilted
        control = self.motion_ctrl.steer_wsk(
            wsk=self.wsk,
            motion=self.motion,
            tgt_wsk_dr_w=tgt_wsk_dr_n_w,
            tgt_body_yaw_w=spline_angle + self.config.body.tilt * self.wsk.orientation,
        )

        if self.config.debug and np.array_equiv(self.spline.keypoints[-1], self.wsk.tip_r_w):
            ds = self.spline.keypoint_distance * self.spline.n_keypoints
            poi = {
                "wsk": self.wsk.r_w,
                "d_defl": self.wsk.r_w + defl_doffset_w_n * ds,
                "d_spl": self.wsk.r_w + spl_dk_w_n * ds,
                "d_tgt": self.wsk.r_w + tgt_wsk_dr_n_w * ds,
            }
            self.monitor.draw_spline(self.spline, title="Swiping", **poi)

        return control

    def policy_initiate_whisking(self) -> ControlMessage | None:  # noqa: C901
        # assume the whisker has disengaged just now
        assert self.prev_state == ControllerState.SWIPING

        # 0. If the spline is not defined, keep the control values
        # There is no better strategy anyway
        if not self.spline:
            self.logger.warning("Spline is not defined, cannot initiate whisking")
            self.state = ControllerState.FAILURE
            return None

        # 1. Calculate new control:
        # The goal is to swipe the whisker along the other side of the edge
        # Rotate the whisker tip around the edge to guarantee sufficient swipe distance after reattachment

        # stabilize the spline
        self.spline.stabilize()

        # keep the old swiping orientation
        tgt_orient = self.tgt_orient
        prev_spline = self.spline.copy()
        # the actual edge is continuous, we use the spline middle point for stability
        spl_start = self.spline(0)
        edge = self.spline(1)
        spl_tangent = normalize(edge - spl_start)
        spl_normal = rotate(spl_tangent, -tgt_orient * np.pi / 2)
        # the tip might be oscillating, so use its neutral position
        radius = self.wsk.length / 4
        tgt_tip_r_w = edge + rotate(radius * spl_tangent, tgt_orient * np.pi / 8)
        # total velocity is fixed for the body, so account for its wider radius
        omega = self.config.body.total_v / radius * tgt_orient

        def control_reach_over_the_edge():
            nonlocal tgt_tip_r_w
            if abs(np.linalg.norm(tgt_tip_r_w - self.wsk.tip_r_w)) < radius / 2:
                # rotate the tip around the center by expected angle at dt
                tgt_tip_r_w = edge + rotate(tgt_tip_r_w - edge, omega * self.motion.dt)
            # get the normal and tangent to the circle at the tip
            normal = normalize(tgt_tip_r_w - edge)
            tangent = rotate(normal, tgt_orient * np.pi / 2)
            # get the new whisker rotation and position
            tgt_body_yaw_w = np.arctan2(normal[1], normal[0]) - tgt_orient * self.config.body.tilt
            tgt_wsk_yaw_w = tgt_body_yaw_w - tgt_orient * self.wsk.config.angle_from_body
            tgt_wsk_r_w = tgt_tip_r_w - rotate(self.wsk.neutral_defl_offset, tgt_wsk_yaw_w)

            if self.config.debug:
                space = np.linspace(0, 2 * np.pi, 100)
                self.monitor.draw_spline(
                    prev_spline,
                    title="Reaching Over The Edge",
                    wsk=self.wsk.r_w,
                    tgt_wsk=tgt_wsk_r_w,
                    tip=self.wsk.tip_r_w,
                    tgt_tip=tgt_tip_r_w,
                    circle=edge + radius * np.array([np.cos(space), np.sin(space)]).T,
                    normal=tgt_tip_r_w + normal * radius / 4,
                    tangent=tgt_tip_r_w + tangent * radius / 4,
                )
            return self.motion_ctrl.steer_wsk(
                wsk=self.wsk,
                motion=self.motion,
                tgt_wsk_dr_w=tgt_wsk_r_w - self.wsk.r_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
            )

        # 2. Reset the spline
        self.spline.reset(track=False)

        # 3. Set the desired next state and the exploration instructions
        has_reached_surface = False

        def initial_whisking_instructions() -> ControlMessage | None:
            nonlocal has_reached_surface
            if not has_reached_surface and not self.wsk.is_deflected:
                # while we haven't reached the surface, keep steering the whisker around the edge
                return control_reach_over_the_edge()

            # we've got the first estimate on the other side,
            # use it to construct a reasonable spline and give control to the swiping policy
            has_reached_surface = True
            tgt_tip_outside_w = edge + (radius / 4) * spl_normal
            tgt_tip_dr_w = tgt_tip_outside_w - self.wsk.tip_r_w
            spl_kp_d = self.config.spline.keypoint_distance
            if abs(np.linalg.norm(tgt_tip_dr_w)) < spl_kp_d * 2 and not self.wsk.is_deflected:
                # the length was not enough to reconstruct the full spline
                # and we have already reached the edge and even went beyond
                # fake the spline and start swiping
                if not self.spline.keypoints:
                    self.state = ControllerState.FAILURE
                    return None
                fake_keypoint = self.spline.keypoints[0].copy()
                while not self.spline:
                    fake_keypoint -= spl_normal * spl_kp_d * 5
                    self.spline.prepend_fake_keypoint(keypoint=fake_keypoint)
                # we have faked the swipe back
                # noinspection PyTypeChecker
                self.tgt_orient = -tgt_orient
                return None

            if self.config.debug:
                self.monitor.draw_spline(
                    prev_spline,
                    title="Reaching Over The Edge",
                    wsk=self.wsk.r_w,
                    tip=self.wsk.tip_r_w,
                    tgt_tip_r_w=tgt_tip_outside_w,
                    normal=tgt_tip_outside_w + spl_normal * radius / 4,
                    tangent=tgt_tip_outside_w + spl_tangent * radius / 4,
                )

            return self.motion_ctrl.steer_wsk(
                wsk=self.wsk,
                motion=self.motion,
                tgt_wsk_dr_w=tgt_tip_dr_w,
                tgt_body_yaw_w=None,
            )

        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.WHISKING
        self.exploration_instructions = self.whisking_instructions = initial_whisking_instructions

        return self.exploration_instructions()

    def policy_reattaching(self) -> ControlMessage | None:
        if not self.spline:
            self.logger.warning("Spline is not defined, cannot reattach")
            self.state = ControllerState.FAILURE
            return None

        # 0. Calculate the spline orientation (flip the direction, as we were moving backwards)
        orient = -self.tgt_orient
        self.spline.stabilize()
        spl_start_w, spl_end_w = self.spline(1), self.spline(0)
        spl_tangent_n = normalize(spl_end_w - spl_start_w)
        spl_normal_n = rotate(spl_tangent_n, -orient * np.pi / 2)
        spl_tangent_angle = np.arctan2(spl_tangent_n[1], spl_tangent_n[0])

        # 1. Calculate the target body yaw (tilt a bit more for better edge engagement)
        tgt_body_yaw_w = spl_tangent_angle + (self.config.body.tilt * 2) * orient
        tgt_wsk_yaw_w = tgt_body_yaw_w - orient * self.wsk.config.angle_from_body

        # 2. Calculate the target whisker position
        # Keep in mind that the current deflection is zero
        optimal_tip_overshoot_d = self.wsk.length / 16
        tgt_tip_r_w = spl_start_w - spl_normal_n * optimal_tip_overshoot_d
        tgt_wsk_r_w = tgt_tip_r_w - rotate(self.wsk.neutral_defl_offset, tgt_wsk_yaw_w)
        # Push the target base position further along the spline so that the tip always reaches the edge
        tgt_wsk_r_w += (self.wsk.length / 8) * spl_tangent_n

        self.monitor.draw_spline(
            self.spline,
            title="The other side of the edge",
            wsk=self.wsk.r_w,
            tip=self.wsk.tip_r_w,
            tgt_wsk_r_w=tgt_wsk_r_w,
            tgt_tip_r_w=tgt_tip_r_w,
            spl_tangent=spl_end_w + spl_tangent_n * np.linalg.norm(spl_end_w - spl_start_w) * 2,
            spl_normal=spl_end_w + spl_normal_n * np.linalg.norm(spl_end_w - spl_start_w) * 2,
        )

        # 3. Reset the spline and the tip estimator
        self.spline.reset()

        # 4. Set the exploration policy to reattach the whisker

        def reattach_instructions() -> ControlMessage | None:
            # orientation has changed by now
            return self.motion_ctrl.steer_wsk(
                wsk=self.wsk,
                motion=self.motion,
                tgt_wsk_dr_w=tgt_wsk_r_w - self.wsk.r_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
            )

        self.exploration_instructions = reattach_instructions
        self.state = ControllerState.EXPLORING
        self.desired_next_state = ControllerState.SWIPING
        return self.exploration_instructions()

    def policy_tunneling(self) -> ControlMessage | None:
        if not self.midpoint_spline:
            return None
        spl_start_w, spl_end_w = self.midpoint_spline(0), self.midpoint_spline(1)
        spl_tangent_n = normalize(spl_end_w - spl_start_w)
        return self.motion_ctrl.steer_body(
            motion=self.motion,
            tgt_body_dr_w=spl_tangent_n,
            tgt_body_yaw_w=np.arctan2(spl_tangent_n[1], spl_tangent_n[0]),
        )
