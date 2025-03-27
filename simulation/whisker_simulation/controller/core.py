from collections import defaultdict

import numpy as np

from whisker_simulation.config import Config, ControlMessage, Orientation, WhiskerId
from whisker_simulation.controller.anomaly_detector import AnomalyDetector
from whisker_simulation.controller.body_motion import MotionController
from whisker_simulation.controller.spline import Spline
from whisker_simulation.models import ControllerState, Motion, SensorData, WhiskerData
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
        self.state_after_exploration = ControllerState.SWIPING
        self.data = self.prev_data = initial_data
        self.motion: Motion = Motion(data=self.data, prev_data=self.prev_data)
        self.splines: dict[WhiskerId, Spline] = {
            wsk_id: Spline(name=wsk_id, config=self.config.spline) for wsk_id in self.data.whiskers
        }
        self.midpoint_spline = Spline(name="midpoint", config=self.config.spline)
        self.midpoint_spline.keypoint_distance = self.config.spline.keypoint_distance * 5

        # stats
        self.stat_tip_traj: dict[WhiskerId, list[tuple[float, np.ndarray, bool]]] = defaultdict(list)
        self.stat_body_traj: list[tuple[float, np.ndarray]] = []
        self.stat_retrievals: list[tuple[np.ndarray, np.ndarray]] = []

        # single-whisker control
        self.active_wsk_id: WhiskerId | None = None
        self.is_retrieval_active: bool = False
        self.tgt_orient: Orientation = Orientation.NEUTRAL
        self.whisking_contact: np.ndarray | None = None
        self.whisking_tangent: np.ndarray | None = None
        self.whisking_edge: np.ndarray | None = None
        self.motion_ctrl = MotionController(total_v=self.config.body.total_v)

        # anomaly detector
        self.anomaly_detector = AnomalyDetector(controller=self)
        self.anomaly_blacklist = [
            (ControllerState.EXPLORING, ControllerState.DISENGAGED),
            (ControllerState.WHISKING, ControllerState.DISENGAGED),
        ]

        # additional instruction for different policies
        self.exploration_instructions = None
        self.whisking_instructions = None

        # trajectory clean up
        self.active_edge_r_w: np.ndarray | None = None
        self.active_edge_border: np.ndarray | None = None

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
            case ControllerState.SWIPING | ControllerState.WHISKING:
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

        # update the trajectories
        self.update_trajectories()

        # check the tunneling condition
        whiskers = self.data.whiskers
        left_wsk, right_wsk = whiskers["l0"], whiskers["r0"]
        both_wsk_deflected = left_wsk.is_deflected and right_wsk.is_deflected
        any_wsk_deflected = left_wsk.is_deflected or right_wsk.is_deflected
        tunneling_possible = both_wsk_deflected and left_wsk.defl * right_wsk.defl > 0
        if tunneling_possible and self.state != ControllerState.TUNNELING:
            self.active_wsk_id = None
            self.is_retrieval_active = False
            self.tgt_orient = Orientation.NEUTRAL
            self.state = ControllerState.TUNNELING

        if any_wsk_deflected and not tunneling_possible:
            if self.state == ControllerState.TUNNELING:
                self.tgt_orient = Orientation.NEUTRAL
                self.state_after_exploration = ControllerState.SWIPING
                self.state = ControllerState.EXPLORING
            if self.state == ControllerState.EXPLORING and not self.is_retrieval_active:
                new_active_wsk_id = next(wsk_id for wsk_id, wsk in whiskers.items() if wsk.is_deflected)
                if self.active_wsk_id != new_active_wsk_id:
                    # reset the splines of the other whiskers
                    for wsk_id in whiskers:
                        if wsk_id != new_active_wsk_id:
                            self.splines[wsk_id].reset()
                    self.active_wsk_id = new_active_wsk_id

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
                if self.state != anomaly_state:
                    self.state = anomaly_state

        # run the policy matching the state
        match self.state:
            case ControllerState.EXPLORING:
                # explore the map according to the exploration instructions
                return self.exploration_policy()
            case ControllerState.SWIPING:
                # follow the surface
                self.is_retrieval_active = False
                return self.policy_swiping()
            case ControllerState.WHISKING:
                # pull the whisker back to the edge (simple linear movement)
                return self.whisking_instructions()
            case ControllerState.DISENGAGED:
                if self.prev_state == ControllerState.SWIPING:
                    # the whisker has disengaged, needs to swipe back
                    self.is_retrieval_active = True
                    return self.policy_initiate_whisking()
                if self.prev_state == ControllerState.WHISKING:
                    # we swiped the other side of the edge, now we rotate and engage
                    return self.policy_reattaching()
            case ControllerState.TUNNELING:
                return self.policy_tunneling()
            case ControllerState.FAILURE:
                return self.motion_ctrl.idle()
            case _:
                raise NotImplementedError(f"State {self.state} is not implemented")

    def update_trajectories(self):
        whiskers = self.data.whiskers
        for wsk_id, wsk in whiskers.items():
            is_valid = wsk.is_deflected and self.splines[wsk_id] and not self.is_retrieval_active
            if wsk_id == self.active_wsk_id:
                if self.active_edge_r_w is not None:
                    distance = (
                        np.cross(self.active_edge_r_w - wsk.tip_r_w, self.active_edge_border)
                        / np.linalg.norm(self.active_edge_border)
                        * self.tgt_orient.value
                    )
                    if distance <= 0:
                        is_valid = False
                    elif self.state != ControllerState.WHISKING and distance > self.wsk.length / 4:
                        self.active_edge_r_w = self.active_edge_border = None
            trajectory = self.stat_tip_traj[wsk_id]
            trajectory.append((self.data.time, wsk.tip_r_w, is_valid))
        self.stat_body_traj.append((self.data.time, self.data.body.r_w))

    def exploration_policy(self) -> ControlMessage | None:
        def apply_instructions() -> ControlMessage | None:
            if self.exploration_instructions is None:
                return None
            return self.exploration_instructions()

        if self.active_wsk_id is None or not self.spline:
            self.tgt_orient = Orientation.NEUTRAL
            return apply_instructions()

        # At this point we have a spline to work with
        # This means that exploration has ended and the state should be updated
        self.logger.debug(f"Spline '{self.spline}' has been defined")
        if self.config.debug and self.monitor:
            self.monitor.draw_spline(self.spline, title="Exploration End", wsk=self.wsk.r_w)
        assert self.state_after_exploration is not None
        self.state = self.state_after_exploration
        self.state_after_exploration = None
        if self.tgt_orient == Orientation.NEUTRAL:
            self.tgt_orient = self.wsk.orientation
        assert self.tgt_orient != Orientation.NEUTRAL

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
        tgt_defl_offset_s = self.wsk.defl_model(self.wsk.config.tgt_defl_abs * np.sign(self.wsk.defl))
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
            reverse=np.sign(self.wsk.defl) == -1,
            tgt_wsk_dr_w=tgt_wsk_dr_n_w,
            tgt_body_yaw_w=spline_angle + self.config.body.tilt * self.wsk.orientation.value,
        )

        if self.config.debug and np.array_equiv(self.spline.keypoints[-1], self.wsk.tip_r_w) and self.monitor:
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
        # The goal is to whisk along the other side of the edge
        # Rotate the whisker tip around the edge to guarantee a stable contact

        # stabilize the spline
        self.spline.stabilize()

        # keep the old swiping orientation
        orient = self.tgt_orient
        spline_copy = self.spline.copy()
        self.whisking_edge = edge = self.spline(1)

        side_tangent = normalize(edge - self.spline(0))
        # define the distance from the edge to the desired contact point
        radius = self.wsk.length / 6
        # define desired whisker tilt at the contact
        tilt = np.pi / 8
        # define the staring angle
        angles = (np.linspace(0, np.pi, 100) + np.pi / 4) * orient.value
        # define the circle
        midpoint_offset = self.wsk.length / 8 * rotate(side_tangent, np.pi / 4 * orient.value)
        circle = (edge - midpoint_offset) + radius * rotate(side_tangent, angles.T)
        # define the maximum error to step
        max_yaw_error = np.pi / 16
        # define the current position on the circle
        j = 0

        def control_reach_over_the_edge():
            nonlocal j
            # step the circle if the tip is keeping up with the rotation and the body yaw is
            tgt_tip = circle[j]
            normal = normalize(tgt_tip - edge)
            tangent = rotate(normal, orient.value * np.pi / 2)
            expected_wsk_yaw_w = np.arctan2(tangent[1], tangent[0]) - tilt * orient.value
            angle_diff = (self.wsk.yaw_w - expected_wsk_yaw_w) % (2 * np.pi)
            yaw_ok = angle_diff < max_yaw_error or angle_diff > 2 * np.pi - max_yaw_error
            if yaw_ok:
                j = min(j + 1, len(circle) - 1)

            # calculate the control
            tgt_tip = circle[j]
            normal = normalize(tgt_tip - edge)
            tangent = rotate(normal, orient.value * np.pi / 2)
            # get the new whisker rotation and position
            tgt_wsk_yaw_w = np.arctan2(tangent[1], tangent[0]) - tilt * orient.value
            tgt_wsk_r_w = tgt_tip - rotate(self.wsk.defl_offset_s, tgt_wsk_yaw_w)
            tgt_body_yaw_w = tgt_wsk_yaw_w - self.wsk.config.angle_from_body

            if self.config.debug and self.monitor:
                self.monitor.draw_spline(
                    spline_copy,
                    title="Reaching Over The Edge",
                    circle=circle,
                    normal=tgt_tip + normal * radius / 4,
                    tangent=tgt_tip + tangent * radius / 4,
                    wsk=self.wsk.r_w,
                    tip=self.wsk.tip_r_w,
                    tgt_wsk=tgt_wsk_r_w,
                    tgt_tip=tgt_tip,
                )
            return self.motion_ctrl.steer_wsk(
                wsk=self.wsk,
                motion=self.motion,
                tgt_wsk_dr_w=tgt_wsk_r_w - self.wsk.r_w,
                tgt_body_yaw_w=tgt_body_yaw_w + np.pi / 8 * orient.value,
            )

        # 2. Reset the spline and invalidate all trajectory points up to the edge
        self.spline.reset()
        trajectory = self.stat_tip_traj[self.active_wsk_id]
        index = len(trajectory) - 1
        eps = self.config.spline.keypoint_distance
        while index >= 0 and np.linalg.norm(edge - trajectory[index][1]) > eps:
            t, xy, is_valid = trajectory[index]
            trajectory[index] = (t, xy, False)
            index -= 1

        # 3. Set the desired next state and the exploration instructions
        self.whisking_tangent = None
        self.whisking_contact = None
        self.active_edge_r_w = None
        self.active_edge_border = None

        orient_sum = 0

        def initial_whisking_instructions() -> ControlMessage | None:
            if self.whisking_tangent is None:
                if not self.wsk.is_deflected:
                    # while we haven't reached the surface, keep steering the whisker around the edge
                    return control_reach_over_the_edge()
                self.whisking_tangent = normalize(edge - self.wsk.tip_r_w)
                self.whisking_contact = self.wsk.tip_r_w
                self.stat_retrievals.append((edge, self.wsk.tip_r_w))

                self.active_edge_r_w = edge
                self.active_edge_border = side_tangent

            # keep track of the orientation
            nonlocal orient_sum
            orient_sum += self.wsk.orientation.value

            # whisking is finished - we need to switch to the reattaching policy
            if (
                not self.wsk.is_deflected
                and self.state == ControllerState.WHISKING
                and np.dot(self.wsk.tip_r_w - edge, self.whisking_tangent) > self.wsk.length / 8
            ):
                self.tgt_orient = Orientation(1) if orient_sum > 0 else Orientation(-1)
                self.state = ControllerState.DISENGAGED

            # if we have just landed here, switch the state
            if self.state == ControllerState.EXPLORING:
                self.state = ControllerState.WHISKING

            # whisk back while slightly rotated towards the object
            return self.motion_ctrl.steer_body(
                motion=self.motion,
                tgt_body_dr_w=self.whisking_tangent,
                tgt_body_yaw_w=self.data.body.yaw_w + np.pi / 16 * orient.value,
            )

        self.state = ControllerState.EXPLORING
        self.state_after_exploration = ControllerState.WHISKING
        self.exploration_instructions = self.whisking_instructions = initial_whisking_instructions

        return self.exploration_instructions()

    def policy_reattaching(self) -> ControlMessage | None:
        assert self.tgt_orient != Orientation.NEUTRAL

        # 0. Calculate the spline orientation (flip the direction, as we were moving backwards)
        orient = self.tgt_orient.flip()
        edge, contact = self.whisking_edge, self.whisking_contact
        tangent_n = normalize(contact - edge)
        normal_n = rotate(tangent_n, -np.pi / 2 * orient.value)
        tangent_angle = np.arctan2(tangent_n[1], tangent_n[0])

        # 1. Calculate the target body yaw (tilt a bit more for better edge engagement)
        tgt_body_yaw_w = tangent_angle + np.pi / 8 * orient.value
        tgt_wsk_yaw_w = tgt_body_yaw_w + self.wsk.config.angle_from_body

        # 2. Calculate the target whisker position
        # Keep in mind that the current deflection is zero
        tip_overshoot_normal = -self.wsk.length / 4
        tip_overshoot_tangent = self.wsk.length / 4
        tgt_tip_r_w = edge + normal_n * tip_overshoot_normal + tangent_n * tip_overshoot_tangent
        tgt_wsk_r_w = tgt_tip_r_w - rotate(self.wsk.neutral_defl_offset, tgt_wsk_yaw_w)
        tgt_body_r_w = tgt_wsk_r_w - rotate(self.wsk.config.offset_from_body, tgt_body_yaw_w)

        if self.config.debug and self.monitor:
            self.monitor.draw_spline(
                self.spline,
                title="The other side of the edge",
                edge=edge,
                contact=contact,
                tangent=contact + tangent_n * np.linalg.norm(contact - edge),
                normal=contact + normal_n * np.linalg.norm(contact - edge),
                wsk=self.wsk.r_w,
                tip=self.wsk.tip_r_w,
                tgt_wsk=tgt_wsk_r_w,
                tgt_tip=tgt_tip_r_w,
            )

        # 3. Reset the spline and the tip estimator
        self.spline.reset()

        # 4. Set the exploration policy to reattach the whisker

        def reattach_instructions() -> ControlMessage | None:
            max_yaw_error = np.pi / 8
            if (max_yaw_error < (tgt_body_yaw_w - self.data.body.yaw_w) % (2 * np.pi) < 2 * np.pi - max_yaw_error) or (
                abs(np.dot(tgt_wsk_r_w - self.wsk.r_w, normal_n)) > 2 * abs(tip_overshoot_normal)
            ):
                # the body yaw has not been set correctly, keep rotating
                body_normal_offset = np.dot(tgt_body_r_w - self.data.body.r_w, normal_n) * normal_n
                eps = self.config.spline.keypoint_distance
                return self.motion_ctrl.steer_body(
                    motion=self.motion,
                    tgt_body_dr_w=np.zeros(2) if np.linalg.norm(body_normal_offset) < eps else body_normal_offset,
                    tgt_body_yaw_w=tgt_body_yaw_w,
                )

            return self.motion_ctrl.steer_wsk(
                wsk=self.wsk,
                motion=self.motion,
                tgt_wsk_dr_w=tgt_wsk_r_w - self.wsk.r_w,
                tgt_body_yaw_w=tgt_body_yaw_w,
            )

        self.exploration_instructions = reattach_instructions
        self.state = ControllerState.EXPLORING
        self.state_after_exploration = ControllerState.SWIPING
        return self.exploration_instructions()

    def policy_tunneling(self) -> ControlMessage | None:
        if not self.midpoint_spline:
            return None
        spl_start_w, spl_end_w = self.midpoint_spline(0.5), self.midpoint_spline(1)
        spl_tangent_n = normalize(spl_end_w - spl_start_w)
        wsk_l0, wsk_r0 = self.data.whiskers["l0"], self.data.whiskers["r0"]
        # let the whiskers pull the target body position normally depending on the deflection
        pull_l = -(wsk_l0.defl - wsk_l0.config.tgt_defl_abs) / wsk_l0.config.tgt_defl_abs
        pull_r = (wsk_r0.defl - wsk_r0.config.tgt_defl_abs) / wsk_r0.config.tgt_defl_abs
        total_pull = np.clip(pull_l + pull_r, -1, 1)
        tgt_body_dr_w = spl_tangent_n + total_pull / 2 * rotate(spl_tangent_n, np.pi / 2 * -wsk_r0.orientation.value)
        return self.motion_ctrl.steer_body(
            motion=self.motion,
            tgt_body_dr_w=tgt_body_dr_w,
            tgt_body_yaw_w=np.arctan2(spl_tangent_n[1], spl_tangent_n[0]),
        )
