import numpy as np

from whisker_simulation.controller.deflection_model import DeflectionModel
from whisker_simulation.controller.spline import Spline
from whisker_simulation.models import Control, WorldState
from whisker_simulation.pid import PID
from whisker_simulation.utils import (
    get_monitor,
    normalize,
    unwrap_pid_error,
    rotate_ccw,
)

__all__ = ["BodyMotionController"]

monitor = get_monitor()


class BodyMotionController:
    def __init__(self, *, total_v: float, control_rps: float):
        self.total_v = total_v
        self.initial_dt = 1 / control_rps
        self.tgt_defl = -3e-4
        self.tilt = 0.2
        self.defl_model = DeflectionModel()
        self.yaw_pid = PID(
            kp=0.5,
            ki=0,
            kd=0,
            dt=self.initial_dt,
            out_limits=(-np.pi / 2, np.pi / 2),
        )

    def control(
        self,
        *,
        spline: Spline,
        state: WorldState,
        prev_state: WorldState,
        has_new_keypoint: bool,
    ) -> Control:
        np.set_printoptions(precision=3, suppress=True)

        # 0. Update the time step of the PID controllers
        dt = (
            state.time - prev_state.time
            if state.time != prev_state.time
            else self.initial_dt
        )
        assert dt / self.initial_dt < 0.9, (
            "Time step is smaller than the control period"
        )
        self.yaw_pid.dt = dt

        # Calculate spline curvature
        spl_k0_w = spline(spline.end_kth_point_u(0))
        spl_k1_w = spline(spline.end_kth_point_u(1))
        spl_dk_w_n = normalize(spl_k1_w - spl_k0_w)
        spline_angle = np.arctan2(spl_dk_w_n[1], spl_dk_w_n[0])

        # Calculate the delta offset between the target and current deflection
        zero_defl_offset_l = self.defl_model.get_position(0)
        cur_defl_offset_l = self.defl_model.get_position(state.wr0_yaw_s)
        tgt_defl_offset_l = self.defl_model.get_position(self.tgt_defl)
        defl_doffset_w = rotate_ccw(
            tgt_defl_offset_l - cur_defl_offset_l, state.body_yaw_w
        )
        defl_doffset_w_n = normalize(-defl_doffset_w)
        defl_offset_weight = np.linalg.norm(defl_doffset_w) / np.linalg.norm(
            tgt_defl_offset_l - zero_defl_offset_l
        )

        # Choose the target direction as weighted average of the spline and deflection offset
        k = np.clip(defl_offset_weight * 1.5, 0, 1)
        tgt_body_dr_n = normalize(defl_doffset_w_n * k + spl_dk_w_n * (1 - k))
        vx, vy = tgt_body_dr_n * self.total_v

        # Choose the target yaw so that the body is slightly tilted towards the spline
        nose_yaw_w = state.body_yaw_w + np.pi / 2
        yaw_error = unwrap_pid_error(spline_angle - (nose_yaw_w + self.tilt))
        body_omega_w = self.yaw_pid(yaw_error)

        if has_new_keypoint:
            ds = spline.keypoint_distance * spline.n_keypoints
            poi = {
                "body": state.body_r_w,
                "d_defl": state.body_r_w + defl_doffset_w_n * ds,
                "d_spl": state.body_r_w + spl_dk_w_n * ds,
                "d_tgt": state.body_r_w + tgt_body_dr_n * ds,
            }
            if k < 0.05:
                del poi["d_defl"]
            if k > 0.5:
                monitor.draw_spline(spline, **poi)

        return Control(body_vx_w=vx, body_vy_w=vy, body_omega_w=body_omega_w)
