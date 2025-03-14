import numpy as np

from whisker_simulation.models import ControlMessage, Motion, WhiskerData
from whisker_simulation.pid import PID
from whisker_simulation.utils import normalize, unwrap_pid_error

__all__ = ["MotionController"]


class MotionController:
    def __init__(self, *, total_v: float):
        self.total_v = total_v
        self.yaw_pid = PID(
            kp=0.5,
            ki=0.01,
            kd=0,
            dt=0,  # will be set in the control method
            out_limits=(-np.pi / 3, np.pi / 3),
        )

    def steer_body(
        self,
        *,
        motion: Motion,
        tgt_body_dr_w: np.ndarray,
        tgt_body_yaw_w: float | None,
        reverse: bool = False,
    ):
        data, prev_data = motion.data, motion.prev_data

        # 1. Update the time step of the PID controllers
        self.yaw_pid.dt = data.time - prev_data.time

        # 2. Calculate the yaw error
        if tgt_body_yaw_w is None:
            tgt_body_yaw_w = data.body.yaw_w
        body_yaw_w = data.body.yaw_w
        if reverse:
            body_yaw_w += np.pi
        yaw_error = unwrap_pid_error(tgt_body_yaw_w - body_yaw_w)

        # 3. Calculate the angular velocity
        body_omega_w = self.yaw_pid(yaw_error)
        vx, vy = self.total_v * normalize(tgt_body_dr_w)
        return ControlMessage(body_vx_w=vx, body_vy_w=vy, body_omega_w=body_omega_w)

    def steer_wsk(
        self,
        *,
        wsk: WhiskerData,
        motion: Motion,
        tgt_wsk_dr_w: np.ndarray,
        tgt_body_yaw_w: float | None,
        reverse: bool = False,
    ) -> ControlMessage:
        # 1. Calculate the body control
        ctrl = self.steer_body(
            motion=motion,
            tgt_body_dr_w=tgt_wsk_dr_w,
            tgt_body_yaw_w=tgt_body_yaw_w,
            reverse=reverse,
        )
        # 2. Account for the shift in the pivot point, as the body rotates around its center and not whisker base
        # Compute the correction term from the pivot shift: ω × (A - B)
        pivot_shift_w = wsk.body_offset_w  # TODO: shouldn't it be -wsk.body_offset_w?
        corr = ctrl.body_omega_w * np.array([pivot_shift_w[1], -pivot_shift_w[0]])
        vx, vy = self.total_v * normalize(np.array([ctrl.body_vx_w, ctrl.body_vy_w]) + corr)

        return ControlMessage(body_vx_w=vx, body_vy_w=vy, body_omega_w=ctrl.body_omega_w)

    @classmethod
    def idle(cls) -> ControlMessage:
        return ControlMessage(body_vx_w=0, body_vy_w=0, body_omega_w=0)
