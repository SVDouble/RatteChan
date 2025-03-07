import numpy as np

from whisker_simulation.models import ControlMessage, SensorData
from whisker_simulation.pid import PID
from whisker_simulation.utils import get_monitor, unwrap_pid_error, normalize

__all__ = ["BodyMotionController"]

monitor = get_monitor()


class BodyMotionController:
    def __init__(self, *, total_v: float):
        self.total_v = total_v
        self.yaw_pid = PID(
            kp=1,
            ki=0.001,
            kd=0.001,
            dt=0,  # will be set in the control method
            out_limits=(-np.pi / 3, np.pi / 3),
        )

    def __call__(
        self,
        *,
        data: SensorData,
        prev_data: SensorData,
        tgt_wsk_dr_w: np.ndarray,
        tgt_body_yaw_w: float | None,
        orient: int,
    ) -> ControlMessage:
        np.set_printoptions(precision=3, suppress=True)
        assert orient != 0, "The orientation cannot not be zero"

        # TODO: move the pivot point to the center of the body
        # and adjust the rotation accordingly

        # 0. Update the time step of the PID controllers
        self.yaw_pid.dt = data.time - prev_data.time

        # 1. Calculate the yaw error
        if tgt_body_yaw_w is None:
            tgt_body_yaw_w = data.body.yaw_w
        yaw_error = unwrap_pid_error(tgt_body_yaw_w - data.body.yaw_w)

        # 2. Calculate the angular velocity
        body_omega_w = self.yaw_pid(yaw_error)

        # 3. Account for the shift in the pivot point, as the body rotates around its center and not whisker base
        # Compute the correction term from the pivot shift: ω × (A - B)
        wsk = data("r0")
        pivot_shift_w = wsk.body_offset_w  # TODO: shouldn't it be -wsk.body_offset_w?
        corr = body_omega_w * np.array([pivot_shift_w[1], -pivot_shift_w[0]])
        vx, vy = self.total_v * normalize(self.total_v * normalize(tgt_wsk_dr_w) + corr)

        return ControlMessage(body_vx_w=vx, body_vy_w=vy, body_omega_w=body_omega_w)

    @classmethod
    def idle(cls) -> ControlMessage:
        return ControlMessage(body_vx_w=0, body_vy_w=0, body_omega_w=0)
