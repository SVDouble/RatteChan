import numpy as np

from whisker_simulation.models import Control, WorldState
from whisker_simulation.pid import PID
from whisker_simulation.utils import rotate_ccw

__all__ = ["BodyMotionController"]


class BodyMotionController:
    def __init__(self, *, total_v: float, control_rps: float):
        self.total_v = total_v
        self.tgt_defl = -3.2e-4
        dt = 1 / control_rps
        self.pid_wr0_yaw_w = PID(
            kp=3000,
            ki=10,
            kd=0,
            dt=dt,
            out_limits=(-self.total_v, self.total_v),
        )

        self.pid_body_yaw_w = PID(
            kp=10,
            ki=0.001,
            kd=0,
            dt=dt,
            out_limits=(-2 * np.pi, 2 * np.pi),
        )

    def control(self, *, tgt_body_yaw_w: float, state: WorldState) -> Control:
        # 1. Calculate the linear velocity of the body using the PID controller based on the whisker deflection
        # why -deflection? it just works
        body_vx_s = self.pid_wr0_yaw_w(-(self.tgt_defl - state.wr0_yaw_s))
        body_vy_s = np.sqrt(self.total_v**2 - body_vx_s**2)
        body_v_w = rotate_ccw(np.array([body_vx_s, body_vy_s]), tgt_body_yaw_w)

        # 2. Calculate the angular velocity of the body using the PID controller based on the target body yaw
        body_omega = self.pid_body_yaw_w(tgt_body_yaw_w - state.body_yaw_w)

        return Control(
            body_vx_w=float(body_v_w[0]),
            body_vy_w=float(body_v_w[1]),
            body_omega_w=body_omega,
        )
