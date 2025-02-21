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
    def __init__(self, *, control_rps: float, total_v: float, tilt: float):
        self.initial_dt = 1 / control_rps
        self.total_v = total_v
        self.tilt = tilt
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
        state: WorldState,
        prev_state: WorldState,
        tgt_body_dr_n_w: np.ndarray,
        spline_angle: float,
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

        # 1. Calculate the linear velocities
        vx, vy = tgt_body_dr_n_w * self.total_v

        # 2. Calculate the yaw error so that the body is slightly tilted towards the spline
        nose_yaw_w = state.body_yaw_w + np.pi / 2
        yaw_error = unwrap_pid_error(spline_angle - (nose_yaw_w + self.tilt))

        # 3. Calculate the angular velocity
        body_omega_w = self.yaw_pid(yaw_error)

        return Control(body_vx_w=vx, body_vy_w=vy, body_omega_w=body_omega_w)
