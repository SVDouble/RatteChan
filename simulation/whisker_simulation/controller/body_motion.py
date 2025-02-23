import numpy as np

from whisker_simulation.models import ControlMessage, SensorData
from whisker_simulation.pid import PID
from whisker_simulation.utils import get_monitor, unwrap_pid_error

__all__ = ["BodyMotionController"]

monitor = get_monitor()


class BodyMotionController:
    def __init__(self, *, total_v: float, tilt: float):
        self.total_v = total_v
        self.tilt = tilt
        self.yaw_pid = PID(
            kp=0.5,
            ki=0,
            kd=0,
            dt=0,  # will be set in the control method
            out_limits=(-np.pi / 2, np.pi / 2),
        )

    def __call__(
        self,
        *,
        data: SensorData,
        prev_data: SensorData,
        tgt_body_dr_n_w: np.ndarray,
        tgt_body_yaw_w: float,
    ) -> ControlMessage:
        np.set_printoptions(precision=3, suppress=True)

        # 0. Update the time step of the PID controllers
        self.yaw_pid.dt = data.time - prev_data.time

        # 1. Calculate the linear velocities
        vx, vy = tgt_body_dr_n_w * self.total_v

        # 2. Calculate the yaw error so that the body is slightly tilted towards the spline
        yaw_error = unwrap_pid_error(tgt_body_yaw_w - data.body_yaw_w)

        # 3. Calculate the angular velocity
        body_omega_w = self.yaw_pid(yaw_error)

        return ControlMessage(body_vx_w=vx, body_vy_w=vy, body_omega_w=body_omega_w)
