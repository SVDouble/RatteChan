from typing import Tuple, Optional

import numpy as np

__all__ = ["PID"]


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        out_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.dt: float = dt
        self.integral: float = 0.0
        self.last_error: float = 0.0
        self.out_limits: Tuple[Optional[float], Optional[float]] = out_limits

    def __call__(self, error: float) -> float:
        self.integral += error * self.dt
        derivative = (error - self.last_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        min_out, max_out = self.out_limits
        return np.clip(output, min_out, max_out)
