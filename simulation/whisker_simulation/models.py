from enum import Enum, auto
from functools import cached_property
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from whisker_simulation.utils import rotate_ccw

__all__ = ["SensorData", "ControlMessage", "ControllerState", "MotionAnalyzer"]


class SensorData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: float

    body_x_w: float
    body_y_w: float

    wr0_yaw_w: float
    wr0_defl: float

    # protected so that it is not used unintentionally
    # generally speaking, the body angle depends on the deflection sign - cw or ccw rotation
    # use body motion controller to avoid confusion
    _wr0_angle_s: float = -np.pi / 2

    @computed_field(repr=False)
    @cached_property
    def defl_model(self) -> Any:
        from whisker_simulation.controller.deflection_model import DeflectionModel

        return DeflectionModel()

    @computed_field(repr=False)
    @cached_property
    def body_r_w(self) -> np.ndarray:
        return np.array([self.body_x_w, self.body_y_w])

    @computed_field(repr=False)
    @cached_property
    def tip_r_w(self) -> np.ndarray:
        return self.body_r_w + rotate_ccw(self.defl_model(self.wr0_defl), self.wr0_yaw_w)


class ControlMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    body_vx_w: float
    body_vy_w: float
    body_omega_w: float


class ControllerState(int, Enum):
    EXPLORING = auto()
    SWIPING = auto()
    DISENGAGED = auto()
    WHISKING = auto()
    FAILURE = auto()


class MotionAnalyzer(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    data: SensorData
    prev_data: SensorData

    @computed_field(repr=False)
    @cached_property
    def defl_model(self) -> Any:
        from whisker_simulation.controller.deflection_model import DeflectionModel

        return DeflectionModel()

    @computed_field(repr=False)
    @cached_property
    def dt(self) -> float:
        return self.data.time - self.prev_data.time

    @computed_field(repr=False)
    @cached_property
    def body_dr_w(self) -> np.ndarray:
        return self.data.body_r_w - self.prev_data.body_r_w

    @computed_field(repr=False)
    @cached_property
    def body_v_w(self) -> np.ndarray:
        return self.body_dr_w / self.dt

    @computed_field(repr=False)
    @cached_property
    def body_v(self) -> np.floating:
        return np.linalg.norm(self.body_v_w)

    @computed_field(repr=False)
    @cached_property
    def tip_dr_w(self) -> np.ndarray:
        return self.data.tip_r_w - self.prev_data.tip_r_w

    @computed_field(repr=False)
    @cached_property
    def tip_v_w(self) -> np.ndarray:
        return self.tip_dr_w / self.dt

    @computed_field(repr=False)
    @cached_property
    def tip_v(self) -> np.floating:
        return np.linalg.norm(self.tip_v_w)

    @computed_field(repr=False)
    @cached_property
    def defl_offset_s(self) -> np.ndarray:
        return self.defl_model(self.data.wr0_defl)

    @computed_field(repr=False)
    @cached_property
    def defl_offset_w(self) -> np.ndarray:
        return rotate_ccw(self.defl_offset_s, self.data.wr0_yaw_w)

    @computed_field(repr=False)
    @cached_property
    def tip_drift_r_w(self) -> np.ndarray:
        # assume that the deflection does not change much in one time step
        prev_defl_offset_w = rotate_ccw(self.defl_offset_s, self.prev_data.wr0_yaw_w)
        tip_rot_dr_w = self.defl_offset_w - prev_defl_offset_w
        return self.tip_dr_w - self.body_dr_w - tip_rot_dr_w

    @computed_field(repr=False)
    @cached_property
    def tip_drift_v_w(self) -> np.ndarray:
        return self.tip_drift_r_w / self.dt

    @computed_field(repr=False)
    @cached_property
    def tip_drift_v(self) -> np.floating:
        return np.linalg.norm(self.tip_drift_v_w)
