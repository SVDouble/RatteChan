from enum import Enum, auto
from functools import cached_property

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

__all__ = ["WorldState", "Control", "Mode"]


class WorldState(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: float

    body_x_w: float
    body_y_w: float
    body_yaw_w: float

    wr0_yaw_s: float

    @computed_field(repr=False)
    @cached_property
    def body_r_w(self) -> np.ndarray:
        return np.array([self.body_x_w, self.body_y_w])


class Control(BaseModel):
    model_config = ConfigDict(frozen=True)

    body_vx_w: float
    body_vy_w: float
    body_omega_w: float


class Mode(int, Enum):
    IDLE = auto()
    ENGAGED = auto()
    SLIPPING_BACKWARDS = auto()
    DISENGAGED = auto()
    FAILURE = auto()
