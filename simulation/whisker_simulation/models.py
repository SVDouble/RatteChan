from functools import cached_property

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

__all__ = ["WorldState", "Control"]


class WorldState(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: float
    wr0_deflection: float
    body_x: float
    body_y: float
    body_yaw: float

    @computed_field(repr=False)
    @cached_property
    def body_r(self) -> np.ndarray:
        return np.array([self.body_x, self.body_y])


class Control(BaseModel):
    model_config = ConfigDict(frozen=True)

    body_vx: float
    body_vy: float
    body_omega: float
