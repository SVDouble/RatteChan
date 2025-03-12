from enum import Enum, auto
from functools import cached_property
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from whisker_simulation.config import BodyConfig, Config, WhiskerConfig, WhiskerOrientation, WhiskerId
from whisker_simulation.utils import import_class, rotate

__all__ = [
    "SensorData",
    "ControlMessage",
    "ControllerState",
    "Motion",
    "WhiskerData",
    "BodyData",
    "BodyMotion",
    "WhiskerMotion",
]


class BodyData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    x_w: float
    y_w: float
    z_w: float
    yaw_w: float

    config: BodyConfig

    @computed_field(repr=False)
    @cached_property
    def r_w(self) -> np.ndarray:  # 2d
        return np.array([self.x_w, self.y_w])


class WhiskerData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    defl: float
    body_ref: BodyData
    config: WhiskerConfig

    @computed_field(repr=False)
    @cached_property
    def defl_model(self) -> Any:
        return import_class(self.config.defl_model)()

    @computed_field(repr=False)
    @cached_property
    def body_offset_w(self) -> np.ndarray:
        return rotate(self.config.body_wsk_offset, self.body_ref.yaw_w)

    @computed_field(repr=False)
    @cached_property
    def r_w(self) -> np.ndarray:
        return self.body_ref.r_w + self.body_offset_w

    @computed_field(repr=False)
    @cached_property
    def yaw_w(self) -> float:
        return self.body_ref.yaw_w + self.config.body_wsk_angle

    @computed_field(repr=False)
    @cached_property
    def defl_offset_s(self) -> np.ndarray:
        return self.defl_model(self.defl)

    @computed_field(repr=False)
    @cached_property
    def defl_offset_w(self) -> np.ndarray:
        return rotate(self.defl_model(self.defl), self.yaw_w)

    @computed_field(repr=False)
    @cached_property
    def tip_r_w(self) -> np.ndarray:
        return self.r_w + self.defl_offset_w

    @computed_field(repr=False)
    @cached_property
    def neutral_defl_offset(self) -> np.ndarray:
        return self.defl_model(0)

    @computed_field(repr=False)
    @cached_property
    def length(self) -> np.floating:
        # assume that the whisker is straight if the deflection is 0
        # this might not be the case in the future
        return np.linalg.norm(self.neutral_defl_offset)

    @computed_field(repr=False)
    @cached_property
    def is_deflected(self) -> bool:
        return abs(self.defl) > self.config.defl_threshold

    @computed_field(repr=False)
    @cached_property
    def orientation(self) -> WhiskerOrientation:
        return np.sign(self.defl) if self.is_deflected else 0


class SensorData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: float
    body: BodyData
    whiskers: dict[str, WhiskerData]

    @classmethod
    def from_mujoco_data(cls, data, config: Config) -> Self:
        def get(sensor_name: str):
            return data.sensor(sensor_name).data.item()

        # the coordinate system of the body should be such that the tip of the mouse points at 90 degrees
        # otherwise a correction factor for yaw_w might be required to compensate for the rotational offset
        # between the deflection model and the body coordinate systems respectively
        # rotating the body so that the tip points at 0 degrees seems to cause instability in mujoco
        # so sticking to the current setup for now
        body_data = BodyData(
            x_w=get(config.body.x_sensor_name),
            y_w=get(config.body.y_sensor_name),
            z_w=get(config.body.z_sensor_name),
            yaw_w=get(config.body.yaw_sensor_name),
            config=config.body,
        )
        whiskers = {
            name: WhiskerData(
                defl=get(wsk_config.defl_sensor_name),
                body_ref=body_data,
                config=wsk_config,
            )
            for name, wsk_config in config.whiskers.items()
        }
        return cls(time=data.time, body=body_data, whiskers=whiskers)


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


class BodyMotion(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    body: BodyData
    prev_body: BodyData
    dt: float

    @computed_field(repr=False)
    @cached_property
    def dr_w(self) -> np.ndarray:
        return self.body.r_w - self.prev_body.r_w

    @computed_field(repr=False)
    @cached_property
    def v_w(self) -> np.ndarray:
        return self.dr_w / self.dt

    @computed_field(repr=False)
    @cached_property
    def v(self) -> np.floating:
        return np.linalg.norm(self.v_w)


class WhiskerMotion(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    body_motion: BodyMotion
    wsk: WhiskerData
    prev_wsk: WhiskerData
    dt: float

    @computed_field(repr=False)
    @cached_property
    def tip_dr_w(self) -> np.ndarray:
        return self.wsk.tip_r_w - self.prev_wsk.tip_r_w

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
    def tip_drift_r_w(self) -> np.ndarray:
        # assume that the deflection does not change much in one time step
        prev_defl_offset_w = rotate(self.wsk.defl_offset_s, self.prev_wsk.yaw_w)
        tip_rot_dr_w = self.wsk.defl_offset_w - prev_defl_offset_w
        return self.tip_dr_w - self.body_motion.dr_w - tip_rot_dr_w

    @computed_field(repr=False)
    @cached_property
    def tip_drift_v_w(self) -> np.ndarray:
        return self.tip_drift_r_w / self.dt

    @computed_field(repr=False)
    @cached_property
    def tip_drift_v(self) -> np.floating:
        return np.linalg.norm(self.tip_drift_v_w)


class Motion(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    data: SensorData
    prev_data: SensorData

    @computed_field(repr=False)
    @cached_property
    def dt(self) -> float:
        return self.data.time - self.prev_data.time

    @computed_field(repr=False)
    @cached_property
    def body(self) -> BodyMotion:
        return BodyMotion(body=self.data.body, prev_body=self.prev_data.body, dt=self.dt)

    def for_whisker(self, wsk_id: WhiskerId) -> WhiskerMotion:
        return WhiskerMotion(
            wsk=self.data.whiskers[wsk_id],
            prev_wsk=self.prev_data.whiskers[wsk_id],
            body_motion=self.body,
            dt=self.dt,
        )
