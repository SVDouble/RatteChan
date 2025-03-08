from enum import Enum, auto
from functools import cached_property
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from whisker_simulation.utils import import_class, rotate

__all__ = [
    "SensorData",
    "ControlMessage",
    "ControllerState",
    "Motion",
    "WhiskerId",
    "WhiskerData",
    "BodyData",
    "BodyMotion",
    "WhiskerMotion",
]


class BodyData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    r_w: np.ndarray  # 2D
    z_w: float
    yaw_w: float


type WhiskerId = Literal["r0", "l0"]


class WhiskerData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    defl: float
    body_angle: float
    body_offset_s: np.ndarray

    _body: BodyData
    _defl_model: Any

    def __init__(self, *, _body: BodyData, _defl_model: Any, **data):
        super().__init__(**data)
        self._body = _body
        self._defl_model = _defl_model

    @computed_field(repr=False)
    @cached_property
    def body_offset_w(self) -> np.ndarray:
        return rotate(self.body_offset_s, self._body.yaw_w)

    @computed_field(repr=False)
    @cached_property
    def r_w(self) -> np.ndarray:
        return self._body.r_w + self.body_offset_w

    @computed_field(repr=False)
    @cached_property
    def yaw_w(self) -> float:
        return self._body.yaw_w + self.body_angle

    @computed_field(repr=False)
    @cached_property
    def defl_offset_s(self) -> np.ndarray:
        return self._defl_model(self.defl)

    @computed_field(repr=False)
    @cached_property
    def defl_offset_w(self) -> np.ndarray:
        return rotate(self._defl_model(self.defl), self.yaw_w)

    @computed_field(repr=False)
    @cached_property
    def tip_r_w(self) -> np.ndarray:
        return self.r_w + self.defl_offset_w

    @computed_field(repr=False)
    @cached_property
    def neutral_defl_offset(self) -> np.ndarray:
        return self._defl_model(0)

    @computed_field(repr=False)
    @cached_property
    def length(self) -> np.floating:
        # assume that the whisker is straight if the deflection is 0
        # this might not be the case in the future
        return np.linalg.norm(self.neutral_defl_offset)


class SensorData(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: float
    body: BodyData

    _wsk_r0: WhiskerData
    _wsk_l0: WhiskerData
    _defl_model: Any

    def __init__(self, *, _wsk_r0: WhiskerData, _wsk_l0: WhiskerData, _defl_model: Any, **data):
        super().__init__(**data)
        self._wsk_r0 = _wsk_r0
        self._wsk_l0 = _wsk_l0
        self._defl_model = _defl_model

    def wsk(self, wsk_id: WhiskerId) -> WhiskerData:
        return getattr(self, f"_wsk_{wsk_id}")

    def __call__(self, wsk_id: WhiskerId) -> WhiskerData:
        return self.wsk(wsk_id)

    @classmethod
    def from_mujoco_data(cls, data, config) -> "SensorData":
        sensors = ["body_x_w", "body_y_w", "body_z_w", "body_yaw_w", "wsk_r0_defl", "wsk_l0_defl"]
        sensor_data: dict[str, float] = {sensor: data.sensor(sensor).data.item() for sensor in sensors}
        # the coordinate system of the body should be such that the tip of the mouse points at 90 degrees
        # otherwise a correction factor for yaw_w might be required to compensate for the rotational offset
        # between the deflection model and the body coordinate systems respectively
        # rotating the body so that the tip points at 0 degrees seems to cause instability in mujoco
        # so sticking to the current setup for now
        body_data = BodyData(
            r_w=np.array([sensor_data["body_x_w"], sensor_data["body_y_w"]]),
            z_w=sensor_data["body_z_w"],
            yaw_w=sensor_data["body_yaw_w"],
        )
        defl_model = import_class(config.defl_model)()
        wsk_r0 = WhiskerData(
            defl=sensor_data["wsk_r0_defl"],
            body_angle=config.body_wsk_r0_angle,
            body_offset_s=config.body_wsk_r0_offset,
            _body=body_data,
            _defl_model=defl_model,
        )
        wsk_l0 = WhiskerData(
            defl=sensor_data["wsk_l0_defl"],
            body_angle=config.body_wsk_l0_angle,
            body_offset_s=config.body_wsk_l0_offset,
            _body=body_data,
            _defl_model=defl_model,
        )
        return cls(time=data.time, body=body_data, _wsk_r0=wsk_r0, _wsk_l0=wsk_l0, _defl_model=defl_model)


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

    def wsk(self, wsk_id: WhiskerId) -> WhiskerMotion:
        return WhiskerMotion(wsk=self.data(wsk_id), prev_wsk=self.prev_data(wsk_id), body_motion=self.body, dt=self.dt)
