from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from whisker_simulation.utils import rotate

__all__ = ["Config", "SplineConfig", "WhiskerConfig", "BodyConfig", "WhiskerId", "Orientation"]

np.set_printoptions(precision=3, suppress=True)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type WhiskerId = Literal["r0", "l0"]


class Orientation(IntEnum):
    LEFT = 1  # also: ccw
    RIGHT = -1  # also: cw
    NEUTRAL = 0

    def flip(self) -> Self:
        match self:
            case self.LEFT:
                return self.RIGHT
            case self.RIGHT:
                return self.LEFT
            case self.NEUTRAL:
                return self


class SplineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="spline_")

    keypoint_distance: float = 1e-3
    n_keypoints: int = 7  # must be odd
    smoothness: float = 0.1

    log_level: LogLevel = Field(alias="log_level", default="INFO")


class WhiskerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="wsk_", arbitrary_types_allowed=True)

    name: str
    defl_model: str
    defl_threshold: float = 2e-5
    tgt_defl_abs: float = 3e-4
    disengaged_duration_threshold: float = 0.1

    defl_sensor_name: str
    angle_from_body: float
    offset_from_body: np.ndarray

    @computed_field(repr=False)
    @cached_property
    def side(self) -> Orientation:
        return Orientation.LEFT if 0 < self.angle_from_body % (2 * np.pi) < np.pi else Orientation.RIGHT

    lowpass_cutoff: float = 0.5
    lowpass_baseline: float = 0


class BodyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="body_")

    tilt: float = 0
    total_v: float = 0.05

    x_mj_sensor_name: str = "body_x_w"
    y_mj_sensor_name: str = "body_y_w"
    z_mj_sensor_name: str = "body_z_w"
    yaw_mj_sensor_name: str = "body_yaw_w"
    mj_body_name: str = "platform"


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # simulation setup
    project_path: Path = Path(__file__).parents[1].resolve()
    model_path: Path = project_path / "models/whisker.xml"
    assets_path: Path = project_path / "assets"
    local_assets_path: Path = assets_path / "local"
    control_rps: int = 30

    # recording setup
    recording_duration: float = 160
    recording_camera_fps: int = 30

    show_gui: bool = True
    use_monitor: bool = True
    detect_anomalies: bool = False
    debug: bool = False
    log_level: LogLevel = "INFO"
    generate_demo_assets: bool = True

    # controller parameters
    # the offsets need to be rotated as the body yaw is measured from the tip
    spline: SplineConfig = SplineConfig()
    body: BodyConfig = BodyConfig()
    _body_com_s: np.ndarray = np.array([0, 0.072])
    whiskers: dict[WhiskerId, WhiskerConfig] = {
        "r0": WhiskerConfig(
            defl_model="whisker_simulation.controller.deflection_model.DeflectionModelRight",
            defl_sensor_name="wsk_r0_defl",
            angle_from_body=np.radians(-105),
            offset_from_body=rotate(np.array([0.030, 0.125]) - _body_com_s, -np.pi / 2),
            name="R0",
        ),
        "l0": WhiskerConfig(
            defl_model="whisker_simulation.controller.deflection_model.DeflectionModelLeft",
            defl_sensor_name="wsk_l0_defl",
            angle_from_body=np.radians(105),
            offset_from_body=rotate(np.array([-0.030, 0.125]) - _body_com_s, -np.pi / 2),
            name="L0",
        ),
    }
