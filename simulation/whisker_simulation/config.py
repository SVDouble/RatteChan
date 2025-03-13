from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from whisker_simulation.utils import rotate

__all__ = ["Config", "SplineConfig", "WhiskerConfig", "BodyConfig", "WhiskerId", "WhiskerOrientation"]

np.set_printoptions(precision=3, suppress=True)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type WhiskerId = Literal["r0", "l0"]
type WhiskerOrientation = Literal[-1, 0, 1]


class SplineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="spline_")

    keypoint_distance: float = 1e-3
    n_keypoints: int = 7  # must be odd
    smoothness: float = 0.1

    log_level: LogLevel = Field(alias="log_level", default="INFO")


class WhiskerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="wsk_", arbitrary_types_allowed=True)

    defl_model: str = "whisker_simulation.controller.deflection_model.DeflectionModel"
    defl_threshold: float = 2e-5
    tgt_defl_abs: float = 3e-4
    disengaged_duration_threshold: float = 0.1

    defl_sensor_name: str
    angle_from_body: float
    offset_from_body: np.ndarray
    side: Literal["left", "right"]


class BodyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="body_")

    tilt: float = 0.2
    total_v: float = 0.05

    x_sensor_name: str = "body_x_w"
    y_sensor_name: str = "body_y_w"
    z_sensor_name: str = "body_z_w"
    yaw_sensor_name: str = "body_yaw_w"


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # simulation setup
    model_path: Path = "models/whisker.xml"
    control_rps: int = 30

    # recording setup
    recording_duration: float = 160
    recording_camera_fps: int = 30

    show_gui: bool = True
    use_monitor: bool = True
    detect_anomalies: bool = False
    debug: bool = False
    log_level: LogLevel = "INFO"

    # controller parameters
    # the offsets need to be rotated as the body yaw is measured from the tip
    spline: SplineConfig = SplineConfig()
    body: BodyConfig = BodyConfig()
    _body_com_s: np.ndarray = np.array([0, 0.072])
    whiskers: dict[WhiskerId, WhiskerConfig] = {
        "r0": WhiskerConfig(
            defl_sensor_name="wsk_r0_defl",
            angle_from_body=-np.pi / 2,
            offset_from_body=rotate(np.array([0.030, 0.125]) - _body_com_s, -np.pi / 2),
            side="right",
        ),
        "l0": WhiskerConfig(
            defl_sensor_name="wsk_l0_defl",
            angle_from_body=np.pi / 2,
            offset_from_body=rotate(np.array([-0.030, 0.125]) - _body_com_s, -np.pi / 2),
            side="left",
        ),
    }
