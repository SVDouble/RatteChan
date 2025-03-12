from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from whisker_simulation.utils import rotate

__all__ = ["Config", "SplineConfig"]

np.set_printoptions(precision=3, suppress=True)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class SplineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="spline_")

    keypoint_distance: float = 1e-3
    n_keypoints: int = 7  # must be odd
    smoothness: float = 0.1

    log_level: LogLevel = Field(alias="log_level", default="INFO")


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # simulation setup
    model_path: Path = "models/whisker.xml"
    control_rps: int = 30

    # simulation model setup
    # the offsets need to be rotated as the body yaw is measured from the tip
    _body_com_s: np.ndarray = np.array([0, 0.072])
    body_wsk_r0_angle: float = -np.pi / 2
    body_wsk_r0_offset: np.ndarray = rotate(np.array([0.030, 0.125]) - _body_com_s, -np.pi / 2)
    body_wsk_l0_angle: float = np.pi / 2
    body_wsk_l0_offset: np.ndarray = rotate(np.array([-0.030, 0.125]) - _body_com_s, -np.pi / 2)

    # recording setup
    recording_duration: float = 160
    recording_camera_fps: int = 30

    show_gui: bool = True
    use_monitor: bool = True
    detect_anomalies: bool = False
    debug: bool = False
    log_level: LogLevel = "INFO"

    defl_model: str = "whisker_simulation.controller.deflection_model.DeflectionModel"

    # controller parameters
    wsk_defl_threshold: float = 2e-5
    wsk_tgt_defl_abs: float = 3e-4
    body_tilt: float = 0.2
    body_total_v: float = 0.05
    spline: SplineConfig = SplineConfig()
