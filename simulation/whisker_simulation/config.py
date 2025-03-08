from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from whisker_simulation.utils import rotate

__all__ = ["Config"]


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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    defl_model: str = "whisker_simulation.controller.deflection_model.DeflectionModel"
