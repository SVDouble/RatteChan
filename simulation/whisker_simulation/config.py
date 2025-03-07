from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

__all__ = ["Config"]


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # simulation setup
    model_path: Path = "models/whisker.xml"
    control_rps: int = 30

    # simulation model setup
    body_wr0_angle: float = -np.pi / 2
    body_wr0_offset_s: np.ndarray = np.array([0.030, 0.025])
    body_wl0_angle: float = np.pi / 2
    body_wl0_offset_s: np.ndarray = np.array([-0.030, 0.025])

    # recording setup
    recording_duration: float = 160
    recording_camera_fps: int = 30

    show_gui: bool = True
    use_monitor: bool = True
    detect_anomalies: bool = False
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    defl_model: str = "whisker_simulation.controller.deflection_model.DeflectionModel"
