from pathlib import Path
from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

__all__ = ["Config"]


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    # simulation setup
    model_path: Path = "models/whisker.xml"
    control_rps: int = 100

    # recording setup
    recording_duration: float = 160
    recording_camera_fps: int = 30

    show_gui: bool = True
    use_monitor: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
