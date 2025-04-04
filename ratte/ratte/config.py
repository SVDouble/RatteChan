from enum import IntEnum, StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import glfw
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from slugify import slugify

from ratte.utils import rotate

__all__ = [
    "Config",
    "SplineConfig",
    "WhiskerConfig",
    "BodyConfig",
    "WhiskerId",
    "Orientation",
    "RendererConfig",
    "ExperimentConfig",
    "MujocoBodyConfig",
    "MujocoGeomConfig",
    "MujocoMeshConfig",
    "ControlMessage",
    "Flag",
]

glfw.ERROR_REPORTING = "ignore"
np.set_printoptions(precision=3, suppress=True)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type WhiskerId = Literal["r0", "l0"]


class MujocoMeshConfig(BaseModel):
    name: str
    file: Path
    scale: list[float]

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = self.model_dump(exclude={"file"})
        kwargs["file"] = str(self.file)
        return kwargs


class MujocoGeomConfig(BaseModel):
    name: str
    type: Literal["mesh"]
    mesh: MujocoMeshConfig
    pos: list[float]
    euler: list[float]
    rgba: list[float]

    def to_kwargs(self) -> dict[str, Any]:
        import mujoco

        kwargs = self.model_dump(exclude={"mesh", "type"})
        kwargs["meshname"] = self.mesh.name
        kwargs["type"] = mujoco.mjtGeom.mjGEOM_MESH
        return kwargs


class MujocoBodyConfig(BaseModel):
    name: str
    geoms: list[MujocoGeomConfig]


class ControlMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    body_vx_w: float
    body_vy_w: float
    body_omega_w: float


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


class RendererConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="renderer_")

    platform_camera: str = "platform_cam"
    width: int = 1280
    height: int = 720
    fps: int = 30
    duration: float | None = None

    # contour resolution
    test_camera: str = "test_body_cam"
    test_camera_width: int = 4000
    test_camera_height: int = 4000


class Flag(StrEnum):
    # TODO: add single whisker option
    USE_PLATFORM = "use_platform"

    # For swiping policy testing
    ADD_OBJ_CIRCLE = "add_obj_circle"
    ADD_OBJ_ROUNDED_RECTANGLE = "add_obj_rounded_rectangle"
    ADD_OBJ_COMPLEX = "add_obj_complex"

    # For retrieval policy testing
    ADD_OBJ_OCTAGON = "add_obj_octagon"
    ADD_OBJ_BOX = "add_obj_box"
    ADD_OBJ_PRISM = "add_obj_prism"
    ADD_OBJ_WALL = "add_obj_wall"

    # For tunnel policy resting
    ADD_OBJ_SMOOTH_TUNNEL = "add_obj_smooth_tunnel"
    ADD_OBJ_ZIGZAG_TUNNEL = "add_obj_zigzag_tunnel"
    ADD_OBJ_CURVED_TUNNEL = "add_obj_curved_tunnel"


class ExperimentConfig(BaseSettings):
    name: str
    category: str = "default"
    flags: set[Flag]
    initial_control: ControlMessage
    timeout: float = 0
    min_loop_time: float = 5
    loop_eps: float = 5e-3
    characteristic_length: float
    metrics_placement: tuple[float, float] = (0.5, 0.5)

    @property
    def slug(self) -> str:
        return slugify(
            self.name, stopwords=["of", "with", "and", "at", "policy", "estimation", "navigation", "contour"]
        )

    def __str__(self) -> str:
        return f"Experiment('{self.slug}')"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        env_file=Path(__file__).parents[1].resolve() / "simulation.env",
    )

    project_path: Path = Path(__file__).parents[1].resolve()
    outputs_path: Path = project_path / "../ratte"
    model_path: Path = project_path / "models/platform.xml"
    assets_path: Path = project_path / "assets"
    local_assets_path: Path = assets_path / "local"

    # general setup
    debug: bool = False
    log_level: LogLevel = "INFO"
    generate_demo_assets: bool = True
    log_anomalies: bool = False

    # simulation setup
    control_rps: int = 30
    experiments: list[ExperimentConfig] = Field(default_factory=list)
    skip_categories: set[str] = Field(default_factory=set)

    # recording setup
    renderer: RendererConfig = RendererConfig()
    export_video: bool = False

    # viewer setup
    show_gui: bool = True
    use_monitor: bool = True
    track_time: bool = True

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
