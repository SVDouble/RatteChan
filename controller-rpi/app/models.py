from typing import Any, Annotated

from pydantic import BaseModel, BeforeValidator

__all__ = ["Sensor", "SensorData"]


def parse_hex_address(v: Any):
    if isinstance(v, str) and v.startswith("0x"):
        return int(v, 16)
    return v  # Assume it's already an integer


class Sensor(BaseModel):
    bus: int
    address: Annotated[int, BeforeValidator(parse_hex_address)]
    name: str


class SensorData(BaseModel):
    x: int | None = None
    y: int | None = None
    z: int | None = None
    t: int | None = None
