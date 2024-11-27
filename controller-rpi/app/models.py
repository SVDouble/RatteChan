from pydantic import BaseModel

__all__ = ["SensorData"]


class SensorData(BaseModel):
    x: int | None = None
    y: int | None = None
    z: int | None = None
    t: int | None = None
