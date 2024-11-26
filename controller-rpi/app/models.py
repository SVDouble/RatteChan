from pydantic import BaseModel

__all__ = ["SensorData"]


class SensorData(BaseModel):
    x: int
    y: int
    z: int
