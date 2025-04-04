from typing import Self

import aiomqtt

from ratte.sensors.models import SensorData
from ratte.sensors.settings import Settings

__all__ = ["Repository"]


class Repository:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.mqtt = aiomqtt.Client(self._settings.mqtt_broker, self._settings.mqtt_port)

    async def __aenter__(self) -> Self:
        await self.mqtt.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mqtt.__aexit__(exc_type, exc_val, exc_tb)

    async def publish_sensor_data(self, name: str, data: SensorData):
        topic = f"{self._settings.mqtt_topic}/{name}"
        await self.mqtt.publish(topic, data.model_dump_json(), qos=0)
