from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import Sensor

__all__ = ["Settings"]


class Settings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(env_file=".env")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    mqtt_broker: str
    mqtt_port: int = 1883
    mqtt_topic: str

    publish_frequency_hz: float = 100

    sensors: list[Sensor]
