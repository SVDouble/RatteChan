from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings"]


class Settings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(env_file=".env")

    broker_url: str
    broker_port: int = 1883
    mqtt_topic: str
    publish_frequency_hz: float = 100

    sensor_addresses: list[int]
    clock_pin: int = 11
    mosi_pin: int = 10
    miso_pin: int = 9
