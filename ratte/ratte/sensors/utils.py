import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["get_logger", "get_settings"]

console = Console(color_system="256", width=150, style="blue")


@lru_cache()
def get_settings():
    from ratte.sensors.settings import Settings

    return Settings()


@lru_cache()
def get_logger(module_name):
    settings = get_settings()

    logger = logging.getLogger(module_name)
    handler = RichHandler(console=console, enable_link_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    return logger


@lru_cache()
def get_repository():
    from ratte.sensors.repository import Repository

    settings = get_settings()
    return Repository(settings)
