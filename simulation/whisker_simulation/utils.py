import logging
from functools import lru_cache

from rich.logging import RichHandler, Console

__all__ = ["get_config", "get_logger", "get_monitor"]


console = Console(color_system="256", width=150, style="blue")


@lru_cache()
def get_config():
    from whisker_simulation.config import Config

    return Config()


@lru_cache()
def get_logger(module_name):
    logger = logging.getLogger(module_name)
    handler = RichHandler(console=console, enable_link_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    config = get_config()
    logger.setLevel(config.log_level)
    return logger


@lru_cache()
def get_monitor():
    from whisker_simulation.monitor import Monitor

    config = get_config()
    if config.use_monitor:
        return Monitor()

    class Dummy:
        def __getattr__(self, name):
            # Return a no-op function for any attribute access.
            return lambda *args, **kwargs: None

    return Dummy()
