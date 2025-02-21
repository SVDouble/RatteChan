import logging
from functools import lru_cache

import numpy as np
from rich.logging import RichHandler, Console

__all__ = [
    "get_config",
    "get_logger",
    "get_monitor",
    "rotate_ccw",
    "normalize",
    "unwrap_pid_error",
]


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


def rotate_ccw(v: np.ndarray, theta: float) -> np.ndarray:
    # noinspection PyPep8Naming
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ v


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / np.linalg.norm(v)


def unwrap_pid_error(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi
