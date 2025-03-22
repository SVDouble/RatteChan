import importlib
import logging
import os
from functools import lru_cache

import numpy as np
from rich.logging import Console, RichHandler

__all__ = ["get_logger", "rotate", "normalize", "unwrap_pid_error", "import_class", "prettify"]


def create_console(**kwargs) -> Console:
    return Console(color_system="256", width=150, style="blue", markup=True, **kwargs)


def prettify(obj) -> str:
    with open(os.devnull, "w") as f:
        console = create_console(record=True, file=f)
        console.print(obj)
        return console.export_text()


@lru_cache()
def get_logger(module_name: str, *, log_level: str):
    logger = logging.getLogger(module_name)
    handler = RichHandler(console=create_console(), enable_link_path=False, rich_tracebacks=True, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def rotate(v: np.ndarray, theta: float) -> np.ndarray:
    # noinspection PyPep8Naming
    r_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return r_matrix @ v


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / np.linalg.norm(v)


def unwrap_pid_error(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def import_class(class_str: str):
    module_name, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
