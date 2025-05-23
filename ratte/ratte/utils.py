import importlib
import logging
import os
from functools import lru_cache

import numpy as np
from rich.logging import Console, RichHandler

__all__ = [
    "get_logger",
    "rotate",
    "normalize",
    "unwrap_pid_error",
    "import_class",
    "prettify",
    "format_mean_std",
    "combine_mean_std",
]


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


def rotate(v: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    theta = np.atleast_1d(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    r_matrix = np.moveaxis(r_matrix, -1, 0)  # shape (N, 2, 2)
    result = np.einsum("nij,j->ni", r_matrix, v)
    return result[0] if result.shape[0] == 1 else result


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


def format_mean_std(mean: float | np.floating, std: float | np.floating) -> tuple[str, str]:
    if std == 0:
        return f"{mean:.2f}", "0"

    # Determine rounding exponent
    exp = int(np.floor(np.log10(abs(std))))
    first_digit = int(std / (10**exp))
    sig = 2 if first_digit == 1 else 1
    decimals = -exp + (sig - 1)

    # Round to nearest significant digit
    rounding_factor = 10 ** (-decimals)
    mean_rounded = rounding_factor * round(mean / rounding_factor)
    std_rounded = rounding_factor * round(std / rounding_factor)

    # Format without scientific notation
    mean_str = f"{mean_rounded:.0f}" if decimals <= 0 else f"{mean_rounded:.{decimals}f}"
    std_str = f"{std_rounded:.0f}" if decimals <= 0 else f"{std_rounded:.{decimals}f}"

    return mean_str, std_str


def combine_mean_std(experiments: list[tuple[int, np.floating, np.floating]]) -> tuple[int, np.floating, np.floating]:
    ns, means, stds = zip(*experiments, strict=True)
    ns, means, stds = np.array(ns), np.array(means), np.array(stds)

    total_n = np.sum(ns)
    combined_mean = np.sum(ns * means) / total_n
    combined_var = np.sum(ns * (stds**2 + (means - combined_mean) ** 2)) / total_n
    combined_std = np.sqrt(combined_var)

    return total_n, combined_mean, combined_std
