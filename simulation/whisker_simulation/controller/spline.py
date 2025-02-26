from collections import deque

import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline

from whisker_simulation.models import SensorData
from whisker_simulation.utils import get_logger, get_monitor

__all__ = ["Spline"]

logger = get_logger(__file__)
monitor = get_monitor()


class Spline:
    def __init__(self, *, keypoint_distance: float, n_keypoints: int, smoothness: float = 0.1):
        # spline parameters
        self.keypoint_distance = keypoint_distance
        self.n_keypoints = n_keypoints
        self.smoothness = smoothness
        self.keypoint_to_body_ratio = 5

        # stateful variables
        self.keypoints = deque(maxlen=self.n_keypoints)
        self.spl: BSpline | None = None
        self.prev_body_r_w = None

    def add_keypoint(self, *, keypoint: np.ndarray, data: SensorData) -> bool:
        has_new_point = False
        # add the new tip point
        if self.keypoints:
            prev_keypoint = np.array(self.keypoints[-1])
            keypoint_d = np.linalg.norm(np.array(keypoint) - prev_keypoint)
            body_d = np.linalg.norm(data.body_r_w - self.prev_body_r_w)
            ratio_ok = 1 / self.keypoint_to_body_ratio < keypoint_d / body_d < self.keypoint_to_body_ratio
            distance_ok = min(keypoint_d, body_d) >= self.keypoint_distance
            if distance_ok and ratio_ok:
                has_new_point = True
        if not self.keypoints:
            has_new_point = True
        if has_new_point:
            self.keypoints.append(keypoint)
            self.prev_body_r_w = data.body_r_w
            monitor.add_keypoint(data.time, keypoint)
        if len(self.keypoints) < self.n_keypoints:
            return has_new_point
        if not has_new_point:
            return False

        # Update the spline
        keypoints = np.array(self.keypoints)
        # Fit spline with the interior knots
        self.spl, _ = interpolate.make_splprep(keypoints.T, s=self.smoothness)
        return True

    def reset(self) -> None:
        self.keypoints.clear()
        self.spl = None
        self.prev_body_r_w = None

    def end_kth_point_u(self, k: float) -> float:
        return 1 + k / (len(self.keypoints) - 1)

    def __call__(self, u) -> np.ndarray:
        return self.spl(u)

    def __bool__(self) -> bool:
        return self.spl is not None
