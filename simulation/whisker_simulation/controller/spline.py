from collections import deque
from typing import Self

import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline

from whisker_simulation.config import SplineConfig
from whisker_simulation.models import SensorData
from whisker_simulation.monitor import Monitor
from whisker_simulation.utils import get_logger

__all__ = ["Spline"]


class Spline:
    def __init__(self, *, config: SplineConfig, monitor: Monitor, track: bool = True):
        self.config = config
        self.monitor = monitor
        self.logger = get_logger(__file__, log_level=config.log_level)

        # spline parameters
        self.keypoint_distance = config.keypoint_distance
        self.n_history = config.n_keypoints // 2
        self.n_keypoints = config.n_keypoints
        self.smoothness = config.smoothness
        self.keypoint_to_body_ratio = 5

        # stateful variables
        self.keypoints = deque(maxlen=self.n_keypoints + self.n_history)
        self.spl: BSpline | None = None
        self.prev_data: SensorData | None = None

        # monitor parameters
        self.track = track

    def add_keypoint(self, *, keypoint: np.ndarray, data: SensorData) -> bool:
        has_new_point = False
        # add the new tip point
        if self.keypoints:
            prev_keypoint = np.array(self.keypoints[-1])
            keypoint_d = np.linalg.norm(keypoint - prev_keypoint)
            body_d = np.linalg.norm(data.body.r_w - self.prev_data.body.r_w)
            ratio_ok = 1 / self.keypoint_to_body_ratio < keypoint_d / body_d < self.keypoint_to_body_ratio
            distance_ok = min(keypoint_d, body_d) >= self.keypoint_distance
            if distance_ok and ratio_ok:
                has_new_point = True
        if not self.keypoints:
            has_new_point = True
        if has_new_point:
            self.keypoints.append(keypoint.copy())
            self.prev_data = data
            if self.track and len(self) > 2:
                self.monitor.add_keypoint(data.time, keypoint.copy())
        if len(self) < self.n_keypoints:
            return has_new_point
        if not has_new_point:
            return False

        # Update the spline
        self._update()
        return True

    def _update(self):
        keypoints = np.array(self.keypoints)[-self.n_keypoints :, :]
        self.spl, _ = interpolate.make_splprep(keypoints.T, s=self.smoothness)

    def prepend_fake_keypoint(self, keypoint: np.ndarray) -> None:
        self.keypoints.appendleft(keypoint.copy())
        if len(self) == self.n_keypoints:
            self._update()

    def stabilize(self):
        # remove the newest points as they might be unstable
        if (k_to_remove := len(self.keypoints) - len(self)) > 0:
            for _ in range(k_to_remove):
                self.keypoints.pop()
            # update the spline
            self._update()

    def reset(self, track: bool = True) -> None:
        self.keypoints.clear()
        self.spl = None
        self.prev_data = None
        self.track = track

    def copy(self) -> Self:
        spl = self.__class__(config=self.config, monitor=self.monitor, track=self.track)
        spl.keypoints = self.keypoints.copy()
        spl.prev_data = self.prev_data.model_copy(deep=True) if self.prev_data else None
        spl._update()
        return spl

    def end_kth_point_u(self, k: float) -> float:
        return 1 + k / (len(self) - 1)

    def __call__(self, u) -> np.ndarray:
        return self.spl(u)

    def __bool__(self) -> bool:
        return self.spl is not None

    def __len__(self):
        return min(len(self.keypoints), self.n_keypoints)
