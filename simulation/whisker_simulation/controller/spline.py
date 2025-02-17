from collections import deque

import numpy as np
import scipy.interpolate as interpolate

from whisker_simulation.models import WorldState
from whisker_simulation.utils import get_monitor, get_logger

__all__ = ["Spline"]

logger = get_logger(__file__)
monitor = get_monitor()


class Spline:
    def __init__(self):
        # spline parameters
        self.degree = 3
        self.n_keypoints = 7
        self.n_knots = 5
        self.smoothness = 0.1
        self.keypoint_distance = 1e-3  # TODO: use velocity to determine this

        # stateful variables
        self.keypoints = deque(maxlen=self.n_keypoints)
        self.coefficients = None
        self.prev_body_r_w = None

    def add_keypoint(self, *, keypoint: np.ndarray, state: WorldState) -> bool:
        has_new_point = False
        # add the new tip point
        if self.keypoints:
            prev_keypoint = np.array(self.keypoints[-1])
            keypoint_d = np.linalg.norm(np.array(keypoint) - prev_keypoint)
            body_d = np.linalg.norm(state.body_r_w - self.prev_body_r_w)
            if min(keypoint_d, body_d) >= self.keypoint_distance:
                has_new_point = True
        if not self.keypoints:
            has_new_point = True
        if has_new_point:
            self.keypoints.append(keypoint)
            self.prev_body_r_w = state.body_r_w
            monitor.add_keypoint(state.time, keypoint)
        if len(self.keypoints) <= self.degree:
            return has_new_point
        if not has_new_point:
            return False

        # update the spline
        keypoints = np.array(self.keypoints)
        # Compute arc-length parameterization for the keypoints
        deltas = np.diff(np.array(keypoints), axis=0)
        distances = np.sqrt((deltas**2).sum(axis=1))
        arc_length = np.concatenate(([0], np.cumsum(distances)))
        u = arc_length / arc_length[-1]  # normalized parameter [0,1]
        # Compute interior quantiles from the normalized parameter as knot locations
        quantile_levels = np.linspace(0, 1, self.n_knots)[1:-1]
        knots = np.quantile(u, quantile_levels)
        # Fit a parametric spline through the keypoints with the computed knots
        # noinspection PyTupleAssignmentBalance
        tck, _ = interpolate.splprep(
            keypoints.T, t=knots, k=self.degree, s=self.smoothness
        )
        self.coefficients = tck
        return True

    def interpolate(self, u: float) -> np.ndarray:
        return interpolate.splev(u, self.coefficients)

    def end_kth_point_u(self, k: float) -> float:
        return 1 + k / (len(self.keypoints) - 1)

    def __bool__(self) -> bool:
        return self.coefficients is not None
