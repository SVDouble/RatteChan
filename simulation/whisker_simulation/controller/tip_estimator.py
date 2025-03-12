from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter

__all__ = ["TipLocalEstimator"]


class TipLocalEstimator:
    def __init__(self, initial_pos: np.ndarray):
        self.initial_filter_x = initial_pos.reshape(-1, 1)

        self.filter = KalmanFilter(dim_x=2, dim_z=2)
        self.filter.H = np.eye(2)
        self.filter.P *= 10
        self.filter.Q = np.eye(2) * 0.01
        self.filter.x = self.initial_filter_x
        self.deque = deque(maxlen=20)

    def update(self, pos: np.ndarray) -> None:
        # filter the tip position using the kalman filter
        self.deque.append(pos)
        pos = pos.reshape(-1, 1)  # kalman filter expects a column vector
        if len(self.deque) > 1:
            self.filter.R = np.cov(np.array(self.deque), rowvar=False)
        else:
            self.filter.x = pos
        self.filter.predict()
        self.filter.update(pos)

    def get(self) -> np.ndarray:
        return self.filter.x.flatten()

    def reset(self) -> None:
        self.deque.clear()
        self.filter.x = self.initial_filter_x
        self.filter.R = np.eye(2)
