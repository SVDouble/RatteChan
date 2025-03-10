from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter

from whisker_simulation.controller.deflection_model import DeflectionModel
from whisker_simulation.models import SensorData
from whisker_simulation.utils import rotate_ccw

__all__ = ["TipEstimator"]


class TipEstimator:
    def __init__(self, defl_model: DeflectionModel, initial_data: SensorData):
        self.defl_model = defl_model
        self.initial_x = self.defl_model(initial_data.wr0_defl).reshape(-1, 1)

        self.tip_s_filter = KalmanFilter(dim_x=2, dim_z=2)
        self.tip_s_filter.H = np.eye(2)
        self.tip_s_filter.P *= 10
        self.tip_s_filter.Q = np.eye(2) * 0.01
        self.tip_s_filter.x = self.initial_x
        self.tip_s_deque = deque(maxlen=20)

    def update_wr0_yaw_s(self, data: SensorData) -> None:
        # get local tip position from deflection
        tip_s = self.defl_model(data.wr0_defl)

        # filter the tip position using the kalman filter
        self.tip_s_deque.append(tip_s)
        tip_s = tip_s.reshape(-1, 1)  # kalman filter expects a column vector
        if len(self.tip_s_deque) > 1:
            self.tip_s_filter.R = np.cov(np.array(self.tip_s_deque), rowvar=False)
        else:
            self.tip_s_filter.x = tip_s
        self.tip_s_filter.predict()
        self.tip_s_filter.update(tip_s)

    def get_w(self, data: SensorData) -> np.ndarray:
        tip_s = self.tip_s_filter.x.flatten()
        return data.body_r_w + rotate_ccw(tip_s, data.wr0_yaw_w)

    def reset(self) -> None:
        self.tip_s_deque.clear()
        self.tip_s_filter.x = self.initial_x
        self.tip_s_filter.R = np.eye(2)
