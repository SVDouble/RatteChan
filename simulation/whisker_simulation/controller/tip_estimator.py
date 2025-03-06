from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter

from whisker_simulation.models import SensorData, WhiskerId

__all__ = ["TipLocalEstimator"]


class TipLocalEstimator:
    def __init__(self, wsk_id: WhiskerId, initial_data: SensorData):
        self.wsk_id = wsk_id
        self.defl_model = initial_data(self.wsk_id)._defl_model
        self.initial_filter_x = initial_data(self.wsk_id).defl_offset_s.reshape(-1, 1)

        self.filter = KalmanFilter(dim_x=2, dim_z=2)
        self.filter.H = np.eye(2)
        self.filter.P *= 10
        self.filter.Q = np.eye(2) * 0.01
        self.filter.x = self.initial_filter_x
        self.deque = deque(maxlen=20)

    def update_wr0_yaw_s(self, data: SensorData) -> None:
        # get local tip position from deflection
        tip_s = self.defl_model(data(self.wsk_id).defl)

        # filter the tip position using the kalman filter
        self.deque.append(tip_s)
        tip_s = tip_s.reshape(-1, 1)  # kalman filter expects a column vector
        if len(self.deque) > 1:
            self.filter.R = np.cov(np.array(self.deque), rowvar=False)
        else:
            self.filter.x = tip_s
        self.filter.predict()
        self.filter.update(tip_s)

    def get_tip_s(self) -> np.ndarray:
        return self.filter.x.flatten()

    def reset(self) -> None:
        self.deque.clear()
        self.filter.x = self.initial_filter_x
        self.filter.R = np.eye(2)
