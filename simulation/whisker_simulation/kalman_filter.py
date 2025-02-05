import numpy as np
from filterpy.kalman import KalmanFilter

__all__ = ['WhiskerKalmanFilter']


class WhiskerKalmanFilter:
    def __init__(self, dim=2, init_state=(-0.015, 0.098), init_var=10, q_scale=0.01):
        self.kf = KalmanFilter(dim_x=dim, dim_z=dim)
        self.kf.x = np.array(init_state)
        self.kf.F = np.eye(dim)
        self.kf.H = np.eye(dim)
        self.kf.P *= init_var
        self.kf.R = np.eye(dim)
        self.kf.Q = np.eye(dim) * q_scale

    def update_noise_matrices(self, measurements):
        zset = np.array(measurements)
        self.kf.R = np.cov(zset[:, 0], zset[:, 1])

    def predict_and_update(self, tip_s):
        self.kf.predict()
        self.kf.update(tip_s)
        return self.kf.x.copy()
