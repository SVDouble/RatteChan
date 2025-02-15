__all__ = ["Monitor"]

import numpy as np


class Monitor:
    def __init__(self):
        # contains pairs (time, (x, y))
        self.keypoint_history = []
        self.first_keypoint = None
        self.distance_eps = 1e-2

    def add_keypoint(self, time: float, keypoint: np.ndarray):
        if self.first_keypoint is None:
            self.first_keypoint = (time, keypoint)
        self.keypoint_history.append((time, keypoint))
        if (
            np.linalg.norm(keypoint - self.first_keypoint[1]) < 0.01
            and time - self.first_keypoint[0] > 10
        ):
            self.plot_keypoints()
            self.keypoint_history = [(time, keypoint)]
            self.first_keypoint = (time, keypoint)

    def plot_keypoints(self):
        import matplotlib.pyplot as plt

        times, points = zip(*self.keypoint_history)
        x, y = zip(*points)
        plt.plot(x, y)
        plt.axis("equal")
        plt.show()
