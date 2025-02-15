__all__ = ["Monitor"]

import numpy as np
from scipy import interpolate


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

    def draw_spline(self, spline, keypoints, u: np.ndarray, **kwargs: np.ndarray):
        import matplotlib.pyplot as plt

        u_fine = np.linspace(0, 1, 100)
        spline_points = interpolate.splev(u_fine, spline)
        predicted = interpolate.splev(u, spline)
        plt.figure()
        plt.plot(spline_points[0], spline_points[1], "r-", label="Spline")
        keypoints = np.array(keypoints)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="b", label="Keypoints")
        plt.scatter(
            predicted[0], predicted[1], c="g", marker="*", s=100, label="Predicted"
        )
        for i, (key, p) in enumerate(kwargs.items()):
            plt.scatter(p[0], p[1], c="cmykw"[i], marker="x", s=100, label=key.title())
        plt.legend()
        plt.title("Spline Fit to Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()
