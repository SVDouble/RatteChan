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
        if np.linalg.norm(keypoint - self.first_keypoint[1]) < 0.01 and time - self.first_keypoint[0] > 10:
            self.plot_keypoints()
            self.keypoint_history = [(time, keypoint)]
            self.first_keypoint = (time, keypoint)

    def plot_keypoints(self):
        import matplotlib.pyplot as plt

        times, points = zip(*self.keypoint_history, strict=True)
        x, y = zip(*points, strict=True)
        plt.plot(x, y)
        plt.axis("equal")
        plt.show()

    def draw_spline(self, spline, title: str = "Map", **kwargs: np.ndarray):
        import matplotlib.pyplot as plt

        from whisker_simulation.controller.spline import Spline

        spline: Spline

        if spline.spl is None:
            return

        plt.figure()

        spline_points = spline(np.linspace(0, 1, 100))
        plt.plot(spline_points[0], spline_points[1], "r-", label="Spline")

        keypoints = np.array(spline.keypoints)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="b", label="Keypoints")

        pred_tip = spline([spline.end_kth_point_u(0), spline.end_kth_point_u(1)])
        plt.scatter(pred_tip[0], pred_tip[1], c="g", marker="*", s=100, label="Predicted")

        for i, (key, p) in enumerate(kwargs.items()):
            plt.scatter(p[0], p[1], c="cmykrbg"[i], marker="x", s=100, label=key.title())

        plt.legend()
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()

    def plot_defl_profile(self, defl_model):
        import matplotlib.pyplot as plt

        deflections = np.linspace(-6e-4, 6e-4, 100)
        points = defl_model(deflections)
        plt.scatter(points[:, 0], points[:, 1], c=deflections, cmap="viridis")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Deflection Model")
        plt.colorbar(label="Deflection")  # shows which color corresponds to which deflection
        plt.show()
