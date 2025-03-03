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

    def draw_spline(self, spline, *, title: str, **kwargs: np.ndarray):
        import matplotlib.pyplot as plt

        from whisker_simulation.controller.spline import Spline

        spline: Spline

        if spline.spl is None:
            return

        cmap = plt.get_cmap("Set1")
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[cmap(i) for i in range(cmap.N)])
        plt.figure()

        spline_points = spline(np.linspace(0, 1, 100))
        plt.plot(spline_points[0], spline_points[1], linestyle="-", label="Spline")

        keypoints = np.array(spline.keypoints)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], label="Keypoints")

        spl_end = spline(1)
        plt.scatter(spl_end[0], spl_end[1], marker="*", s=100, label="Spline End")
        for i, (key, p) in enumerate(kwargs.items()):
            if p.shape == (2,):
                plt.scatter(p[0], p[1], marker="x", s=100, label=key.title())
            elif p.shape[1] == 2:
                plt.plot(p[:, 0], p[:, 1], linestyle="-", label=key.title())
            else:
                raise ValueError("Invalid shape for keypoint")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.tight_layout()
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
