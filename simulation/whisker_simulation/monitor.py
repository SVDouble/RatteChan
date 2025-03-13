import mujoco
import mujoco.viewer
import numpy as np

from whisker_simulation.models import SensorData

__all__ = ["Monitor"]


class Trajectory:
    def __init__(self, color: np.ndarray, kp_d: float):
        self.keypoint_distance = kp_d

        self.kp_color = color
        self.kp_size = 0.001
        self.conn_color = color
        self.conn_width = 2
        self.mat = np.eye(3).flatten()

        self.kp: np.ndarray | None = None
        self.prev_kp: np.ndarray | None = None
        self.n_kp: int = 0

    def add_keypoint(self, keypoint: np.ndarray):
        last_point = self.kp if self.kp is not None else self.prev_kp
        if last_point is not None and np.linalg.norm(keypoint - last_point) < self.keypoint_distance:
            if self.kp is not None:
                self.kp, self.prev_kp = None, self.kp
            return
        self.prev_kp, self.kp = last_point, keypoint
        self.n_kp += 1

    def render(self, viewer: mujoco.viewer.Handle):
        if self.kp is None:
            return
        self.draw_point(viewer, self.kp)
        if self.prev_kp is not None:
            self.draw_connector(viewer, self.prev_kp, self.kp)

    def new_geom_id(self, viewer: mujoco.viewer.Handle):
        if not hasattr(viewer, "next_geom_id"):
            viewer.next_geom_id = viewer.user_scn.ngeom
        new_geom_id = viewer.next_geom_id % len(viewer.user_scn.geoms)
        viewer.next_geom_id += 1
        viewer.user_scn.ngeom = min(viewer.next_geom_id, len(viewer.user_scn.geoms))
        return new_geom_id

    def draw_point(self, viewer: mujoco.viewer.Handle, keypoint: np.ndarray):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[self.new_geom_id(viewer)],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.kp_size, 0, 0],
            pos=keypoint,
            mat=self.mat,
            rgba=self.kp_color,
        )

    def draw_connector(self, viewer: mujoco.viewer.Handle, from_kp: np.ndarray, to_kp: np.ndarray):
        geom = viewer.user_scn.geoms[self.new_geom_id(viewer)]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.kp_size, 0, 0],
            pos=from_kp,
            mat=self.mat,
            rgba=self.conn_color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            self.conn_width,
            from_kp,
            to_kp,
        )


class Monitor:
    def __init__(self):
        # contains pairs (time, (x, y))
        self.keypoint_history = []
        self.first_keypoint = None
        self.distance_eps = 1e-2
        self.rendering_distance = 1e-2

        self.body_trajectory = Trajectory(color=np.array([1, 1, 0, 1]), kp_d=0.005)
        self.wsk_r0_tip_trajectory = Trajectory(color=np.array([0, 1, 0, 1]), kp_d=0.005)
        self.wsk_l0_tip_trajectory = Trajectory(color=np.array([1, 0, 0, 1]), kp_d=0.005)
        # self.spline_kps: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)

    def add_keypoint(self, name: str, time: float, keypoint: np.ndarray):
        if self.first_keypoint is None:
            self.first_keypoint = (time, keypoint)
        self.keypoint_history.append((time, keypoint))
        if np.linalg.norm(keypoint - self.first_keypoint[1]) < self.distance_eps and time - self.first_keypoint[0] > 10:
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
        for key, p in kwargs.items():
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

    def on_simulation_step(self, viewer: mujoco.viewer.Handle, data: SensorData):
        self.body_trajectory.add_keypoint(np.array([*data.body.r_w, data.body.z_w]))
        self.wsk_r0_tip_trajectory.add_keypoint(np.array([*data.whiskers["r0"].tip_r_w, data.body.z_w]))
        self.wsk_l0_tip_trajectory.add_keypoint(np.array([*data.whiskers["l0"].tip_r_w, data.body.z_w]))

        self.body_trajectory.render(viewer)
        self.wsk_r0_tip_trajectory.render(viewer)
        self.wsk_l0_tip_trajectory.render(viewer)

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
