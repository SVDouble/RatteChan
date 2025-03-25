from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import mujoco
import mujoco.viewer
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from whisker_simulation.config import Config, ExperimentConfig
from whisker_simulation.contours import Contour
from whisker_simulation.models import SensorData, Stats
from whisker_simulation.utils import combine_mean_std, format_mean_std

__all__ = ["Monitor"]


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


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

    def reset(self):
        self.kp = None
        self.prev_kp = None
        self.n_kp = 0

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
    def __init__(self, config: Config):
        self.config = config

        # contains pairs (time, (x, y))
        self.distance_eps = 1e-2
        self.rendering_distance = 1e-2

        self.body_trajectory = Trajectory(color=np.array([1, 1, 0, 1]), kp_d=0.005)
        self.wsk_r0_tip_trajectory = Trajectory(color=np.array([0, 1, 0, 1]), kp_d=0.005)
        self.wsk_l0_tip_trajectory = Trajectory(color=np.array([1, 0, 0, 1]), kp_d=0.005)

    def reset(self):
        self.body_trajectory.reset()
        self.wsk_r0_tip_trajectory.reset()
        self.wsk_l0_tip_trajectory.reset()

    def draw_spline(self, spline, *, title: str, **kwargs: np.ndarray):
        import matplotlib.pyplot as plt

        from whisker_simulation.controller.spline import Spline

        spline: Spline

        # cmap = plt.get_cmap("Set1")
        # plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[cmap(i) for i in range(cmap.N)])
        plt.figure()

        if spline:
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

        deflections = np.linspace(-6e-4, 6e-4, 1000)
        points = defl_model(deflections)
        f = plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c=deflections, cmap="viridis")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Deflection Profile of the Deflection Model")
        plt.colorbar(label="Whisker Deflection")
        plt.show()
        f.savefig(self.config.outputs_path / "deflection_profile.pdf", backend="pdf")

    def summarize_experiment(
        self,
        *,
        stats: list[Stats],
        exp_config: ExperimentConfig,
        plot_path: Path,
        body_xy: np.ndarray,
        edges: np.ndarray | None,
        contacts: np.ndarray | None,
        include_platform_trajectory: bool = False,
        preserve_axis_names: bool = False,
    ):
        if not stats:
            return

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        swap = False
        for stat in stats:
            ref_outer_contour = stat.ref_contour.outer_contour()
            # decide whether to swap axes based on outer contour’s bounding box
            h = ref_outer_contour.xy[:, 1].max() - ref_outer_contour.xy[:, 1].min()
            w = ref_outer_contour.xy[:, 0].max() - ref_outer_contour.xy[:, 0].min()
            swap = h / w > 1.25
        x, y = (1, 0) if swap else (0, 1)
        metrics = []

        # Legend text
        ref_contour_legend_text = "Reference Contour\n" + r"$C_{\text{ref}} = \{\mathbf{x}_i\}_{i=1}^N$"
        est_contour_legend_text = "Estimated Contour\n" + r"$C_{\text{est}} = \{\mathbf{y}_i\}_{i=1}^N$"

        legend_separate_points = None

        for i, stat in enumerate(stats):
            ref_contour = stat.ref_contour
            est_d = ref_contour.distance_to_points(stat.est_xy)
            valid_d = est_d[stat.est_mask]
            d_mean, d_std = np.mean(valid_d), np.std(valid_d)
            metrics.append((len(valid_d), d_mean, d_std))
            ref_outer_contour, ref_inner_contour = ref_contour.outer_contour(), ref_contour.inner_contour()

            ax.fill(ref_outer_contour.xy[:, x], ref_outer_contour.xy[:, y], color="C0", alpha=0.2)
            if ref_inner_contour is not None:
                ax.fill(ref_inner_contour.xy[:, x], ref_inner_contour.xy[:, y], color="white")

            # Plot the reference contour
            ax.plot(
                ref_contour.xy[:, x],
                ref_contour.xy[:, y],
                label=(ref_contour_legend_text if i == 0 else None),
                linestyle="--",
                linewidth=1,
                color="C0",
                zorder=2,
            )

            # Find the outliers in the estimated contour
            mask = np.abs(est_d - d_mean) > d_std
            labeled, num = label(mask)
            min_run_length = len(est_d) // 100

            outliers_mask = np.isin(
                labeled, [i for i in range(1, num + 1) if np.sum(labeled == i) >= min_run_length]
            ).flatten()

            # Plot the estimated contour, make sure that the gaps are not plotted
            est_xy = stat.est_xy.copy()
            est_xy[~stat.est_mask | outliers_mask] = np.nan
            ax.plot(
                est_xy[:, x],
                est_xy[:, y],
                label=(r"for $|d_i - \bar d| \leq \sigma_d$" if i == 0 else None),
                linewidth=1.5,
                color="C2",
                alpha=0.8,
                zorder=3,
            )

            # Plot the outliers in the estimated contour
            outliers_xy = stat.est_xy.copy()
            outliers_xy[~stat.est_mask | ~outliers_mask] = np.nan
            ax.plot(
                outliers_xy[:, x],
                outliers_xy[:, y],
                label=(r"for $|d_i - \bar d| > \sigma_d$" if i == 0 else None),
                linewidth=1.5,
                color="C3",
                alpha=0.8,
                zorder=4,
            )

            # Plot the start and end points
            finite_est_xy = stat.est_xy[stat.est_mask & np.all(np.isfinite(stat.est_xy), axis=1)]
            start, end = finite_est_xy[0], finite_est_xy[-1]
            if legend_separate_points is None:
                legend_separate_points = np.linalg.norm(start - end) > self.distance_eps
            if legend_separate_points:
                ax.scatter(start[x], start[y], marker="s", s=30, color="C0", label="Entry Point", zorder=6)
                ax.scatter(end[x], end[y], marker="s", s=30, color="C1", label="Exit Point", zorder=6)
            else:
                ax.scatter(start[x], start[y], marker="s", s=30, color="C1", label="Entry/Exit Point", zorder=6)

        # Combine the metrics
        _, combined_mean, combined_std = combine_mean_std(metrics)
        mean, std = format_mean_std(combined_mean * 1e3, combined_std * 1e3)
        text = rf"$\bar d \pm \sigma_d = {mean} \pm {std}\,\mathrm{{mm}}$"
        px, py = exp_config.metrics_placement
        ax.text(px, py, text, transform=ax.transAxes, ha="center", va="center")

        # handle the tunneling case: both whiskers are deflected at the same time
        if len(stats) == 2:
            # Compute the centerline (an estimate)
            side_a, side_b = stats[0].ref_contour.xy, stats[1].ref_contour.xy
            d_ab, idx_ab = stats[1].ref_contour.kdtree.query(side_a)
            cl_a = (side_a + side_b[idx_ab]) / 2
            d_ba, idx_ba = cKDTree(side_b[idx_ab]).query(side_a)
            cl_b = (side_b[idx_ab] + side_a[idx_ba]) / 2
            cl = (cl_a + cl_b) / 2
            cl_contour = Contour(cl)

            # Estimate the average distance to the centerline
            tunnel_mask = stats[0].est_mask & stats[1].est_mask
            cl_d = cl_contour.distance_to_points(body_xy)
            cl_mean_d = np.mean(cl_d[tunnel_mask])
            cl_std_d = np.std(cl_d[tunnel_mask])

            # Plot the centerline
            ax.plot(
                cl[:, x],
                cl[:, y],
                label="Tunnel Centerline",
                linewidth=1,
                color="C4",
                alpha=1,
                zorder=4,
            )

            # Plot the midpoints where the tunneling occurs
            valid_body_xy = body_xy.copy()
            valid_body_xy[~tunnel_mask] = np.nan
            ax.plot(
                valid_body_xy[:, x],
                valid_body_xy[:, y],
                label="Platform COM",
                linewidth=1.5,
                color="C5",
                alpha=0.8,
                zorder=5,
            )

            # Add the metrics for the tunneling case
            mean, std = format_mean_std(cl_mean_d * 1e3, cl_std_d * 1e3)
            text = rf"$\bar d_{{\mathrm{{mid}}}} \pm \sigma_{{d_{{\mathrm{{mid}}}}}} = {mean} \pm {std}\,\mathrm{{mm}}$"
            py -= 0.05
            ax.text(px, py, text, transform=ax.transAxes, ha="center", va="center")

        # Plot the body trajectory
        if include_platform_trajectory:
            ax.plot(
                body_xy[:, x],
                body_xy[:, y],
                label="Platform Trajectory",
                linestyle=":",
                linewidth=1.5,
                color="C7",
                zorder=1,
            )

        if contacts is not None:
            # Plot the contacts
            ax.scatter(
                contacts[:, x],
                contacts[:, y],
                label="Retrieval Contact\n" + r"$C_{\text{retr}} = \{\mathbf{r}_j\}_{j=1}^P \subseteq C_{\text{est}}$",
                color="black",
                marker="x",
                s=20,
                zorder=6,
            )

            # Plot the edges
            ax.scatter(
                edges[:, x],
                edges[:, y],
                label="Edges\n" + r"$E = \{\mathbf{e}_j\}_{j=1}^M \subseteq C_{\text{est}}$",
                color="black",
                alpha=0,
                marker="^",
                s=10,
                zorder=6,
            )

            # Add average contact retrieval distance as text
            cr_d = np.linalg.norm(contacts - edges, axis=1)
            cr_d_mean, cr_d_std = np.mean(cr_d), np.std(cr_d)
            cr_d_mean, cr_d_std = format_mean_std(cr_d_mean * 1e3, cr_d_std * 1e3)
            text = (
                r"$\bar d_{\mathrm{retr}} \pm \sigma_{d_{\mathrm{retr}}}"
                + rf"= {cr_d_mean} \pm {cr_d_std}\,\mathrm{{mm}}$"
            )
            py -= 0.05
            ax.text(px, py, text, transform=ax.transAxes, ha="center", va="center")

        if not preserve_axis_names:
            swap = False
        ax.set_xlabel("Y Coordinate (m)" if swap else "X Coordinate (m)")
        ax.set_ylabel("X Coordinate (m)" if swap else "Y Coordinate (m)")
        ax.axis("equal")

        # Set the title
        ax.set_title(exp_config.name)

        # Built the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        h, l = ax.get_legend_handles_labels()

        pa, pb = (3, 5) if legend_separate_points else (3, 4)
        ax.legend(
            (h[:1] + [""] + h[1:3] + [""] + h[pb:] + [""] + h[pa:pb]),
            (
                l[:1]
                + [est_contour_legend_text]
                + l[1:3]
                + [r"$d_i = \|\mathbf{x}_i - \mathbf{y}_i\|$"]
                + l[pb:]
                + [r"$d_{\text{retr}, j} = \|\mathbf{r}_j - \mathbf{e}_j\|$"]
                + l[pa:pb]
            ),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
            ncol=1,
            handler_map={str: LegendTitle()},
        )

        plt.savefig(str(plot_path), format=plot_path.suffix[1:], bbox_inches="tight")
        plt.show()
