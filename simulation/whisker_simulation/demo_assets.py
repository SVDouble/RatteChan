import math
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy import interpolate

from whisker_simulation.config import Config

__all__ = ["generate_demo_assets", "has_demo_assets"]


# Resample a curve uniformly by arc length using a spline.
def resample_curve_spline(curve, resolution):
    spline, _ = interpolate.make_splprep(curve.T, s=0, k=3)
    u_dense = np.linspace(0, 1, 1000)
    dense = spline(u_dense).T
    dists = np.sqrt(np.sum(np.diff(dense, axis=0) ** 2, axis=1))
    s_dense = np.concatenate(([0], np.cumsum(dists)))
    total_length = s_dense[-1]
    new_s = np.arange(0, total_length, resolution)
    if new_s[-1] != total_length:
        new_s = np.append(new_s, total_length)
    new_u = np.interp(new_s, s_dense, u_dense)
    return spline(new_u).T


# Create wall segments from a curve; return XML geoms and segment corners for plotting.
def create_walls_from_curve(
    curve: np.ndarray,
    resolution: float,
    wall_thickness: float,
    wall_height: float,
    color: str,
    overlap: float = 0.01,  # extra length added to each segment for overlap
) -> tuple[list[ET.Element], np.ndarray, list[np.ndarray]]:
    resampled: np.ndarray = resample_curve_spline(curve, resolution)  # uniformly resampled curve
    geoms: list[ET.Element] = []
    segments: list[np.ndarray] = []
    n: int = len(resampled)
    for i in range(n - 1):
        p0, p1 = resampled[i], resampled[i + 1]
        mid = (p0 + p1) / 2
        dx, dy = p1 - p0
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        effective_length = length + overlap  # extend segment length for overlap
        geom = ET.Element(
            "geom",
            attrib={
                "type": "box",
                "size": f"{effective_length / 2:.6f} {wall_thickness / 2:.6f} {wall_height:.6f}",
                "pos": f"{mid[0]:.6f} {mid[1]:.6f} 0",
                "euler": f"0 0 {angle:.6f}",
                "rgba": color,
            },
        )
        geoms.append(geom)
        theta = math.radians(angle)
        dx_half, dy_half = effective_length / 2, wall_thickness / 2
        corners_local = np.array([[dx_half, dy_half], [-dx_half, dy_half], [-dx_half, -dy_half], [dx_half, -dy_half]])
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        segments.append((R @ corners_local.T).T + mid)
    # Add extra wall segment only if the curve is closed (i.e. first and last points nearly coincide).
    if np.linalg.norm(resampled[0] - resampled[-1]) < 1e-1:
        p0, p1 = resampled[-1], resampled[0]
        mid = (p0 + p1) / 2
        dx, dy = p1 - p0
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        effective_length = length + overlap
        geom = ET.Element(
            "geom",
            attrib={
                "type": "box",
                "size": f"{effective_length / 2:.6f} {wall_thickness / 2:.6f} {wall_height:.6f}",
                "pos": f"{mid[0]:.6f} {mid[1]:.6f} 0",
                "euler": f"0 0 {angle:.6f}",
                "rgba": color,
            },
        )
        geoms.append(geom)
        theta = math.radians(angle)
        dx_half, dy_half = effective_length / 2, wall_thickness / 2
        corners_local = np.array([[dx_half, dy_half], [-dx_half, dy_half], [-dx_half, -dy_half], [dx_half, -dy_half]])
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        segments.append((R @ corners_local.T).T + mid)
    return geoms, resampled, segments


# Generate MuJoCo XML, plot walls, and save the XML.
def generate_mujoco_xml(
    curves: list[np.ndarray],
    resolution: float,
    wall_thickness: float,
    wall_height: float,
    color: str,
    model_name: str,
    body_names: list[str],
    output_file: str,
) -> None:
    # Create MuJoCo XML structure.
    mujoco = ET.Element("mujoco", attrib={"model": model_name})
    worldbody = ET.SubElement(mujoco, "worldbody")

    plt.figure()  # Single plot for all curves.
    for curve, body_name in zip(curves, body_names, strict=True):
        body = ET.SubElement(worldbody, "body", attrib={"name": body_name})
        # Create walls for this curve.
        geoms, resampled, segments = create_walls_from_curve(curve, resolution, wall_thickness, wall_height, color)
        for g in geoms:
            body.append(g)
        # Plot the resampled curve.
        plt.plot(resampled[:, 0], resampled[:, 1], label=f"Curve {body_name}")
        # Plot each wall segment.
        for corners in segments:
            plt.gca().add_patch(Polygon(corners, closed=True, edgecolor="r", facecolor="none"))
    plt.title("All Curves")
    plt.axis("equal")
    plt.legend()

    xml_str = ET.tostring(mujoco, encoding="unicode")
    xml_pretty = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_file, "w") as f:
        f.write(xml_pretty)
    plt.show()


def remove_curve_segment(curve: np.ndarray, cut_start: float, cut_end: float) -> np.ndarray:
    """
    Remove a segment from a curve defined by cut_start and cut_end percentages of the total arc length.
    Returns the remaining curve as a single concatenated np.ndarray.

    Parameters:
      curve: np.ndarray of shape (n, 2) representing the curve.
      cut_start: float, starting percentage (0-100) of arc length to remove.
      cut_end: float, ending percentage (0-100) of arc length to remove.

    Returns:
      np.ndarray of shape (m, 2): the remaining curve after removal.
    """
    # Compute cumulative arc length.
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cum_length[-1]

    # Convert percentage values to arc length values.
    removal_start_arc = total_length * (cut_start / 100.0)
    removal_end_arc = total_length * (cut_end / 100.0)

    # Interpolate a point at a given target arc length.
    def interp_point(target: float) -> np.ndarray:
        i = int(np.searchsorted(cum_length, target))
        if i == 0:
            return curve[0]
        if i >= len(curve):
            return curve[-1]
        t = (target - cum_length[i - 1]) / (cum_length[i] - cum_length[i - 1])
        return curve[i - 1] + t * (curve[i] - curve[i - 1])

    removal_start_pt = interp_point(removal_start_arc)
    removal_end_pt = interp_point(removal_end_arc)

    # Determine indices where removal starts and ends.
    i_start = int(np.searchsorted(cum_length, removal_start_arc))
    i_end = int(np.searchsorted(cum_length, removal_end_arc))

    # Create the two parts: from removal_end to end, and from start to removal_start.
    part_after = curve[i_end:].tolist()
    part_after.insert(0, removal_end_pt)
    part_before = curve[:i_start].tolist()
    part_before.append(removal_start_pt)

    # For a closed curve, join the two parts to form one continuous remainder.
    # (Assumes the curve is ordered; if open, the join creates a discontinuity.)
    new_curve = np.array(part_after + part_before)
    return new_curve


def smooth_blend(t: float | np.ndarray) -> float | np.ndarray:
    """
    A cubic blend that stays at 1 with zero derivative at t=0,
    goes to 0 with zero derivative at t=1. Specifically:
      B(t) = 1 - 3t^2 + 2t^3
    so B(0)=1, B'(0)=0, B(1)=0, B'(1)=0.
    We'll use this to smoothly reduce the offset toward 0.
    """
    return 1.0 - 3.0 * (t**2) + 2.0 * (t**3)


def segment_arclength(points: np.ndarray) -> float:
    """Compute total arc length of a polyline."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def resample_by_arclength(points: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (S, XY) where:
      S is a (n_samples,) array of uniform arc-length parameters from [0..L].
      XY is a (n_samples,2) array of coordinates resampled from 'points'.
    """
    L = segment_arclength(points)
    if L < 1e-12 or len(points) < 2:
        return np.array([0.0]), points.copy()
    # Original cumulative length
    diffs = np.diff(points, axis=0)
    seglens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate(([0], np.cumsum(seglens)))
    s_new = np.linspace(0, cumlen[-1], n_samples)
    x_samp = np.interp(s_new, cumlen, points[:, 0])
    y_samp = np.interp(s_new, cumlen, points[:, 1])
    xy = np.column_stack((x_samp, y_samp))
    return s_new, xy


def build_offset(
    xy: np.ndarray, svals: np.ndarray, f_len: float, displacement: float, ref_radii: np.ndarray
) -> np.ndarray:
    """
    Given:
      xy        : shape (N,2) coordinates
      svals     : shape (N,) arc-length param from 0..L
      f_len     : length fraction for endpoints (e.g. 0.15 means 15% at each end)
      displacement : radial scaling factor (<1 pulls in, >1 pushes out)
      ref_radii : reference radius at each sample
    Return an array of new radii that transitions from an offset near the endpoints
    to zero offset in the middle, using smooth_blend(t).
    """
    l = svals[-1] if len(svals) > 1 else 0
    if l < 1e-12:
        return np.hypot(xy[:, 0], xy[:, 1])

    r = np.hypot(xy[:, 0], xy[:, 1])
    # For the first f_len * L, we want an offset that goes from max at s=0 to 0 at s=f_len*L
    # For the last f_len * L, we want an offset that goes from 0 at s=(1-f_len)*L to max at s=L
    # "max" means (displacement-1)*(r - r_ref).
    # We'll define a local param t in [0..1], then use smooth_blend(t).

    l_end = f_len * l
    new_r = r.copy()
    for i, s in enumerate(svals):
        # Determine if we are in the start region, middle region, or end region
        if s < l_end:
            # start region
            t = s / l_end  # from 0..1
            # blend goes from 1 at t=0 to 0 at t=1
            blend = smooth_blend(t)
        elif s > (1 - f_len) * l:
            # end region
            dist_from_end = l - s
            t = dist_from_end / l_end  # from 0..1
            # blend goes from 1 at t=0 to 0 at t=1
            blend = smooth_blend(t)
        else:
            blend = 0.0
        offset = (displacement - 1.0) * (r[i] - ref_radii[i]) * blend
        new_r[i] = r[i] + offset
    return new_r


def create_ref_interpolator(ref_curve: np.ndarray, n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a uniform-sampled table of angles -> reference radius, for radial offset.
    We'll do a simple nearest-angle or linear angle interpolation.
    This is simpler if your shape is star-like around (0,0).
    Returns (angles, radii).
    """
    # Convert reference curve to polar and sort by angle
    x, y = ref_curve[:, 0], ref_curve[:, 1]
    th = np.unwrap(np.arctan2(y, x))
    r = np.hypot(x, y)
    # Sort by ascending angle
    idx_sort = np.argsort(th)
    th_sorted = th[idx_sort]
    r_sorted = r[idx_sort]
    # Extend 2π for wrap-around
    th_sorted = np.concatenate((th_sorted, [th_sorted[0] + 2 * math.pi]))
    r_sorted = np.concatenate((r_sorted, [r_sorted[0]]))
    # Create a uniform angle grid from [th_sorted[0]..th_sorted[-1]]
    angle_min, angle_max = th_sorted[0], th_sorted[-1]
    angle_table = np.linspace(angle_min, angle_max, n_samples)
    radius_table = np.interp(angle_table, th_sorted, r_sorted)
    return angle_table, radius_table


def interp_ref_radius(theta: float, angle_table: np.ndarray, radius_table: np.ndarray) -> float:
    """Linear interpolation of reference radius given an angle."""
    # Map theta into [angle_min..angle_min+2π]
    while theta < angle_table[0]:
        theta += 2 * math.pi
    while theta > angle_table[-1]:
        theta -= 2 * math.pi
    return np.interp(theta, angle_table, radius_table)


def wrap_segment_preserve_length(
    seg: np.ndarray,
    angle_table: np.ndarray,
    radius_table: np.ndarray,
    wrap_fraction: float,
    displacement: float,
    num_iter: int = 5,
    num_samples: int = 200,
) -> np.ndarray:
    """
    Smoothly push/pull the endpoints of 'seg' in a radial sense, preserving its original arc length.
    1) Resample 'seg' by arc length to get (svals, xy)
    2) Build new radius array with a smooth blend near endpoints
    3) Iteratively scale that offset so final length == original length
    4) Re-sample final_xy back to the original number of points in 'seg'
    """
    l_orig = segment_arclength(seg)
    if l_orig < 1e-12 or len(seg) < 2:
        return seg

    # 1) Resample the original segment
    svals, xy = resample_by_arclength(seg, num_samples)
    th_xy = np.arctan2(xy[:, 1], xy[:, 0])
    # Reference radii at each sample
    ref_r = np.array([interp_ref_radius(a, angle_table, radius_table) for a in th_xy])

    # 2 & 3) Iteratively adjust offset to preserve length
    alpha = 1.0
    for _ in range(num_iter):
        new_r = build_offset(xy, svals, wrap_fraction, displacement, ref_r * alpha)
        new_xy = np.column_stack((new_r * np.cos(th_xy), new_r * np.sin(th_xy)))
        l_new = segment_arclength(new_xy)
        if l_new < 1e-12:
            break
        alpha *= l_orig / l_new

    # Final pass
    final_r = build_offset(xy, svals, wrap_fraction, displacement, ref_r * alpha)
    final_xy = np.column_stack((final_r * np.cos(th_xy), final_r * np.sin(th_xy)))

    # 4) Get the arc-length parameter for final_xy, then interpolate back to len(seg) points
    if len(final_xy) < 2:
        return final_xy  # Degenerate
    svals_new, _ = resample_by_arclength(final_xy, final_xy.shape[0])  # Param for final_xy
    svals_final = np.linspace(0, svals_new[-1], len(seg))  # New param for output

    x_out = np.interp(svals_final, svals_new, final_xy[:, 0])
    y_out = np.interp(svals_final, svals_new, final_xy[:, 1])
    return np.column_stack((x_out, y_out))


def wrap_cutouts(
    target_curve: np.ndarray,
    reference_curve: np.ndarray,
    wrap_fraction: float,
    displacement: float,
    gap_threshold_factor: float = 3.0,
    num_iter: int = 5,
    num_samples: int = 200,
) -> np.ndarray:
    """
    1) Split 'target_curve' by large gaps into segments.
    2) For each segment, smoothly push/pull endpoints radially, preserving its length.
    3) Concatenate segments and return as one curve.

    wrap_fraction: fraction of segment length at each end to blend (0..0.5).
    displacement : 1 => no shift, <1 => move inward, >1 => move outward
    """
    if len(target_curve) < 2:
        return target_curve

    # Identify large gaps
    diffs = np.linalg.norm(np.diff(target_curve, axis=0), axis=1)
    median_diff = np.median(diffs) if len(diffs) else 0.0
    gap_indices = np.where(diffs > gap_threshold_factor * median_diff)[0]

    segments = []
    start = 0
    for gidx in gap_indices:
        segments.append(target_curve[start : gidx + 1])
        start = gidx + 1
    segments.append(target_curve[start:])

    # Precompute reference curve's angle->radius table
    angle_table, radius_table = create_ref_interpolator(reference_curve)

    wrapped_segments: list[np.ndarray] = []
    for seg in segments:
        if len(seg) < 2:
            wrapped_segments.append(seg)
        else:
            seg_wrapped = wrap_segment_preserve_length(
                seg,
                angle_table,
                radius_table,
                wrap_fraction=wrap_fraction,
                displacement=displacement,
                num_iter=num_iter,
                num_samples=num_samples,
            )
            wrapped_segments.append(seg_wrapped)

    return np.concatenate(wrapped_segments, axis=0)


def generate_sine_wave():
    x = np.linspace(0, 2, 100)
    curve1 = np.column_stack((x, np.sin(x)))
    d = 0.5
    curve2 = np.column_stack((x, np.sin(x) + d))
    return curve1, curve2


# Generate squiggly circle curves: an outer curve with sine-modulated radius and an inner curve offset by gap_percent.
def generate_squiggly_circles(base_radius=1.0, sine_amp=0.1, sine_freq=5, gap_percent=20, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # angular positions
    r_outer = base_radius + sine_amp * np.sin(sine_freq * theta)  # sine-modulated outer radius
    outer = np.column_stack((r_outer * np.cos(theta), r_outer * np.sin(theta)))  # outer curve coordinates
    gap = gap_percent / 100 * base_radius  # constant gap based on base_radius
    inner = np.column_stack(
        ((r_outer - gap) * np.cos(theta), (r_outer - gap) * np.sin(theta))
    )  # inner curve with constant width
    return outer, inner


# Generate rectangle with rounded corners and slight noise to prevent duplicates
def generate_rounded_rectangle(w, h, r, num_points_corner=50, num_points_edge=20, epsilon=1e-6):
    r = min(r, w / 2, h / 2)
    theta = np.linspace(0, np.pi / 2, num_points_corner, endpoint=False)

    corners = [
        (w / 2 - r, h / 2 - r, 0),
        (-w / 2 + r, h / 2 - r, np.pi / 2),
        (-w / 2 + r, -h / 2 + r, np.pi),
        (w / 2 - r, -h / 2 + r, 3 * np.pi / 2),
    ]

    points = []
    for i, (cx, cy, angle_offset) in enumerate(corners):
        # Rounded corner points
        corner_x = cx + r * np.cos(theta + angle_offset)
        corner_y = cy + r * np.sin(theta + angle_offset)
        points.append(np.column_stack((corner_x, corner_y)))

        # Straight edge points
        next_cx, next_cy, next_angle = corners[(i + 1) % 4]
        next_corner_start_x = next_cx + r * np.cos(next_angle)
        next_corner_start_y = next_cy + r * np.sin(next_angle)

        edge_x = np.linspace(corner_x[-1], next_corner_start_x, num_points_edge + 2, endpoint=False)[1:]
        edge_y = np.linspace(corner_y[-1], next_corner_start_y, num_points_edge + 2, endpoint=False)[1:]
        points.append(np.column_stack((edge_x, edge_y)))

    # Stack all points and add small random noise
    all_points = np.vstack(points)
    noise = np.random.uniform(-epsilon, epsilon, all_points.shape)
    return all_points + noise


def generate_rounded_rectangle_model(output: Path):
    rectangle = generate_rounded_rectangle(w=0.5, h=1, r=0.1)
    generate_mujoco_xml(
        curves=[rectangle],
        resolution=0.01,
        wall_thickness=0.02,
        wall_height=0.1,
        color="0.2 0.5 0.1 1",
        model_name=output.name,
        body_names=["c0"],
        output_file=str(output),
    )


def generate_tunnel_model(output: Path):
    outer, inner = generate_squiggly_circles(gap_percent=25)
    outer = remove_curve_segment(outer, 0, 20)
    outer = wrap_cutouts(outer, inner, 0.1, 1.5)

    # Generate curves and plot them.
    generate_mujoco_xml(
        curves=[outer, inner],
        resolution=0.01,
        wall_thickness=0.02,
        wall_height=0.1,
        color="0.2 0.5 0.1 1",
        model_name=output.name,
        body_names=["c0", "c1"],
        output_file=str(output),
    )


assets = {
    "rounded_rectangle": generate_rounded_rectangle_model,
    "tunnel": generate_tunnel_model,
}


def has_demo_assets(config: Config) -> bool:
    return config.local_assets_path.exists() and all(
        (config.local_assets_path / f"{model}.xml").exists() for model in assets.keys()
    )


def generate_demo_assets(config: Config):
    config.local_assets_path.mkdir(exist_ok=True)
    for model_name, generator in assets.items():
        generator(config.local_assets_path / f"{model_name}.xml")


if __name__ == "__main__":
    generate_demo_assets(config=Config())
