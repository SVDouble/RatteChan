import math
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy import interpolate

assets = Path("../assets").absolute()


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
    if np.linalg.norm(resampled[0] - resampled[-1]) < 1e-5:
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
    for curve, body_name in zip(curves, body_names):
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


def create_ref_interpolator(ref_curve: np.ndarray) -> Callable[[float], float]:
    # Convert reference curve to polar coordinates.
    angles = np.arctan2(ref_curve[:, 1], ref_curve[:, 0])
    angles = np.unwrap(angles)
    radii = np.sqrt(ref_curve[:, 0] ** 2 + ref_curve[:, 1] ** 2)
    # Append the first point shifted by 2π for periodicity.
    angles = np.concatenate((angles, [angles[0] + 2 * np.pi]))
    radii = np.concatenate((radii, [radii[0]]))

    def interp(theta: float) -> float:
        theta_mod = np.mod(theta - angles[0], 2 * np.pi) + angles[0]
        return np.interp(theta_mod, angles, radii)

    return interp


def wrap_cutouts(
    target_curve: np.ndarray,
    reference_curve: np.ndarray,
    wrap_percent: float,  # fraction of points at each cutout end to modify (e.g., 0.1 for 10%)
    displacement: float,  # scaling factor: 1 means no change, -0.5 moves endpoints inward (closer to ref), 1.5 moves them outward (away from ref)
    gap_threshold_factor: float = 3.0,  # factor over median distance to detect a gap
) -> np.ndarray:
    """
    For each cutout (detected as a large gap) in target_curve, modify the endpoints
    by gradually adjusting their radial coordinate. The new radius is calculated as:
       new_r = r + (displacement - 1) * (r - r_ref) * weight
    where r_ref is the reference radius (from an interpolator) at the point's angle,
    and weight (from 0 to 1) controls the gradual transition.
    Returns the modified curve as one concatenated np.ndarray.
    """
    diffs = np.linalg.norm(np.diff(target_curve, axis=0), axis=1)
    median_diff = np.median(diffs)
    gap_indices = np.where(diffs > gap_threshold_factor * median_diff)[0]

    segments: list[np.ndarray] = []
    start = 0
    for gap_idx in gap_indices:
        segments.append(target_curve[start : gap_idx + 1])
        start = gap_idx + 1
    segments.append(target_curve[start:])

    ref_interp = create_ref_interpolator(reference_curve)
    modified_segments: list[np.ndarray] = []
    for seg in segments:
        n_points = len(seg)
        if n_points < 2:
            modified_segments.append(seg)
            continue
        mod_seg = seg.copy()
        n_wrap = max(2, int(wrap_percent * n_points))
        # Modify start endpoint of segment.
        for i in range(n_wrap):
            weight = (n_wrap - i) / n_wrap  # weight: 1 at the very endpoint, 0 at the boundary of unmodified region
            x, y = mod_seg[i]
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
            r_ref = ref_interp(theta)
            # New radius: positive displacement moves endpoints away from reference.
            new_r = r + (displacement - 1) * (r - r_ref) * weight
            mod_seg[i] = np.array([new_r * math.cos(theta), new_r * math.sin(theta)])
        # Modify end endpoint of segment.
        for i in range(n_points - n_wrap, n_points):
            weight = (i - (n_points - n_wrap)) / (n_wrap - 1) if n_wrap > 1 else 1.0
            x, y = mod_seg[i]
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
            r_ref = ref_interp(theta)
            new_r = r + (displacement - 1) * (r - r_ref) * weight
            mod_seg[i] = np.array([new_r * math.cos(theta), new_r * math.sin(theta)])
        modified_segments.append(mod_seg)
    return np.concatenate(modified_segments, axis=0)


def generate_sine_wave():
    x = np.linspace(0, 2, 100)
    curve1 = np.column_stack((x, np.sin(x)))
    d = 0.5
    curve2 = np.column_stack((x, np.sin(x) + d))
    return curve1, curve2


# Generate squiggly circle curves: an outer curve with sine-modulated radius and an inner curve offset by gap_percent.
def generate_squiggly_circles(base_radius=1.0, sine_amp=0.1, sine_freq=5, gap_percent=20, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # angular positions
    # Outer curve: base radius modulated by a sine wave.
    r_outer = base_radius + sine_amp * np.sin(sine_freq * theta)
    outer = np.column_stack((r_outer * np.cos(theta), r_outer * np.sin(theta)))
    # Inner curve: each radius reduced by gap_percent.
    inner = np.column_stack(
        ((r_outer * (1 - gap_percent / 100)) * np.cos(theta), (r_outer * (1 - gap_percent / 100)) * np.sin(theta))
    )
    return outer, inner


def run():
    outer, inner = generate_squiggly_circles()
    outer = remove_curve_segment(outer, 0, 20)
    outer = wrap_cutouts(outer, inner, 0.1, 2)

    # Generate curves and plot them.
    generate_mujoco_xml(
        curves=[outer, inner],
        resolution=0.01,
        wall_thickness=0.02,
        wall_height=0.1,
        color="0.2 0.5 0.1 1",
        model_name="walls",
        body_names=["wall1", "wall2"],
        output_file=str(assets / "walls.xml"),
    )


if __name__ == "__main__":
    run()
