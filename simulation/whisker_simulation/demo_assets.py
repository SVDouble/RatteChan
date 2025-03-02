import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from obj2mjcf.cli import Args as ObjArgs
from obj2mjcf.cli import CoacdArgs, process_obj
from shapely.geometry import Polygon

assets = Path("../assets").absolute()


def create_demo_cog():
    N, R, b, height = 8, 0.5, 0.1, 0.1  # number of spikes, circle radius, chord length at circle, extrusion height
    apex_angles = np.linspace(10, 120, N)  # varying apex angles for spikes (degrees)
    gamma_deg = np.degrees(np.arcsin(b / (2 * R)))  # half-central angle for chord
    num_points_side = 10  # subdivision points along each spike side
    num_points_arc = 20  # subdivision points along base (circle) arcs

    def circle_pt(deg):
        rad = np.radians(deg)
        return R * np.cos(rad), R * np.sin(rad)

    def subdivide_line(p0, p1, num_points):
        # Returns list of points along the line from p0 to p1 (including endpoints)
        return [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t) for t in np.linspace(0, 1, num_points)]

    # Compute spike geometry: for each spike, get left base point, apex, and right base point on circle
    spikes = []
    for i in range(N):
        phi = 360 * i / N  # central orientation of spike
        L_angle = phi - gamma_deg  # left base angle on circle
        R_angle = phi + gamma_deg  # right base angle on circle
        L_pt = circle_pt(L_angle)
        R_pt = circle_pt(R_angle)
        M = ((L_pt[0] + R_pt[0]) / 2, (L_pt[1] + R_pt[1]) / 2)  # midpoint of chord
        alpha = np.radians(apex_angles[i])  # spike apex angle in radians
        s = b / (2 * np.sin(alpha / 2))  # side length of spike triangle
        h = s * np.cos(alpha / 2)  # height from chord to apex
        m_len = np.hypot(M[0], M[1])
        nx, ny = M[0] / m_len, M[1] / m_len  # outward unit vector (from center)
        apex = (M[0] + h * nx, M[1] + h * ny)  # apex point of spike
        spikes.append((L_pt, apex, R_pt))

    # Build full outline by connecting each spike (with subdivided sides) and then adding a subdivided arc along the circle base between spikes
    outline = []
    for i in range(N):
        L_pt, apex, R_pt = spikes[i]
        # Subdivide left side (L_pt to apex) and right side (apex to R_pt)
        side1 = subdivide_line(L_pt, apex, num_points_side)
        side2 = subdivide_line(apex, R_pt, num_points_side)
        spike_pts = side1[:-1] + side2  # avoid duplicating the apex point
        outline.extend(spike_pts)

        # Determine arc on the circle from current spike's right base to next spike's left base
        next_L_pt = spikes[(i + 1) % N][0]
        # Convert current R_pt and next_L_pt to angles
        a1 = np.degrees(np.arctan2(R_pt[1], R_pt[0]))
        a2 = np.degrees(np.arctan2(next_L_pt[1], next_L_pt[0]))
        if a2 <= a1:
            a2 += 360
        arc_pts = [circle_pt(a) for a in np.linspace(a1, a2, num_points_arc, endpoint=False)[1:]]
        outline.extend(arc_pts)

    outline.append(outline[0])  # close the polygon

    poly = Polygon(outline)
    mesh = trimesh.creation.extrude_polygon(poly, height=height).subdivide()  # extrude and subdivide for high-res mesh

    # 2D preview of the outline
    x, y = poly.exterior.xy
    plt.plot(x, y, marker="o", markersize=3)
    plt.axis("equal")
    plt.title("2D Outline with Subdivided Spike Sides and Base Arcs")
    plt.show()

    cog = assets / "cog"
    if cog.exists():
        shutil.rmtree(cog)
    cog = assets / "cog.obj"
    mesh.export(cog)
    coacd_args = CoacdArgs(preprocess_resolution=50)
    obj_args = ObjArgs(decompose=True, save_mjcf=True, coacd_args=coacd_args, obj_dir="")
    process_obj(cog, obj_args)
    cog.unlink()
