from functools import cached_property, partial
from typing import Self

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

__all__ = ["extract_contours", "plot_contours", "Contour", "ObjectContour"]


class Contour:
    def __init__(self, xy: np.ndarray):
        self.xy: np.ndarray = xy

    @cached_property
    def kdtree(self) -> cKDTree:
        # KDTree for fast nearest neighbor search
        return cKDTree(self.xy)

    def transform(self, f) -> Self:
        self.xy = f(self.xy)
        if "kdtree" in self.__dict__:  # Invalidate cached property
            # noinspection PyPropertyAccess
            del self.kdtree
        return self

    def approximate(self, eps: float = 1e-3) -> Self:
        self.transform(partial(cv2.approxPolyDP, eps=eps, closed=True))
        return self

    def contour_distance_mean(self, contour: Self) -> float:
        if not isinstance(contour, Contour):
            raise TypeError("Expected Contour instance as argument")
        dist_ref, _ = contour.kdtree.query(self.xy)
        return float(np.mean(dist_ref))

    def contour_distance_std(self, contour: Self) -> float:
        if not isinstance(contour, Contour):
            raise TypeError("Expected Contour instance as argument")
        dist_ref, _ = contour.kdtree.query(self.xy)
        return float(np.std(dist_ref))

    def distance(self, point: np.ndarray) -> float:
        return float(self.kdtree.query(point)[0])


class ObjectContour(Contour):
    def __init__(
        self,
        xy: np.ndarray,
        *,
        index: int,
        hierarchy: np.ndarray,
        contour_map: dict[int, Self],
        obj_mask: np.ndarray,
    ):
        super().__init__(xy)
        self.index: int = index
        self.hierarchy: np.ndarray = hierarchy
        self.obj_mask: np.ndarray = obj_mask
        self.contour_map: dict[int, Self] = contour_map

    def outer_contour(self) -> Self:
        idx = self.index
        # climb until no parent
        while (parent := self.hierarchy[idx, 3]) != -1:
            idx = parent
        return self.contour_map[idx]

    def inner_contour(self) -> Self | None:
        outer = self.outer_contour()
        child = self.hierarchy[outer.index, 2]  # first child of outer
        return self.contour_map.get(child)


def get_compound_contours(frame: np.ndarray, dilate_threshold: int = 1) -> list[ObjectContour]:
    h, w, _ = frame.shape
    # Get unique labels (each pixel is an (objid, objtype) pair)
    flat = frame.reshape(-1, 2)
    unique_labels, counts = np.unique(flat, axis=0, return_counts=True)
    # Exclude background (-1,-1) if present
    labels = []
    cnt_list = []
    for lbl, cnt in zip(unique_labels, counts, strict=True):
        if not (lbl[0] == -1 and lbl[1] == -1):
            labels.append(tuple(lbl))
            cnt_list.append(cnt)
    if not labels:
        return []
    # Assume floor is the label with maximum count
    floor_label = labels[np.argmax(cnt_list)]
    # Only consider non-floor labels for merging
    non_floor = [lbl for lbl in labels if lbl != floor_label]
    # For each label, extract its pixel coordinates (rows, cols)
    label_coords = {}
    for lbl in non_floor:
        ys, xs = np.where((frame[..., 0] == lbl[0]) & (frame[..., 1] == lbl[1]))
        if len(ys) > 0:
            label_coords[lbl] = np.column_stack((ys, xs))
    # Compute bounding boxes for each label: (min_y, min_x, max_y, max_x)
    bboxes = {}
    for lbl, coords in label_coords.items():
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        bboxes[lbl] = (min_y, min_x, max_y, max_x)
    # Use union-find to group labels whose bounding boxes are adjacent/overlapping (with a threshold)
    label_list = list(label_coords.keys())
    parent = {lbl: lbl for lbl in label_list}

    def find(lbl):
        if parent[lbl] != lbl:
            parent[lbl] = find(parent[lbl])
        return parent[lbl]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            a, b = label_list[i], label_list[j]
            min_y_a, min_x_a, max_y_a, max_x_a = bboxes[a]
            min_y_b, min_x_b, max_y_b, max_x_b = bboxes[b]
            if (
                max_y_a >= min_y_b - dilate_threshold
                and max_y_b >= min_y_a - dilate_threshold
                and max_x_a >= min_x_b - dilate_threshold
                and max_x_b >= min_x_a - dilate_threshold
            ):
                union(a, b)
    groups = {}
    for lbl in label_list:
        root = find(lbl)
        groups.setdefault(root, []).append(lbl)
    # For each group, merge their coordinates and compute a contour using a minimal mask
    compound_contours = []
    for group in groups.values():
        all_coords = np.vstack([label_coords[lbl] for lbl in group])
        # Get minimal bounding rectangle for the group
        min_y, min_x = all_coords.min(axis=0)
        max_y, max_x = all_coords.max(axis=0)
        offset = np.array([min_x, min_y])
        # Create a small binary mask for just these points
        mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        shifted = all_coords - np.array([min_y, min_x])
        mask[shifted[:, 0], shifted[:, 1]] = 255

        # Plot the merged mask for this group (before contour extraction)
        # plt.figure()
        # plt.imshow(mask, cmap="gray")
        # plt.title("Merged Group Mask")
        # plt.gca().invert_yaxis()  # match image coordinates
        # plt.show()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        assert hierarchy.shape[0] == 1, "Expected a single object hierarchy"
        contour_map = {}
        for i, cnt in enumerate(contours):
            cnt = cnt.squeeze(1)
            # Offset back to original image coordinates
            cnt[:] += offset
            contour_map[i] = ObjectContour(cnt, index=i, hierarchy=hierarchy[0], contour_map=contour_map, obj_mask=mask)
        compound_contours.extend(contour_map.values())
    return compound_contours


def pixel_to_world(
    contour: np.ndarray,
    center: np.ndarray,
    world_width: float,
    world_height: float,
    image_shape: tuple[int, ...],
):
    """Transforms contour points from pixel to world coordinates.
    Assumes pixel (0,0) at top-left; world Y increases upward."""
    h, w, _ = image_shape
    cx, cy = center
    pts_world = np.empty_like(contour, dtype=float)
    # Map x: from [0, W] to [cx - world_width/2, cx + world_width/2]
    pts_world[:, 0] = cx + (contour[:, 0] - w / 2) * (world_width / w)
    # Map y: from [0, H] to [cy + world_height/2, cy - world_height/2] so that top pixel becomes highest y
    pts_world[:, 1] = cy - (contour[:, 1] - h / 2) * (world_height / h)
    return pts_world


def extract_contours(frame: np.ndarray, center: np.ndarray, width: float, height: float) -> list[ObjectContour]:
    compound_contours = get_compound_contours(frame, dilate_threshold=1)
    transform = partial(pixel_to_world, center=center, world_width=width, world_height=height, image_shape=frame.shape)
    return [contour.transform(transform) for contour in compound_contours]


def plot_contours(contours: list[Contour | np.ndarray]):
    # Plot the world contours with legend (only contour points are plotted)
    plt.figure(figsize=(10, 8))
    legend_entries = []
    rng = np.random.default_rng(42)
    for i, cnt in enumerate(contours, start=1):
        if isinstance(cnt, Contour):
            cnt = cnt.xy
        # Generate a vivid random color
        color = rng.uniform(0, 1, 3)
        plt.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=2)
        legend_entries.append(mpatches.Patch(color=color, label=f"Compound Obj {i}"))
    plt.legend(handles=legend_entries, loc="upper right")
    plt.xlabel("World X")
    plt.ylabel("World Y")
    plt.title("Compound Object Contours in World Coordinates")
    plt.axis("equal")
    plt.show()
