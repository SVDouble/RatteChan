import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["extract_contours", "plot_contours"]


def get_compound_contours(frame: np.ndarray, dilate_threshold: int = 1, approximate: bool = False) -> list[np.ndarray]:
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

        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        for cnt in cnts:
            if approximate:
                # Approximate the contour with a custom epsilon
                eps = 1e-3
                cnt = cv2.approxPolyDP(cnt, eps, closed=True)
            cnt = cnt.squeeze(1)
            # Offset back to original image coordinates
            cnt[:, 0] += min_x
            cnt[:, 1] += min_y
            compound_contours.append(cnt)
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


def extract_contours(frame: np.ndarray, center: np.ndarray, width: float, height: float):
    compound_contours = get_compound_contours(frame, dilate_threshold=1)
    world_contours = [pixel_to_world(cnt, center, width, height, frame.shape) for cnt in compound_contours]
    return world_contours


def plot_contours(contours: list[np.ndarray]):
    # Plot the world contours with legend (only contour points are plotted)
    plt.figure(figsize=(10, 8))
    legend_entries = []
    rng = np.random.default_rng(42)
    for i, cnt in enumerate(contours, start=1):
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
