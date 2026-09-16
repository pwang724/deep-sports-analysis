"""COCO-17 joint names, skeleton edges and drawing helpers."""
from __future__ import annotations

import cv2
import numpy as np

COCO17 = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee",
    "r_knee", "l_ankle", "r_ankle",
]

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

# BGR colours, one per backend, for the annotated video.
COLORS = {"rfdetr": (0, 200, 255), "vitpose": (255, 80, 200)}


def draw_pose(frame: np.ndarray, xy: np.ndarray, scores: np.ndarray, color, track_id: int, box, thr: float = 0.3) -> None:
    """Draw one skeleton and its track ID onto a BGR frame in place."""
    for a, b in SKELETON:
        if scores[a] > thr and scores[b] > thr:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2)
    for k in range(len(COCO17)):
        if scores[k] > thr:
            cv2.circle(frame, tuple(xy[k].astype(int)), 3, color, -1)
    x1, y1 = int(box[0]), int(box[1])
    cv2.putText(frame, f"id {track_id}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def joints_to_columns(xy: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Flatten (17, 2) coordinates and (17,) scores into `<joint>_x/_y/_c` columns."""
    out: dict[str, float] = {}
    for k, name in enumerate(COCO17):
        out[f"{name}_x"] = float(xy[k, 0])
        out[f"{name}_y"] = float(xy[k, 1])
        out[f"{name}_c"] = float(scores[k])
    return out
