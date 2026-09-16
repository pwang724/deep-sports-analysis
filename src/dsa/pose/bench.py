"""Score detectors against labelled player boxes.

A labelled player counts as found when some detection above the confidence
threshold overlaps its box with IoU >= 0.5. Pure numpy/pandas, unit tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd



def iou(a: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against (N, 4) boxes."""
    if len(boxes) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], boxes[:, 2]) - np.maximum(a[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], boxes[:, 3]) - np.maximum(a[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(area, 1e-9)


def gt_recall(preds: list[tuple[np.ndarray, np.ndarray]], labels: list[dict[str, np.ndarray]],
              threshold: float, min_iou: float = 0.5) -> dict:
    """Per-player recall and mean matched IoU over images.

    `preds[i]` is (xyxy, conf) for image i; `labels[i]` maps player name to its
    xyxy box. Also reports detections per image above the threshold, which
    includes legitimate non-players (umpire, ball kids), so it is a rough
    false-positive signal rather than a precision.
    """
    found = {name: [] for name in ("top", "bottom")}
    ious = {name: [] for name in found}
    n_det = []
    for (xyxy, conf), boxes in zip(preds, labels):
        keep = xyxy[conf >= threshold]
        n_det.append(len(keep))
        for name, gt in boxes.items():
            best = iou(gt, keep).max() if len(keep) else 0.0
            found[name].append(best >= min_iou)
            if best >= min_iou:
                ious[name].append(best)
    out = {"threshold": threshold}
    for name in found:
        out[f"{name}_recall"] = float(np.mean(found[name])) if found[name] else float("nan")
        out[f"{name}_iou"] = float(np.mean(ious[name])) if ious[name] else float("nan")
    out["dets_per_image"] = float(np.mean(n_det))
    return out


def gt_report(runs: dict[str, tuple[list, dict]], labels: list[dict], thresholds=(0.25, 0.4)) -> pd.DataFrame:
    """One row per (detector, threshold) from `gt_recall`, plus seconds per image."""
    rows = []
    for name, (preds, summary) in runs.items():
        for t in thresholds:
            rows.append({"detector": name, **gt_recall(preds, labels, t), "sec_per_image": summary["sec_per_image"]})
    return pd.DataFrame(rows)
