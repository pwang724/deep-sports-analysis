"""Score pose backends against hand-labelled joints.

Dataset: Tennis Player Actions (Mendeley, CC BY 4.0). 2,000 images, one player
each, 18 OpenPose joints in COCO JSON. The first 17 are COCO order; the 18th is
the neck, which neither model predicts, so it is dropped.

Backends scored, all on the labelled box so the comparison is fair:
  rfdetr_full   RF-DETR Keypoint on the whole image (bottom-up, its native mode)
  rfdetr_crop   RF-DETR Keypoint on a padded crop of the box (same pixels ViTPose sees)
  vitpose       ViTPose top-down on the box

Metrics, on joints the labeller marked visible:
  OKS   COCO object keypoint similarity per instance, mean over images
  PCK@t fraction of joints within t * box height of the label
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from dsa.pose.backends import load_vitpose, run_vitpose
from dsa.pose.bench import iou
from dsa.pose.skeleton import COCO17

# COCO per-joint falloff constants for OKS
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
BACKENDS = ("rfdetr_full", "rfdetr_crop", "vitpose")


class EvalModels:
    """RF-DETR Keypoint Preview (evaluation only; the tracker no longer uses it) plus ViTPose."""

    def __init__(self, vitpose_path: str | Path, device: str, dtype: torch.dtype = torch.float32):
        from rfdetr import RFDETRKeypointPreview

        self.rfdetr = RFDETRKeypointPreview()
        if dtype != torch.float32:
            self.rfdetr.inference(compile=False, dtype=dtype)
        self.vitpose = load_vitpose(vitpose_path, device, dtype)
        self.device = device


def crop_box(shape: tuple[int, int], xyxy: np.ndarray, pad: float) -> np.ndarray:
    """Expand a box by `pad` of its size on every side and clip to the image. Returns int xyxy."""
    H, W = shape
    x1, y1, x2, y2 = xyxy
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    return np.array([max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)]).astype(int)


def run_rfdetr_crop(rf, rgb: np.ndarray, xyxy: np.ndarray, pad: float = 0.3, threshold: float = 0.1):
    """Top-down use of the keypoint model: run it on a padded crop of each box.

    Gives RF-DETR the same pixels ViTPose sees. In each crop the detection that
    best overlaps the box is kept. Returns (N, 17, 2) xy, (N, 17) scores and an
    (N,) mask of boxes where a person was found.
    """
    n = len(xyxy)
    xy, sc, found = np.zeros((n, 17, 2)), np.zeros((n, 17)), np.zeros(n, dtype=bool)
    for i, box in enumerate(np.asarray(xyxy, dtype=float)):
        cx1, cy1, cx2, cy2 = crop_box(rgb.shape[:2], box, pad)
        kp = rf.predict(rgb[cy1:cy2, cx1:cx2], threshold=threshold, include_source_image=False)
        if not len(kp.xy):
            continue
        j = int(iou(box - [cx1, cy1, cx1, cy1], kp.data["xyxy"].astype(float)).argmax())
        xy[i] = kp.xy[j] + [cx1, cy1]
        sc[i] = kp.confidence[j]
        found[i] = True
    return xy, sc, found


def oks(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray, area: float) -> float:
    d2 = ((pred - gt) ** 2).sum(1)
    e = d2 / (2 * (SIGMAS ** 2) * (area + np.spacing(1)))
    m = vis > 0
    return float(np.exp(-e[m]).mean()) if m.any() else np.nan


def iter_labels(root: Path, limit: int | None = None):
    """Yield (action, file_name, image_path, gt (17,2), vis (17,), box xyxy) for usable labels."""
    for ann_file in sorted((root / "annotations").glob("*.json")):
        action = ann_file.stem
        d = json.load(open(ann_file))
        imgs = {im["id"]: im for im in d["images"]}
        anns = d["annotations"][:limit] if limit else d["annotations"]
        for ann in anns:
            im = imgs[ann["image_id"]]
            k = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)[:17]
            bx, by, bw, bh = ann["bbox"]
            if bw < 4 or bh < 4 or (k[:, 2] > 0).sum() < 5:
                continue  # degenerate label
            yield action, im["file_name"], root / "images" / action / im["file_name"], k[:, :2], k[:, 2], np.array([bx, by, bx + bw, by + bh], float)


def _score(name: str, xy: np.ndarray | None, gt, vis, box, action, file_name) -> dict:
    if xy is None:
        return {"action": action, "image": file_name, "backend": name, "detected": False, "oks": np.nan}
    bh = box[3] - box[1]
    err = np.hypot(*(xy - gt).T) / bh
    r = {"action": action, "image": file_name, "backend": name, "detected": True,
         "oks": oks(xy, gt, vis, (box[2] - box[0]) * bh),
         "pck05": float((err[vis > 0] < 0.05).mean()), "pck10": float((err[vis > 0] < 0.10).mean())}
    for i, jn in enumerate(COCO17):
        r[f"{jn}_err"] = err[i] if vis[i] > 0 else np.nan
    return r


def evaluate(root: str | Path, models: EvalModels, limit: int | None = None, log_every: int = 200) -> tuple[pd.DataFrame, dict]:
    """Run every backend over the dataset. Returns per-image rows and seconds per image per backend."""
    root = Path(root)
    rows, timing, n = [], {b: 0.0 for b in BACKENDS}, 0
    for action, file_name, path, gt, vis, box in iter_labels(root, limit):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t = time.time()
        kp = models.rfdetr.predict(rgb, threshold=0.3, include_source_image=False)
        timing["rfdetr_full"] += time.time() - t
        full_xy = None
        if len(kp.xy):
            ious = iou(box, kp.data["xyxy"].astype(float))
            j = int(ious.argmax())
            if ious[j] > 0.3:
                full_xy = kp.xy[j]

        t = time.time()
        crop_xy, _, found = run_rfdetr_crop(models.rfdetr, rgb, box[None])
        timing["rfdetr_crop"] += time.time() - t

        t = time.time()
        vp_xy, _ = run_vitpose(*models.vitpose, models.device, rgb, box[None])
        timing["vitpose"] += time.time() - t
        n += 1

        rows.append(_score("rfdetr_full", full_xy, gt, vis, box, action, file_name))
        rows.append(_score("rfdetr_crop", crop_xy[0] if found[0] else None, gt, vis, box, action, file_name))
        rows.append(_score("vitpose", vp_xy[0], gt, vis, box, action, file_name))
        if n % log_every == 0:
            print(f"{action} {n} images", flush=True)
    return pd.DataFrame(rows), {b: timing[b] / max(n, 1) for b in BACKENDS}


def summarize(df: pd.DataFrame, sec_per_image: dict) -> dict:
    """Per-backend detection rate, mean OKS, PCK, timing, and per-joint median error."""
    out = {}
    for b, g in df.groupby("backend"):
        gg = g[g.detected]
        out[b] = {"detected": float(g.detected.mean()), "mean_oks": float(gg.oks.mean()),
                  "pck05": float(gg.pck05.mean()), "pck10": float(gg.pck10.mean()),
                  "sec_per_image": sec_per_image.get(b),
                  "joint_err_pct": {jn: float(100 * gg[f"{jn}_err"].median()) for jn in COCO17},
                  "oks_by_action": {a: float(x) for a, x in gg.groupby("action").oks.mean().items()}}
    return out


def print_summary(summary: dict) -> None:
    names = list(summary)
    print(f"{'backend':14s} {'detected':>9s} {'OKS':>7s} {'PCK@.05':>8s} {'PCK@.10':>8s} {'s/img':>7s}")
    for b in names:
        s = summary[b]
        print(f"{b:14s} {s['detected']:9.1%} {s['mean_oks']:7.3f} {s['pck05']:8.3f} {s['pck10']:8.3f} {s['sec_per_image'] or 0:7.3f}")
    print("\nper-joint median error, % of box height:")
    print(f"{'joint':12s}" + "".join(f"{b:>14s}" for b in names))
    for jn in COCO17:
        print(f"{jn:12s}" + "".join(f"{summary[b]['joint_err_pct'][jn]:14.1f}" for b in names))
    print("\nmean OKS by action:")
    actions = sorted({a for b in names for a in summary[b]["oks_by_action"]})
    print(f"{'action':16s}" + "".join(f"{b:>14s}" for b in names))
    for a in actions:
        print(f"{a:16s}" + "".join(f"{summary[b]['oks_by_action'].get(a, float('nan')):14.3f}" for b in names))
