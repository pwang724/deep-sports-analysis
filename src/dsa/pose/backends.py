"""The models behind the tracker: a person detector and ViTPose for joints.

Detection is delegated to `dsa.pose.detectors`. ViTPose-Plus is top-down: it
gets one box, warps the crop to 256x192 and predicts 17 COCO joints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dsa.pose.detectors import Detector, make_detector


@dataclass
class PoseModels:
    detector: Detector
    vitpose: tuple[object, object]   # (processor, model)
    device: str


def load_vitpose(path: str | Path, device: str, dtype: torch.dtype = torch.float32):
    """Load a ViTPose checkpoint from a local directory. Returns (processor, model)."""
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    proc = AutoProcessor.from_pretrained(str(path))
    model = VitPoseForPoseEstimation.from_pretrained(str(path), dtype=dtype).to(device).eval()
    return proc, model


def load_models(detector: str, imgsz: int, vitpose_path: str | Path, device: str,
                dtype: torch.dtype = torch.float32) -> PoseModels:
    """Build the detector by registry name at `imgsz` and load ViTPose, both in `dtype`."""
    det = make_detector(detector, imgsz=imgsz, dtype=str(dtype).removeprefix("torch."), device=device)
    return PoseModels(detector=det, vitpose=load_vitpose(vitpose_path, device, dtype), device=device)


def sanitize_boxes(xyxy: np.ndarray, min_side: float = 2.0) -> np.ndarray:
    """Widen boxes thinner than `min_side` px; the ViTPose affine warp is singular on them."""
    xyxy = np.asarray(xyxy, dtype=float).copy()
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    xyxy[:, 2] = xyxy[:, 0] + np.maximum(w, min_side)
    xyxy[:, 3] = xyxy[:, 1] + np.maximum(h, min_side)
    return xyxy


@torch.inference_mode()
def run_vitpose(proc, model, device: str, rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-down joints for boxes on one RGB frame. Returns (N, 17, 2) xy and (N, 17) scores."""
    if len(xyxy) == 0:
        return np.zeros((0, 17, 2)), np.zeros((0, 17))
    xywh = sanitize_boxes(xyxy)
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    boxes = [xywh.tolist()]
    inputs = proc(rgb, boxes=boxes, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    # ViTPose-Plus is a mixture of experts over datasets; expert 0 is COCO.
    dataset_index = torch.zeros(len(xywh), dtype=torch.long, device=device)
    out = model(**inputs, dataset_index=dataset_index)
    out.heatmaps = out.heatmaps.float()  # post-processing expects fp32
    res = proc.post_process_pose_estimation(out, boxes=boxes)[0]
    xy = np.stack([r["keypoints"].cpu().numpy() for r in res])
    sc = np.stack([r["scores"].cpu().numpy() for r in res])
    return xy, sc
