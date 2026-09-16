"""Person detectors behind one interface: RGB frame in, person boxes out.

Each detector returns (xyxy, conf) as float arrays of shape (N, 4) and (N,),
keeping every person box above `min_conf` so callers can apply their own
threshold. Chosen by measurement, see docs/RESULTS-pose.md section 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Detector(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 4)), np.zeros(0)


@dataclass
class RFDetr:
    """RF-DETR detection model at an overridden input resolution. The default detector.

    `imgsz` must be a multiple of 32. Weights download into `$RF_HOME` on first use.
    """

    size: str = "medium"
    imgsz: int = 1152
    min_conf: float = 0.1
    dtype: str = "float16"
    device: str | None = None   # picked by the library: CUDA, then MPS, then CPU

    def __post_init__(self):
        import rfdetr
        import torch
        from rfdetr.assets.coco_classes import COCO_CLASSES

        cls = getattr(rfdetr, f"RFDETR{self.size.capitalize()}")
        self.model = cls(resolution=self.imgsz)
        if self.dtype != "float32":
            self.model.inference(compile=False, dtype=getattr(torch, self.dtype))
        self.person = next(i for i, n in COCO_CLASSES.items() if n == "person")
        self.name = f"rfdetr-{self.size}@{self.imgsz}"

    def detect(self, rgb):
        d = self.model.predict(rgb, threshold=self.min_conf)
        keep = d.class_id == self.person
        if not keep.any():
            return _empty()
        return d.xyxy[keep].astype(float), d.confidence[keep].astype(float)


@dataclass
class Yolo:
    """Ultralytics YOLO, person class only. AGPL; kept as the fast alternative."""

    weights: str = "yolo11m.pt"
    imgsz: int = 1280
    min_conf: float = 0.1
    dtype: str = "float16"
    device: str | None = None

    def __post_init__(self):
        from ultralytics import YOLO

        self.model = YOLO(self.weights)
        self.name = f"{Path(self.weights).stem}@{self.imgsz}"

    def detect(self, rgb):
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics reads numpy arrays as BGR
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.min_conf, classes=[0], half=self.dtype == "float16",
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return _empty()
        return r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float)


DETECTORS = {"rfdetr": RFDetr, "yolo": Yolo}


def make_detector(kind: str, **kwargs) -> Detector:
    """Build a detector by registry name, e.g. make_detector('rfdetr', imgsz=1152)."""
    if kind not in DETECTORS:
        raise ValueError(f"unknown detector {kind!r}; choose from {sorted(DETECTORS)}")
    return DETECTORS[kind](**kwargs)
