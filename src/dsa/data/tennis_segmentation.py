"""TennisSegmentation: 197 broadcast end-on frames with masks for both players.

Source: huggingface.co/datasets/julia-wenkmann/TennisSegmentation (no stated
license; used for evaluation only). Masks are converted to boxes here. Two
frames carry a "player" mask that covers most of the court, so any label
larger than `MAX_AREA_FRAC` of the frame is dropped. Every frame has a ball
mask; in 8 the ball is two interlaced copies, so its centre is the mean of all
ball pixels. Frames are 1280 x 720 (108) or 1920 x 1080 (89).
"""
from __future__ import annotations

import glob
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_IDS = {"top": 3, "bottom": 4}   # far player, near player (id2label.json)
BALL_ID = 2
MAX_AREA_FRAC = 0.1


@dataclass
class Sample:
    index: int
    image: np.ndarray                 # RGB (H, W, 3)
    boxes: dict[str, np.ndarray]      # player name -> xyxy, only players with a valid label
    ball: np.ndarray | None = None    # (x, y) centre of the ball mask, None if there is none


def mask_to_boxes(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Tight xyxy box per player class present in a label mask, minus degenerate labels."""
    H, W = mask.shape
    boxes = {}
    for name, cid in CLASS_IDS.items():
        ys, xs = np.where(mask == cid)
        if not len(xs):
            continue
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
        if (box[2] - box[0]) * (box[3] - box[1]) / (W * H) <= MAX_AREA_FRAC:
            boxes[name] = box
    return boxes


def mask_to_ball(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask == BALL_ID)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5]) if len(xs) else None


def load(root: str | Path) -> list[Sample]:
    """Read every parquet shard under `root/data` and return decoded samples in file order."""
    files = sorted(glob.glob(str(Path(root) / "data" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    samples = []
    for i, row in df.iterrows():
        image = np.array(Image.open(io.BytesIO(row["pixel_values"]["bytes"])).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(row["label"]["bytes"])))
        samples.append(Sample(i, image, mask_to_boxes(mask), mask_to_ball(mask)))
    return samples
