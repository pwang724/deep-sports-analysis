"""Detect, track and pose players in a video locally.

    python scripts/joints.py data/raw/clip.mp4 --start 60 --duration 20 --stride 2

See dsa.pose.tracking for the output format. For GPU runs use dsa/cloud/modal_app.py.
"""
from __future__ import annotations

import argparse
import json

import torch

from dsa.pose.backends import load_models
from dsa.pose.detectors import DETECTORS
from dsa.pose.tracking import TrackConfig, track_segment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--duration", type=float, default=None, help="seconds; default to end of video")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--detector", default="rfdetr", choices=sorted(DETECTORS))
    p.add_argument("--imgsz", type=int, default=1152, help="detector input size")
    p.add_argument("--threshold", type=float, default=0.4, help="detector confidence")
    p.add_argument("--vitpose", default="models/vitpose-plus-huge")
    p.add_argument("--out", default="output/joints")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    models = load_models(a.detector, a.imgsz, a.vitpose, a.device)
    summary = track_segment(a.video, a.start, a.duration, a.out, models, TrackConfig(stride=a.stride, threshold=a.threshold))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
