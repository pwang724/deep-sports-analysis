"""Benchmark person detectors against labelled frames, one detector per GPU container.

    modal run src/dsa/cloud/bench_detectors.py

Scores every candidate on TennisSegmentation (197 broadcast frames with both
players boxed) and prints recall at IoU 0.5 per player.
"""
import time
from pathlib import Path

import modal

from dsa.cloud.modal_app import GPU, MODELS_DIR, VOL, base_image, volume

DATASET_DIR = VOL / "datasets" / "tennis_segmentation"

# YOLO weights download on first use into the volume.
bench_image = (
    base_image.uv_pip_install("ultralytics==8.3.240")
    .env({"YOLO_CONFIG_DIR": str(MODELS_DIR / "yolo")})
    .add_local_python_source("dsa")
)

app = modal.App("deep-sports-bench-detectors", image=bench_image)

CANDIDATES = [
    ("rfdetr", {"size": "medium", "imgsz": 1152}),
    ("yolo", {"weights": "yolo11m.pt", "imgsz": 1280}),
]
THRESHOLDS = (0.25, 0.4)


def _load_detector(kind: str, kwargs: dict):
    import os

    from dsa.pose.detectors import make_detector

    os.makedirs(MODELS_DIR / "yolo", exist_ok=True)
    os.chdir(MODELS_DIR / "yolo")  # ultralytics downloads bare weight names into cwd
    return make_detector(kind, **kwargs)


@app.function(gpu=GPU, volumes={str(VOL): volume}, cpu=4, memory=16_384, timeout=60 * 60)
def bench_gt(kind: str, kwargs: dict) -> dict:
    """Run one detector over the labelled frames; return per-image boxes and timing."""
    from dsa.data.tennis_segmentation import load

    det = _load_detector(kind, kwargs)
    samples = load(DATASET_DIR)
    preds, elapsed = [], 0.0
    for i, s in enumerate(samples):
        t = time.time()
        preds.append(det.detect(s.image))
        if i >= 3:
            elapsed += time.time() - t
    return {"detector": det.name, "kind": kind, "kwargs": kwargs, "images": len(samples),
            "sec_per_image": elapsed / max(len(samples) - 3, 1), "gpu": GPU, "preds": preds}


@app.local_entrypoint()
def main(out: str = "output/bench_detectors"):
    """Score every candidate on the labelled frames and print recall at IoU 0.5."""
    import pandas as pd

    from dsa.data.tennis_segmentation import load
    from dsa.pose.bench import gt_report

    labels = [s.boxes for s in load(Path("data/tennis_segmentation"))]
    results = list(bench_gt.starmap([(k, kw) for k, kw in CANDIDATES]))
    runs = {r["detector"]: (r.pop("preds"), r) for r in results}

    table = gt_report(runs, labels, THRESHOLDS)
    out_dir = Path(out)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "tennis_segmentation.csv", index=False)
    for name, (preds, _) in runs.items():  # per-image boxes, so misses and extras can be inspected
        rows = [{"image": i, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": c}
                for i, (xyxy, conf) in enumerate(preds) for b, c in zip(xyxy, conf)]
        pd.DataFrame(rows, columns=["image", "x1", "y1", "x2", "y2", "conf"]).to_parquet(out_dir / "preds" / f"{name}.parquet", index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(runs)} detectors on TennisSegmentation, {len(labels)} labelled frames, IoU >= 0.5, {GPU}\n")
    print(table.round(3).to_string(index=False))
