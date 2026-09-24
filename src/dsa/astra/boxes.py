"""Astra as a person detector, scored on TennisSegmentation.

TennisSegmentation has 197 broadcast frames (1280 x 720) with both players
labelled; the far player is about 80 px tall. Astra sees the whole frame and
boxes every person. A player counts as found when some Astra box overlaps its
label with IoU >= 0.5 (dsa.pose.bench). RF-DETR Medium's stored boxes from
the detector benchmark (output/bench_detectors/preds) are scored on the same
frames. Other people (ball kids, umpire) are unlabelled, so boxes per frame is
a rough count, not a precision.

    python -m dsa.astra.boxes --n 100
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import ask
from dsa.data.paths import SOURCES
from dsa.data.tennis_segmentation import load
from dsa.pose.bench import gt_recall

SCHEMA = {
    "type": "object",
    "properties": {"people": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "number"} for k in ("x1", "y1", "x2", "y2")},
        "required": ["x1", "y1", "x2", "y2"], "additionalProperties": False}}},
    "required": ["people"],
    "additionalProperties": False,
}


def prompt(w: int, h: int) -> str:
    return (
        f"The attached image is {w} x {h} pixels, a frame of tennis. Draw a tight bounding box around "
        "every person visible: both players, however small or far away, and anyone else on or beside "
        "the court (ball kids, umpire, line judges); spectators in the stands can be skipped. x1, y1 is "
        "the top-left corner and x2, y2 the bottom-right, in pixels from the top-left of the image. "
        "Include the whole body but not the racket. Look at the image directly; do not run any commands."
    )


def label_one(i: int, image: np.ndarray, frames_dir: Path, cache_dir: Path) -> tuple[int, np.ndarray, float]:
    path = frames_dir / f"{i:03d}.png"
    if not path.exists():
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    h, w = image.shape[:2]
    answer, sec = ask(prompt(w, h), [path], SCHEMA, cache_dir)
    boxes = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in answer["people"]], float).reshape(-1, 4)
    return i, boxes, sec


def misses_sheet(samples, preds: dict, out: Path, n: int = 8) -> None:
    """Frames where Astra missed a player at IoU 0.5: labels green, Astra red."""
    from dsa.pose.bench import iou

    tiles = []
    for s in samples:
        boxes = preds[s.index]
        if all(len(boxes) and iou(gt, boxes).max() >= 0.5 for gt in s.boxes.values()):
            continue
        bgr = cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR)
        for b, color in [(b, (0, 200, 0)) for b in s.boxes.values()] + [(b, (0, 0, 255)) for b in boxes]:
            cv2.rectangle(bgr, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), color, 2)
        tiles.append(cv2.resize(bgr, (640, 360)))
        if len(tiles) == n:
            break
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(SOURCES / "tennis_segmentation"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rfdetr", default="output/bench_detectors/preds/rfdetr-medium@1152.parquet")
    p.add_argument("--out", default="output/astra_eval/boxes")
    a = p.parse_args()

    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    all_samples = load(a.root)
    rng = np.random.default_rng(0)
    samples = [all_samples[i] for i in sorted(rng.choice(len(all_samples), min(a.n, len(all_samples)), replace=False))]
    print(f"{len(samples)} frames", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda s: label_one(s.index, s.image, out / "frames", out / "cache"), samples))
    astra = {i: boxes for i, boxes, _ in results}
    secs = [sec for *_, sec in results if sec > 0]
    pd.DataFrame([{"image": i, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for i, boxes in astra.items() for b in boxes]
                 ).to_parquet(out / "boxes.parquet", index=False)
    misses_sheet(samples, astra, out / "misses.jpg")

    labels = [s.boxes for s in samples]
    rf = pd.read_parquet(a.rfdetr)
    rows = [{"labeler": "astra", **gt_recall([(astra[s.index], np.ones(len(astra[s.index]))) for s in samples], labels, 0.5)}]
    for t in (0.25, 0.4):
        preds = [(g[["x1", "y1", "x2", "y2"]].to_numpy(), g.conf.to_numpy())
                 for g in (rf[rf.image == s.index] for s in samples)]
        rows.append({"labeler": "rfdetr-medium", **gt_recall(preds, labels, t)})
    table = pd.DataFrame(rows).rename(columns={"top_recall": "far_recall", "top_iou": "far_iou",
                                               "bottom_recall": "near_recall", "bottom_iou": "near_iou"})
    table.loc[table.labeler == "astra", "threshold"] = np.nan
    table.to_csv(out / "summary.csv", index=False)
    (out / "sec_per_call.json").write_text(json.dumps(float(np.mean(secs)) if secs else None))
    print(f"\n{len(samples)} frames, IoU >= 0.5\n")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
