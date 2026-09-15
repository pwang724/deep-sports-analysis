"""Score both pose backends against hand-labelled tennis joints.

Dataset: Tennis Player Actions (Mendeley, CC BY 4.0). 2,000 images, one player
each, 18 OpenPose joints in COCO JSON. The first 17 are COCO order; the 18th is
the neck, which neither model predicts, so it is dropped.

Metrics, computed only on joints the labeller marked visible:
  OKS   COCO object keypoint similarity per instance, mean over images
  PCK@t fraction of joints within t * box_height of the label (t = 0.05, 0.10)
  per-joint median error in % of box height

RF-DETR runs on the full image; its detection is matched to the label by IoU.
ViTPose runs top-down on the labelled box, and also on RF-DETR's box so the
two are compared under the same detector.

    python src/eval_gt.py --root "data/tennis_player_actions/Tennis Player Actions Dataset for Human Pose Estimation"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from joints import COCO17, load_rfdetr, load_vitpose, run_vitpose

# COCO keypoint sigmas for OKS
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def oks(pred, gt, vis, area):
    d2 = ((pred - gt) ** 2).sum(1)
    e = d2 / (2 * (SIGMAS ** 2) * (area + np.spacing(1)))
    m = vis > 0
    return float(np.exp(-e[m]).mean()) if m.any() else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=None, help="images per action, for a quick run")
    p.add_argument("--vitpose", default="models/vitpose-plus-huge")
    p.add_argument("--out", default="output/eval_gt")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rf = load_rfdetr()
    vp = load_vitpose(a.vitpose, a.device)

    rows = []
    timing = {"rfdetr": 0.0, "vitpose_gtbox": 0.0, "vitpose_rfbox": 0.0, "n": 0}
    for ann_file in sorted((root / "annotations").glob("*.json")):
        action = ann_file.stem
        d = json.load(open(ann_file))
        imgs = {im["id"]: im for im in d["images"]}
        anns = d["annotations"][: a.limit] if a.limit else d["annotations"]
        for ann in anns:
            im = imgs[ann["image_id"]]
            bgr = cv2.imread(str(root / "images" / action / im["file_name"]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            k = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)[:17]
            gt, vis = k[:, :2], k[:, 2]
            bx, by, bw, bh = ann["bbox"]
            gt_box = np.array([bx, by, bx + bw, by + bh], dtype=float)
            area = bw * bh

            t = time.time()
            kp = rf.predict(rgb, threshold=0.3, include_source_image=False)
            timing["rfdetr"] += time.time() - t
            rf_xy = rf_sc = rf_box = None
            if len(kp.xy):
                ious = [iou(gt_box, b) for b in kp.data["xyxy"]]
                j = int(np.argmax(ious))
                if ious[j] > 0.3:
                    rf_xy, rf_sc, rf_box = kp.xy[j], kp.confidence[j], kp.data["xyxy"][j].astype(float)

            t = time.time()
            v_xy, v_sc = run_vitpose(*vp, a.device, rgb, gt_box[None])
            timing["vitpose_gtbox"] += time.time() - t
            v2_xy = v2_sc = None
            if rf_box is not None:
                t = time.time()
                v2_xy, v2_sc = run_vitpose(*vp, a.device, rgb, rf_box[None])
                timing["vitpose_rfbox"] += time.time() - t
            timing["n"] += 1

            for name, xy in [("rfdetr", rf_xy), ("vitpose_gtbox", v_xy[0]), ("vitpose_rfbox", None if v2_xy is None else v2_xy[0])]:
                if xy is None:
                    rows.append({"action": action, "image": im["file_name"], "backend": name, "detected": False, "oks": np.nan})
                    continue
                err = np.hypot(*(xy - gt).T) / bh
                r = {"action": action, "image": im["file_name"], "backend": name, "detected": True, "oks": oks(xy, gt, vis, area),
                     "pck05": float((err[vis > 0] < 0.05).mean()), "pck10": float((err[vis > 0] < 0.10).mean())}
                for i, jn in enumerate(COCO17):
                    r[f"{jn}_err"] = err[i] if vis[i] > 0 else np.nan
                rows.append(r)
            if timing["n"] % 100 == 0:
                print(f"{action} {timing['n']} images  rf {timing['rfdetr']/timing['n']:.2f}s  vp {timing['vitpose_gtbox']/timing['n']:.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_image.parquet", index=False)
    print(f"\nimages: {timing['n']}")
    summary = {}
    for b, g in df.groupby("backend"):
        det = g.detected.mean()
        gg = g[g.detected]
        s = {"detected": float(det), "mean_oks": float(gg.oks.mean()), "pck05": float(gg.pck05.mean()), "pck10": float(gg.pck10.mean()),
             "sec_per_image": timing[b] / max(timing["n"], 1)}
        summary[b] = s
        print(f"\n{b:14s} detected {det:5.1%}  mean OKS {s['mean_oks']:.3f}  PCK@0.05 {s['pck05']:.3f}  PCK@0.10 {s['pck10']:.3f}  {s['sec_per_image']:.2f}s/img")
    print("\nper-joint median error, % of box height:")
    print(f"{'joint':12s}" + "".join(f"{b:>15s}" for b in summary))
    for jn in COCO17:
        line = f"{jn:12s}"
        for b in summary:
            g = df[(df.backend == b) & df.detected]
            line += f"{100 * g[f'{jn}_err'].median():15.1f}"
        print(line)
    print("\nby action, mean OKS:")
    print(df[df.detected].pivot_table(index="action", columns="backend", values="oks", aggfunc="mean").round(3).to_string())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
