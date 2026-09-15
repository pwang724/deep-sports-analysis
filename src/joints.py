"""Track player joints in a video with two backends and write them to a table.

Backend A: RF-DETR Keypoint Preview. One pass per frame gives every person's box
and 17 COCO joints. ByteTrack assigns stable IDs across frames.

Backend B: ViTPose-Plus on the same tracked boxes. Top-down, so it only ever sees
one person crop at a time, which is what it was trained on.

Output: one parquet in long format, one row per (frame, track, backend), with
box, detection confidence and 17 joints (x, y, conf). Plus an annotated mp4 so
the numbers can be eyeballed.

    python src/joints.py data/raw/clip.mp4 --start 60 --duration 20 --stride 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch

COCO17 = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee",
    "r_knee", "l_ankle", "r_ankle",
]
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]
COLORS = {"rfdetr": (0, 200, 255), "vitpose": (255, 80, 200)}  # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--duration", type=float, default=None, help="seconds")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--threshold", type=float, default=0.4, help="RF-DETR box confidence")
    p.add_argument("--backends", default="rfdetr,vitpose")
    p.add_argument("--vitpose", default="models/vitpose-plus-huge")
    p.add_argument("--out", default="output/joints")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def load_rfdetr():
    from rfdetr import RFDETRKeypointPreview

    return RFDETRKeypointPreview()


def load_vitpose(path: str, device: str):
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    proc = AutoProcessor.from_pretrained(path)
    model = VitPoseForPoseEstimation.from_pretrained(path).to(device).eval()
    return proc, model


@torch.inference_mode()
def run_vitpose(proc, model, device: str, rgb: np.ndarray, xyxy: np.ndarray):
    """Return (N, 17, 2) xy and (N, 17) scores for boxes xyxy on one RGB frame."""
    if len(xyxy) == 0:
        return np.zeros((0, 17, 2)), np.zeros((0, 17))
    xywh = xyxy.copy().astype(float)
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    inputs = proc(rgb, boxes=[xywh.tolist()], return_tensors="pt").to(device)
    # ViTPose-Plus is mixture-of-experts over datasets; index 0 is COCO.
    dataset_index = torch.zeros(len(xywh), dtype=torch.long, device=device)
    out = model(**inputs, dataset_index=dataset_index)
    res = proc.post_process_pose_estimation(out, boxes=[xywh.tolist()])[0]
    xy = np.stack([r["keypoints"].cpu().numpy() for r in res])
    sc = np.stack([r["scores"].cpu().numpy() for r in res])
    return xy, sc


def draw(frame: np.ndarray, xy: np.ndarray, sc: np.ndarray, color, track_id, box, thr=0.3):
    for a, b in SKELETON:
        if sc[a] > thr and sc[b] > thr:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2)
    for k in range(17):
        if sc[k] > thr:
            cv2.circle(frame, tuple(xy[k].astype(int)), 3, color, -1)
    x1, y1 = int(box[0]), int(box[1])
    cv2.putText(frame, f"id {track_id}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    a = parse_args()
    backends = a.backends.split(",")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(a.start * fps)
    f1 = int((a.start + a.duration) * fps) if a.duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    rf = load_rfdetr()
    vp = load_vitpose(a.vitpose, a.device) if "vitpose" in backends else None
    tracker = sv.ByteTrack(frame_rate=fps / a.stride)

    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (W, H))
    rows: list[dict] = []
    timing = {"rfdetr": 0.0, "vitpose": 0.0, "frames": 0}

    for fi in range(f0, f1):
        ok, bgr = cap.read()
        if not ok:
            break
        if (fi - f0) % a.stride:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t = time.time()
        kp = rf.predict(rgb, threshold=a.threshold, include_source_image=False)
        timing["rfdetr"] += time.time() - t

        n = len(kp.xy)
        if n:
            det = sv.Detections(
                xyxy=kp.data["xyxy"].astype(float),
                confidence=np.asarray(kp.detection_confidence, dtype=float) if kp.detection_confidence is not None else None,
                class_id=np.asarray(kp.class_id),
                data={"idx": np.arange(n)},
            )
            det = tracker.update_with_detections(det)
        else:
            det = sv.Detections.empty()
            tracker.update_with_detections(det)

        if len(det) and vp is not None:
            t = time.time()
            vxy, vsc = run_vitpose(*vp, a.device, rgb, det.xyxy)
            timing["vitpose"] += time.time() - t

        for j in range(len(det)):
            src = int(det.data["idx"][j])
            tid = int(det.tracker_id[j])
            box = det.xyxy[j]
            base = {"frame": fi, "t": fi / fps, "track_id": tid, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "det_conf": float(det.confidence[j]) if det.confidence is not None else np.nan}
            xy, sc = kp.xy[src], kp.confidence[src]
            rows.append({**base, "backend": "rfdetr", **{f"{COCO17[k]}_{c}": v for k in range(17) for c, v in zip("xyc", (xy[k, 0], xy[k, 1], sc[k]))}})
            draw(bgr, xy, sc, COLORS["rfdetr"], tid, box)
            if vp is not None:
                rows.append({**base, "backend": "vitpose", **{f"{COCO17[k]}_{c}": v for k in range(17) for c, v in zip("xyc", (vxy[j, k, 0], vxy[j, k, 1], vsc[j, k]))}})
                draw(bgr, vxy[j], vsc[j], COLORS["vitpose"], tid, box)

        writer.write(bgr)
        timing["frames"] += 1
        if timing["frames"] % 50 == 0:
            print(f"frame {fi}  people {len(det)}  rfdetr {timing['rfdetr']/timing['frames']:.2f}s/f  vitpose {timing['vitpose']/timing['frames']:.2f}s/f", flush=True)

    writer.release()
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "joints.parquet", index=False)
    summary = {"video": a.video, "start": a.start, "duration": a.duration, "stride": a.stride, "fps": fps,
               "frames_processed": timing["frames"], "rows": len(df), "tracks": int(df.track_id.nunique()) if len(df) else 0,
               "sec_per_frame": {k: timing[k] / max(timing["frames"], 1) for k in ("rfdetr", "vitpose")}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
