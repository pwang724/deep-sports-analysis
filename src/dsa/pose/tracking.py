"""Detect, track and pose every person over a stretch of video.

Per processed frame: the detector finds people, ByteTrack gives each a
stable ID, ViTPose estimates 17 joints per tracked box.

Output per run: `joints.parquet`, one row per (frame, track) with the box,
detection confidence and joints; `annotated.mp4` with skeletons drawn;
`summary.json` with frame counts and seconds per frame per stage.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from dsa.pose.backends import PoseModels, run_vitpose
from dsa.pose.skeleton import draw_pose, joints_to_columns
from dsa.video import H264Writer

SKELETON_COLOR = (255, 80, 200)  # BGR


@dataclass
class TrackConfig:
    stride: int = 2            # process every Nth frame
    threshold: float = 0.4     # detector confidence


def track_segment(video: str | Path, start: float, duration: float | None, out_dir: str | Path,
                  models: PoseModels, cfg: TrackConfig, track_id_offset: int = 0) -> dict:
    """Track players from `start` for `duration` seconds and write outputs to `out_dir`.

    `track_id_offset` keeps ByteTrack IDs unique when segments of one video
    are processed independently. Returns the summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    f0 = int(start * fps)
    f1 = int((start + duration) * fps) if duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    tracker = sv.ByteTrack(frame_rate=fps / cfg.stride)
    writer = H264Writer(out_dir / "annotated.mp4", fps / cfg.stride, size)
    rows: list[dict] = []
    timing = {"detect": 0.0, "vitpose": 0.0}
    frames = 0

    for fi in range(f0, f1):
        ok, bgr = cap.read()
        if not ok:
            break
        if (fi - f0) % cfg.stride:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t = time.time()
        xyxy, conf = models.detector.detect(rgb)
        timing["detect"] += time.time() - t
        keep = conf >= cfg.threshold
        xyxy, conf = xyxy[keep], conf[keep]

        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(xyxy), dtype=int)) if len(xyxy) else sv.Detections.empty()
        det = tracker.update_with_detections(det)

        t = time.time()
        vxy, vsc = run_vitpose(*models.vitpose, models.device, rgb, det.xyxy)
        timing["vitpose"] += time.time() - t

        for j in range(len(det)):
            tid = int(det.tracker_id[j]) + track_id_offset
            box = det.xyxy[j]
            rows.append({"frame": fi, "t": fi / fps, "track_id": tid, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                         "det_conf": float(det.confidence[j]), **joints_to_columns(vxy[j], vsc[j])})
            draw_pose(bgr, vxy[j], vsc[j], SKELETON_COLOR, tid, box)

        writer.write(bgr)
        frames += 1
        if frames % 50 == 0:
            print(f"frame {fi}  people {len(det)}  detect {timing['detect'] / frames:.3f}s/f  "
                  f"vitpose {timing['vitpose'] / frames:.3f}s/f", flush=True)

    writer.release()
    cap.release()

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "joints.parquet", index=False)
    summary = {
        "video": str(video), "start": start, "duration": duration, "fps": fps,
        "detector": models.detector.name, "config": asdict(cfg),
        "frames_processed": frames, "rows": len(df), "tracks": int(df.track_id.nunique()) if len(df) else 0,
        "sec_per_frame": {k: v / max(frames, 1) for k, v in timing.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
