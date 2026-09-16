"""Score the tracker (RF-DETR + ByteTrack) on a clip: coverage, identity stability, speed, cost.

    modal run src/dsa/cloud/bench_tracking.py --start 320.5 --duration 34.5

Runs per segment on its own GPU container and returns per-frame boxes with
track ids; the local side scores them with dsa.pose.track_metrics. Use it to
tune ByteTrack or to compare a candidate tracker against the current one.
"""
import io
import time
from pathlib import Path

import modal

from dsa.cloud.modal_app import GPU, VIDEOS_DIR, VOL, image, volume

GPU_USD_PER_SEC = {"L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583}

app = modal.App("deep-sports-bench-tracking", image=image)


def _read_frames(video_name: str, start: float, duration: float, stride: int):
    """Decode every stride-th frame of the segment as RGB. Returns (frame indices, frames, fps, (W, H))."""
    import cv2

    cap = cv2.VideoCapture(str(VIDEOS_DIR / video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    f0, f1 = int(start * fps), int((start + duration) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    idx, frames = [], []
    for fi in range(f0, f1):
        ok, bgr = cap.read()
        if not ok:
            break
        if (fi - f0) % stride == 0:
            idx.append(fi)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return idx, frames, fps, size


def _pack(rows, idx, size, seconds, name):
    import pandas as pd

    buf = io.BytesIO()
    pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]).to_parquet(buf, index=False)
    return {"tracker": name, "frames": idx, "size": size, "seconds": seconds, "rows": buf.getvalue()}


@app.function(gpu=GPU, volumes={str(VOL): volume}, cpu=4, memory=16_384, timeout=60 * 60)
def run_rfdetr(video_name: str, start: float, duration: float, stride: int, threshold: float) -> dict:
    """RF-DETR Medium boxes linked by ByteTrack; timing covers detection and tracking."""
    import numpy as np
    import supervision as sv

    from dsa.pose.detectors import make_detector

    det = make_detector("rfdetr", imgsz=1152)
    idx, frames, fps, size = _read_frames(video_name, start, duration, stride)
    tracker = sv.ByteTrack(frame_rate=fps / stride)
    rows, t0 = [], time.time()
    for fi, rgb in zip(idx, frames):
        xyxy, conf = det.detect(rgb)
        keep = conf >= threshold
        d = sv.Detections(xyxy=xyxy[keep], confidence=conf[keep], class_id=np.zeros(int(keep.sum()), dtype=int)) if keep.any() else sv.Detections.empty()
        d = tracker.update_with_detections(d)
        rows += [dict(frame=fi, track_id=int(t), x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=float(c)) for b, c, t in zip(d.xyxy, d.confidence, d.tracker_id)]
    return _pack(rows, idx, size, time.time() - t0, "rfdetr+bytetrack")


@app.local_entrypoint()
def main(video: str = "uso2026_final_highlights.h264.mp4", start: float = 320.5, duration: float = 34.5,
         segment_len: float = 15.0, stride: int = 2, threshold: float = 0.4, out: str = "output/bench_tracking"):
    import pandas as pd

    from dsa.pose.segments import plan_segments
    from dsa.pose.track_metrics import tracker_report

    segs = plan_segments(start, duration, segment_len)
    rf = run_rfdetr.starmap([(video, s.start, s.duration, stride, threshold) for s in segs])

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for results in (list(rf),):
        # Offset ids per segment so both trackers are scored the same way; identity resets at segment edges for both.
        parts, frames = [], []
        for k, r in enumerate(results):
            df = pd.read_parquet(io.BytesIO(r["rows"]))
            df["track_id"] += k * 100_000
            parts.append(df)
            frames += r["frames"]
        rows = pd.concat(parts, ignore_index=True)
        W, H = results[0]["size"]
        name = results[0]["tracker"]
        rows.to_parquet(out_dir / f"{name}.parquet", index=False)
        rep = tracker_report(rows, frames, W, H)
        sec_per_frame = sum(r["seconds"] for r in results) / max(len(frames), 1)
        video_fps = len(frames) / duration
        usd_per_min = sec_per_frame * video_fps * 60 * GPU_USD_PER_SEC[GPU[0]]
        table.append({"tracker": name, **rep, "sec_per_frame": sec_per_frame, "usd_per_video_min": usd_per_min})

    t = pd.DataFrame(table)
    t.to_csv(out_dir / "report.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n{video} [{start}s, +{duration}s], stride {stride}, {len(segs)} segments, {GPU[0]}\n")
    print(t.round(3).to_string(index=False))
