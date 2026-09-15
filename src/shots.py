"""Find the end-on court segments in a broadcast clip.

The main broadcast camera is fixed behind the baseline, so every rally frame
looks nearly identical at thumbnail scale. Compare each sampled frame to one
reference end-on frame; runs of matching frames are rally segments.

    python src/shots.py data/raw/clip.mp4 --ref-time 0 --out output/shots.json
"""
from __future__ import annotations

import argparse
import json
import subprocess

import cv2
import numpy as np

SMALL = (48, 27)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--ref-time", type=float, default=0.0, help="seconds of a known end-on frame")
    p.add_argument("--sample-fps", type=float, default=4.0)
    p.add_argument("--threshold", type=float, default=0.12, help="mean abs diff below this = end-on")
    p.add_argument("--min-len", type=float, default=2.0, help="drop segments shorter than this (s)")
    p.add_argument("--out", default="output/shots.json")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Sequential decode at thumbnail size via ffmpeg: seeking per frame is far too slow on 1080p60.
    w, h = SMALL
    cmd = ["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"fps={a.sample_fps},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255
    times = [i / a.sample_fps for i in range(len(frames))]
    ref_t = frames[int(a.ref_time * a.sample_fps)]
    scores = [float(np.abs(f - ref_t).mean()) for f in frames]

    on = np.array(scores) < a.threshold
    segs = []
    start = None
    for t, flag in zip(times, on):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, times[-1]))
    segs = [(s, e) for s, e in segs if e - s >= a.min_len]

    out = {"video": a.video, "fps": fps, "ref_time": a.ref_time, "threshold": a.threshold,
           "segments": [{"start": round(s, 2), "end": round(e, 2), "len": round(e - s, 2)} for s, e in segs],
           "total_on_seconds": round(sum(e - s for s, e in segs), 1), "clip_seconds": round(n / fps, 1)}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{len(segs)} segments, {out['total_on_seconds']}s end-on of {out['clip_seconds']}s")
    for s in out["segments"][:40]:
        print(f"  {s['start']:7.2f} -> {s['end']:7.2f}  ({s['len']:.1f}s)")
    # score histogram so the threshold can be sanity-checked
    h, edges = np.histogram(scores, bins=12)
    print("score histogram:", " ".join(f"{e:.2f}:{c}" for e, c in zip(edges, h)))


if __name__ == "__main__":
    main()
