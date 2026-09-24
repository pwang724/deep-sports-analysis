"""RacketVision tennis -> frames, rackets, ball.

Labels for every rally are small files under tennis/all/<match>/:
  racket/<rally>/<frame>.json  [{bbox_xywh, keypoints 5 x 3}] (no category
                               field in the tennis subset); a file exists only
                               for frames with at least one racket
  csv/<rally>_ball.csv         Frame, Visibility (0 / 1), X, Y
Frames index tennis/videos/<match>_<rally>.mp4 (1920 x 1080, 10 s, fps varies
by match: 25, 29.97 or 60). Every racket frame also has a ball row; about
two thirds of ball frames have no racket file. Frames with neither are not
labelled.

Racket points are (top, bottom of head, handle, left, right), the order of
schema.RACKET. Their flag is 1 (labelled) or 0 with x, y = 0, 0 (not placed:
hidden or off-frame, the source does not say which), so 1 -> vis 2 and 0 ->
NaN. Ball Visibility 0 has X, Y = 0, 0: visible = False with no position.

Labels are on disk for all 431 rallies in tennis/info/{train,val,test}.json
(350 train, 38 val, 43 test; one rally per match). Only 3 videos are local
(matches 1, 10, 100); the whole tennis subset, videos included, is on the
Modal volume (/vol/datasets/racketvision, dsa.cloud.train_ball.fetch_rv).
Media paths are recorded either way; fps is read from the video when it is on
disk, else NaN (the volume's frames720/index.csv has every rally's fps).
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from dsa.data.convert import frame_row, rel, run, xyxy
from dsa.data.paths import SOURCES

NAME = "racketvision"
ROOT = SOURCES / "racketvision/tennis"
W, H = 1920, 1080


def video_fps(path) -> float:
    if not path.exists():
        return np.nan
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or np.nan


def convert() -> dict[str, pd.DataFrame]:
    splits = {tuple(mr): s for s in ("train", "val", "test") for mr in json.load(open(ROOT / "info" / f"{s}.json"))}
    frames, rackets, ball = [], [], []
    for (match, rally), split in sorted(splits.items()):
        csv = ROOT / "all" / match / "csv" / f"{rally}_ball.csv"
        if not csv.exists():
            continue
        media_id = f"{match}_{rally}"
        video = ROOT / "videos" / f"{media_id}.mp4"
        media, fps = rel(video), video_fps(video)
        balls = pd.read_csv(csv).set_index("Frame")
        racket_files = {int(p.stem): p for p in (ROOT / "all" / match / "racket" / rally).glob("*.json")
                        if not p.name.startswith("._")}
        for frame in sorted(set(balls.index) | set(racket_files)):
            row = frame_row(NAME, media_id, int(frame), split, media, fps, W, H)
            frames.append(row)
            if frame in balls.index:
                b = balls.loc[frame]
                vis = bool(b.Visibility)
                ball.append({"sample": row["sample"], "x": float(b.X) if vis else np.nan,
                             "y": float(b.Y) if vis else np.nan, "visible": vis, "labeler": NAME})
            if frame in racket_files:
                for k, r in enumerate(json.load(open(racket_files[frame]))):
                    kp = np.array(r["keypoints"], float)
                    kp[kp[:, 2] == 0] = np.nan
                    kp[:, 2] *= 2
                    rackets.append({"sample": row["sample"], "racket": k, "box": xyxy(r["bbox_xywh"]),
                                    "kp": kp.ravel().tolist(), "person": np.nan, "labeler": NAME})
    return {"frames": pd.DataFrame(frames), "rackets": pd.DataFrame(rackets), "ball": pd.DataFrame(ball)}


if __name__ == "__main__":
    run(NAME)
