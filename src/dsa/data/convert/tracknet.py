"""TrackNet tennis -> frames, ball, events.

TrackNet/Dataset/game<g>/Clip<c>/: numbered JPEGs (1280 x 720, 30 fps) and a
Label.csv with one row per image: visibility 0 (no ball), 1 (visible), 2 (hard
to see), 3 (occluded), the ball centre, and status 0 flying / 1 hit / 2 bounce
(blank when there is no ball). 10 games, 95 clips, 19,835 frames.

Each clip is treated as a video: media is the clip folder, frame the image
number (<media>/<frame:04d>.jpg). Visibility 1 and 2 give a visible ball.
Visibility 3 (82 frames) is a ball hidden behind a player or the net whose
position the labeller still marked: stored as visible = False with its x, y
kept, so it counts as "not visible" for the visibility head and its position
is there for anything that wants the trajectory. Visibility 0 is visible =
False with no position.

Status 1 is any racket contact, serves included, so it becomes type "shot";
status 2 is "bounce". The hitter / bounce side is not labelled. Status is
given on every frame, so each clip is a span covering "shot,bounce".

Split: games 1-7 train, 8-10 test (WASB's split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsa.data.convert import events_frame, frame_row, rel, run, span
from dsa.data.paths import SOURCES

NAME = "tracknet"
ROOT = SOURCES / "tracknet/TrackNet/Dataset"
FPS, W, H = 30.0, 1280, 720
EVENT = {1: "shot", 2: "bounce"}


def convert() -> dict[str, pd.DataFrame]:
    frames, ball, events, spans = [], [], [], []
    for clip in sorted(p for p in ROOT.glob("game*/Clip*") if p.is_dir() and not p.name.startswith("._")):
        game = int(clip.parent.name.removeprefix("game"))
        split = "train" if game <= 7 else "test"
        media_id, media = f"{clip.parent.name}/{clip.name}", rel(clip)
        labels = pd.read_csv(clip / "Label.csv")
        spans.append(span(NAME, split, media, 0, len(labels), FPS, "shot,bounce"))
        for r in labels.itertuples(index=False):
            frame = int(r[0].removesuffix(".jpg"))
            vis, x, y, status = int(r[1]), float(r[2]), float(r[3]), r[4]
            row = frame_row(NAME, media_id, frame, split, media, FPS, W, H)
            frames.append(row)
            ball.append({"sample": row["sample"], "x": x if vis else np.nan, "y": y if vis else np.nan,
                         "visible": vis in (1, 2), "labeler": NAME})
            if not pd.isna(status) and int(status) in EVENT:
                events.append({"source": NAME, "split": split, "media": media, "frame": frame, "fps": FPS,
                               "type": EVENT[int(status)], "labeler": NAME})
    return {"frames": pd.DataFrame(frames), "ball": pd.DataFrame(ball), "events": events_frame(events),
            "spans": pd.DataFrame(spans)}


if __name__ == "__main__":
    run(NAME)
