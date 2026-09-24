"""TennisCourtDetector -> frames, court.

data/data_train.json and data_val.json: {id, kps (14 x 2), metric} per 1280 x
720 broadcast frame, kps already in COURT_POINTS order (its court_reference.py).
`metric` is not a label and is dropped. Every point is labelled, including
points outside the frame (x from -880 to 1729): those keep their projected
x, y with vis 0 (not in view); points inside the frame get vis 2. There is no
occlusion flag, so a point hidden behind a player is still vis 2.

Only 100 val images are unpacked (the rest are in tennis_court_det_dataset.zip);
media is recorded as data/images/<id>.png regardless.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dsa.data.convert import frame_row, rel, run
from dsa.data.paths import SOURCES

NAME = "tennis_court_detector"
ROOT = SOURCES / "tennis_court_detector/data"
W, H = 1280, 720


def convert() -> dict[str, pd.DataFrame]:
    frames, court = [], []
    for split in ("train", "val"):
        for item in json.load(open(ROOT / f"data_{split}.json")):
            row = frame_row(NAME, item["id"], -1, split, rel(ROOT / "images" / f"{item['id']}.png"), np.nan, W, H)
            frames.append(row)
            xy = np.asarray(item["kps"], float)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
            court.append({"sample": row["sample"], "kp": np.c_[xy, np.where(inside, 2.0, 0.0)].ravel().tolist(),
                          "labeler": NAME})
    return {"frames": pd.DataFrame(frames), "court": pd.DataFrame(court)}


if __name__ == "__main__":
    run(NAME)
