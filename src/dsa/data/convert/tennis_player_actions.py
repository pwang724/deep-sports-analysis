"""Tennis Player Actions -> frames, people.

2,000 1280 x 720 JPEGs of one player each, 500 per action, with one COCO-style
file per action (annotations/<action>.json): a box and 18 OpenPose-style
points, the 17 COCO body points then the neck. COCO-Annotator flags: 2
visible, 1 labelled but occluded, 0 not placed (21 points in all), so 0 ->
NaN. The action becomes `stroke`: forehand, backhand, serve, ready (from
ready_position). Every label is kept, including 6 boxes under 4 px wide that
dsa.pose.eval_pose skips when scoring. Split: test (the phase-0 scoring set).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.convert import frame_row, rel, run, xyxy
from dsa.data.paths import SOURCES

NAME = "tennis_player_actions"
ROOT = SOURCES / "tennis_player_actions/Tennis Player Actions Dataset for Human Pose Estimation"
STROKE = {"forehand": "forehand", "backhand": "backhand", "serve": "serve", "ready_position": "ready"}
POINTS = [schema.KP[j] for j in schema.BODY + ["neck"]]


def convert() -> dict[str, pd.DataFrame]:
    frames, people = {}, []
    for action, stroke in STROKE.items():
        d = json.load(open(ROOT / "annotations" / f"{action}.json"))
        images = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = images[a["image_id"]]
            media_id = f"{action}/{im['file_name'].rsplit('.', 1)[0]}"
            frames[media_id] = frame_row(NAME, media_id, -1, "test", rel(ROOT / "images" / action / im["file_name"]),
                                         np.nan, im["width"], im["height"])
            src = np.array(a["keypoints"], float).reshape(18, 3)
            src[src[:, 2] == 0] = np.nan
            kp = schema.empty_kp()
            kp[POINTS] = src
            people.append({"sample": frames[media_id]["sample"], "person": 0, "track": None, "box": xyxy(a["bbox"]),
                           "kp": schema.flat(kp), "stroke": stroke, "labeler": NAME})
    return {"frames": pd.DataFrame(list(frames.values())), "people": pd.DataFrame(people)}


if __name__ == "__main__":
    run(NAME)
