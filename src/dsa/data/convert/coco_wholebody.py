"""COCO-WholeBody val -> frames, people.

coco_wholebody_val_v1.0.json: COCO val2017 person annotations plus hand, face
and foot points. Kept: the box (xywh -> xyxy), the 17 body points and the 6
foot points, whose order is left big toe, left small toe, left heel, right
big toe, right small toe, right heel (the paper's; checked on the data: the
first three sit by the left ankle, heels closest). Hands and face are dropped.

COCO v = 0 means "not labelled" (it also covers points out of frame or too
hidden to guess), so 0 -> NaN; 1 (labelled, occluded) and 2 (visible) carry
over. Foot points are all NaN when foot_valid is false and use only 0 / 2.
Crowd regions (227) are skipped. People without any body points (4,425 small
or truncated people) are kept for their box. Images with no non-crowd person
get no rows.

Only a few images are on disk (fetched one by one as used); every label is
converted, with media val2017/<file_name>. Split: val.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.convert import frame_row, rel, run, xyxy
from dsa.data.paths import SOURCES

NAME = "coco_wholebody"
ROOT = SOURCES / "coco_wholebody"
FEET = [schema.KP[j] for j in schema.FEET]


def coco_kp(values, n: int) -> np.ndarray:
    kp = np.array(values, float).reshape(n, 3)
    kp[kp[:, 2] == 0] = np.nan
    return kp


def convert() -> dict[str, pd.DataFrame]:
    d = json.load(open(ROOT / "coco_wholebody_val_v1.0.json"))
    images = {im["id"]: im for im in d["images"]}
    frames, people, count = {}, [], {}
    for a in sorted(d["annotations"], key=lambda a: (a["image_id"], a["id"])):
        if a["iscrowd"]:
            continue
        im = images[a["image_id"]]
        media_id = im["file_name"].rsplit(".", 1)[0]
        if media_id not in frames:
            frames[media_id] = frame_row(NAME, media_id, -1, "val", rel(ROOT / "val2017" / im["file_name"]), np.nan,
                                         im["width"], im["height"])
        kp = schema.empty_kp()
        kp[:17] = coco_kp(a["keypoints"], 17)
        if a.get("foot_valid"):
            kp[FEET] = coco_kp(a["foot_kpts"], 6)
        k = count[media_id] = count.get(media_id, -1) + 1
        people.append({"sample": frames[media_id]["sample"], "person": k, "track": None, "box": xyxy(a["bbox"]),
                       "kp": schema.flat(kp), "stroke": None, "labeler": NAME})
    return {"frames": pd.DataFrame(list(frames.values())), "people": pd.DataFrame(people)}


if __name__ == "__main__":
    run(NAME)
